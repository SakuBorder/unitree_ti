import numpy as np
import os
import yaml
from tqdm import tqdm
import os.path as osp

from amp_rsl_rl.utils import torch_utils
import joblib
import torch
import torch.multiprocessing as mp
import copy
import gc
from amp_rsl_rl.utils.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from scipy.spatial.transform import Rotation as sRot
import random
from amp_rsl_rl.utils.flags import flags
from amp_rsl_rl.utils.motion_lib_base import MotionLibBase, DeviceCache, compute_motion_dof_vels, FixHeightMode
from amp_rsl_rl.utils.torch_humanoid_batch import Humanoid_Batch
from easydict import EasyDict

def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    
    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


class MotionLibTaihu(MotionLibBase):

    def __init__(self, motion_file, device, fix_height=FixHeightMode.no_fix, masterfoot_conifg=None, 
                 min_length=-1, im_eval=False, multi_thread=True, extend_hand=True, extend_head=True, 
                 mjcf_file="resources/robots/taihu/ti.xml", sim_timestep=1/50):

        super().__init__(motion_file=motion_file, device=device, fix_height=fix_height, 
                        masterfoot_conifg=masterfoot_conifg, min_length=min_length, 
                        im_eval=im_eval, multi_thread=multi_thread, sim_timestep=sim_timestep)
        
        cfg = EasyDict({
            'asset': EasyDict({'assetFileName': mjcf_file}),
            'extend_config': []
        })
        self.mesh_parsers = Humanoid_Batch(cfg)
        self.isaac_gym_joint_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11]
        
        self.use_joint_mapping = True
        self.debug_dof = False
        self.sign_flip_set = {2,4,6,7,8,9} 
        
        return
    
    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        
        with torch.no_grad():
            device = pose_aa.device
            B, seq_len = pose_aa.shape[:2]
            
            motion_res = mesh_parsers.fk_batch(pose_aa, trans, return_full=True)
            
            left_ankle_pos = motion_res['global_translation'][:, :, 6]  # L_ANKLE_R_S
            right_ankle_pos = motion_res['global_translation'][:, :, 12]  # R_ANKLE_R_S
            
            foot_height_offset = 0.1
            left_foot_height = left_ankle_pos[:, :, 2] - foot_height_offset
            right_foot_height = right_ankle_pos[:, :, 2] - foot_height_offset
            
            min_foot_height = torch.min(left_foot_height, right_foot_height)
            
            if fix_height_mode == FixHeightMode.full_fix:
                height_adjustment = -min_foot_height
            elif fix_height_mode == FixHeightMode.ankle_fix:
                target_ankle_height = 0.1
                min_ankle_height = torch.min(left_ankle_pos[:, :, 2], right_ankle_pos[:, :, 2])
                height_adjustment = target_ankle_height - min_ankle_height
            else:
                height_adjustment = torch.zeros_like(min_foot_height)
            
            trans_fixed = trans.clone()
            trans_fixed[:, :, 2] += height_adjustment
            
            diff_fix = height_adjustment.mean().item()
            
        return trans_fixed, diff_fix
        
    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1, target_heading=None):

        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        
        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        torch.cuda.empty_cache()
        gc.collect()

        total_len = 0.0
        self.num_joints = len(skeleton_trees[0].node_names)
        num_motion_to_load = len(skeleton_trees)

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(len(skeleton_trees)) + start_idx, self._num_unique_motions).to(self._device)

        self._curr_motion_ids = sample_idxes
        self.one_hot_motions = torch.nn.functional.one_hot(self._curr_motion_ids, num_classes=self._num_unique_motions).to(self._device)
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        print("\n****************************** Current motion keys ******************************")
        print("Sampling motion:", sample_idxes[:30])
        if len(self.curr_motion_keys) < 100:
            print(self.curr_motion_keys)
        else:
            print(self.curr_motion_keys[:30], ".....")
        print("*********************************************************************************\n")

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        mp.set_sharing_strategy('file_descriptor')

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = 1 
        # num_jobs = min(mp.cpu_count(), 64)

        # if num_jobs <= 8 or not self.multi_thread:
        #     num_jobs = 1
        # if flags.debug:
        #     num_jobs = 1
        
        res_acc = {}
        jobs = motion_data_list
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))

        jobs = [(ids[i:i + chunk], jobs[i:i + chunk], skeleton_trees[i:i + chunk], 
                gender_betas[i:i + chunk], self.fix_height, self.mesh_parsers, 
                self._masterfoot_conifg, target_heading, max_len) for i in range(0, len(jobs), chunk)]
        
        job_args = [jobs[i] for i in range(len(jobs))]
        for i in range(1, len(jobs)):
            worker_args = (*job_args[i], queue, i)
            worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
            worker.start()
        res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))

        for i in tqdm(range(len(jobs) - 1)):
            res = queue.get()
            res_acc.update(res)

        for f in tqdm(range(len(res_acc))):
            motion_file_data, curr_motion = res_acc[f]
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            
            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            del curr_motion
            
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)
        self._num_motions = len(motions)

        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        
        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
            self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
            self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
            self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)
        
        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)
            print(f"DOF position data loaded: {self.dof_pos.shape}")
            
            # 调试输出原始DOF数据
            if self.debug_dof:
                print(f"Raw DOF sample: {self.dof_pos[0, :10]}")
        
        if flags.real_traj:
            self.q_gts = torch.cat(self.q_gts, dim=0).float().to(self._device)
            self.q_grs = torch.cat(self.q_grs, dim=0).float().to(self._device)
            self.q_gavs = torch.cat(self.q_gavs, dim=0).float().to(self._device)
            self.q_gvs = torch.cat(self.q_gvs, dim=0).float().to(self._device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = self.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        print(f"Taihu robot with {self.num_joints} joints ({self.num_joints} DOF)")
        return motions

    def apply_isaac_gym_joint_mapping(self, dof_pos):
        if not self.use_joint_mapping:
            return dof_pos

        mapped_dof = torch.zeros_like(dof_pos)

        # if dof_pos.shape[-1] >= 30 and len(self.isaac_gym_joint_mapping) == 30:
        for i, mapping_idx in enumerate(self.isaac_gym_joint_mapping):
            if i < mapped_dof.shape[-1] and mapping_idx < dof_pos.shape[-1]:
                val = dof_pos[:, mapping_idx]         

                if i in self.sign_flip_set:
                    val = -val

                mapped_dof[:, i] = val               
        return mapped_dof
    # else:
    #     print(f"Warning: Cannot apply mapping. DOF shape: {dof_pos.shape}, mapping len: {len(self.isaac_gym_joint_mapping)}")
    #     return dof_pos

    def get_motion_state(self, motion_ids, motion_times, offset=None):

        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        if hasattr(self, 'dof_pos'):
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
        else:
            local_rot0 = self.lrs[f0l]
            local_rot1 = self.lrs[f1l]
            
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, 
                rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        
        if hasattr(self, 'dof_pos'):
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
            # import ipdb; ipdb.set_trace()

            dof_pos = self.apply_isaac_gym_joint_mapping(dof_pos)
            
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof_smpl(local_rot)
            
            dof_pos = self.apply_isaac_gym_joint_mapping(dof_pos)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        

        root_rot = rb_rot[..., 0, :].clone()  
        

        # if root_rot.shape[-1] == 4:
        #     root_rot = root_rot[:, [3, 0, 1, 2]]  # (x,y,z,w) -> (w,x,y,z)
        
        return_dict = {}
        
        if hasattr(self, 'gts_t'):
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            
            rg_rot_t = torch_utils.slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
            
            return_dict['rg_pos_t'] = rg_pos_t
            return_dict['rg_rot_t'] = rg_rot_t
            return_dict['body_vel_t'] = body_vel_t
            return_dict['body_ang_vel_t'] = body_ang_vel_t
        
        if flags.real_traj:
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = torch_utils.slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel
            
        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": root_rot,  
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_pos": rg_pos,  
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
        })
        # print("Joint names:", self.mesh_parsers.node_names)
        return return_dict
    
    def set_joint_mapping_mode(self, use_mapping=True, debug=False):

        self.use_joint_mapping = use_mapping
        self.debug_dof = debug
        print(f"Joint mapping mode: {'ON' if use_mapping else 'OFF'}")
        print(f"Debug mode: {'ON' if debug else 'OFF'}")
        
    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, gender_betas, fix_height, 
                                mesh_parsers, masterfoot_config, target_heading, max_len, queue, pid):

        np.random.seed(np.random.randint(5000) * pid)
        res = {}
        assert (len(ids) == len(motion_data_list))
        
        for f in range(len(motion_data_list)):
            curr_id = ids[f]
            curr_file = motion_data_list[f]
            
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                # 直接加载pkl文件
                loaded_data = joblib.load(curr_file)
                
                # 如果加载的数据直接就是运动数据字典
                if isinstance(loaded_data, dict) and 'root_trans_offset' in loaded_data:
                    curr_file = loaded_data
                # 如果加载的数据是包含运动数据的字典
                elif isinstance(loaded_data, dict):
                    # 从文件名推导key
                    filename = motion_data_list[f].split("/")[-1].split(".")[0]
                    
                    # 尝试各种可能的key匹配
                    possible_keys = [
                        filename,
                        filename.replace('_poses', ''),
                        filename.replace(' t2_poses', '_poses'),
                    ]
                    
                    # 添加所有包含文件名的key
                    for key in loaded_data.keys():
                        if filename in str(key) or str(key).endswith(filename):
                            possible_keys.append(key)
                    
                    # 尝试匹配key
                    matched_key = None
                    for key in possible_keys:
                        if key in loaded_data:
                            matched_key = key
                            break
                    
                    if matched_key is not None:
                        curr_file = loaded_data[matched_key]
                        if pid == 0:  # 只在主进程打印
                            print(f"[MotionLibTaihu] Using key '{matched_key}' for file {filename}")
                    else:
                        # 如果都找不到，使用第一个可用的key
                        first_key = list(loaded_data.keys())[0]
                        curr_file = loaded_data[first_key]
                        if pid == 0:  # 只在主进程打印警告
                            print(f"[MotionLibTaihu] Warning: key '{filename}' not found in {motion_data_list[f]}. "
                                f"Available keys: {list(loaded_data.keys())[:5]}{'...' if len(loaded_data.keys()) > 5 else ''}. "
                                f"Using fallback key '{first_key}'.")
                else:
                    raise ValueError(f"Unexpected data structure in {motion_data_list[f]}: {type(loaded_data)}")
            
            # 验证数据结构
            if not isinstance(curr_file, dict):
                raise ValueError(f"Expected dict data structure, got {type(curr_file)} for file {motion_data_list[f]}")
            
            # 检查必需的字段
            required_fields = ['root_trans_offset', 'pose_aa', 'fps']
            missing_fields = [field for field in required_fields if field not in curr_file]
            if missing_fields:
                available_fields = list(curr_file.keys())
                raise ValueError(f"Missing required fields {missing_fields} in {motion_data_list[f]}. "
                            f"Available fields: {available_fields}")

            # 处理序列长度
            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            # 提取运动数据
            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            dt = 1/curr_file['fps']

            B, J, N = pose_aa.shape

            # 处理目标朝向
            if target_heading is not None:
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(torch_utils.calc_heading_quat_inv(
                    torch.from_numpy(start_root_rot.as_quat()[None, ])))
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())
                trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T))

            # 执行前向运动学
            try:
                curr_motion = mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full=True, dt=dt)
                curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items()})
            except Exception as e:
                raise RuntimeError(f"Forward kinematics failed for {motion_data_list[f]}: {e}")
            
            res[curr_id] = (curr_file, curr_motion)

        if queue is not None:
            queue.put(res)
        else:
            return res