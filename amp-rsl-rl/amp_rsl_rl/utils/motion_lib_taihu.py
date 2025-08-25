import numpy as np
import os.path as osp
import joblib
import torch
from scipy.spatial.transform import Rotation as sRot
import random

from amp_rsl_rl.utils import torch_utils
from amp_rsl_rl.utils.motion_lib_base import MotionLibBase, FixHeightMode
from amp_rsl_rl.utils.torch_humanoid_batch import Humanoid_Batch
from easydict import EasyDict


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


class MotionLibTaihu(MotionLibBase):

    def __init__(self, motion_file, device, fix_height=FixHeightMode.no_fix,
                 masterfoot_conifg=None, min_length=-1, im_eval=False,
                 multi_thread=False, extend_hand=True, extend_head=True,
                 mjcf_file="resources/robots/taihu/ti.xml", sim_timestep=1/50):

        super().__init__(motion_file=motion_file, device=device,
                         fix_height=fix_height,
                         masterfoot_conifg=masterfoot_conifg,
                         min_length=min_length, im_eval=im_eval,
                         multi_thread=multi_thread, sim_timestep=sim_timestep)

        cfg = EasyDict({
            'asset': EasyDict({'assetFileName': mjcf_file}),
            'extend_config': []
        })
        self.mesh_parsers = Humanoid_Batch(cfg)
        self.isaac_gym_joint_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.use_joint_mapping = True
        self.debug_dof = False
        self.sign_flip_set = {2, 4, 6, 7, 8, 9}

    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers,
                         fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0

        with torch.no_grad():
            motion_res = mesh_parsers.fk_batch(pose_aa, trans, return_full=True)

            left_ankle_pos = motion_res['global_translation'][:, :, 6]
            right_ankle_pos = motion_res['global_translation'][:, :, 12]

            foot_height_offset = 0.1
            left_foot_height = left_ankle_pos[:, :, 2] - foot_height_offset
            right_foot_height = right_ankle_pos[:, :, 2] - foot_height_offset

            min_foot_height = torch.min(left_foot_height, right_foot_height)

            if fix_height_mode == FixHeightMode.full_fix:
                height_adjustment = -min_foot_height
            elif fix_height_mode == FixHeightMode.ankle_fix:
                target_ankle_height = 0.1
                min_ankle_height = torch.min(left_ankle_pos[:, :, 2],
                                             right_ankle_pos[:, :, 2])
                height_adjustment = target_ankle_height - min_ankle_height
            else:
                height_adjustment = torch.zeros_like(min_foot_height)

            trans_fixed = trans.clone()
            trans_fixed[:, :, 2] += height_adjustment

            diff_fix = height_adjustment.mean().item()

        return trans_fixed, diff_fix

    def apply_isaac_gym_joint_mapping(self, dof_pos):
        if not self.use_joint_mapping:
            return dof_pos

        mapped_dof = torch.zeros_like(dof_pos)
        for i, mapping_idx in enumerate(self.isaac_gym_joint_mapping):
            if i < mapped_dof.shape[-1] and mapping_idx < dof_pos.shape[-1]:
                val = dof_pos[:, mapping_idx]
                if i in self.sign_flip_set:
                    val = -val
                mapped_dof[:, i] = val
        return mapped_dof

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        state = super().get_motion_state(motion_ids, motion_times, offset)
        state['dof_pos'] = self.apply_isaac_gym_joint_mapping(state['dof_pos'])
        return state

    def set_joint_mapping_mode(self, use_mapping=True, debug=False):
        self.use_joint_mapping = use_mapping
        self.debug_dof = debug
        print(f"Joint mapping mode: {'ON' if use_mapping else 'OFF'}")
        print(f"Debug mode: {'ON' if debug else 'OFF'}")

    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees,
                                  gender_betas, fix_height, mesh_parsers,
                                  masterfoot_config, target_heading, max_len,
                                  queue, pid):
        np.random.seed(np.random.randint(5000) * pid)
        res = {}
        assert len(ids) == len(motion_data_list)

        for f in range(len(motion_data_list)):
            curr_id = ids[f]
            curr_file = motion_data_list[f]

            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                loaded_data = joblib.load(curr_file)
                if isinstance(loaded_data, dict) and 'root_trans_offset' in loaded_data:
                    curr_file = loaded_data
                elif isinstance(loaded_data, dict):
                    filename = motion_data_list[f].split("/")[-1].split(".")[0]
                    possible_keys = [filename,
                                     filename.replace('_poses', ''),
                                     filename.replace(' t2_poses', '_poses')]
                    for key in loaded_data.keys():
                        if filename in str(key) or str(key).endswith(filename):
                            possible_keys.append(key)
                    matched_key = None
                    for key in possible_keys:
                        if key in loaded_data:
                            matched_key = key
                            break
                    if matched_key is not None:
                        curr_file = loaded_data[matched_key]
                    else:
                        first_key = list(loaded_data.keys())[0]
                        curr_file = loaded_data[first_key]
                else:
                    raise ValueError(
                        f"Unexpected data structure in {motion_data_list[f]}: {type(loaded_data)}"
                    )

            if not isinstance(curr_file, dict):
                raise ValueError(
                    f"Expected dict data structure, got {type(curr_file)} for file {motion_data_list[f]}"
                )

            required_fields = ['root_trans_offset', 'pose_aa', 'fps']
            missing_fields = [field for field in required_fields if field not in curr_file]
            if missing_fields:
                available_fields = list(curr_file.keys())
                raise ValueError(
                    f"Missing required fields {missing_fields} in {motion_data_list[f]}. "
                    f"Available fields: {available_fields}"
                )

            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            dt = 1 / curr_file['fps']

            if target_heading is not None:
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(
                    torch_utils.calc_heading_quat_inv(
                        torch.from_numpy(start_root_rot.as_quat()[None, :])
                    )
                )
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot
                pose_aa[:, 0] = torch.tensor(
                    (heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec()
                )
                trans = torch.matmul(
                    trans,
                    torch.from_numpy(heading_delta.as_matrix().squeeze().T),
                )

            curr_motion = mesh_parsers.fk_batch(
                pose_aa[None, :], trans[None, :], return_full=True, dt=dt
            )
            curr_motion = EasyDict({
                k: v.squeeze() if torch.is_tensor(v) else v
                for k, v in curr_motion.items()
            })

            res[curr_id] = (curr_file, curr_motion)

        if queue is not None:
            queue.put(res)
        else:
            return res