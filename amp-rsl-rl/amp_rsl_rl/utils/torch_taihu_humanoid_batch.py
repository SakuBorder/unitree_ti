import torch 
import numpy as np
import amp_rsl_rl.utils.rotation_conversions as tRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import smpl_sim.poselib.core.rotation3d as pRot


class Taihu_Humanoid_Batch:
    def __init__(self, mjcf_file="path/to/taihu.xml", extend_hand=False, extend_head=False, device=torch.device("cpu")):
        self.mjcf_data = mjcf_data = self.from_mjcf(mjcf_file)
        self.dof_axis = mjcf_data["dof_axis"].to(device)  # 从 MJCF 解析的旋转轴
        self.extend_hand = extend_hand
        self.extend_head = extend_head
        
        if extend_hand:
            self.model_names = mjcf_data['node_names'] + ["left_hand_link", "right_hand_link"]
            left_wrist_idx = self.find_joint_index("L_WRIST_R_S")
            right_wrist_idx = self.find_joint_index("R_WRIST_R_S")
            self._parents = torch.cat((mjcf_data['parent_indices'], torch.tensor([left_wrist_idx, right_wrist_idx]))).to(device)
            
            arm_length = 0.12 
            self._offsets = torch.cat((mjcf_data['local_translation'], torch.tensor([[arm_length, 0, 0], [arm_length, 0, 0]])), dim=0)[None, ].to(device)
            self._local_rotation = torch.cat((mjcf_data['local_rotation'], torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])), dim=0)[None, ].to(device)
            self._remove_idx = 2
        else:
            self._parents = mjcf_data['parent_indices']
            self.model_names = mjcf_data['node_names']
            self._offsets = mjcf_data['local_translation'][None, ].to(device)
            self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
            self._remove_idx = 0
            
        if extend_head:
            if extend_hand:
                self._remove_idx = 3
            else:
                self._remove_idx = 1
            self.model_names = self.model_names + ["head_link"]
            neck_idx = self.find_joint_index("NECK_Y_S")
            self._parents = torch.cat((self._parents, torch.tensor([neck_idx]).to(device))).to(device)
            
            head_length = 0.25  
            self._offsets = torch.cat((self._offsets, torch.tensor([[[0, 0, head_length]]]).to(device)), dim=1).to(device)
            self._local_rotation = torch.cat((self._local_rotation, torch.tensor([[[1, 0, 0, 0]]]).to(device)), dim=1).to(device)
        if extend_hand:
            hand_axis = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(2, 3)
            self.dof_axis = torch.cat([self.dof_axis, hand_axis], dim=0)
        
        if extend_head:
            head_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device)
            self.dof_axis = torch.cat([self.dof_axis, head_axis], dim=0)
        self.joints_range = mjcf_data['joints_range'].to(device)
        self._local_rotation_mat = tRot.quaternion_to_matrix(self._local_rotation).float()

    def plot_joint_axes(pose_mat, joint_idx, axis):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, pose_mat[joint_idx, 0, 0], pose_mat[joint_idx, 0, 1], pose_mat[joint_idx, 0, 2], color='r')  # X轴
        ax.quiver(0, 0, 0, pose_mat[joint_idx, 1, 0], pose_mat[joint_idx, 1, 1], pose_mat[joint_idx, 1, 2], color='g')  # Y轴
        ax.quiver(0, 0, 0, pose_mat[joint_idx, 2, 0], pose_mat[joint_idx, 2, 1], pose_mat[joint_idx, 2, 2], color='b')  # Z轴
        ax.quiver(0, 0, 0, axis[0], axis[1], axis[2], color='k', linewidth=3)  # 旋转轴
        plt.title(f"Joint {joint_idx} Rotation Axis")
        plt.show()

    def find_joint_index(self, joint_name):
        try:
            return self.mjcf_data['node_names'].index(joint_name)
        except ValueError:
            # 
            for i, name in enumerate(self.mjcf_data['node_names']):
                if joint_name.replace("_S", "") in name:
                    return i
            return 0 
        
    def from_mjcf(self, path):
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        dof_axis = []  

        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            if "euler" in xml_node.attrib:
                euler = np.fromstring(xml_node.attrib.get("euler"), dtype=float, sep=" ")
                from scipy.spatial.transform import Rotation as R
                quat = R.from_euler('xyz', euler).as_quat()  # [x, y, z, w]
                quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # 转为 [w, x, y, z]
            else:
                quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            
            curr_index = node_index
            node_index += 1
            

            all_joints = xml_node.findall("joint")
            for joint in all_joints:
                if joint.attrib.get("range") is not None: 
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
                else:
                    if not joint.attrib.get("type") == "free":
                        joints_range.append([-np.pi, np.pi])
                
                if "axis" in joint.attrib:
                    axis = np.fromstring(joint.attrib["axis"], dtype=float, sep=" ")
                    dof_axis.append(axis)
                else:
                    dof_axis.append(np.array([0.0, 1.0, 0.0]))
            
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
                
            return node_index

        _add_xml_node(xml_body_root, -1, 0)
        
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "dof_axis": torch.from_numpy(np.array(dof_axis, dtype=np.float32))  # 新增：返回 dof_axis
        }

    def fk_batch(self, pose, trans, convert_to_mat=True, return_full=False, dt=1/30):
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2]
        pose = pose[..., :len(self._parents), :]  
        
        if self.extend_hand and self.extend_head and pose.shape[-2] == 31: 
            pose = torch.cat([pose, torch.zeros(B, seq_len, 1, 3).to(device).type(dtype)], dim=-2)
        elif self.extend_hand and pose.shape[-2] == 32:  # 30个关节+2个手部
            pose = torch.cat([pose, torch.zeros(B, seq_len, 1, 3).to(device).type(dtype)], dim=-2)

        if convert_to_mat:
            pose_quat = tRot.axis_angle_to_quaternion(pose)
            pose_mat = tRot.quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
            
        if len(pose_mat.shape) != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        
        J = pose_mat.shape[2] - 1  
        
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        
        return_dict = EasyDict()
        
        wbody_rot = tRot.wxyz_to_xyzw(tRot.matrix_to_quaternion(wbody_mat))
        
        if self.extend_hand or self.extend_head:
            if return_full:
                return_dict.global_velocity_extend = self._compute_velocity(wbody_pos, dt) 
                return_dict.global_angular_velocity_extend = self._compute_angular_velocity(wbody_rot, dt)
                
            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_mat_extend = wbody_mat.clone()
            return_dict.global_rotation_extend = wbody_rot.clone()
            
            if self._remove_idx > 0:
                wbody_pos = wbody_pos[..., :-self._remove_idx, :]
                wbody_mat = wbody_mat[..., :-self._remove_idx, :, :]
                wbody_rot = wbody_rot[..., :-self._remove_idx, :]
        
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot
            
        if return_full:
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
            return_dict.local_rotation = tRot.wxyz_to_xyzw(pose_quat)
            return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[..., 0, :]
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity
            
            if self.extend_hand or self.extend_head:
                return_dict.dof_pos = pose.sum(dim=-1)[..., 1:][..., :-self._remove_idx] if self._remove_idx > 0 else pose.sum(dim=-1)[..., 1:]
            else:
                return_dict.dof_pos = pose.sum(dim=-1)[..., 1:]
            
            dof_vel = ((return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1]) / dt)
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim=1)
            return_dict.fps = int(1/dt)
        
        return return_dict
    
    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]])
                rot_mat = torch.matmul(rotations_world[self._parents[i]], torch.matmul(self._local_rotation_mat[:, (i):(i + 1)], rotations[:, :, (i - 1):i, :]))
                
                positions_world.append(jpos)
                rotations_world.append(rot_mat)
        
        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world
    
    @staticmethod
    def _compute_velocity(p, time_delta, gaussian_filter=True):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if gaussian_filter:
            velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)
        return velocity
    
    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, gaussian_filter=True):
        diff_quat_data = pRot.quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = pRot.quat_mul_norm(r[..., 1:, :, :], pRot.quat_inverse(r[..., :-1, :, :]))
        diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if gaussian_filter:
            angular_velocity = torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"))
        return angular_velocity


Humanoid_Batch = Taihu_Humanoid_Batch