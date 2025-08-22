from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

class TiV2Robot(LeggedRobot):

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(11 + 2*self.num_actions + 12, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[
        9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.  # previous actions
        noise_vec[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = 0.  # sin/cos phase

        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.feet_rot = self.feet_state[:, :, 3:7]
        self.feet_rpy[:,0] = get_euler_xyz_in_tensor(self.feet_rot[:,0])
        self.feet_rpy[:,1] = get_euler_xyz_in_tensor(self.feet_rot[:,1])
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt




    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.feet_rot = self.feet_state[:, :, 3:7]
        self.feet_rpy[:,0] = get_euler_xyz_in_tensor(self.feet_rot[:,0])
        self.feet_rpy[:,1] = get_euler_xyz_in_tensor(self.feet_rot[:,1])
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8 #1.05
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        # self.phase[self.phase>0.6] = 0.6 + 0.2 * torch.sqrt((self.phase[self.phase>0.6] - 0.6) / 0.4)  
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        # print(1,self.leg_phase[self.leg_phase>0.6])
        # self.leg_phase[self.leg_phase>0.6] = 0.6 + 0.2 * torch.sqrt((self.leg_phase[self.leg_phase>0.6] - 0.6) / 0.4)  
        # print(2,self.leg_phase[self.leg_phase>0.6])
        # self.leg_phase
        # if original_phase > 0.8:
        #     mapped = torch.sqrt((original_phase - 0.8) / 0.2)  
        return super()._post_physics_step_callback()

    # def compute_observations(self):
    #     """ Computes observations
    #     """

    #     sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1) # 1
    #     cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1) # 1
    #     self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, # 3
    #                               self.projected_gravity, # 3
    #                               self.commands[:, :3] * self.commands_scale, # 3
    #                               (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #                               self.dof_vel * self.obs_scales.dof_vel,
    #                               self.actions,
    #                               sin_phase,
    #                               cos_phase
    #                               ), dim=-1)
    #     self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, # 3
    #                                          self.base_ang_vel * self.obs_scales.ang_vel,
    #                                          self.projected_gravity,
    #                                          self.commands[:, :3] * self.commands_scale,
    #                                          (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #                                          self.dof_vel * self.obs_scales.dof_vel,
    #                                          self.actions,
    #                                          sin_phase,
    #                                          cos_phase
    #                                          ), dim=-1)
    #     if self.add_noise:
    #         self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1) # 1
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1) # 1
        current_obs = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, # 3
                                  self.projected_gravity, # 3
                                  self.commands[:, :3] * self.commands_scale, # 3
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions,
                                  sin_phase,
                                  cos_phase
                                  ), dim=-1)
        current_actor_obs = torch.clone(current_obs)
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_proprioceptive_obs_length], current_actor_obs[:, :self.num_one_step_obs]), dim=-1)   

        # current_critic_obs = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,current_obs), dim=-1)
        current_critic_obs = torch.cat((current_obs,self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1) # 1
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1) # 1
        current_obs = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, # 3
                                  self.projected_gravity, # 3
                                  self.commands[:, :3] * self.commands_scale, # 3
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions,
                                  sin_phase,
                                  cos_phase
                                  ), dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        return torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)[env_ids]
            
            
    def _reward_contact(self):
        # res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        right = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            right += (~(contact ^ is_stance))
            # right[~self.is_vel[...,0]] += ((contact) * torch.exp(-torch.norm(self.base_lin_vel[:,],dim=1)))[~self.is_vel[...,0]] 

        res = 2 * (right==2) 
        # print(res)
        return res

    # def _reward_contact(self):
    #     res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    #     for i in range(self.feet_num):
    #         is_stance = self.leg_phase[:, i] < 0.55
    #         contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
    #         res += ~(contact ^ is_stance)
    #     return res
    
    def _reward_standing(self):
        """
        奖励机器人保持站立稳定（所有脚都接触地面）。
        """
        # 检查每只脚的 z 向接触力是否大于阈值（表示接触地面）
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0  # shape: [envs, feet_num]

        # 判断是否所有脚都接触地面（按行求和等于脚数即为全接触）
        standing = torch.sum(contact, dim=1) == self.feet_num  # shape: [envs]

        # 奖励：如果全部脚都接触地面则奖励1，否则0
        reward = standing.float()
        # print(reward)

        # return reward*(~self.is_vel[...,0])
        return reward

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        # print(self.feet_pos[:, :, 2])
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.12) * ~contact
        # return torch.sum(pos_error, dim=(1))*self.is_vel[...,0]
        return torch.sum(pos_error, dim=(1))

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [1, 2, 7, 8]]), dim=1)

    def _reward_ankle_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [5,11]]), dim=1)
    
    ##惩罚 足部滑动（接触地面时不要有xy方向的速度），[0,+00]
    # def _reward_foot_slip(self): 
    #     # Penalize foot slipping
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     return torch.sum(torch.norm(self.feet_vel[:,:,:2], dim=2) * contact, dim=1)
    
    # # 惩罚 足部接触力，[0,+00]
    def _reward_foot_contact_forces(self):
        # penalize high contact forces
        # print(self.contact_forces[:, self.feet_indices, :])
        # print(torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  300).clip(min=0.), dim=1))
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  450).clip(min=0.), dim=1)
    
    # # 惩罚 接触动量（缓慢着陆），[0,+00]
    def _reward_contact_momentum(self):
        # encourage soft contacts
        # foot_contact_momentum_z = torch.clip(self.feet_vel[:, :, 2], max=0) * torch.clip(self.contact_forces[:, self.feet_indices, 2] - 50, min=0)
        # return torch.sum((foot_contact_momentum_z != 0)*self.first_contacts*torch.exp(foot_contact_momentum_z/100), dim=1)
        foot_contact_momentum_z = torch.clip(self.feet_vel[:, :, 2], max=0) * torch.clip(self.contact_forces[:, self.feet_indices, 2] - 50, min=0)
        return torch.sum(foot_contact_momentum_z, dim=1)    

    # def _reward_feet_ground_parallel(self):
    #     feet_heights, feet_heights_var = self._get_feet_heights()
    #     # continue_contact = (self.feet_air_time >= 3* self.dt) * self.contact_filt
    #     return torch.sum(feet_heights_var, dim=1)
    
    # def _reward_feet_parallel(self):
    #     left_foot_pos = self.rigid_body_states_view[:, self.left_foot_indices[0:3], :3].clone()
    #     right_foot_pos = self.rigid_body_states_view[:, self.right_foot_indices[0:3], :3].clone()
    #     feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=2)
    #     feet_distances_var = torch.var(feet_distances, dim=1)
    #     return feet_distances_var

    # def _get_feet_heights(self, env_ids=None):
    #     """ Samples heights of the terrain at required points around each robot.
    #         The points are offset by the base's position and rotated by the base's yaw

    #     Args:
    #         env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

    #     Raises:
    #         NameError: [description]

    #     Returns:
    #         [type]: [description]
    #     """
    #     left_foot_pos = self.rigid_body_states_view[:, self.left_foot_indices, :3].clone()
    #     right_foot_pos = self.rigid_body_states_view[:, self.right_foot_indices, :3].clone()
    #     if self.cfg.terrain.mesh_type == 'plane':
    #         left_foot_height = torch.mean(left_foot_pos[:, :, 2], dim = -1, keepdim=True)
    #         left_foot_height_var = torch.var(left_foot_pos[:, :, 2], dim = -1, keepdim=True)
    #         right_foot_height = torch.mean(right_foot_pos[:, :, 2], dim = -1, keepdim=True)
    #         right_foot_height_var = torch.var(right_foot_pos[:, :, 2], dim = -1, keepdim=True)
    #         return torch.cat((left_foot_height, right_foot_height), dim=-1), torch.cat((left_foot_height_var, right_foot_height_var), dim=-1)
    #     elif self.cfg.terrain.mesh_type == 'none':
    #         raise NameError("Can't measure height with terrain mesh type 'none'")


    # # def _reward_Bart_tracking_x_vel(self):
    # #     rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    # #     # stand_index = torch.where(self.is_vel == False)[0]
    # #     # walk_index = torch.where(self.is_vel == True)[0]
    # #     rew[~self.is_vel[:,0]] = torch.exp(-5 * torch.abs(self.base_lin_vel[~self.is_vel[:,0],0] - self.commands[~self.is_vel[:,0], 0]))
    # #     rew[self.is_vel[:,0]] = torch.exp(-5 * ((torch.abs(self.base_lin_vel[self.is_vel[:,0],0] - self.commands[self.is_vel[:,0], 0]))**2))
    # #     return rew

    # # def _reward_Bart_tracking_y_vel(self):
    # #     rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    # #     # stand_index = torch.where(self.is_vel == False)[0]
    # #     # walk_index = torch.where(self.is_vel == True)[0]
    # #     rew[~self.is_vel[:,0]] = torch.exp(-5 * torch.abs(self.base_lin_vel[~self.is_vel[:,0],1] - self.commands[~self.is_vel[:,0], 1]))
    # #     rew[self.is_vel[:,0]] = torch.exp(-5 * ((torch.abs(self.base_lin_vel[self.is_vel[:,0],1] - self.commands[self.is_vel[:,0], 1]))**2))
    # #     return rew
    
    # # def _reward_Bart_tracking_rot_vel(self):
    # #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    # #     return torch.exp(-300*ang_vel_error)
    
    # # def _reward_Bart_orientation(self):
    # #     # Penalize non flat base orientation
    # #     # print(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1))
    # #     return torch.exp(-30*torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1))

    # # def _reward_Bart_feet_contact(self):
    # #     rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    # #     # stand_index = torch.where(self.is_vel == False)[0]
    # #     # walk_index = torch.where(self.is_vel == True)[0]
    # #     rew[~self.is_vel[:,0]] = 1.0

    # #     # 检查每只脚的 z 向接触力是否大于阈值（表示接触地面）
    # #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.0  # shape: [envs, feet_num]
    # #     # 判断是否所有脚都接触地面（按行求和等于脚数即为全接触）
    # #     one_feet_contact = torch.sum(contact, dim=1) == 1  # shape: [envs]
    # #     # one_feet_contact_index = torch.where(one_feet_contact == True)[0] 

    # #     one_feet_contact_index = one_feet_contact.unsqueeze(1)
    # #     self.last_one_feet_contact_time[one_feet_contact_index[:,0]] = self.episode_length_buf[one_feet_contact_index[:,0]].unsqueeze(1) * self.dt
    # #     walk_one_feet_contact_index = self.is_vel & (self.episode_length_buf.unsqueeze(1) * self.dt - self.last_one_feet_contact_time <= 0.2)
    # #     # walk_one_feet_contact_index =  self.is_vel & one_feet_contact.unsqueeze(1)
    # #     rew[walk_one_feet_contact_index[:,0]] = 1.0
    # #     # rew[~walk_one_feet_contact_index[:,0]] = 0.0
    # #     # print(self.is_vel,walk_one_feet_contact_index)
    # #     return rew
    
    # # def _reward_Bart_base_height(self):
    # #     base_height = self.root_states[:, 2]
    # #     # print(base_height)
    # #     return torch.exp(-20*torch.abs(base_height - self.cfg.rewards.base_height_target))
    
    # # def _reward_Bart_feet_airtime(self):
    # #     rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    # #     rew[~self.is_vel[:,0]] = 1.0
    # #     rew[self.is_vel[:,0]] = torch.sum((self.feet_air_time[self.is_vel[:,0]] - 0.5) * self.first_contacts[self.is_vel[:,0]], dim=1)
    # #     # print(self.is_vel,rew,self.first_contacts)
    # #     return rew
    
    # # def _reward_Bart_feet_orientation(self):
    # #     rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    # #     is_rot_vel = torch.abs(self.commands[:, 2]) > 0.2
    # #     rew[is_rot_vel] = torch.exp(-torch.sum(torch.sum(self.feet_rpy[is_rot_vel,:2], dim=2),dim=1))
    # #     rew[~is_rot_vel] = torch.exp(-torch.sum(torch.sum(self.feet_rpy[~is_rot_vel,:3], dim=2),dim=1))
    # #     # print(self.feet_rpy)
    # #     return rew

    # # def _reward_Bart_feet_position(self):
    # #     rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    # #     # rew[self.is_vel[:,0]] = 1
    # #     cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    # #     footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    # #     for i in range(len(self.feet_indices)):
    # #         footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
    # #     foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
    # #     rew = torch.exp(-3*torch.abs(foot_leteral_dis - 0.27))
    # #     # print(foot_leteral_dis)
    # #     return rew

    # # def _reward_Bart_base_acc(self):
    # #     # Penalize dof accelerations
    # #     return torch.exp(-0.01*torch.sum(torch.abs((self.last_root_vel[:,:3] - self.last_root_vel[:, :3]) / self.dt), dim=1))
    
    # # def _reward_Bart_action_diff(self):
    # #     # Penalize changes in actions
    # #     return torch.exp(-0.02*torch.sum(torch.abs((self.last_actions - self.actions) / self.dt), dim=1))
    

    # # def _reward_Bart_torque(self):
    # #     # penalize torques too close to the limit
    # #     return torch.exp(-0.02*torch.sum((torch.abs(self.torques)/self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)/12)

    # # def _reward_Bart_hip_pos(self):
    # #     return torch.exp(-300*torch.sum(torch.square(self.dof_pos[:, [1, 2, 7, 8]]), dim=1))
    

    def _reward_feet_parallel(self):
        left_foot_left_height = self.rigid_body_states_view[:, [self.left_foot_indices[2]], 2]
        left_foot_right_height = self.rigid_body_states_view[:, [self.left_foot_indices[4]], 2]
        left_foot_height_dis = torch.norm(left_foot_left_height - left_foot_right_height, dim=1)
        right_foot_left_height = self.rigid_body_states_view[:, [self.right_foot_indices[2]], 2]
        right_foot_right_height = self.rigid_body_states_view[:, [self.right_foot_indices[4]], 2] 
        right_foot_height_dis = torch.norm(right_foot_left_height - right_foot_right_height, dim=1)
        return left_foot_height_dis+right_foot_height_dis
    

    def _reward_feet_heading_alignment(self):

        left_quat = self.rigid_body_states_view[:, self.feet_indices[0], 3:7]
        right_quat = self.rigid_body_states_view[:, self.feet_indices[1], 3:7]

        forward_left_feet = quat_apply(left_quat, self.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, self.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])


        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))

        rew = (heading_diff_left + heading_diff_right)*(~(torch.norm(self.commands[:, [2]], dim=1) > 0.2))
            
        return rew