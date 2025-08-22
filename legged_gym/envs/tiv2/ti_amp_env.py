# legged_gym/envs/tiv2/ti_amp_env.py
from legged_gym.envs.base.legged_robot import LeggedRobot
import torch
from isaacgym.torch_utils import *

class TiV2AMPRobot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # AMP相关的观测维度
        self.num_amp_obs = self._get_amp_obs_dim()
        
    def _get_amp_obs_dim(self):
        """计算AMP观测的维度"""
        # 关节位置 + 关节速度 + 基座线速度(局部) + 基座角速度(局部)
        return self.num_dof * 2 + 3 + 3
    
    def compute_amp_observations(self):
        """计算AMP所需的观测"""
        # 将关节速度和基座速度转换到局部坐标系
        base_lin_vel_local = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        base_ang_vel_local = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        
        amp_obs = torch.cat([
            self.dof_pos,  # 关节位置
            self.dof_vel,  # 关节速度
            base_lin_vel_local,  # 局部线速度
            base_ang_vel_local,  # 局部角速度
        ], dim=-1)
        
        return amp_obs
    
    def step(self, actions):
        # 执行原始step
        obs, privileged_obs, rewards, dones, infos, termination_ids, termination_obs = super().step(actions)
        
        # 添加AMP观测到infos
        amp_obs = self.compute_amp_observations()
        infos["observations"]["amp"] = amp_obs
        
        return obs, privileged_obs, rewards, dones, infos, termination_ids, termination_obs