# legged_gym/envs/tiv2/ti_amp_env.py
from legged_gym.envs.tiv2.tiv2_env import TiV2Robot
import torch
from isaacgym.torch_utils import *  # quat_rotate, quat_rotate_inverse, etc.
from isaacgym import gymtorch
import torch.nn.functional as F
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor


class TiV2AMPRobot(TiV2Robot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # ===== 新增：AMP历史观测配置 =====
        self._num_amp_obs_steps = getattr(cfg.env, 'num_amp_obs_steps', 2)  # 默认2步历史
        self._num_amp_obs_per_step = self._get_amp_obs_per_step_dim()  # 单步AMP观测维度
        self.num_amp_obs = self._num_amp_obs_steps * self._num_amp_obs_per_step  # 总AMP观测维度
        
        # 创建AMP观测历史缓冲区
        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), 
            device=self.device, dtype=torch.float
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]  # 当前帧
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]  # 历史帧
        
        self.amp_data = None  # 由 runner 注入

        # 确保 phase 属性存在
        if not hasattr(self, 'phase'):
            print("[TiV2AMPRobot] Adding missing 'phase' attribute")
            self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # 调试信息
        print(f"[TiV2AMPRobot] Actor obs dim: {self.num_obs}")
        print(f"[TiV2AMPRobot] Critic obs dim: {self.num_privileged_obs}")
        print(f"[TiV2AMPRobot] AMP obs dim: {self.num_amp_obs} ({self._num_amp_obs_steps} steps × {self._num_amp_obs_per_step} per step)")

    def set_amp_data(self, amp_data):
        """由 Runner 注入 AMPLoader 句柄"""
        self.amp_data = amp_data

    def _get_amp_obs_per_step_dim(self):
        """计算单步AMP观测维度：12 dof pos + 12 dof vel + 3 lin vel + 3 ang vel = 30"""
        return self.num_dof * 2 + 3 + 3  # 30维

    def _get_amp_obs_dim(self):
        """保持向后兼容的接口"""
        return self.num_amp_obs

    def _compute_amp_observations_single_step(self):
        """计算当前帧的AMP观测（不包含历史）"""
        base_lin_vel_local = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        base_ang_vel_local = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        amp_obs = torch.cat([
            self.dof_pos,           # 关节位置 (12)
            self.dof_vel,           # 关节速度 (12)
            base_lin_vel_local,     # 局部线速度 (3)
            base_ang_vel_local,     # 局部角速度 (3)
        ], dim=-1)  # 30 维

        return amp_obs

    def _update_hist_amp_obs(self, env_ids=None):
        """滚动更新AMP历史观测"""
        if env_ids is None:
            # 更新所有环境
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            # 更新指定环境
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]

    def _compute_amp_observations(self, env_ids=None):
        """计算当前帧AMP观测并更新到缓冲区"""
        if env_ids is None:
            self._curr_amp_obs_buf[:] = self._compute_amp_observations_single_step()
        else:
            self._curr_amp_obs_buf[env_ids] = self._compute_amp_observations_single_step()[env_ids]

    def compute_amp_observations(self):
        """对外接口：返回完整的AMP观测（包含历史）"""
        return self._amp_obs_buf.view(-1, self.num_amp_obs)

    def post_physics_step(self):
        """重写post_physics_step，添加AMP观测更新"""
        # 先更新AMP观测历史（在父类逻辑之前）
        self._update_hist_amp_obs()
        self._compute_amp_observations()
        
        # 调用父类的post_physics_step，它应该正确处理终止逻辑
        # 注意：我们直接调用LeggedRobot的post_physics_step，跳过TiV2Robot
        return super(TiV2Robot, self).post_physics_step()

    def get_observations(self):
        """重写 get_observations 以兼容 AMPOnPolicyRunner"""
        try:
            super().compute_observations()
        except AttributeError as e:
            print(f"[Warning] Parent compute_observations failed: {e}")
            print("[Warning] Using observation buffers directly")

        # 获取观测
        actor_obs = self.obs_buf
        critic_obs = self.privileged_obs_buf if self.privileged_obs_buf is not None else self.obs_buf
        amp_obs = self.compute_amp_observations()  # 包含历史的AMP观测

        # 调试信息（仅第一次）
        if not hasattr(self, '_debug_printed'):
            print(f"[Debug] Actor obs shape: {actor_obs.shape}")
            print(f"[Debug] Critic obs shape: {critic_obs.shape}")
            print(f"[Debug] AMP obs shape: {amp_obs.shape}")
            self._debug_printed = True

        # 确保 critic_obs 维度正确
        if critic_obs.shape[-1] != self.num_privileged_obs:
            print(f"[Warning] Critic obs dim mismatch: got {critic_obs.shape[-1]}, expected {self.num_privileged_obs}")
            if critic_obs.shape[-1] > self.num_privileged_obs:
                critic_obs = critic_obs[..., :self.num_privileged_obs]
            else:
                padding = self.num_privileged_obs - critic_obs.shape[-1]
                critic_obs = torch.cat(
                    [critic_obs, torch.zeros(critic_obs.shape[0], padding, device=critic_obs.device)],
                    dim=-1
                )

        # 构建 extras 字典
        extras = {
            "observations": {
                "amp": amp_obs,
                "critic": critic_obs
            }
        }

        return actor_obs, extras

    def step(self, actions):
        """重写 step：保持返回7元组"""
        ret = super().step(actions)

        if not isinstance(ret, tuple):
            raise RuntimeError(f"[TiV2AMPRobot.step] Unexpected return type from super().step: {type(ret)}")

        if len(ret) == 7:
            obs, privileged_obs, rewards, dones, infos, termination_ids, termination_obs = ret
        elif len(ret) == 4:
            obs, rewards, dones, infos = ret
            privileged_obs = self.get_privileged_observations()
            termination_ids = None
            termination_obs = None
            print("[TiV2AMPRobot.step][Warning] Parent step returned 4-tuple; "
                  "filled privileged_obs via get_privileged_observations(). Please fix parent to return 7-tuple.")
        else:
            raise RuntimeError(f"[TiV2AMPRobot.step] Unsupported super().step() signature with len={len(ret)}")

        # 确保 privileged_obs 维度正确
        if privileged_obs is not None and privileged_obs.shape[-1] != self.num_privileged_obs:
            if privileged_obs.shape[-1] > self.num_privileged_obs:
                privileged_obs = privileged_obs[..., :self.num_privileged_obs]
            else:
                padding = self.num_privileged_obs - privileged_obs.shape[-1]
                privileged_obs = torch.cat(
                    [privileged_obs, torch.zeros(privileged_obs.shape[0], padding, device=privileged_obs.device)],
                    dim=-1
                )

        # 把 critic 观测放到 infos，便于调试
        if isinstance(infos, dict):
            infos.setdefault("observations", {})
            infos["observations"]["critic"] = privileged_obs if privileged_obs is not None else obs

        return obs, privileged_obs, rewards, dones, infos, termination_ids, termination_obs

    def get_privileged_observations(self):
        """确保 privileged observations 返回正确维度"""
        try:
            priv_obs = super().get_privileged_observations()
        except AttributeError as e:
            print(f"[Warning] Parent get_privileged_observations failed: {e}")
            priv_obs = self.privileged_obs_buf if hasattr(self, 'privileged_obs_buf') else None

        # 维度检查和修正
        if priv_obs is not None and priv_obs.shape[-1] != self.num_privileged_obs:
            if priv_obs.shape[-1] > self.num_privileged_obs:
                priv_obs = priv_obs[..., :self.num_privileged_obs]
            else:
                padding = self.num_privileged_obs - priv_obs.shape[-1]
                priv_obs = torch.cat(
                    [priv_obs, torch.zeros(priv_obs.shape[0], padding, device=priv_obs.device)],
                    dim=-1
                )

        return priv_obs

    def compute_observations(self):
        """重写父类的 compute_observations 以避免 phase 属性错误"""
        try:
            super().compute_observations()
        except AttributeError as e:
            if "'phase'" in str(e):
                print(f"[Warning] Phase attribute missing in parent compute_observations: {e}")
                if not hasattr(self, 'phase'):
                    self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
                try:
                    super().compute_observations()
                except Exception as e2:
                    print(f"[Error] Still failed after adding phase: {e2}")
                    self._manual_compute_observations()
            else:
                raise e

    def _post_physics_step_callback(self):
        """重新实现父类TiV2Robot的_post_physics_step_callback逻辑"""
        # 更新足部状态
        self.update_feet_state()
        
        # 相位计算逻辑
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

    def update_feet_state(self):
        """更新足部状态（来自TiV2Robot）"""
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.feet_rot = self.feet_state[:, :, 3:7]
        self.feet_rpy[:,0] = get_euler_xyz_in_tensor(self.feet_rot[:,0])
        self.feet_rpy[:,1] = get_euler_xyz_in_tensor(self.feet_rot[:,1])
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

    def _reward_alive(self):
        """奖励存活"""
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _init_amp_obs_for_reset(self, env_ids):
        """为重置的环境初始化AMP观测历史"""
        if len(env_ids) == 0:
            return
            
        # 计算当前帧观测
        self._compute_amp_observations(env_ids)
        
        # 将当前观测复制到所有历史帧（简单策略）
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)  # [N, 1, obs_dim]
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs.expand(-1, self._num_amp_obs_steps - 1, -1)

    def reset_idx(self, env_ids):
        """
        覆盖父类reset，添加专家帧热启动 + AMP历史初始化
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids)
        env_ids = env_ids.to(self.device, dtype=torch.long).reshape(-1)

        if env_ids.numel() == 0:
            return

        # 1) 父类复位
        super().reset_idx(env_ids)

        # 2) 专家状态覆盖（如果有数据）
        if getattr(self, "amp_data", None) is not None:
            n = env_ids.numel()
            try:
                quat_xyzw, qpos, qvel, vlin_local, vang_local = self.amp_data.get_state_for_reset(n)
                
                # 设备/精度对齐
                todev = lambda t: t.to(device=self.device, dtype=torch.float32)
                quat_xyzw  = todev(quat_xyzw)
                qpos       = todev(qpos)
                qvel       = todev(qvel)
                vlin_local = todev(vlin_local)
                vang_local = todev(vang_local)

                # 归一化四元数
                quat_xyzw = torch.nn.functional.normalize(quat_xyzw, dim=-1)

                # DOF维度对齐
                if qpos.shape[1] != self.num_dof:
                    if qpos.shape[1] > self.num_dof:
                        qpos = qpos[:, :self.num_dof]
                        qvel = qvel[:, :self.num_dof]
                    else:
                        pad = self.num_dof - qpos.shape[1]
                        qpos = torch.nn.functional.pad(qpos, (0, pad))
                        qvel = torch.nn.functional.pad(qvel, (0, pad))

                # 局部→世界速度变换
                vlin_world = quat_rotate(quat_xyzw, vlin_local)
                vang_world = quat_rotate(quat_xyzw, vang_local)

                # 写入状态缓冲区
                ids = env_ids.long()
                self.root_states[ids, 3:7]   = quat_xyzw
                self.root_states[ids, 7:10]  = vlin_world
                self.root_states[ids, 10:13] = vang_world

                if hasattr(self, "base_quat"):
                    self.base_quat[ids] = quat_xyzw
                if hasattr(self, "base_lin_vel"):
                    self.base_lin_vel[ids] = vlin_world
                if hasattr(self, "base_ang_vel"):
                    self.base_ang_vel[ids] = vang_world

                self.dof_pos[ids, :] = qpos
                self.dof_vel[ids, :] = qvel
                if hasattr(self, "dof_state") and self.dof_state.ndim == 3:
                    self.dof_state[ids, :, 0] = qpos
                    self.dof_state[ids, :, 1] = qvel

                # push到仿真
                try:
                    index_device = self.root_states.device
                    env_ids_i32 = env_ids.to(device=index_device, dtype=torch.int32)

                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(self.root_states),
                        gymtorch.unwrap_tensor(env_ids_i32),
                        env_ids_i32.numel(),
                    )

                    if hasattr(self, "dof_state") and self.dof_state.ndim == 3:
                        self.gym.set_dof_state_tensor_indexed(
                            self.sim,
                            gymtorch.unwrap_tensor(self.dof_state),
                            gymtorch.unwrap_tensor(env_ids_i32),
                            env_ids_i32.numel(),
                        )
                except Exception as e:
                    print(f"[TiV2AMPRobot] Warning: failed to push AMP reset to sim: {e}")
                    
            except Exception as e:
                print(f"[TiV2AMPRobot] Warning: get_state_for_reset failed ({e}), fallback to default reset.")

        # 3) 初始化AMP观测历史
        self._init_amp_obs_for_reset(env_ids)

    # ===== 新增：支持从专家数据构建AMP观测demo（用于判别器训练）=====
    def fetch_amp_obs_demo(self, num_samples):
        """
        从专家数据采样AMP观测demo，用于判别器训练
        返回形状: [num_samples, num_amp_obs_steps * obs_per_step]
        """
        if self.amp_data is None:
            # 如果没有专家数据，返回零张量
            return torch.zeros((num_samples, self.num_amp_obs), device=self.device)
        
        try:
            # 简化版本：直接从当前专家数据采样
            # 更完整的实现需要考虑时间步长和历史构建
            demo_obs_list = []
            for _ in range(self._num_amp_obs_steps):
                # 为每个时间步采样专家状态
                quat_xyzw, qpos, qvel, vlin_local, vang_local = self.amp_data.get_state_for_reset(num_samples)
                
                # 构建单步AMP观测
                demo_obs = torch.cat([
                    qpos.to(self.device),
                    qvel.to(self.device),
                    vlin_local.to(self.device),
                    vang_local.to(self.device)
                ], dim=-1)
                demo_obs_list.append(demo_obs)
            
            # 堆叠成历史序列并展平
            amp_obs_demo = torch.stack(demo_obs_list, dim=1)  # [num_samples, steps, obs_per_step]
            return amp_obs_demo.view(num_samples, -1)  # [num_samples, total_obs]
            
        except Exception as e:
            print(f"[TiV2AMPRobot] Warning: fetch_amp_obs_demo failed ({e}), returning zeros")
            return torch.zeros((num_samples, self.num_amp_obs), device=self.device)