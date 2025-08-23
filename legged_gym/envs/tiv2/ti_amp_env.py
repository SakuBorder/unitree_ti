# legged_gym/envs/tiv2/ti_amp_env.py
from legged_gym.envs.tiv2.tiv2_env import TiV2Robot
import torch
from isaacgym.torch_utils import *


class TiV2AMPRobot(TiV2Robot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.num_amp_obs = self._get_amp_obs_dim()

        # 确保 phase 属性存在（父类 compute_observations 可能需要）
        if not hasattr(self, 'phase'):
            print("[TiV2AMPRobot] Adding missing 'phase' attribute")
            self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # 调试信息：确认观测维度
        print(f"[TiV2AMPRobot] Actor obs dim: {self.num_obs}")
        print(f"[TiV2AMPRobot] Critic obs dim: {self.num_privileged_obs}")
        print(f"[TiV2AMPRobot] AMP obs dim: {self.num_amp_obs}")

    def _get_amp_obs_dim(self):
        """计算 AMP 观测的维度：12 dof pos + 12 dof vel + 3 lin vel + 3 ang vel = 30"""
        return self.num_dof * 2 + 3 + 3  # 12 + 12 + 3 + 3 = 30

    def compute_amp_observations(self):
        """计算 AMP 所需的观测"""
        # 将基座速度转换到局部坐标系
        base_lin_vel_local = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        base_ang_vel_local = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        amp_obs = torch.cat([
            self.dof_pos,           # 关节位置 (12)
            self.dof_vel,           # 关节速度 (12)
            base_lin_vel_local,     # 局部线速度 (3)
            base_ang_vel_local,     # 局部角速度 (3)
        ], dim=-1)  # 总共 30 维

        return amp_obs

    def get_observations(self):
        """重写 get_observations 以兼容 AMPOnPolicyRunner"""
        # 安全地调用父类的 compute_observations()
        try:
            super().compute_observations()
        except AttributeError as e:
            print(f"[Warning] Parent compute_observations failed: {e}")
            print("[Warning] Using observation buffers directly")

        # 获取观测
        actor_obs = self.obs_buf  # 历史观测栈 (例如 282 维)
        critic_obs = self.privileged_obs_buf if self.privileged_obs_buf is not None else self.obs_buf
        amp_obs = self.compute_amp_observations()  # AMP 观测 (30 维)

        # 调试信息（仅在第一次调用时显示）
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

        # 构建 extras 字典（runner 当前不会用到，但保留不影响）
        extras = {
            "observations": {
                "amp": amp_obs,
                "critic": critic_obs
            }
        }

        # 返回 (actor_obs, extras)：runner 会忽略 extras，并另行调用 get_privileged_observations()
        return actor_obs, extras

    def step(self, actions):
        """重写 step：保持返回**7元组**，从而让 runner 正确拿到 50 维的 privileged_obs"""
        # 调用父类 step（期望返回 7 元组）
        # (obs, privileged_obs, rewards, dones, infos, termination_ids, termination_obs)
        ret = super().step(actions)

        # 兼容父类不同返回签名：若只返回 4 元组，则补齐为 7 元组（不推荐，但兜底）
        if not isinstance(ret, tuple):
            raise RuntimeError(f"[TiV2AMPRobot.step] Unexpected return type from super().step: {type(ret)}")

        if len(ret) == 7:
            obs, privileged_obs, rewards, dones, infos, termination_ids, termination_obs = ret
        elif len(ret) == 4:
            # 强烈建议修正父类使其返回 7 元组，这里只是最后兜底
            obs, rewards, dones, infos = ret
            privileged_obs = self.get_privileged_observations()
            termination_ids = None
            termination_obs = None
            print("[TiV2AMPRobot.step][Warning] Parent step returned 4-tuple; "
                  "filled privileged_obs via get_privileged_observations(). Please fix parent to return 7-tuple.")
        else:
            raise RuntimeError(f"[TiV2AMPRobot.step] Unsupported super().step() signature with len={len(ret)}")

        # 确保 privileged_obs 维度为 num_privileged_obs
        if privileged_obs is not None and privileged_obs.shape[-1] != self.num_privileged_obs:
            if privileged_obs.shape[-1] > self.num_privileged_obs:
                privileged_obs = privileged_obs[..., :self.num_privileged_obs]
            else:
                padding = self.num_privileged_obs - privileged_obs.shape[-1]
                privileged_obs = torch.cat(
                    [privileged_obs, torch.zeros(privileged_obs.shape[0], padding, device=privileged_obs.device)],
                    dim=-1
                )

        # （可选）把 critic 观测放到 infos，便于调试；runner 不依赖这个字段
        if isinstance(infos, dict):
            infos.setdefault("observations", {})
            infos["observations"]["critic"] = privileged_obs if privileged_obs is not None else obs

        # **关键**：返回 7 元组，保持 API 与父类一致
        return obs, privileged_obs, rewards, dones, infos, termination_ids, termination_obs

    def get_privileged_observations(self):
        """确保 privileged observations 返回正确维度"""
        try:
            priv_obs = super().get_privileged_observations()
        except AttributeError as e:
            print(f"[Warning] Parent get_privileged_observations failed: {e}")
            # 如果父类方法失败，使用 privileged_obs_buf
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
                # 如果是 phase 属性问题，创建一个临时的 phase
                if not hasattr(self, 'phase'):
                    self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
                # 重试调用
                try:
                    super().compute_observations()
                except Exception as e2:
                    print(f"[Error] Still failed after adding phase: {e2}")
                    # 如果还是失败，手动更新观测缓冲区
                    self._manual_compute_observations()
            else:
                raise e

    def _manual_compute_observations(self):
        """手动计算观测（如果父类方法失败）"""
        print("[Warning] Using manual observation computation")
        # 这里可以实现基本的观测计算逻辑；作为最后回退方案
        pass

    def _reward_alive(self):
        """奖励存活"""
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
