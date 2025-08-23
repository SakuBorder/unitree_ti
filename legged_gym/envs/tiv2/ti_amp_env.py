# legged_gym/envs/tiv2/ti_amp_env.py
from legged_gym.envs.tiv2.tiv2_env import TiV2Robot
import torch
from isaacgym.torch_utils import *  # quat_rotate, quat_rotate_inverse, etc.
from isaacgym import gymtorch
import torch.nn.functional as F


class TiV2AMPRobot(TiV2Robot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.num_amp_obs = self._get_amp_obs_dim()
        self.amp_data = None  # 由 runner 注入（见 set_amp_data）

        # 确保 phase 属性存在（父类 compute_observations 可能需要）
        if not hasattr(self, 'phase'):
            print("[TiV2AMPRobot] Adding missing 'phase' attribute")
            self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # 调试信息：确认观测维度
        print(f"[TiV2AMPRobot] Actor obs dim: {self.num_obs}")
        print(f"[TiV2AMPRobot] Critic obs dim: {self.num_privileged_obs}")
        print(f"[TiV2AMPRobot] AMP obs dim: {self.num_amp_obs}")

    # --------- Runner 注入 AMPLoader 句柄（便于 reset 热启动）---------
    def set_amp_data(self, amp_data):
        """由 Runner 注入 AMPLoader 句柄。"""
        self.amp_data = amp_data

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

    # ------------------- 关键新增：专家帧“热启动”复位 -------------------
    def reset_idx(self, env_ids):
        """
        覆盖父类 reset，以“专家帧热启动”重置这些 env：
        - 若未注入 amp_data，则回退到父类默认行为。
        - 若已注入，从专家数据随机采样 (quat_wxyz, qpos, qvel, vlin_local, vang_local)，
        写入 root_states 与 dof 状态，并 push 到仿真。
        """
        # 1) 规范化 env_ids -> 1D Long tensor on device
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids)
        env_ids = env_ids.to(self.device, dtype=torch.long).reshape(-1)

        # ★ 空批次直接早退，避免 0 样本采样/警告
        if env_ids.numel() == 0:
            return

        # 2) 先让父类做地形/统计复位与一次 push
        super().reset_idx(env_ids)

        # 3) 没有 amp_data 就保留默认复位
        if getattr(self, "amp_data", None) is None:
            return

        # 4) 用专家帧覆盖
        n = env_ids.numel()
        try:
            quat, qpos, qvel, vlin_local, vang_local = self.amp_data.get_state_for_reset(n)
        except Exception as e:
            # 真异常才打印（空批次不会走到这里）
            print(f"[TiV2AMPRobot] Warning: get_state_for_reset failed ({e}), fallback to default reset.")
            return

        # 设备/精度对齐
        todev = lambda t: t.to(device=self.device, dtype=torch.float32)
        quat        = todev(quat)        # (N, 4) wxyz
        qpos        = todev(qpos)        # (N, dof?)
        qvel        = todev(qvel)        # (N, dof?)
        vlin_local  = todev(vlin_local)  # (N, 3)
        vang_local  = todev(vang_local)  # (N, 3)

        # DOF 维度对齐（数据集维度与环境不同则裁/补）
        if qpos.shape[1] != self.num_dof:
            if qpos.shape[1] > self.num_dof:
                qpos = qpos[:, :self.num_dof]
                qvel = qvel[:, :self.num_dof]
            else:
                pad = self.num_dof - qpos.shape[1]
                qpos = torch.nn.functional.pad(qpos, (0, pad))
                qvel = torch.nn.functional.pad(qvel, (0, pad))

        # 局部 → 世界 的速度变换
        vlin_world = quat_rotate(quat, vlin_local)
        vang_world = quat_rotate(quat, vang_local)

        # 写入本地缓冲区
        ids = env_ids.long()  # 已在 self.device 上
        # 根状态（仅覆盖姿态/速度，位置沿用父类默认出生点）
        self.root_states[ids, 3:7]   = quat
        self.root_states[ids, 7:10]  = vlin_world
        self.root_states[ids, 10:13] = vang_world

        if hasattr(self, "base_quat"):
            self.base_quat[ids] = quat
        if hasattr(self, "base_lin_vel"):
            self.base_lin_vel[ids] = vlin_world
        if hasattr(self, "base_ang_vel"):
            self.base_ang_vel[ids] = vang_world

        # 关节状态
        self.dof_pos[ids, :] = qpos
        self.dof_vel[ids, :] = qvel
        if hasattr(self, "dof_state") and self.dof_state.ndim == 3:
            self.dof_state[ids, :, 0] = qpos
            self.dof_state[ids, :, 1] = qvel

        # 5) push 到仿真（索引张量必须与状态张量在同一设备；GPU 管线必须传 GPU 张量）
        try:
            index_device = self.root_states.device  # 通常是 cuda:0（GPU PhysX）
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

