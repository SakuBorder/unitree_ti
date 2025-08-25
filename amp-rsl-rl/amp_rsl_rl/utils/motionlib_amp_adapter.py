# motionlib_amp_adapter.py
import os
import torch
from typing import List, Optional, Tuple

# 与 HumanoidAMP 的实现保持一致：使用 torch_utils（你环境里已有）
from amp_rsl_rl.utils import torch_utils  # 若路径不同，换成你的 torch_utils 导入
from isaacgym.torch_utils import quat_rotate, quat_mul  # 若你只用 torch_utils 也可全换成它

class MotionLibAMPAdapter:
    """
    用 MotionLib 在线生成 AMP 训练对儿 (obs_t, obs_{t+1}) 的适配器。
    暴露与 AMPLoader 相同/近似的接口，方便替换：
      - feed_forward_generator(num_mini_batch, mini_batch_size)
      - get_state_for_reset(number_of_samples)
    其余：sample_item_dim（=K*D），num_amp_obs_per_step（=D）
    """

    def __init__(
        self,
        motion_lib,                     # 已实例化的 MotionLibTaihu / MotionLibBase 子类
        dt: float,                      # 仿真步长 * 控制 decimation（与环境 delta_t 一致）
        history_steps: int = 2,         # K（≥2）
        history_stride: int = 1,        # S
        expect_dof_obs_dim: int = 12,   # 与环境一致
        expect_key_bodies: int = 2,     # 关键体数量（如：双脚/双手）
        key_body_ids: Optional[List[int]] = None,  # 在 motion_lib 返回的 rg_pos 中要取的身体索引
        use_root_h: Optional[bool] = None,         # 是否包含 root height；若 None 则读 AMP_USE_ROOT_H
        device: torch.device = torch.device("cpu"),
    ):
        self.motion_lib = motion_lib
        self.dt = float(dt)
        self.K = max(2, int(history_steps))
        self.S = max(1, int(history_stride))
        self.expect_dof_obs_dim = int(expect_dof_obs_dim)
        self.expect_key_bodies = int(expect_key_bodies)
        self.key_body_ids = key_body_ids
        self.device = device

        # 根高开关：与 HumanoidAMP 保持一致，默认读环境变量
        if use_root_h is None:
            use_root_h = str(os.environ.get("AMP_USE_ROOT_H", "0")) == "1"
        self.use_root_h = bool(use_root_h)

        # 试采 1 次确定单步维度
        _probe = self._sample_amp_window(batch_size=1)
        self.num_amp_obs_per_step = int(_probe["obs"].shape[-1] // self.K)
        self.sample_item_dim = int(_probe["obs"].shape[-1])  # = K*D

    # ============== 对外接口（兼容 AMPLoader）==============

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Yield num_mini_batch 次，每次输出 (obs, next_obs)，形状 [B, K*D]。"""
        for _ in range(num_mini_batch):
            out = self._sample_amp_window(batch_size=mini_batch_size)
            yield out["obs"], out["next_obs"]

    @torch.no_grad()
    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        """
        返回 (quat_xyzw, q, dq, v_lin_local, v_ang_local)，形状：
          quat: [N, 4]; q/dq: [N, J]; v_lin/v_ang: [N, 3]
        注：J 由 motion_lib 的 dof_pos 推断；速度旋到 heading 局部系。
        """
        n = int(number_of_samples)
        if n <= 0:
            return (
                torch.empty((0, 4), device=self.device),
                torch.empty((0, 0), device=self.device),
                torch.empty((0, 0), device=self.device),
                torch.empty((0, 3), device=self.device),
                torch.empty((0, 3), device=self.device),
            )

        mids = self.motion_lib.sample_motions(n)
        # 用 K=1 的时间，避免负时间
        t0 = self.motion_lib.sample_time(mids, truncate_time=0.0)

        st = self.motion_lib.get_motion_state(mids, t0)
        # 取根姿、DOF、速度
        root_rot = st["root_rot"]              # [N, 4]  xyzw（MotionLibTaihu 已是 xyzw）
        dof_pos  = st["dof_pos"]               # [N, J]
        dof_vel  = st["dof_vel"].reshape(n, -1)  # [N, J]（你的实现里本来就展平）
        v_lin_w  = st["root_vel"]              # [N, 3] 世界
        v_ang_w  = st["root_ang_vel"]          # [N, 3] 世界

        # 旋到“朝向”系
        heading_inv = torch_utils.calc_heading_quat_inv(root_rot)      # [N,4] xyzw
        v_lin_local = quat_rotate(heading_inv, v_lin_w)                # [N,3]
        v_ang_local = quat_rotate(heading_inv, v_ang_w)                # [N,3]

        return (root_rot, dof_pos, dof_vel, v_lin_local, v_ang_local)

    # ============== 内部：采样并构造 AMP K 窗口 ==============

    @torch.no_grad()
    def _sample_amp_window(self, batch_size: int):
        B = int(batch_size)
        K, S, dt = self.K, self.S, self.dt

        mids = self.motion_lib.sample_motions(B)

        # 确保能向过去回溯 K-1 步（每步跨度 S*dt）
        truncate_time = (K - 1) * S * dt + 1e-8
        t0 = self.motion_lib.sample_time(mids, truncate_time=truncate_time) + truncate_time

        # 组装 times: [B, K]，从 t0, t0 - S*dt, ..., t0 - (K-1)S*dt
        steps = torch.arange(0, K, device=self.device, dtype=torch.float32) * (S * dt)
        times = t0.unsqueeze(1) - steps.unsqueeze(0)  # [B, K]
        times_next = times + dt                       # 下一时刻窗口（整体往前 dt）

        # 批量展平查询
        obs = self._build_flattened_obs(mids, times)         # [B, K*D]
        next_obs = self._build_flattened_obs(mids, times_next)

        return {"obs": obs, "next_obs": next_obs}

    def _build_flattened_obs(self, mids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        给定 [B, K] 的 times，调用 motion_lib.get_motion_state 逐列拼接 K 帧，
        每帧经 build_amp_obs_per_step 后展平为 [B, K*D]
        """
        B, K = times.shape
        per_steps: List[torch.Tensor] = []

        for j in range(K):
            st = self.motion_lib.get_motion_state(mids, times[:, j])
            # 可能含：'rg_pos'（全部刚体世界位置）——可用于关键体
            key_world = None
            if self.key_body_ids is not None and "rg_pos" in st:
                key_world = st["rg_pos"][:, self.key_body_ids, :]  # [B, Kb, 3]

            step = self._build_amp_obs_per_step(
                base_pos_world=st["root_pos"],            # [B,3]
                base_quat_xyzw=st["root_rot"],            # [B,4] xyzw
                base_lin_vel_world=st["root_vel"],        # [B,3]
                base_ang_vel_world=st["root_ang_vel"],    # [B,3]
                dof_pos=st["dof_pos"],                    # [B,J]（已按 mapping）
                dof_vel=st["dof_vel"].reshape(B, -1),     # [B,J]
                key_body_pos_world=key_world,             # [B,Kb,3] 或 None
            )                                             # [B, D]
            per_steps.append(step)

        obs = torch.stack(per_steps, dim=1).reshape(B, -1)  # [B, K, D] → [B, K*D]
        return obs

    def _build_amp_obs_per_step(
        self,
        base_pos_world: torch.Tensor,      # [B,3]
        base_quat_xyzw: torch.Tensor,      # [B,4]  xyzw
        base_lin_vel_world: torch.Tensor,  # [B,3]
        base_ang_vel_world: torch.Tensor,  # [B,3]
        dof_pos: torch.Tensor,             # [B,J]
        dof_vel: torch.Tensor,             # [B,J]
        key_body_pos_world: Optional[torch.Tensor],  # [B,Kb,3] or None
    ) -> torch.Tensor:
        """
        与 HumanoidAMP.build_amp_observations 一致的单步 D 维构造：
        [root_h(1), root_rot_tan_norm(6), v_lin_local(3), v_ang_local(3),
         dof_obs( expect_dof_obs_dim ), dof_vel(expect_dof_obs_dim),
         key_local(3*expect_key_bodies)]
        """
        B = base_pos_world.shape[0]
        dev = base_pos_world.device
        dtype = base_pos_world.dtype

        # root_height
        if self.use_root_h:
            root_h = base_pos_world[:, 2:3]         # [B,1]
        else:
            root_h = torch.zeros((B, 1), device=dev, dtype=dtype)

        # heading 及局部旋转
        heading_inv = torch_utils.calc_heading_quat_inv(base_quat_xyzw)     # [B,4] xyzw
        root_rot_local = quat_mul(heading_inv, base_quat_xyzw)              # [B,4] xyzw
        root_rot_tan_norm = torch_utils.quat_to_tan_norm(root_rot_local)    # [B,6]

        # 根速度旋到 heading 系
        v_lin_local = quat_rotate(heading_inv, base_lin_vel_world)          # [B,3]
        v_ang_local = quat_rotate(heading_inv, base_ang_vel_world)          # [B,3]

        # dof 裁剪/补零
        if dof_pos.shape[1] >= self.expect_dof_obs_dim:
            dof_obs = dof_pos[:, :self.expect_dof_obs_dim]
            dof_vel_obs = dof_vel[:, :self.expect_dof_obs_dim]
        else:
            pad = self.expect_dof_obs_dim - dof_pos.shape[1]
            z = torch.zeros((B, pad), device=dev, dtype=dtype)
            dof_obs = torch.cat([dof_pos, z], dim=-1)
            dof_vel_obs = torch.cat([dof_vel, z], dim=-1)

        # 关键体：相对根 + 旋到 heading 系
        if key_body_pos_world is None or key_body_pos_world.numel() == 0:
            key_local_flat = torch.zeros((B, 3 * self.expect_key_bodies), device=dev, dtype=dtype)
        else:
            Kb = key_body_pos_world.shape[1]
            rel = key_body_pos_world - base_pos_world.unsqueeze(1)          # [B,Kb,3]
            heading_expand = heading_inv.unsqueeze(1).expand(-1, Kb, -1)    # [B,Kb,4]
            rel_flat = rel.reshape(-1, 3)
            head_flat = heading_expand.reshape(-1, 4)
            local = quat_rotate(head_flat, rel_flat).reshape(B, Kb, 3)      # [B,Kb,3]
            key_local_flat = local.reshape(B, 3 * Kb)
            if Kb < self.expect_key_bodies:
                pad = (self.expect_key_bodies - Kb) * 3
                key_local_flat = torch.cat([key_local_flat,
                                            torch.zeros((B, pad), device=dev, dtype=dtype)], dim=-1)
            elif Kb > self.expect_key_bodies:
                key_local_flat = key_local_flat[:, : 3 * self.expect_key_bodies]

        return torch.cat(
            [root_h, root_rot_tan_norm, v_lin_local, v_ang_local, dof_obs, dof_vel_obs, key_local_flat],
            dim=-1
        )
