from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import torch
from isaacgym.torch_utils import quat_rotate, quat_mul, quat_rotate_inverse


# ---------------- math helpers (xyzw) ----------------
def calc_heading_quat_inv_xyzw(q_xyzw: torch.Tensor) -> torch.Tensor:
    """保留世界系 yaw，返回其逆（xyzw）。"""
    try:
        from amp_rsl_rl.utils import torch_utils
        return torch_utils.calc_heading_quat_inv(q_xyzw)
    except Exception:
        q = torch.zeros_like(q_xyzw)
        q[..., 3] = 1.0
        return q


def quat_to_tan_norm_xyzw(q_xyzw: torch.Tensor) -> torch.Tensor:
    """quat->6D tan-norm；若不可用则退化为 quat+零填充。"""
    try:
        from amp_rsl_rl.utils import torch_utils
        return torch_utils.quat_to_tan_norm(q_xyzw)
    except Exception:
        pad = torch.zeros(q_xyzw.shape[:-1] + (2,), device=q_xyzw.device, dtype=q_xyzw.dtype)
        return torch.cat([q_xyzw, pad], dim=-1)


# ---------------- per-step AMP obs builder ----------------
def build_amp_obs_per_step(
    base_pos_world: torch.Tensor,      # [B,3]
    base_quat_xyzw: torch.Tensor,      # [B,4]
    base_lin_vel_world: torch.Tensor,  # [B,3]
    base_ang_vel_world: torch.Tensor,  # [B,3]
    dof_pos: torch.Tensor,             # [B,J]
    dof_vel: torch.Tensor,             # [B,J]
    key_body_pos_world: Optional[torch.Tensor],  # [B,Kb,3] or None
    use_root_h: bool,
    expect_dof_obs_dim: int,
    expect_key_bodies: int,
) -> torch.Tensor:
    B = base_pos_world.shape[0]
    dev, dtype = base_pos_world.device, base_pos_world.dtype

    root_h = base_pos_world[:, 2:3] if use_root_h else torch.zeros((B, 1), device=dev, dtype=dtype)
    heading_inv = calc_heading_quat_inv_xyzw(base_quat_xyzw)       # [B,4]
    root_local = quat_mul(heading_inv, base_quat_xyzw)             # [B,4]
    root_rot_tan = quat_to_tan_norm_xyzw(root_local)               # [B,6]
    v_lin_local = quat_rotate(heading_inv, base_lin_vel_world)     # [B,3]
    v_ang_local = quat_rotate(heading_inv, base_ang_vel_world)     # [B,3]

    Jexp = int(expect_dof_obs_dim)
    if dof_pos.shape[1] >= Jexp:
        dof_obs = dof_pos[:, :Jexp]
        dof_vel_obs = dof_vel[:, :Jexp]
    else:
        pad = Jexp - dof_pos.shape[1]
        z = torch.zeros((B, pad), device=dev, dtype=dtype)
        dof_obs = torch.cat([dof_pos, z], dim=-1)
        dof_vel_obs = torch.cat([dof_vel, z], dim=-1)

    Kbexp = int(expect_key_bodies)
    if key_body_pos_world is None or key_body_pos_world.numel() == 0 or Kbexp <= 0:
        key_local_flat = torch.zeros((B, 3 * max(0, Kbexp)), device=dev, dtype=dtype)
    else:
        Kb = key_body_pos_world.shape[1]
        rel = key_body_pos_world - base_pos_world.unsqueeze(1)   # [B,Kb,3]
        hi = heading_inv.unsqueeze(1).expand(-1, Kb, -1).reshape(-1, 4)
        local = quat_rotate(hi, rel.reshape(-1, 3)).reshape(B, Kb, 3)
        key_local_flat = local.reshape(B, 3 * Kb)
        if Kb < Kbexp:
            pad = (Kbexp - Kb) * 3
            key_local_flat = torch.cat([key_local_flat, torch.zeros((B, pad), device=dev, dtype=dtype)], dim=-1)
        elif Kb > Kbexp:
            key_local_flat = key_local_flat[:, : 3 * Kbexp]

    return torch.cat([root_h, root_rot_tan, v_lin_local, v_ang_local, dof_obs, dof_vel_obs, key_local_flat], dim=-1)


# ---------------- env helpers ----------------
def unpack_env_observations(env) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    got = env.get_observations()
    obs = got[0] if isinstance(got, tuple) else got
    priv = None
    if hasattr(env, "get_privileged_observations"):
        priv = env.get_privileged_observations()
    return obs, priv


def unpack_env_step(ret: Any) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    if isinstance(ret, tuple):
        if len(ret) >= 7:
            obs, priv, rewards, dones, infos, term_ids, term_priv = ret[:7]
            return obs, priv, rewards, dones, infos, term_ids, term_priv
        if len(ret) == 4:
            obs, rewards, dones, infos = ret
            return obs, None, rewards, dones, infos, None, None
    raise RuntimeError(f"Unsupported env.step return: {type(ret)}")


def merge_amp_cfg(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    amp = dict(train_cfg.get("amp", {}))
    for k in (
        "amp_data_path","dataset_names","dataset_weights","slow_down_factor",
        "num_amp_obs","dt","decimation","replay_buffer_size","reward_scale",
        "joint_names","style_weight","task_weight",
        "num_amp_obs_steps","history_steps","history_stride",
        "motion_file","mjcf_file","extend_hand","extend_head",
        "expect_dof_obs_dim","key_body_ids","bootstrap_motions","motionlib_class",
        "use_joint_mapping","debug_joint_mapping",
    ):
        if k in train_cfg and k not in amp:
            amp[k] = train_cfg[k]
    return amp


def build_amp_obs_from_obs(env, obs: torch.Tensor, num_amp_obs: int) -> torch.Tensor:
    """从 env 内部状态快速构建 AMP obs；不可用则退化为裁剪/填充 obs。"""
    if hasattr(env, 'compute_amp_observations'):
        return env.compute_amp_observations()

    try:
        if all(hasattr(env, a) for a in ('dof_pos', 'dof_vel', 'base_quat', 'root_states')):
            base_lin_vel_local = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
            base_ang_vel_local = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
            amp_obs = torch.cat([env.dof_pos, env.dof_vel, base_lin_vel_local, base_ang_vel_local], dim=-1)
            return _fit_dim(amp_obs, num_amp_obs)
    except Exception:
        pass

    return _fit_dim(obs, num_amp_obs)


def _fit_dim(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[-1] == n:
        return x
    if x.shape[-1] > n:
        return x[..., :n]
    pad = n - x.shape[-1]
    return torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device, dtype=x.dtype)], dim=-1)
