# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import List, Union, Tuple, Generator, Optional
from dataclasses import dataclass
import os
import joblib

import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

# === 与环境一致的四元数工具 ===
from isaacgym.torch_utils import quat_rotate, quat_from_angle_axis, quat_mul


def download_amp_dataset_from_hf(
    destination_dir: Path,
    robot_folder: str,
    files: list,
    repo_id: str = "ami-iit/amp-dataset",
) -> list:
    """
    Downloads AMP dataset files from Hugging Face and saves them to `destination_dir`.
    Ensures real file copies (not symlinks or hard links).
    """
    from huggingface_hub import hf_hub_download

    destination_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = []

    for file in files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{robot_folder}/{file}",
            repo_type="dataset",
            local_files_only=False,
        )
        local_copy = destination_dir / file
        # Deep copy to avoid symlinks
        with open(file_path, "rb") as src_file, open(local_copy, "wb") as dst_file:
            dst_file.write(src_file.read())

        # Remove extension for dataset name
        if file.endswith(".npy"):
            dataset_names.append(file.replace(".npy", ""))
        elif file.endswith(".pkl"):
            dataset_names.append(file.replace(".pkl", ""))
        else:
            dataset_names.append(file)

    return dataset_names


# =========================
# 工具：与环境对齐的旋转处理
# =========================

def _calc_heading_quat_inv(q_xyzw: torch.Tensor) -> torch.Tensor:
    """从四元数提取“朝向”（绕 z）并取逆，xyzw。"""
    ref = torch.zeros_like(q_xyzw[..., :3])
    ref[..., 0] = 1.0
    dir_xy = quat_rotate(q_xyzw, ref)
    heading = torch.atan2(dir_xy[..., 1], dir_xy[..., 0])
    axis = torch.zeros_like(ref)
    axis[..., 2] = 1.0
    return quat_from_angle_axis(-heading, axis)

def _quat_to_tan_norm(q_xyzw: torch.Tensor) -> torch.Tensor:
    """把四元数表示成 6 维 tan-norm（旋转 x 轴、z 轴后拼接）。"""
    ref_tan = torch.zeros_like(q_xyzw[..., :3]); ref_tan[..., 0] = 1.0
    ref_norm = torch.zeros_like(q_xyzw[..., :3]); ref_norm[..., 2] = 1.0
    tan  = quat_rotate(q_xyzw, ref_tan)
    norm = quat_rotate(q_xyzw, ref_norm)
    return torch.cat([tan, norm], dim=-1)  # (..., 6)

def _build_amp_obs_per_step(
    base_pos_world: torch.Tensor,      # (T, 3)
    base_quat_xyzw: torch.Tensor,      # (T, 4)
    base_lin_vel_world: torch.Tensor,  # (T, 3)
    base_ang_vel_world: torch.Tensor,  # (T, 3)
    dof_pos: torch.Tensor,             # (T, J)
    dof_vel: torch.Tensor,             # (T, J)
    key_body_pos_world: Optional[torch.Tensor],  # (T, K, 3) or None
    expect_dof_obs_dim: int = 12,
    expect_key_bodies: int = 2,
) -> torch.Tensor:
    """
    与 Humanoid AMP 对齐的单步观测：
      [root_h(1), root_rot_tan_norm(6), local_root_vel(3), local_root_ang_vel(3),
       dof_obs(12), dof_vel(12), local_key_body_pos(3*K)] -> K=2 → 43 维
    """
    T = base_pos_world.shape[0]
    device = base_pos_world.device
    dtype  = base_pos_world.dtype

    # 1) root_h
    root_h = base_pos_world[:, 2:3]  # (T,1)

    # 2) 根姿态做 heading 归一 → tan-norm(6)
    heading_inv = _calc_heading_quat_inv(base_quat_xyzw)     # (T,4)
    root_rot_local = quat_mul(heading_inv, base_quat_xyzw)   # (T,4)
    root_rot_tan_norm = _quat_to_tan_norm(root_rot_local)    # (T,6)

    # 3) 根速度旋到“朝向坐标系”
    local_root_vel     = quat_rotate(heading_inv, base_lin_vel_world)   # (T,3)
    local_root_ang_vel = quat_rotate(heading_inv, base_ang_vel_world)   # (T,3)

    # 4) dof obs（取前 12 维，不足则补零）
    if dof_pos.shape[1] >= expect_dof_obs_dim:
        dof_obs = dof_pos[:, :expect_dof_obs_dim]
        dof_vel_obs = dof_vel[:, :expect_dof_obs_dim]
    else:
        pad = expect_dof_obs_dim - dof_pos.shape[1]
        dof_obs = torch.cat([dof_pos, torch.zeros(T, pad, device=device, dtype=dtype)], dim=-1)
        dof_vel_obs = torch.cat([dof_vel, torch.zeros(T, pad, device=device, dtype=dtype)], dim=-1)

    # 5) 关键体位：相对根 + 旋到朝向系 → 展平 (T, 3*K)，K=2 不足补零，多了裁剪
    if key_body_pos_world is None or key_body_pos_world.numel() == 0:
        local_key_flat = torch.zeros((T, 3 * expect_key_bodies), device=device, dtype=dtype)
    else:
        K = key_body_pos_world.shape[1]
        root_pos_expand = base_pos_world.unsqueeze(1)                 # (T,1,3)
        key_rel = key_body_pos_world - root_pos_expand                # (T,K,3)

        heading_expand = heading_inv.unsqueeze(1).expand(-1, K, -1)   # (T,K,4)
        key_rel_flat = key_rel.reshape(-1, 3)
        heading_flat = heading_expand.reshape(-1, 4)
        local_key = quat_rotate(heading_flat, key_rel_flat).reshape(T, K, 3)
        local_key_flat = local_key.reshape(T, 3 * K)

        if K < expect_key_bodies:
            pad = (expect_key_bodies - K) * 3
            local_key_flat = torch.cat([local_key_flat, torch.zeros(T, pad, device=device, dtype=dtype)], dim=-1)
        elif K > expect_key_bodies:
            local_key_flat = local_key_flat[:, : 3 * expect_key_bodies]

    # 6) 拼接 43 维
    obs = torch.cat([
        root_h,
        root_rot_tan_norm,
        local_root_vel,
        local_root_ang_vel,
        dof_obs,
        dof_vel_obs,
        local_key_flat,
    ], dim=-1)
    return obs  # (T, 43)


@dataclass
class MotionData:
    """
    Data class representing motion data for humanoid agents.

    Attributes (shapes by time T):
        - joint_positions: (T, J)
        - joint_velocities: (T, J)
        - base_positions: (T, 3)                  <-- 新增，供 root_h 使用
        - base_lin_velocities_mixed: (T, 3)
        - base_ang_velocities_mixed: (T, 3)
        - base_lin_velocities_local: (T, 3)
        - base_ang_velocities_local: (T, 3)
        - base_quat: (T, 4) quaternion in **xyzw**
        - key_body_positions_world: (T, K, 3) or None  <-- 可选
    """

    joint_positions: Union[torch.Tensor, np.ndarray]
    joint_velocities: Union[torch.Tensor, np.ndarray]
    base_positions: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
    base_quat: Union[Rotation, torch.Tensor, np.ndarray]
    key_body_positions_world: Optional[Union[torch.Tensor, np.ndarray]] = None
    device: torch.device = torch.device("cpu")

    # Optional metadata (filled by loader)
    _meta: Optional[dict] = None
    _group_idx: Optional[int] = None
    _path: Optional[Path] = None
    _format: Optional[str] = None

    def __post_init__(self) -> None:
        def to_tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = to_tensor(self.joint_positions)
        if isinstance(self.joint_velocities, np.ndarray):
            self.joint_velocities = to_tensor(self.joint_velocities)

        if isinstance(self.base_positions, np.ndarray):
            self.base_positions = to_tensor(self.base_positions)

        if isinstance(self.base_lin_velocities_mixed, np.ndarray):
            self.base_lin_velocities_mixed = to_tensor(self.base_lin_velocities_mixed)
        if isinstance(self.base_ang_velocities_mixed, np.ndarray):
            self.base_ang_velocities_mixed = to_tensor(self.base_ang_velocities_mixed)
        if isinstance(self.base_lin_velocities_local, np.ndarray):
            self.base_lin_velocities_local = to_tensor(self.base_lin_velocities_local)
        if isinstance(self.base_ang_velocities_local, np.ndarray):
            self.base_ang_velocities_local = to_tensor(self.base_ang_velocities_local)

        if isinstance(self.base_quat, Rotation):
            quat_xyzw = self.base_quat.as_quat()  # (T,4) xyzw
            self.base_quat = torch.tensor(quat_xyzw, device=self.device, dtype=torch.float32)
        elif isinstance(self.base_quat, np.ndarray):
            if self.base_quat.shape[-1] == 4:
                self.base_quat = torch.tensor(self.base_quat, device=self.device, dtype=torch.float32)

        if self.key_body_positions_world is not None and isinstance(self.key_body_positions_world, np.ndarray):
            self.key_body_positions_world = to_tensor(self.key_body_positions_world)

    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:
        """
        旧版的 30 维/步；保留以兼容（不在 Loader 中使用）。
        建议改用 Loader 内部的 _build_amp_obs_per_step。
        """
        return torch.cat(
            (
                self.joint_positions[indices],
                self.joint_velocities[indices],
                self.base_lin_velocities_local[indices],
                self.base_ang_velocities_local[indices],
            ),
            dim=1,
        )

    def get_state_for_reset(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return (quat, q, dq, v_lin_local, v_ang_local) for env resets."""
        return (
            self.base_quat[indices],
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
        )

    def get_random_sample_for_reset(self, items: int = 1) -> Tuple[torch.Tensor, ...]:
        indices = torch.randint(0, len(self), (items,), device=self.device)
        return self.get_state_for_reset(indices)


class AMPLoader:
    """
    Loader/processor for AMP datasets (NPY original, PKL retargeted).

    新增：每步 43 维 AMP 观测（与环境完全对齐）+ 历史滑窗（K 步）。
    兼容接口：
      - feed_forward_generator() 仍返回 (state, next_state) 形如 [B, K*D]
      - get_state_for_reset() 仍返回单帧（用于环境复位）
    """

    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        dataset_names: List[str],
        dataset_weights: List[float],
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None] = None,
        history_steps: Optional[int] = None,
        history_stride: Optional[int] = None,
    ) -> None:
        # 设备 / 路径
        self.device = torch.device(device) if isinstance(device, str) else device
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)

        # --- 历史窗口配置 ---
        env_k = os.environ.get("AMP_HISTORY_STEPS")
        env_s = os.environ.get("AMP_HISTORY_STRIDE")
        self.history_steps = int(history_steps if history_steps is not None else (int(env_k) if env_k else 1))
        self.history_stride = int(history_stride if history_stride is not None else (int(env_s) if env_s else 1))
        if self.history_steps < 1:
            self.history_steps = 1
        if self.history_stride < 1:
            self.history_stride = 1

        # ---------- 1) 解析 dataset_names ----------
        expanded_items: List[Tuple[int, Path, str]] = []
        group_to_files: List[List[Tuple[Path, str]]] = []

        for g_idx, name in enumerate(dataset_names):
            base_path = dataset_path_root / name
            files_in_group: List[Tuple[Path, str]] = []

            if base_path.is_dir():
                for p in sorted(base_path.rglob("*.pkl")):
                    files_in_group.append((p, "pkl"))
                for p in sorted(base_path.rglob("*.npy")):
                    files_in_group.append((p, "npy"))
                if not files_in_group:
                    print(f"[AMPLoader] WARNING: empty group folder: {base_path}")
            else:
                p_npy = dataset_path_root / f"{name}.npy"
                p_pkl = dataset_path_root / f"{name}.pkl"
                if p_npy.exists():
                    files_in_group.append((p_npy, "npy"))
                elif p_pkl.exists():
                    files_in_group.append((p_pkl, "pkl"))
                else:
                    print(f"[AMPLoader] WARNING: missing dataset: {p_npy} / {p_pkl}")

            group_to_files.append(files_in_group)
            for p, fmt in files_in_group:
                expanded_items.append((g_idx, p, fmt))

        if not expanded_items:
            raise FileNotFoundError(
                f"No dataset files found under {dataset_path_root} for names={dataset_names}"
            )

        # ---------- 2) 自动构建 expected_joint_names（仅当未提供，且存在 npy） ----------
        if expected_joint_names is None:
            joint_union: List[str] = []
            seen = set()
            for _, p, fmt in expanded_items:
                if fmt == "npy":
                    try:
                        info = np.load(str(p), allow_pickle=True).item()
                        for j in info.get("joints_list", []):
                            if j not in seen:
                                seen.add(j)
                                joint_union.append(j)
                    except Exception as e:
                        print(f"[AMPLoader] Skip joint list from {p}: {e}")
            expected_joint_names = joint_union if joint_union else []

        # ---------- 3) 按文件逐个加载 ----------
        self.motion_data: List[MotionData] = []
        file_group_indices: List[int] = []
        file_lengths: List[int] = []

        for g_idx, path, fmt in expanded_items:
            if fmt == "npy":
                md = self.load_npy_data(path, simulation_dt, slow_down_factor, expected_joint_names)
            else:
                md = self.load_pkl_data(path, simulation_dt, slow_down_factor, expected_joint_names)

            if md is None:
                print(f"[AMPLoader] WARNING: failed to load {path}")
                continue

            md._group_idx = g_idx
            md._path = Path(path)
            md._format = fmt

            self.motion_data.append(md)
            file_group_indices.append(g_idx)
            file_lengths.append(len(md))

        if not self.motion_data:
            raise ValueError("No valid motion data loaded")

        # ---------- 4) 每文件构造“43 维/步”的观测，再按历史窗口拼接 ----------
        per_file_obs: List[torch.Tensor] = []
        per_file_states: List[torch.Tensor] = []

        for data in self.motion_data:
            T = len(data)
            # 43 维/步
            per_step = _build_amp_obs_per_step(
                base_pos_world           = data.base_positions,
                base_quat_xyzw           = data.base_quat,
                base_lin_vel_world       = data.base_lin_velocities_mixed,
                base_ang_vel_world       = data.base_ang_velocities_mixed,
                dof_pos                  = data.joint_positions,
                dof_vel                  = data.joint_velocities,
                key_body_pos_world       = getattr(data, "key_body_positions_world", None),  # 可 None
                expect_dof_obs_dim       = 12,
                expect_key_bodies        = 2,
            )  # [T, 43]
            per_file_obs.append(per_step)

            # reset 用的逐帧状态（与环境复位接口一致）
            idx = torch.arange(T, device=self.device)
            quat, jp, jv, blv, bav = data.get_state_for_reset(idx)
            per_file_states.append(torch.cat([quat, jp, jv, blv, bav], dim=1))

        self.per_step_dim = int(per_file_obs[0].shape[1])  # -> 43

        # 历史窗口：x_t, x_{t-1}, ...
        all_obs_windows: List[torch.Tensor] = []
        all_next_obs_windows: List[torch.Tensor] = []
        window_counts_per_file: List[int] = []

        K = self.history_steps
        S = self.history_stride
        for obs in per_file_obs:
            T = obs.shape[0]
            if T < 2:
                window_counts_per_file.append(0)
                continue

            i_min = (K - 1) * S if K > 1 else 0
            i_max = T - 2
            if i_max < i_min:
                window_counts_per_file.append(0)
                continue

            starts = torch.arange(i_min, i_max + 1, device=self.device)
            inds_s  = torch.stack([starts - j * S for j in range(K)], dim=1)      # [N, K]
            inds_sp = torch.stack([starts + 1 - j * S for j in range(K)], dim=1)  # [N, K]

            win_s  = obs[inds_s].reshape(starts.shape[0], -1)   # [N, K*D]
            win_sp = obs[inds_sp].reshape(starts.shape[0], -1)  # [N, K*D]

            all_obs_windows.append(win_s)
            all_next_obs_windows.append(win_sp)
            window_counts_per_file.append(int(win_s.shape[0]))

        if len(all_obs_windows) == 0:
            # 回退：无窗口则单步
            obs_list, next_obs_list, reset_states = [], [], []
            for data in self.motion_data:
                T = len(data)
                idx = torch.arange(T, device=self.device)
                obs = _build_amp_obs_per_step(
                    data.base_positions, data.base_quat,
                    data.base_lin_velocities_mixed, data.base_ang_velocities_mixed,
                    data.joint_positions, data.joint_velocities,
                    getattr(data, "key_body_positions_world", None),
                    expect_dof_obs_dim=12, expect_key_bodies=2
                )
                next_idx = torch.clamp(idx + 1, max=T - 1)
                next_obs = obs[next_idx]

                obs_list.append(obs)
                next_obs_list.append(next_obs)

                quat, jp, jv, blv, bav = data.get_state_for_reset(idx)
                reset_states.append(torch.cat([quat, jp, jv, blv, bav], dim=1))

            self.all_obs = torch.cat(obs_list, dim=0)           # [M, 43]
            self.all_next_obs = torch.cat(next_obs_list, dim=0) # [M, 43]
            self.all_states = torch.cat(reset_states, dim=0)
            self.sample_item_dim = int(self.all_obs.shape[1])   # 43
            self.history_steps_effective = 1
        else:
            self.all_obs = torch.cat(all_obs_windows, dim=0)         # [M, K*43]
            self.all_next_obs = torch.cat(all_next_obs_windows, dim=0)
            self.all_states = torch.cat(per_file_states, dim=0)      # 逐帧复位池
            self.sample_item_dim = int(self.all_obs.shape[1])        # K*43
            self.history_steps_effective = K

        # ---------- 5) 组级采样权重 -> 每一窗口 ----------
        num_groups = len(dataset_names)
        if len(dataset_weights) < num_groups:
            dataset_weights = list(dataset_weights) + [1.0] * (num_groups - len(dataset_weights))

        group_weights = torch.tensor(dataset_weights[:num_groups], dtype=torch.float32, device=self.device)
        if group_weights.sum() <= 0:
            group_weights = torch.ones_like(group_weights)
        group_weights = group_weights / group_weights.sum()

        file_weights_source = window_counts_per_file if len(all_obs_windows) > 0 else file_lengths

        group_total_units = torch.zeros(num_groups, dtype=torch.float64, device=self.device)
        for units, g in zip(file_weights_source, file_group_indices):
            group_total_units[g] += float(units)

        per_unit_chunks = []
        for units, g in zip(file_weights_source, file_group_indices):
            if group_total_units[g] > 0 and units > 0:
                w_per_unit = (group_weights[g].double() / group_total_units[g]).item()
            else:
                w_per_unit = 0.0
            per_unit_chunks.append(torch.full((int(units),), w_per_unit, dtype=torch.float64, device=self.device))

        if per_unit_chunks:
            per_unit = torch.cat(per_unit_chunks, dim=0)
            per_unit = per_unit / per_unit.sum().clamp_min(1e-12)
        else:
            per_unit = torch.ones((self.all_obs.shape[0],), dtype=torch.float64, device=self.device)
            per_unit = per_unit / per_unit.sum().clamp_min(1e-12)

        self.per_frame_weights = per_unit.to(dtype=torch.float32)

        # ---------- 6) 汇总打印 ----------
        print("[AMPLoader] Summary:")
        kind = "windows" if self.history_steps_effective > 1 else "frames"
        for gi, name in enumerate(dataset_names):
            print(
                f"  Group {gi} '{name}': weight={group_weights[gi].item():.3f}, "
                f"{kind}={int(group_total_units[gi].item())}"
            )
        print(
            f"  Total files: {len(self.motion_data)}, "
            f"per-step dim D={self.per_step_dim}, "
            f"history K={self.history_steps_effective}, "
            f"sample dim={self.sample_item_dim} "
            f"({self.history_steps_effective}*{self.per_step_dim if self.history_steps_effective>1 else self.sample_item_dim})"
        )

        print("[AMPLoader] Detailed listing (per group & file):")
        root_str = str(dataset_path_root.resolve())
        for gi, name in enumerate(dataset_names):
            group_files = [md for md in self.motion_data if md._group_idx == gi]
            if not group_files:
                print(f"  [Group {gi} '{name}'] (empty)")
                continue
            total_frames = int(sum(int(getattr(md, "_meta", {}).get("resampled_frames", len(md))) for md in group_files))
            print(f"  [Group {gi} '{name}'] files={len(group_files)}, total_resampled_frames={total_frames}")
            for md in group_files:
                rel = str(md._path.resolve())
                if rel.startswith(root_str + "/"):
                    rel = rel[len(root_str) + 1 :]
                m = md._meta or {}
                fmt = m.get("format", md._format or "?")
                ofps = m.get("orig_fps", "?")
                oT = m.get("orig_frames", "?")
                rdt = m.get("resampled_dt", "?")
                rT = m.get("resampled_frames", len(md))
                print(f"    - {rel} | fmt={fmt} | orig:fps={ofps}, frames={oT} | resampled:dt={rdt}, frames={rT}")

    # ---------- 基础插值/数值工具 ----------

    def _resample_data_Rn(
        self,
        data: Union[List[np.ndarray], np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> np.ndarray:
        """Resample data in R^n space using linear interpolation."""
        if isinstance(data, list):
            data = np.array(data)
        f = interp1d(original_keyframes, data, axis=0, fill_value="extrapolate")
        return f(target_keyframes)

    def _resample_data_SO3(
        self,
        raw_quaternions: Union[List[np.ndarray], np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> Rotation:
        """Resample rotations using SLERP (clip target times into [t0, tN])."""
        if isinstance(raw_quaternions, list):
            raw_quaternions = np.array(raw_quaternions)

        t_orig = np.asarray(original_keyframes, dtype=np.float64)
        t_tgt = np.asarray(target_keyframes, dtype=np.float64)

        t_min, t_max = float(t_orig[0]), float(t_orig[-1])
        t_tgt = np.clip(t_tgt, t_min, t_max)

        rot = Rotation.from_quat(raw_quaternions)  # xyzw
        slerp = Slerp(t_orig, rot)
        return slerp(t_tgt)

    def _compute_ang_vel(
        self,
        data: List[Rotation],
        dt: float,
        local: bool = False,
    ) -> np.ndarray:
        """Compute angular velocities from rotation sequence."""
        R_prev = data[:-1]
        R_next = data[1:]

        if local:
            rel = R_prev.inv() * R_next          # R_i^-1 * R_{i+1}
        else:
            rel = R_next * R_prev.inv()          # R_{i+1} * R_i^-1

        rotvec = rel.as_rotvec() / dt
        return np.vstack((rotvec, rotvec[-1]))

    def _compute_raw_derivative(self, data: np.ndarray, dt: float) -> np.ndarray:
        """Compute numerical derivative with simple finite difference."""
        d = (data[1:] - data[:-1]) / dt
        return np.vstack([d, d[-1:]])

    # ---------- NPY / PKL 加载 ----------

    def load_npy_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
    ) -> Optional[MotionData]:
        """Loads and processes one NPY motion dataset."""
        try:
            data = np.load(str(dataset_path), allow_pickle=True).item()
        except Exception as e:
            print(f"Failed to load NPY file {dataset_path}: {e}")
            return None

        if not expected_joint_names:
            expected_joint_names = data.get("joints_list", [])

        dataset_joint_names = data.get("joints_list", [])

        # build index map for expected_joint_names
        idx_map: List[Union[int, None]] = []
        for j in expected_joint_names:
            if j in dataset_joint_names:
                idx_map.append(dataset_joint_names.index(j))
            else:
                idx_map.append(None)

        # reorder & fill joint positions
        jp_list: List[np.ndarray] = []
        for frame in data["joint_positions"]:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jp_list.append(arr)

        dt = 1.0 / data["fps"] / float(slow_down_factor)
        T = len(jp_list)
        t_orig = np.linspace(0, T * dt, T)
        T_new = int(T * dt / simulation_dt)
        t_new = np.linspace(0, T * dt, T_new)

        resampled_joint_positions = self._resample_data_Rn(jp_list, t_orig, t_new)
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )

        resampled_base_positions = self._resample_data_Rn(
            data["root_position"], t_orig, t_new
        )
        resampled_base_orientations = self._resample_data_SO3(
            data["root_quaternion"], t_orig, t_new
        )

        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )

        resampled_base_ang_vel_mixed = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=False
        )

        resampled_base_lin_vel_local = np.stack(
            [
                R.as_matrix().T @ v
                for R, v in zip(
                    resampled_base_orientations, resampled_base_lin_vel_mixed
                )
            ]
        )
        resampled_base_ang_vel_local = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=True
        )

        # 可选关键体位（如数据中存在）
        key_world = None
        for k in ["key_body_positions_world", "keypoints_world", "keypoints"]:
            if k in data:
                try:
                    key_world = self._resample_data_Rn(data[k], t_orig, t_new)  # (T,K,3)
                except Exception:
                    key_world = None
                break

        md = MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_positions=resampled_base_positions,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=resampled_base_ang_vel_mixed,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            base_ang_velocities_local=resampled_base_ang_vel_local,
            base_quat=resampled_base_orientations,
            key_body_positions_world=key_world,
            device=self.device,
        )
        md._meta = {
            "format": "npy",
            "orig_fps": float(data.get("fps", 30.0)),
            "orig_frames": int(len(data.get("joint_positions", []))),
            "resampled_dt": float(simulation_dt),
            "resampled_frames": int(len(md)),
        }
        return md

    def load_pkl_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
    ) -> Optional[MotionData]:
        """Loads and processes one PKL motion dataset from retargeting."""
        try:
            pkl_data = joblib.load(str(dataset_path))
        except Exception as e:
            print(f"Failed to load PKL file {dataset_path}: {e}")
            return None

        if not isinstance(pkl_data, dict):
            print(f"Unexpected PKL format in {dataset_path}")
            return None

        motion_key = list(pkl_data.keys())[0]
        data = pkl_data[motion_key]

        if "root_trans_offset" not in data or "dof" not in data:
            print(f"Missing required fields in PKL file {dataset_path}")
            return None

        root_trans = data["root_trans_offset"]      # (T, 3)
        dof_positions = data["dof"]                 # (T, dof) or (T, dof, 1)
        root_quat_xyzw = data.get("root_rot", None) # (T, 4) in xyzw
        fps = float(data.get("fps", 30.0))

        # to numpy & squeeze
        if isinstance(root_trans, torch.Tensor):
            root_trans = root_trans.numpy()
        if isinstance(dof_positions, torch.Tensor):
            dof_positions = dof_positions.numpy()
        if isinstance(root_quat_xyzw, torch.Tensor):
            root_quat_xyzw = root_quat_xyzw.numpy()
        if dof_positions.ndim == 3 and dof_positions.shape[-1] == 1:
            dof_positions = dof_positions.squeeze(-1)

        # 时间轴
        dt_src = 1.0 / fps / float(slow_down_factor)
        T = int(dof_positions.shape[0])
        if T < 2:
            print(f"Motion too short in {dataset_path}")
            return None

        t_orig = np.arange(T, dtype=np.float64) * dt_src
        total_duration = float(t_orig[-1])

        T_new = int(np.floor(total_duration / simulation_dt)) + 1
        t_new = np.arange(T_new, dtype=np.float64) * simulation_dt
        if t_new[-1] > total_duration:
            t_new[-1] = total_duration

        # 关节重排（如提供 expected_joint_names）
        dof_names = data.get("dof_names", None)
        if dof_names is not None and expected_joint_names:
            name_to_src = {n: i for i, n in enumerate(dof_names)}
            mapped = np.zeros((T, len(expected_joint_names)), dtype=dof_positions.dtype)
            for i, n in enumerate(expected_joint_names):
                if n in name_to_src:
                    mapped[:, i] = dof_positions[:, name_to_src[n]]
                else:
                    mapped[:, i] = 0.0
            dof_positions = mapped

        # 重采样
        resampled_joint_positions = self._resample_data_Rn(dof_positions, t_orig, t_new)
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )

        resampled_base_positions = self._resample_data_Rn(root_trans, t_orig, t_new)
        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )

        if root_quat_xyzw is not None:
            resampled_base_orientations = self._resample_data_SO3(
                root_quat_xyzw, t_orig, t_new
            )
        else:
            resampled_base_orientations = Rotation.identity(len(t_new))

        resampled_base_ang_vel_mixed = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=False
        )
        resampled_base_ang_vel_local = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=True
        )

        Rmats = resampled_base_orientations.as_matrix()
        resampled_base_lin_vel_local = np.einsum(
            "nij,nj->ni", np.transpose(Rmats, (0, 2, 1)), resampled_base_lin_vel_mixed
        )

        # 关键体位：多数 retarget pkl 没有，缺省为 None（Loader 里会用零占位）
        key_world = None

        md = MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_positions=resampled_base_positions,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=resampled_base_ang_vel_mixed,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            base_ang_velocities_local=resampled_base_ang_vel_local,
            base_quat=resampled_base_orientations,  # xyzw -> tensor in __post_init__
            key_body_positions_world=key_world,
            device=self.device,
        )
        md._meta = {
            "format": "pkl",
            "orig_fps": float(fps),
            "orig_frames": int(T),
            "resampled_dt": float(simulation_dt),
            "resampled_frames": int(len(md)),
        }
        return md

    # ---------- 统一入口（可选） ----------

    def load_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
    ) -> Optional[MotionData]:
        """Auto-detect format by suffix and load."""
        if dataset_path.suffix == ".pkl":
            return self.load_pkl_data(dataset_path, simulation_dt, slow_down_factor, expected_joint_names)
        elif dataset_path.suffix == ".npy":
            return self.load_npy_data(dataset_path, simulation_dt, slow_down_factor, expected_joint_names)
        else:
            print(f"Unsupported file format: {dataset_path.suffix}")
            return None

    # ---------- 采样与重置 ----------

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Yield mini-batches of (state, next_state) pairs. Shapes: [B, K*D]."""
        for _ in range(num_mini_batch):
            idx = torch.multinomial(self.per_frame_weights, mini_batch_size, replacement=True)
            yield self.all_obs[idx], self.all_next_obs[idx]

    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly sample full states for environment resets from the flat buffer.
        Returns: (quat, q, dq, v_lin_local, v_ang_local)
        """
        n = int(number_of_samples) if number_of_samples is not None else 0
        device = self.device
        dtype = torch.float32

        if getattr(self, "motion_data", None) and len(self.motion_data) > 0:
            joint_dim = int(self.motion_data[0].joint_positions.shape[1])
        else:
            if hasattr(self, "all_states"):
                total_dim = int(self.all_states.shape[1])  # 4 + J + J + 3 + 3
                joint_dim = max(0, (total_dim - 10) // 2)
            else:
                joint_dim = 0

        if n <= 0:
            empty_quat = torch.empty((0, 4), device=device, dtype=dtype)
            empty_jpos = torch.empty((0, joint_dim), device=device, dtype=dtype)
            empty_jvel = torch.empty((0, joint_dim), device=device, dtype=dtype)
            empty_vlin = torch.empty((0, 3), device=device, dtype=dtype)
            empty_vang = torch.empty((0, 3), device=device, dtype=dtype)
            return empty_quat, empty_jpos, empty_jvel, empty_vlin, empty_vang

        if not hasattr(self, "per_frame_weights") or self.per_frame_weights.numel() == 0:
            empty_quat = torch.empty((n, 4), device=device, dtype=dtype)
            empty_jpos = torch.empty((n, joint_dim), device=device, dtype=dtype)
            empty_jvel = torch.empty((n, joint_dim), device=device, dtype=dtype)
            empty_vlin = torch.empty((n, 3), device=device, dtype=dtype)
            empty_vang = torch.empty((n, 3), device=device, dtype=dtype)
            return empty_quat, empty_jpos, empty_jvel, empty_vlin, empty_vang

        idx = torch.multinomial(self.per_frame_weights, n, replacement=True)
        # 注意：all_states 仍是“逐帧”的重置池，不受历史窗口影响
        if not hasattr(self, "all_states") or self.all_states.numel() == 0:
            empty_quat = torch.empty((n, 4), device=device, dtype=dtype)
            empty_jpos = torch.empty((n, joint_dim), device=device, dtype=dtype)
            empty_jvel = torch.empty((n, joint_dim), device=device, dtype=dtype)
            empty_vlin = torch.empty((n, 3), device=device, dtype=dtype)
            empty_vang = torch.empty((n, 3), device=device, dtype=dtype)
            return empty_quat, empty_jpos, empty_jvel, empty_vlin, empty_vang

        # 为避免索引越界（weights 对应窗口或帧），这里对 all_states 采用均匀采样
        state_idx = torch.randint(0, self.all_states.shape[0], (n,), device=device)
        full = self.all_states[state_idx]  # [n, 4 + J + J + 3 + 3]
        return torch.split(full, [4, joint_dim, joint_dim, 3, 3], dim=1)
