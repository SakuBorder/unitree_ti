# ----------------- AMPOnPolicyRunner (motionlib version) -----------------
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import inspect
import os
import statistics
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.utils import store_code_state

from amp_rsl_rl.utils import Normalizer, export_policy_as_onnx
from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils.motion_lib_taihu import MotionLibTaihu  # ✅ 直接导入，KISS

try:
    from amp_rsl_rl.networks import ActorCriticMoE  # type: ignore
    _HAS_MOE = True
except Exception:
    ActorCriticMoE = None  # type: ignore
    _HAS_MOE = False


# ----------------------------------------------------------------------
# MotionLib -> AMP adapter
# ----------------------------------------------------------------------
from isaacgym.torch_utils import quat_rotate, quat_mul


class MotionLibAMPAdapter:
    """
    Wrap a MotionLib-like object into an AMP expert 'data source':
      - sample_pairs via feed_forward_generator(...)
      - flat observation of length K*D, where K=history_steps and D=per-step features
    """

    def __init__(
        self,
        motion_lib,                     # instance of MotionLibTaihu or compatible
        dt: float,                      # env control timestep (dt * decimation)
        history_steps: int = 2,         # K
        history_stride: int = 1,        # stride (in env dt units)
        expect_dof_obs_dim: int = 12,   # env dof_obs_size
        expect_key_bodies: int = 2,     # number of key bodies encoded into obs
        key_body_ids: Optional[List[int]] = None,  # indices in motion_lib.rg_pos to use
        use_root_h: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.motion_lib = motion_lib
        self.dt = float(dt)
        self.K = max(2, int(history_steps))
        self.S = max(1, int(history_stride))
        self.expect_dof_obs_dim = int(expect_dof_obs_dim)
        self.expect_key_bodies = int(expect_key_bodies)
        self.key_body_ids = key_body_ids
        self.use_root_h = bool(use_root_h)
        self.device = device

        # probe 1 sample to determine per-step dim (D)
        _probe = self._sample_amp_window(batch_size=1)
        self.num_amp_obs_per_step = int(_probe["obs"].shape[-1] // self.K)
        self.sample_item_dim = int(_probe["obs"].shape[-1])  # = K*D

    # ---- AMPLoader-like API ----
    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for _ in range(num_mini_batch):
            out = self._sample_amp_window(batch_size=mini_batch_size)
            yield out["obs"], out["next_obs"]

    @torch.no_grad()
    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns: (root_quat_xyzw, dof_pos, dof_vel, v_lin_local, v_ang_local)
        Shapes:
          root_quat: [N,4], dof_pos: [N,J], dof_vel: [N,J], v_*: [N,3]
        """
        n = int(number_of_samples)
        if n <= 0:
            z = torch.empty
            dev = self.device
            return z((0, 4), device=dev), z((0, 0), device=dev), z((0, 0), device=dev), z((0, 3), device=dev), z((0, 3), device=dev)

        mids = self.motion_lib.sample_motions(n)
        t0 = self.motion_lib.sample_time(mids, truncate_time=0.0)

        st = self.motion_lib.get_motion_state(mids, t0)
        root_rot = st["root_rot"]                # [N,4] xyzw
        dof_pos = st["dof_pos"]                  # [N,J]
        dof_vel = st["dof_vel"].reshape(n, -1)   # [N,J]
        v_lin_w = st["root_vel"]                 # [N,3]
        v_ang_w = st["root_ang_vel"]             # [N,3]

        heading_inv = calc_heading_quat_inv_xyzw(root_rot)  # [N,4]
        v_lin_local = quat_rotate(heading_inv, v_lin_w)
        v_ang_local = quat_rotate(heading_inv, v_ang_w)
        return (root_rot, dof_pos, dof_vel, v_lin_local, v_ang_local)

    # ---- internal helpers ----
    @torch.no_grad()
    def _sample_amp_window(self, batch_size: int):
        B = int(batch_size)
        K, S, dt = self.K, self.S, self.dt

        mids = self.motion_lib.sample_motions(B)

        # ensure we can look back K-1 steps of size S*dt
        truncate_time = (K - 1) * S * dt + 1e-8
        t0 = self.motion_lib.sample_time(mids, truncate_time=truncate_time) + truncate_time

        steps = torch.arange(0, K, device=self.device, dtype=torch.float32) * (S * dt)
        times = t0.unsqueeze(1) - steps.unsqueeze(0)    # [B,K]
        times_next = times + dt                         # slide window forward by +dt

        obs = self._build_flattened_obs(mids, times)
        next_obs = self._build_flattened_obs(mids, times_next)
        return {"obs": obs, "next_obs": next_obs}

    def _build_flattened_obs(self, mids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        B, K = times.shape
        per = []
        for j in range(K):
            st = self.motion_lib.get_motion_state(mids, times[:, j])

            key_world = None
            if self.key_body_ids is not None and "rg_pos" in st:
                key_world = st["rg_pos"][:, self.key_body_ids, :]  # [B,Kb,3]

            step = build_amp_obs_per_step(
                base_pos_world=st["root_pos"],
                base_quat_xyzw=st["root_rot"],
                base_lin_vel_world=st["root_vel"],
                base_ang_vel_world=st["root_ang_vel"],
                dof_pos=st["dof_pos"],
                dof_vel=st["dof_vel"].reshape(B, -1),
                key_body_pos_world=key_world,
                use_root_h=self.use_root_h,
                expect_dof_obs_dim=self.expect_dof_obs_dim,
                expect_key_bodies=self.expect_key_bodies,
            )  # [B,D]
            per.append(step)
        return torch.stack(per, dim=1).reshape(B, -1)


# ---- math helpers (xyzw convention) ----
def calc_heading_quat_inv_xyzw(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Keep only yaw of q (world), return its inverse as xyzw. Matches IsaacGym's AMP heading handling."""
    try:
        from amp_rsl_rl.utils import torch_utils
        return torch_utils.calc_heading_quat_inv(q_xyzw)
    except Exception:
        # fallback: identity inverse (no heading) to keep things running
        q = torch.zeros_like(q_xyzw)
        q[..., 3] = 1.0
        return q


def quat_to_tan_norm_xyzw(q_xyzw: torch.Tensor) -> torch.Tensor:
    try:
        from amp_rsl_rl.utils import torch_utils
        return torch_utils.quat_to_tan_norm(q_xyzw)
    except Exception:
        # fallback: raw quaternion padded to 6D
        pad = torch.zeros(q_xyzw.shape[:-1] + (2,), device=q_xyzw.device, dtype=q_xyzw.dtype)
        return torch.cat([q_xyzw[..., :4], pad], dim=-1)


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

    # root height
    root_h = base_pos_world[:, 2:3] if use_root_h else torch.zeros((B, 1), device=dev, dtype=dtype)

    # heading & local root rot (tan-norm 6D)
    heading_inv = calc_heading_quat_inv_xyzw(base_quat_xyzw)          # [B,4]
    root_local = quat_mul(heading_inv, base_quat_xyzw)                # [B,4]
    root_rot_tan = quat_to_tan_norm_xyzw(root_local)                  # [B,6]

    # local root velocities
    v_lin_local = quat_rotate(heading_inv, base_lin_vel_world)        # [B,3]
    v_ang_local = quat_rotate(heading_inv, base_ang_vel_world)        # [B,3]

    # dof pos/vel (crop or pad to expect_dof_obs_dim)
    Jexp = int(expect_dof_obs_dim)
    if dof_pos.shape[1] >= Jexp:
        dof_obs = dof_pos[:, :Jexp]
        dof_vel_obs = dof_vel[:, :Jexp]
    else:
        pad = Jexp - dof_pos.shape[1]
        z = torch.zeros((B, pad), device=dev, dtype=dtype)
        dof_obs = torch.cat([dof_pos, z], dim=-1)
        dof_vel_obs = torch.cat([dof_vel, z], dim=-1)

    # key bodies: relative to root, rotated by heading_inv
    Kbexp = int(expect_key_bodies)
    if key_body_pos_world is None or key_body_pos_world.numel() == 0:
        key_local_flat = torch.zeros((B, 3 * Kbexp), device=dev, dtype=dtype)
    else:
        Kb = key_body_pos_world.shape[1]
        rel = key_body_pos_world - base_pos_world.unsqueeze(1)        # [B,Kb,3]
        h_flat = heading_inv.unsqueeze(1).expand(-1, Kb, -1).reshape(-1, 4)
        rel_flat = rel.reshape(-1, 3)
        local = quat_rotate(h_flat, rel_flat).reshape(B, Kb, 3)       # [B,Kb,3]
        key_local_flat = local.reshape(B, 3 * Kb)
        if Kb < Kbexp:
            pad = (Kbexp - Kb) * 3
            key_local_flat = torch.cat([key_local_flat, torch.zeros((B, pad), device=dev, dtype=dtype)], dim=-1)
        elif Kb > Kbexp:
            key_local_flat = key_local_flat[:, : 3 * Kbexp]

    return torch.cat([root_h, root_rot_tan, v_lin_local, v_ang_local, dof_obs, dof_vel_obs, key_local_flat], dim=-1)


# ----------------------------------------------------------------------
# Utilities for env unpacking
# ----------------------------------------------------------------------
def _unpack_env_observations(env: VecEnv) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    got = env.get_observations()
    if isinstance(got, tuple):
        obs = got[0]
    else:
        obs = got
    priv = None
    if hasattr(env, "get_privileged_observations"):
        priv = env.get_privileged_observations()
    return obs, priv


def _unpack_env_step(ret: Any) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    if isinstance(ret, tuple):
        if len(ret) >= 7:
            obs, priv, rewards, dones, infos, term_ids, term_priv = ret[:7]
            return obs, priv, rewards, dones, infos, term_ids, term_priv
        if len(ret) == 4:
            obs, rewards, dones, infos = ret
            return obs, None, rewards, dones, infos, None, None
    raise RuntimeError(
        f"Unsupported env.step(...) return signature: type={type(ret)}, len={len(ret) if isinstance(ret, tuple) else 'N/A'}"
    )


def _merge_amp_cfg(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    amp = dict(train_cfg.get("amp", {}))
    for k in (
        # legacy keys (kept so older cfgs still work)
        "amp_data_path", "dataset_names", "dataset_weights", "slow_down_factor",
        "num_amp_obs", "dt", "decimation", "replay_buffer_size", "reward_scale",
        "joint_names", "style_weight", "task_weight",
        "num_amp_obs_steps", "history_steps", "history_stride",
        # motionlib-centric keys
        "motion_file", "mjcf_file", "extend_hand", "extend_head",
        "expect_dof_obs_dim", "key_body_ids", "bootstrap_motions", "motionlib_class",
        "use_joint_mapping", "debug_joint_mapping",
    ):
        if k in train_cfg and k not in amp:
            amp[k] = train_cfg[k]
    return amp


# ----------------------------------------------------------------------
# AMPOnPolicyRunner (KISS import for MotionLibTaihu)
# ----------------------------------------------------------------------
class AMPOnPolicyRunner:
    """AMP + PPO Runner (TensorBoard only, memory-safe). MotionLib-backed expert sampling."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        from pathlib import Path

        self.env = env
        self.device = device

        # --- cfg ---
        self.cfg = train_cfg
        self.runner_cfg = dict(train_cfg.get("runner", {}))
        self.policy_cfg = dict(train_cfg.get("policy", {}))
        self.alg_cfg = dict(train_cfg.get("algorithm", {}))
        self.discriminator_cfg = dict(train_cfg.get("discriminator", {}))
        self.amp_cfg = _merge_amp_cfg(train_cfg)

        # --- reward mix ---
        self.task_w = float(self.amp_cfg.get("task_weight", 0.5))
        self.style_w = float(self.amp_cfg.get("style_weight", 0.5))
        _sum_w = self.task_w + self.style_w
        if _sum_w <= 0.0:
            self.task_w, self.style_w = 0.5, 0.5
        else:
            self.task_w /= _sum_w
            self.style_w /= _sum_w

        # --- obs dims ---
        obs0, priv0 = _unpack_env_observations(self.env)
        if not isinstance(obs0, torch.Tensor):
            raise RuntimeError("env.get_observations() must return a torch.Tensor or (Tensor, extras).")
        num_actor_obs = int(obs0.shape[-1])
        num_critic_obs = int(priv0.shape[-1]) if isinstance(priv0, torch.Tensor) else num_actor_obs

        num_actions = getattr(self.env, "num_actions", None)
        if num_actions is None:
            raise RuntimeError("env.num_actions is required by AMPOnPolicyRunner.")

        # --- policy selection ---
        policy_name = self.runner_cfg.get("policy_class_name", self.policy_cfg.get("class_name", "ActorCritic"))
        policies: Dict[str, Any] = {"ActorCritic": ActorCritic}
        if _HAS_MOE and ActorCriticMoE is not None:
            policies["ActorCriticMoE"] = ActorCriticMoE
        if policy_name not in policies:
            print(f"[AMPOnPolicyRunner] Policy '{policy_name}' not found; falling back to 'ActorCritic'.")
            policy_name = "ActorCritic"
        policy_cls = policies[policy_name]

        policy_kwargs = {k: v for k, v in self.policy_cfg.items() if k not in ("class_name",)}
        self.actor_critic: ActorCritic | ActorCriticRecurrent | Any = policy_cls(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            **policy_kwargs,
        ).to(self.device)

        # --- AMP timing (dt, decimation) ---
        dt_cfg = getattr(getattr(self.env, "cfg", None), "sim", None)
        dt = getattr(dt_cfg, "dt", None)
        deci_cfg = getattr(getattr(self.env, "cfg", None), "control", None)
        decimation = getattr(deci_cfg, "decimation", None) or getattr(getattr(self.env, "cfg", None), "decimation", None)
        if dt is None:
            dt = float(self.amp_cfg.get("dt", 1.0 / 60.0))
        if decimation is None:
            decimation = int(self.amp_cfg.get("decimation", 1))
        delta_t = float(dt) * int(decimation)

        # --- AMP dims/history ---
        num_amp_obs_env = getattr(self.env, "num_amp_obs", None)
        num_amp_obs_cfg = self.amp_cfg.get("num_amp_obs", None)
        num_amp_obs = int(num_amp_obs_env if num_amp_obs_env is not None else (num_amp_obs_cfg if num_amp_obs_cfg is not None else num_actor_obs))

        history_steps = self.amp_cfg.get("history_steps", None)
        if history_steps is None:
            history_steps = self.amp_cfg.get("num_amp_obs_steps", None)
        if history_steps is None:
            history_steps = getattr(self.env, "_num_amp_obs_steps", 2)
        history_steps = int(history_steps)
        history_stride = int(self.amp_cfg.get("history_stride", 1))

        # --- MotionLib config ---
        amp_data_path = self.amp_cfg.get("amp_data_path", None)
        motion_file = self.amp_cfg.get("motion_file", None) or amp_data_path
        if motion_file is None:
            raise RuntimeError("AMP config requires 'motion_file' (or legacy 'amp_data_path').")

        mjcf_file = self.amp_cfg.get("mjcf_file", "/home/dy/dy/code/unitree_ti/assert/ti5/ti5_12dof.xml")
        extend_hand = bool(self.amp_cfg.get("extend_hand", True))
        extend_head = bool(self.amp_cfg.get("extend_head", True))
        expect_dof_obs_dim = int(self.amp_cfg.get("expect_dof_obs_dim", 12))
        key_body_ids = self.amp_cfg.get("key_body_ids", [6, 12])  # default: L/R ankles in many rigs
        bootstrap_motions = int(self.amp_cfg.get("bootstrap_motions", min(32, getattr(self.env, "num_envs", 32))))
        use_joint_mapping = bool(self.amp_cfg.get("use_joint_mapping", True))
        debug_joint_mapping = bool(self.amp_cfg.get("debug_joint_mapping", False))

        # --- use_root_h flag from env.observations.amp.root_height ---
        use_root_h = bool(
            getattr(getattr(self.env, "cfg", None), "observations", None)
            and getattr(self.env.cfg.observations, "amp", None)
            and getattr(self.env.cfg.observations.amp, "root_height", False)
        )
        os.environ["AMP_USE_ROOT_H"] = "1" if use_root_h else "0"

        # --- Instantiate MotionLib (KISS: use the imported class directly) ---
        # Optional: honor a different 'motionlib_class' value, but still use MotionLibTaihu (compatibility)
        motionlib_class_name = self.amp_cfg.get("motionlib_class", "MotionLibTaihu")
        if motionlib_class_name != "MotionLibTaihu":
            print(f"[AMPOnPolicyRunner] motionlib_class='{motionlib_class_name}' not supported in KISS mode; using MotionLibTaihu.")

        motion_lib = MotionLibTaihu(
            motion_file=motion_file,
            device=self.device,
            mjcf_file=mjcf_file,
            extend_hand=extend_hand,
            extend_head=extend_head,
            sim_timestep=delta_t,
        )
        if hasattr(motion_lib, "set_joint_mapping_mode"):
            motion_lib.set_joint_mapping_mode(use_mapping=use_joint_mapping, debug=debug_joint_mapping)

        # --- Bootstrap motions once ---
        node_names = getattr(motion_lib.mesh_parsers, "model_names", None) or getattr(motion_lib.mesh_parsers, "body_names", None)
        if node_names is None:
            raise RuntimeError("MotionLib.mesh_parsers must expose 'model_names' or 'body_names'.")

        class _Skel:
            __slots__ = ("node_names",)
            def __init__(self, names): self.node_names = names

        num_boot = max(1, bootstrap_motions)
        skeleton_trees = [_Skel(node_names) for _ in range(num_boot)]
        gender_betas = [torch.zeros(17) for _ in range(num_boot)]  # placeholder
        limb_weights = [1.0] * len(node_names)

        motion_lib.load_motions(
            skeleton_trees=skeleton_trees,
            gender_betas=gender_betas,
            limb_weights=limb_weights,
            random_sample=True,
            start_idx=0,
            max_len=-1,
            target_heading=None,
        )

        # --- Build adapter (replaces AMPLoader) ---
        amp_data = MotionLibAMPAdapter(
            motion_lib=motion_lib,
            dt=delta_t,
            history_steps=history_steps,
            history_stride=history_stride,
            expect_dof_obs_dim=expect_dof_obs_dim,
            expect_key_bodies=len(key_body_ids),
            key_body_ids=key_body_ids,
            use_root_h=use_root_h,
            device=self.device,
        )
        self.amp_data = amp_data

        # Allow env to access amp_data if it wants (e.g., reset from motion states)
        if hasattr(self.env, "set_amp_data"):
            self.env.set_amp_data(self.amp_data)

        # Try initial reset
        try:
            all_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
            if hasattr(self.env, "reset_idx"):
                self.env.reset_idx(all_ids)
            else:
                self.env.reset()
        except Exception as e:
            print(f"[Runner] Soft-warn: initial AMP reset failed: {e}")

        # --- dataset summary ---
        num_motions = getattr(motion_lib, "num_motions", lambda: None)()
        total_len = getattr(motion_lib, "get_total_length", lambda: None)()
        sample_preview = "n/a"
        try:
            g = amp_data.feed_forward_generator(num_mini_batch=1, mini_batch_size=2)
            ex_obs, ex_next = next(g)
            sample_preview = f"obs={tuple(ex_obs.shape)}, next_obs={tuple(ex_next.shape)}"
        except Exception as e:
            sample_preview = f"peek failed: {e}"

        from pathlib import Path
        self._amp_data_text = (
            f"motion_file: {Path(motion_file)}\n"
            f"mjcf_file: {mjcf_file}\n"
            f"motions_loaded: {num_motions if num_motions is not None else 'n/a'}\n"
            f"total_length(s): {total_len:.3f}" if isinstance(total_len, (int, float)) else "total_length(s): n/a"
        )
        print("[AMP DATA] ==== MotionLib summary ====")
        for line in self._amp_data_text.splitlines():
            print("[AMP DATA] " + line)
        print(
            f"[AMP DATA] delta_t={delta_t:.6f}, history_steps={int(history_steps)}, "
            f"history_stride={int(history_stride)}, expect_dof_obs_dim={expect_dof_obs_dim}, "
            f"key_body_ids={key_body_ids}"
        )
        print(f"[AMP DATA] sample preview: {sample_preview}")
        print("[AMP DATA] =================================")

        # --- AMP bits ---
        self.amp_normalizer = Normalizer(num_amp_obs, device=self.device)
        self.discriminator = Discriminator(
            input_dim=num_amp_obs * 2,
            hidden_layer_sizes=self.discriminator_cfg.get("hidden_dims", [1024, 512]),
            reward_scale=float(self.discriminator_cfg.get("reward_scale", 1.0)),
            device=self.device,
            loss_type=self.discriminator_cfg.get("loss_type", "BCEWithLogits"),
        ).to(self.device)

        # --- sanity checks (AFTER discriminator is created) ---
        if self.amp_data.sample_item_dim != num_amp_obs:
            raise ValueError(
                f"AMP obs dim mismatch: env={num_amp_obs}, data={self.amp_data.sample_item_dim} (K*D). "
                f"Check history_steps/stride, expect_dof_obs_dim, key_body_ids, and env's AMP obs definition."
            )
        disc_expected = 2 * num_amp_obs
        if self.discriminator.input_dim != disc_expected:
            raise ValueError(
                f"Discriminator input_dim={self.discriminator.input_dim} should equal {disc_expected} (=2*num_amp_obs)"
            )

        # --- Algorithm ---
        algo_name = self.runner_cfg.get("algorithm_class_name", self.alg_cfg.get("class_name", "AMP_PPO"))
        if algo_name != "AMP_PPO":
            print(f"[AMPOnPolicyRunner] Algorithm '{algo_name}' not supported; falling back to 'AMP_PPO'.")

        raw_algo_kwargs = {k: v for k, v in self.alg_cfg.items() if k != "class_name"}
        allowed = set(inspect.signature(AMP_PPO.__init__).parameters) - {"self"}
        algo_kwargs = {k: v for k, v in raw_algo_kwargs.items() if k in allowed}

        dropped = sorted(set(raw_algo_kwargs) - set(algo_kwargs))
        if dropped:
            print(f"[AMPOnPolicyRunner] Dropped unsupported AMP_PPO args: {dropped}")

        self.alg: AMP_PPO = AMP_PPO(
            actor_critic=self.actor_critic,
            discriminator=self.discriminator,
            amp_data=self.amp_data,
            amp_normalizer=self.amp_normalizer,
            device=self.device,
            **algo_kwargs,
        )

        # --- rollout/logging ---
        self.num_steps_per_env = int(self.runner_cfg.get("num_steps_per_env", 24))
        self.save_interval = int(self.runner_cfg.get("save_interval", 50))
        self.empirical_normalization = bool(self.runner_cfg.get("empirical_normalization", False))

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.critic_obs_normalizer = torch.nn.Identity()

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [num_actions],
        )

        self.log_dir = log_dir
        self.writer: Optional[TensorboardSummaryWriter] = None
        if self.log_dir is not None:
            self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    # ----------------- training loop -----------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if hasattr(self.env, "reset"):
            self.env.reset()

        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf") and hasattr(self.env, "max_episode_length"):
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs, priv = _unpack_env_observations(self.env)
        critic_obs = priv if isinstance(priv, torch.Tensor) else obs

        num_amp_obs = int(getattr(self.env, "num_amp_obs", self.amp_cfg.get("num_amp_obs", obs.shape[-1])))
        amp_obs = self._build_amp_obs_from_obs(obs, num_amp_obs)

        obs = self.obs_normalizer(obs).to(self.device)
        critic_obs = self.critic_obs_normalizer(critic_obs).to(self.device)
        amp_obs = amp_obs.to(self.device)

        self.train_mode()

        ep_infos: List[Dict[str, Any]] = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        print(f"[AMP DEBUG] Shapes | actor_obs={tuple(obs.shape)}, critic_obs={tuple(critic_obs.shape)}, amp_obs={tuple(amp_obs.shape)}")
        if hasattr(self.discriminator, "input_dim"):
            print(f"[AMP DEBUG] Discriminator input_dim={self.discriminator.input_dim} (should be 2*{num_amp_obs})")

        if self.writer is not None and getattr(self, "_amp_data_text", None):
            self.writer.add_text("AMP/DataSummary", f"<pre>{self._amp_data_text}</pre>", 0)

        for it in range(start_iter, tot_iter):
            start = time.time()
            mean_style_reward_log = 0.0
            mean_task_reward_log = 0.0

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    self.alg.act_amp(amp_obs)

                    step_ret = self.env.step(actions)
                    obs_next, priv_next, rewards, dones, infos, term_ids, term_priv = _unpack_env_step(step_ret)
                    critic_next = priv_next if isinstance(priv_next, torch.Tensor) else obs_next

                    next_amp_obs = self._build_amp_obs_from_obs(obs_next, num_amp_obs)

                    obs_next = self.obs_normalizer(obs_next).to(self.device)
                    critic_next = self.critic_obs_normalizer(critic_next).to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)

                    style_rewards = self.discriminator.predict_reward(amp_obs, next_amp_obs, normalizer=self.amp_normalizer)

                    mean_task_reward_log += rewards.mean().item()
                    mean_style_reward_log += style_rewards.mean().item()
                    rewards = self.task_w * rewards + self.style_w * style_rewards

                    self.alg.process_env_step(rewards, dones, infos)
                    self.alg.process_amp_step(next_amp_obs)

                    if term_ids is not None and term_priv is not None:
                        term_ids = term_ids.to(self.device)
                        term_priv = term_priv.to(self.device)
                        critic_fixed = critic_next.clone().detach()
                        critic_fixed[term_ids] = term_priv.clone().detach()
                        critic_next = critic_fixed

                    obs = obs_next
                    critic_obs = critic_next
                    amp_obs = next_amp_obs.detach()

                    if self.writer is not None:
                        if isinstance(infos, dict) and "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if new_ids.numel() > 0:
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs)

            update_out = self.alg.update()
            if not isinstance(update_out, (list, tuple)):
                raise RuntimeError(f"Unexpected update() return type: {type(update_out)}")

            vals = list(update_out)
            KEYS = [
                "mean_value_loss",
                "mean_surrogate_loss",
                "mean_amp_loss",
                "mean_grad_pen_loss",
                "mean_policy_pred",
                "mean_expert_pred",
                "mean_accuracy_policy",
                "mean_accuracy_expert",
                "mean_kl_divergence",
                "mean_swap_loss",
                "mean_actor_sym_loss",
                "mean_critic_sym_loss",
            ]
            if len(vals) < len(KEYS):
                vals += [0.0] * (len(KEYS) - len(vals))
            vals = vals[:len(KEYS)]
            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_accuracy_policy,
                mean_accuracy_expert,
                mean_kl_divergence,
                mean_swap_loss,
                mean_actor_sym_loss,
                mean_critic_sym_loss,
            ) = vals

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            steps_this_iter = float(self.num_steps_per_env)
            mean_style_reward_avg = mean_style_reward_log / max(1.0, steps_this_iter)
            if (it < start_iter + 3) or (it % 100 == 0):
                print(
                    f"[AMP DEBUG] it={it} | amp_loss={mean_amp_loss:.4f} "
                    f"| grad_pen={mean_grad_pen_loss:.4f} "
                    f"| policy_pred≈{mean_policy_pred:.3f} "
                    f"| expert_pred≈{mean_expert_pred:.3f} "
                    f"| acc_pol={mean_accuracy_policy:.3f} "
                    f"| acc_exp={mean_accuracy_expert:.3f} "
                    f"| style_reward(avg/step)={mean_style_reward_avg:.4f}"
                )

            if self.writer is not None:
                to_log = {
                    "it": it,
                    "tot_iter": tot_iter,
                    "collection_time": collection_time,
                    "learn_time": learn_time,
                    "ep_infos": ep_infos,
                    "rewbuffer": rewbuffer,
                    "lenbuffer": lenbuffer,
                    "mean_value_loss": mean_value_loss,
                    "mean_surrogate_loss": mean_surrogate_loss,
                    "mean_amp_loss": mean_amp_loss,
                    "mean_grad_pen_loss": mean_grad_pen_loss,
                    "mean_policy_pred": mean_policy_pred,
                    "mean_expert_pred": mean_expert_pred,
                    "mean_accuracy_policy": mean_accuracy_policy,
                    "mean_accuracy_expert": mean_accuracy_expert,
                    "mean_kl_divergence": mean_kl_divergence,
                    "mean_swap_loss": mean_swap_loss,
                    "mean_actor_sym_loss": mean_actor_sym_loss,
                    "mean_critic_sym_loss": mean_critic_sym_loss,
                    "mean_style_reward_log": mean_style_reward_log,
                    "mean_task_reward_log": mean_task_reward_log,
                    "num_learning_iterations": num_learning_iterations,
                }
                self.log(to_log)

            if self.log_dir is not None and (it % self.save_interval == 0):
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), save_onnx=True)

            ep_infos.clear()
            if it == start_iter and self.log_dir is not None:
                try:
                    store_code_state(self.log_dir, [rsl_rl.__file__])
                except Exception:
                    pass

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"), save_onnx=True)

    # ----------------- utils/log/save -----------------

    def _build_amp_obs_from_obs(self, obs: torch.Tensor, num_amp_obs: int) -> torch.Tensor:
        if hasattr(self.env, 'compute_amp_observations'):
            return self.env.compute_amp_observations()

        try:
            from isaacgym.torch_utils import quat_rotate_inverse
            if hasattr(self.env, 'dof_pos') and hasattr(self.env, 'dof_vel') and hasattr(self.env, 'base_quat') and hasattr(self.env, 'root_states'):
                base_lin_vel_local = quat_rotate_inverse(self.env.base_quat, self.env.root_states[:, 7:10])
                base_ang_vel_local = quat_rotate_inverse(self.env.base_quat, self.env.root_states[:, 10:13])
                amp_obs = torch.cat([
                    self.env.dof_pos,
                    self.env.dof_vel,
                    base_lin_vel_local,
                    base_ang_vel_local,
                ], dim=-1)
                if amp_obs.shape[-1] != num_amp_obs:
                    if amp_obs.shape[-1] > num_amp_obs:
                        amp_obs = amp_obs[..., :num_amp_obs]
                    else:
                        pad = num_amp_obs - amp_obs.shape[-1]
                        amp_obs = torch.cat([amp_obs, torch.zeros(amp_obs.shape[0], pad, device=amp_obs.device)], dim=-1)
                return amp_obs
        except Exception as e:
            print(f"Warning: Failed to construct AMP observations from environment state: {e}")

        amp_obs = obs
        if amp_obs.shape[-1] != num_amp_obs:
            if amp_obs.shape[-1] > num_amp_obs:
                amp_obs = amp_obs[..., :num_amp_obs]
            else:
                pad = num_amp_obs - amp_obs.shape[-1]
                amp_obs = torch.cat([amp_obs, torch.zeros(amp_obs.shape[0], pad, device=amp_obs.device)], dim=-1)
        return amp_obs

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                # KISS: 用 CPU 聚合，避免 GPU 挤占
                vals = []
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    v = ep_info[key]
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().flatten()
                        vals.append(v)
                    else:
                        try:
                            vals.append(torch.tensor([float(v)], dtype=torch.float32))
                        except Exception:
                            pass
                if vals:
                    value = torch.cat(vals).mean().item()
                    if "/" in key:
                        self.writer.add_scalar(key, value, locs["it"])
                        ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar("Episode/" + key, value, locs["it"])
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        try:
            mean_std = float(self.alg.actor_critic.std.mean().item())
        except Exception:
            mean_std = float("nan")
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # ======== 明确区分 Actor 与 Critic 的 loss 命名 ========
        # 保持原有记录（兼容旧看板）
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])   # == Critic loss
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])    # == Actor loss(=surrogate)

        # 新增更明确的命名
        self.writer.add_scalar("Critic/Loss", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Actor/Loss", locs["mean_surrogate_loss"], locs["it"])

        # 其余保持不变
        self.writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar("Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"])
        self.writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
        self.writer.add_scalar("Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"])
        self.writer.add_scalar("Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Loss/mean_kl_divergence", locs["mean_kl_divergence"], locs["it"])
        self.writer.add_scalar("Loss/Swap Loss", locs.get("mean_swap_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/Actor Sym Loss", locs.get("mean_actor_sym_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/Critic Sym Loss", locs.get("mean_critic_sym_loss", 0.0), locs["it"])

        self.writer.add_scalar("Policy/mean_noise_std", mean_std, locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_style_reward", locs["mean_style_reward_log"], locs["it"])
            self.writer.add_scalar("Train/mean_task_reward", locs["mean_task_reward_log"], locs["it"])

        head = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        mean_style_reward_avg = locs["mean_style_reward_log"] / max(1.0, float(self.num_steps_per_env))
        common = (
            f"""{'#' * width}\n"""
            f"""{head.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Critic loss (value):':>{pad}} {locs['mean_value_loss']:.4f}\n"""
            f"""{'Actor loss (surrogate):':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
            f"""{'Grad penalty:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
            f"""{'Disc policy_pred (~P(policy)):':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
            f"""{'Disc expert_pred (~P(expert)):':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
            f"""{'Disc acc(policy/expert):':>{pad}} {locs['mean_accuracy_policy']:.3f} / {locs['mean_accuracy_expert']:.3f}\n"""
            f"""{'Style reward (avg/step):':>{pad}} {mean_style_reward_avg:.4f}\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
        )
        if len(locs["rewbuffer"]) > 0:
            common += (
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        log_string = common + ep_string

        eta_seconds = self.tot_time / (locs["it"] + 1) * (locs["num_learning_iterations"] - locs["it"])
        eta_h, rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(rem, 60)
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"""
        )
        print(log_string)


    def save(self, path, infos=None, save_onnx=False):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = getattr(self, "obs_normalizer").state_dict()
            saved_dict["critic_obs_norm_state_dict"] = getattr(self, "critic_obs_normalizer").state_dict()
        torch.save(saved_dict, path)

        if save_onnx:
            onnx_folder = os.path.dirname(path)
            try:
                iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
            except Exception:
                iteration = self.current_learning_iteration
            onnx_model_name = f"policy_{iteration}.onnx"
            export_policy_as_onnx(
                self.alg.actor_critic,
                normalizer=self.obs_normalizer if hasattr(self, "obs_normalizer") else torch.nn.Identity(),
                path=onnx_folder,
                filename=onnx_model_name,
            )

    def load(self, path, load_optimizer=True, weights_only=False):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=weights_only)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.amp_normalizer = loaded_dict["amp_normalizer"]

        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.runner_cfg.get("empirical_normalization", False):
            if device is not None and hasattr(self, "obs_normalizer"):
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        self.alg.discriminator.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        self.alg.discriminator.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
