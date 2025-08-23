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

from amp_rsl_rl.utils import Normalizer, AMPLoader, export_policy_as_onnx
from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import Discriminator

# 可选：如果有 MoE，尝试导入；没有就静默回退
try:
    from amp_rsl_rl.networks import ActorCriticMoE  # type: ignore
    _HAS_MOE = True
except Exception:
    ActorCriticMoE = None  # type: ignore
    _HAS_MOE = False


# ----------------- 工具函数：兼容不同的 env API -----------------

def _unpack_env_observations(env: VecEnv) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    兼容：
      - get_observations() -> obs
      - get_observations() -> (obs, extras)
    并尝试通过 env.get_privileged_observations() 获取 priv。
    """
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
    """
    统一解包 env.step(...)：
      (obs, priv, rewards, dones, infos, term_ids, term_priv)
    支持 7 元组与 4 元组。
    """
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
    """
    统一抽取 AMP 相关配置：
      - 优先从 train_cfg['amp'] 读取
      - 若有人把 amp 键放在顶层（amp_data_path 等），也一并合入
    """
    amp = dict(train_cfg.get("amp", {}))
    for k in (
        "amp_data_path", "dataset_names", "dataset_weights", "slow_down_factor",
        "num_amp_obs", "dt", "decimation", "replay_buffer_size", "reward_scale",
        "joint_names", "style_weight", "task_weight",
        "num_amp_obs_steps", "history_steps", "history_stride"
    ):
        if k in train_cfg and k not in amp:
            amp[k] = train_cfg[k]
    return amp


class AMPOnPolicyRunner:
    """
    AMP + PPO 训练器（仅 TensorBoard，KISS 版）
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        from pathlib import Path

        self.env = env
        self.device = device

        # -------- 配置拆解 --------
        self.cfg = train_cfg
        self.runner_cfg = dict(train_cfg.get("runner", {}))
        self.policy_cfg = dict(train_cfg.get("policy", {}))
        self.alg_cfg = dict(train_cfg.get("algorithm", {}))
        self.discriminator_cfg = dict(train_cfg.get("discriminator", {}))
        self.amp_cfg = _merge_amp_cfg(train_cfg)

        # === 奖励融合权重（默认 0.5/0.5，并归一化到和为 1） ===
        self.task_w = float(self.amp_cfg.get("task_weight", 0.5))
        self.style_w = float(self.amp_cfg.get("style_weight", 0.5))
        _sum_w = self.task_w + self.style_w
        if _sum_w <= 0.0:
            self.task_w, self.style_w = 0.5, 0.5
        else:
            self.task_w /= _sum_w
            self.style_w /= _sum_w

        # -------- 获取一帧观测维度 --------
        obs0, priv0 = _unpack_env_observations(self.env)
        if not isinstance(obs0, torch.Tensor):
            raise RuntimeError("env.get_observations() must return a torch.Tensor or (Tensor, extras).")
        num_actor_obs = int(obs0.shape[-1])
        num_critic_obs = int(priv0.shape[-1]) if isinstance(priv0, torch.Tensor) else num_actor_obs

        num_actions = getattr(self.env, "num_actions", None)
        if num_actions is None:
            raise RuntimeError("env.num_actions is required by AMPOnPolicyRunner.")

        # -------- 策略网络选择（支持 MoE 回退） --------
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

        # -------- AMP 组件 --------
        # dt/decimation：优先从 env.cfg；取不到用 amp_cfg 兜底
        dt_cfg = getattr(getattr(self.env, "cfg", None), "sim", None)
        dt = getattr(dt_cfg, "dt", None)
        deci_cfg = getattr(getattr(self.env, "cfg", None), "control", None)
        decimation = getattr(deci_cfg, "decimation", None) or getattr(getattr(self.env, "cfg", None), "decimation", None)
        if dt is None:
            dt = float(self.amp_cfg.get("dt", 1.0 / 60.0))
        if decimation is None:
            decimation = int(self.amp_cfg.get("decimation", 1))
        delta_t = float(dt) * int(decimation)

        # 优先使用环境提供的 AMP 维度（含历史），否则用 cfg 中的数值
        num_amp_obs_env = getattr(self.env, "num_amp_obs", None)
        num_amp_obs_cfg = self.amp_cfg.get("num_amp_obs", None)
        num_amp_obs = int(num_amp_obs_env if num_amp_obs_env is not None else (num_amp_obs_cfg if num_amp_obs_cfg is not None else num_actor_obs))

        # 历史窗口设置：优先 amp.num_amp_obs_steps 或 env._num_amp_obs_steps
        history_steps = self.amp_cfg.get("history_steps", None)
        if history_steps is None:
            history_steps = self.amp_cfg.get("num_amp_obs_steps", None)
        if history_steps is None:
            history_steps = getattr(self.env, "_num_amp_obs_steps", 1)
        history_stride = self.amp_cfg.get("history_stride", 1)

        joint_names = self.amp_cfg.get("joint_names", None)
        amp_data_path = self.amp_cfg.get("amp_data_path", None)
        dataset_names = list(self.amp_cfg.get("dataset_names", []) or [])
        dataset_weights = list(self.amp_cfg.get("dataset_weights", []) or [])
        slow_down_factor = self.amp_cfg.get("slow_down_factor", 1)

        # === 目录展开逻辑 ===
        if amp_data_path is None:
            raise RuntimeError("AMP config requires 'amp_data_path'.")
        root = Path(amp_data_path)

        if len(dataset_names) == 1:
            maybe_dir = root / dataset_names[0]
            if maybe_dir.is_dir():
                exts = {".npy", ".npz", ".pt", ".pkl"}
                files = [p for p in maybe_dir.iterdir() if p.is_file() and p.suffix in exts]
                files.sort()
                if len(files) == 0:
                    print(f"[AMP DATA] Warning: directory '{maybe_dir}' is empty or has no supported files {sorted(list(exts))}.")
                dataset_names = [f"{maybe_dir.name}/{p.stem}" for p in files]
                if len(dataset_weights) != len(dataset_names):
                    dataset_weights = [1.0] * len(dataset_names)
                print(f"[AMP DATA] Expanded directory '{maybe_dir}': {len(dataset_names)} files found.")
            elif not maybe_dir.exists():
                print(f"[AMP DATA] Note: '{maybe_dir}' not a directory; will treat '{dataset_names[0]}' as a single clip name.")

        # 权重长度对齐
        if dataset_weights and len(dataset_weights) != len(dataset_names):
            if len(dataset_weights) > len(dataset_names):
                dataset_weights = dataset_weights[:len(dataset_names)]
            else:
                dataset_weights = dataset_weights + [1.0] * (len(dataset_names) - len(dataset_weights))

        # 构造 AMPLoader（传入历史窗口参数）
        amp_data = AMPLoader(
            self.device,
            str(root),
            dataset_names,
            dataset_weights,
            delta_t,
            slow_down_factor,
            joint_names,
            history_steps=history_steps,
            history_stride=history_stride,
        )
        self.amp_data = amp_data  # 暴露供调试

        # === 注入到环境，并确保首次复位就使用专家状态 ===
        if hasattr(self.env, "set_amp_data"):
            self.env.set_amp_data(self.amp_data)
        try:
            all_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
            if hasattr(self.env, "reset_idx"):
                self.env.reset_idx(all_ids)
            else:
                self.env.reset()
        except Exception as e:
            print(f"[Runner] Soft-warn: initial AMP reset failed: {e}")

        # === AMP 数据摘要（打印 + 训练时写入 TensorBoard） ===
        def _get_attr_any(obj, names):
            for name in names:
                if hasattr(obj, name):
                    v = getattr(obj, name)
                    if v is not None:
                        return v
            return None

        num_pairs = _get_attr_any(amp_data, [
            "num_pairs", "num_samples", "size", "N", "length", "total_pairs", "total_samples"
        ])
        if num_pairs is None:
            try:
                num_pairs = len(amp_data)
            except Exception:
                num_pairs = "unknown"

        _ds = dataset_names or []
        _ws = dataset_weights or []
        if not _ws:
            _ws = [1.0] * len(_ds)
        datasets_line = ", ".join([f"{n}(w={w})" for n, w in zip(_ds, _ws)])

        container = _get_attr_any(amp_data, ["datasets", "clips", "sequences", "items", "sources"])
        details_lines = []
        if isinstance(container, (list, tuple)):
            show_k = min(12, len(container))
            for i in range(show_k):
                item = container[i]
                name = _get_attr_any(item, ["name", "id", "file", "path", "basename"]) or f"item_{i}"
                length = _get_attr_any(item, ["num_pairs", "num_samples", "length", "size", "N"])
                if length is None:
                    try:
                        length = len(item)
                    except Exception:
                        length = "?"
                details_lines.append(f"  - #{i}: {name}, len={length}")
            if len(container) > show_k:
                details_lines.append(f"  - ... (+{len(container)-show_k} more)")

        sample_preview = "n/a"
        try:
            sampler = getattr(amp_data, "sample_pairs", None)
            if callable(sampler):
                sample = amp_data.sample_pairs(2)
                def _shape_of(x):
                    try:
                        import torch as _t
                        if isinstance(x, _t.Tensor):
                            return tuple(x.shape)
                    except Exception:
                        pass
                    try:
                        if hasattr(x, "shape"):
                            return tuple(x.shape)
                    except Exception:
                        pass
                    try:
                        return f"len={len(x)}"
                    except Exception:
                        return type(x).__name__
                if isinstance(sample, (list, tuple)):
                    shapes = ", ".join(_shape_of(v) for v in sample)
                    sample_preview = f"tuple[{len(sample)}]: {shapes}"
                elif isinstance(sample, dict):
                    parts = [f"{k}:{_shape_of(v)}" for k, v in sample.items()]
                    sample_preview = "{ " + ", ".join(parts) + " }"
                else:
                    sample_preview = f"type={type(sample).__name__}"
        except Exception as e:
            sample_preview = f"peek failed: {e}"

        pairs_txt = f"{num_pairs}" if isinstance(num_pairs, (int, float)) else str(num_pairs)
        details_block = "\n".join(details_lines) if details_lines else "(no per-dataset detail available)"
        self._amp_data_text = (
            f"path: {str(root)}\n"
            f"datasets: {datasets_line if datasets_line else '<none>'}\n"
            f"delta_t={delta_t:.6f}, decimation={decimation}, slow_down_factor={slow_down_factor}\n"
            f"num_amp_obs={num_amp_obs}, disc_input_dim={num_amp_obs * 2}\n"
            f"history_steps={int(history_steps)}, history_stride={int(history_stride)}\n"
            f"estimated_pairs={pairs_txt}\n"
            f"per-dataset:\n{details_block}\n"
            f"sample_pairs preview: {sample_preview}"
        )
        print("[AMP DATA] ==== Expert dataset summary ====")
        for line in self._amp_data_text.splitlines():
            print("[AMP DATA] " + line)
        print("[AMP DATA] =================================")

        # 归一化器 & 判别器
        self.amp_normalizer = Normalizer(num_amp_obs, device=self.device)
        self.discriminator = Discriminator(
            input_dim=num_amp_obs * 2,  # current + next
            hidden_layer_sizes=self.discriminator_cfg.get("hidden_dims", [1024, 512]),
            reward_scale=float(self.discriminator_cfg.get("reward_scale", 1.0)),
            device=self.device,
            loss_type=self.discriminator_cfg.get("loss_type", "BCEWithLogits"),
        ).to(self.device)

        # -------- 算法 --------
        algo_name = self.runner_cfg.get("algorithm_class_name", self.alg_cfg.get("class_name", "AMP_PPO"))
        if algo_name != "AMP_PPO":
            print(f"[AMPOnPolicyRunner] Algorithm '{algo_name}' not supported; falling back to 'AMP_PPO'.")
        raw_algo_kwargs = {k: v for k, v in self.alg_cfg.items() if k != "class_name"}
        allowed = set(inspect.signature(AMP_PPO.__init__).parameters.keys()) - {"self"}
        algo_kwargs = {k: v for k, v in raw_algo_kwargs.items() if k in allowed}
        dropped = sorted(set(raw_algo_kwargs) - set(algo_kwargs))
        if dropped:
            print(f"[AMPOnPolicyRunner] Dropped unsupported AMP_PPO args: {dropped}")

        self.alg: AMP_PPO = AMP_PPO(
            actor_critic=self.actor_critic,
            discriminator=self.discriminator,
            amp_data=amp_data,
            amp_normalizer=self.amp_normalizer,
            device=self.device,
            **algo_kwargs,
        )

        # -------- rollout/日志 --------
        self.num_steps_per_env = int(self.runner_cfg.get("num_steps_per_env", 24))
        self.save_interval = int(self.runner_cfg.get("save_interval", 50))
        self.empirical_normalization = bool(self.runner_cfg.get("empirical_normalization", False))

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.critic_obs_normalizer = torch.nn.Identity()

        # 存储初始化
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [num_actions],
        )

        # 日志（仅 TensorBoard）
        self.log_dir = log_dir
        self.writer: Optional[TensorboardSummaryWriter] = None
        if self.log_dir is not None:
            self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    # ----------------- 训练循环 -----------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if hasattr(self.env, "reset"):
            self.env.reset()

        # 随机化 episode 起点
        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf") and hasattr(self.env, "max_episode_length"):
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # 初始观测
        obs, priv = _unpack_env_observations(self.env)
        critic_obs = priv if isinstance(priv, torch.Tensor) else obs

        # 初始化 AMP 观测（优先环境提供）
        num_amp_obs = int(getattr(self.env, "num_amp_obs", self.amp_cfg.get("num_amp_obs", obs.shape[-1])))
        amp_obs = self._build_amp_obs_from_obs(obs, num_amp_obs)

        # 归一化/设备
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

        # [AMP DEBUG] 一次性 sanity 打印
        print(f"[AMP DEBUG] Shapes | actor_obs={tuple(obs.shape)}, critic_obs={tuple(critic_obs.shape)}, amp_obs={tuple(amp_obs.shape)}")
        if hasattr(self.discriminator, "input_dim"):
            print(f"[AMP DEBUG] Discriminator input_dim={self.discriminator.input_dim} (should be 2*{num_amp_obs})")

        # AMP 数据摘要 -> TensorBoard
        if self.writer is not None and getattr(self, "_amp_data_text", None):
            self.writer.add_text("AMP/DataSummary", f"<pre>{self._amp_data_text}</pre>", 0)

        for it in range(start_iter, tot_iter):
            start = time.time()
            mean_style_reward_log = 0.0
            mean_task_reward_log = 0.0

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 策略动作
                    actions = self.alg.act(obs, critic_obs)
                    # AMP：缓存当前 amp_obs
                    self.alg.act_amp(amp_obs)

                    # 环境推进
                    step_ret = self.env.step(actions)
                    obs_next, priv_next, rewards, dones, infos, term_ids, term_priv = _unpack_env_step(step_ret)

                    # 选取 critic 下一帧
                    critic_next = priv_next if isinstance(priv_next, torch.Tensor) else obs_next

                    # 构建 next_amp_obs
                    next_amp_obs = self._build_amp_obs_from_obs(obs_next, num_amp_obs)

                    # 归一化 & 设备
                    obs_next = self.obs_normalizer(obs_next).to(self.device)
                    critic_next = self.critic_obs_normalizer(critic_next).to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)

                    # 判别器风格奖励
                    style_rewards = self.discriminator.predict_reward(amp_obs, next_amp_obs, normalizer=self.amp_normalizer)

                    # 原始任务/风格奖励均值（日志用）
                    mean_task_reward_log += rewards.mean().item()
                    mean_style_reward_log += style_rewards.mean().item()

                    # 融合奖励
                    rewards = self.task_w * rewards + self.style_w * style_rewards

                    # 存入算法缓存
                    self.alg.process_env_step(rewards, dones, infos)
                    self.alg.process_amp_step(next_amp_obs)

                    # 终止修正（若有）
                    if term_ids is not None and term_priv is not None:
                        term_ids = term_ids.to(self.device)
                        term_priv = term_priv.to(self.device)
                        critic_fixed = critic_next.clone().detach()
                        critic_fixed[term_ids] = term_priv.clone().detach()
                        critic_next = critic_fixed

                    # 滚动
                    obs = obs_next
                    critic_obs = critic_next
                    amp_obs = torch.clone(next_amp_obs)

                    # 训练日志（episode）
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

                # 学习步骤
                start = stop
                self.alg.compute_returns(critic_obs)

            # --- Update step（返回值长度自适应） ---
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

            # [AMP DEBUG] 前几次与每 100 次打印
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

            # 记录到 TensorBoard & 终端
            if self.writer is not None:
                locals_to_log = locals().copy()
                locals_to_log.update({
                    "mean_swap_loss": mean_swap_loss,
                    "mean_actor_sym_loss": mean_actor_sym_loss,
                    "mean_critic_sym_loss": mean_critic_sym_loss
                })
                self.log(locals_to_log)

            # 定期保存
            if self.log_dir is not None and (it % self.save_interval == 0):
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), save_onnx=True)

            ep_infos.clear()
            if it == start_iter and self.log_dir is not None:
                try:
                    store_code_state(self.log_dir, [rsl_rl.__file__])
                except Exception:
                    pass

        # 末次保存
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"), save_onnx=True)

    # ----------------- 日志/工具/保存 -----------------

    def _build_amp_obs_from_obs(self, obs: torch.Tensor, num_amp_obs: int) -> torch.Tensor:
        """
        正确构建AMP观测 - 优先使用环境提供的专用方法
        """
        # 方法1: 优先使用环境的compute_amp_observations方法（其内部已含历史）
        if hasattr(self.env, 'compute_amp_observations'):
            return self.env.compute_amp_observations()

        # 方法2: 从环境状态重新构造（降级为单步），保持维度 <= num_amp_obs
        try:
            from isaacgym.torch_utils import quat_rotate_inverse
            if hasattr(self.env, 'dof_pos') and hasattr(self.env, 'dof_vel') and hasattr(self.env, 'base_quat') and hasattr(self.env, 'root_states'):
                base_lin_vel_local = quat_rotate_inverse(self.env.base_quat, self.env.root_states[:, 7:10])
                base_ang_vel_local = quat_rotate_inverse(self.env.base_quat, self.env.root_states[:, 10:13])
                amp_obs = torch.cat([
                    self.env.dof_pos,      # 关节位置
                    self.env.dof_vel,      # 关节速度
                    base_lin_vel_local,    # 局部线速度
                    base_ang_vel_local,    # 局部角速度
                ], dim=-1)
                if amp_obs.shape[-1] != num_amp_obs:
                    # 简单裁剪/零填充到指定维度
                    if amp_obs.shape[-1] > num_amp_obs:
                        amp_obs = amp_obs[..., :num_amp_obs]
                    else:
                        pad = num_amp_obs - amp_obs.shape[-1]
                        amp_obs = torch.cat([amp_obs, torch.zeros(amp_obs.shape[0], pad, device=amp_obs.device)], dim=-1)
                return amp_obs
        except Exception as e:
            print(f"Warning: Failed to construct AMP observations from environment state: {e}")

        # 方法3：最后兜底用原 obs（并裁剪/填充）
        amp_obs = obs
        if amp_obs.shape[-1] != num_amp_obs:
            if amp_obs.shape[-1] > num_amp_obs:
                amp_obs = amp_obs[..., :num_amp_obs]
            else:
                pad = num_amp_obs - amp_obs.shape[-1]
                amp_obs = torch.cat([amp_obs, torch.zeros(amp_obs.shape[0], pad, device=amp_obs.device)], dim=-1)
        return amp_obs

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # 统计
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Episode 标量写 TensorBoard
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # 其它标量
        try:
            mean_std = float(self.alg.actor_critic.std.mean().item())
        except Exception:
            mean_std = float("nan")
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # 写 TensorBoard（包含 AMP）
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
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

        # ---- 终端块打印 ----
        head = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        steps_this_iter = float(self.num_steps_per_env)
        mean_style_reward_avg = locs["mean_style_reward_log"] / max(1.0, steps_this_iter)

        common = (
            f"""{'#' * width}\n"""
            f"""{head.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
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

        # ETA
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
