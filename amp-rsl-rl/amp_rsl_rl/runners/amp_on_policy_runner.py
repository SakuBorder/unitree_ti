from __future__ import annotations
import os, time, statistics, inspect
from collections import deque
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.utils import store_code_state

from amp_rsl_rl.utils import Normalizer
from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils.motion_lib_taihu import MotionLibTaihu

from amp_rsl_rl.utils.amp_helpers import (
    unpack_env_observations, unpack_env_step, merge_amp_cfg, build_amp_obs_from_obs
)
from amp_rsl_rl.utils.motion_lib_adapter import MotionLibAMPAdapter
from amp_rsl_rl.utils.amp_logging import AmpLogger
from amp_rsl_rl.utils.amp_saver import save_checkpoint, load_checkpoint


try:
    from amp_rsl_rl.networks import ActorCriticMoE  # type: ignore
    _HAS_MOE = True
except Exception:
    ActorCriticMoE = None  # type: ignore
    _HAS_MOE = False


class AMPOnPolicyRunner:
    """AMP + PPO Runner（模块化版本，KISS）。"""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.device = device

        # ---- cfg ----
        self.cfg = train_cfg
        self.runner_cfg = dict(train_cfg.get("runner", {}))
        self.policy_cfg = dict(train_cfg.get("policy", {}))
        self.alg_cfg = dict(train_cfg.get("algorithm", {}))
        self.discriminator_cfg = dict(train_cfg.get("discriminator", {}))
        self.amp_cfg = merge_amp_cfg(train_cfg)

        # ---- weights (task/style) ----
        task_w = float(self.amp_cfg.get("task_weight", 0.5))
        style_w = float(self.amp_cfg.get("style_weight", 0.5))
        s = task_w + style_w
        self.task_w = (task_w / s) if s > 0 else 0.5
        self.style_w = (style_w / s) if s > 0 else 0.5

        # ---- obs dims ----
        obs0, priv0 = unpack_env_observations(self.env)
        if not isinstance(obs0, torch.Tensor):
            raise RuntimeError("env.get_observations() 应返回 Tensor 或 (Tensor, extras)")
        num_actor_obs = int(obs0.shape[-1])
        num_critic_obs = int(priv0.shape[-1]) if isinstance(priv0, torch.Tensor) else num_actor_obs

        num_actions = getattr(self.env, "num_actions", None)
        if num_actions is None:
            raise RuntimeError("需要 env.num_actions")

        # ---- policy ----
        policy_name = self.runner_cfg.get("policy_class_name", self.policy_cfg.get("class_name", "ActorCritic"))
        policies: Dict[str, Any] = {"ActorCritic": ActorCritic}
        if _HAS_MOE and ActorCriticMoE is not None:
            policies["ActorCriticMoE"] = ActorCriticMoE
        policy_cls = policies.get(policy_name, ActorCritic)

        policy_kwargs = {k: v for k, v in self.policy_cfg.items() if k not in ("class_name",)}
        self.actor_critic: ActorCritic | ActorCriticRecurrent | Any = policy_cls(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            **policy_kwargs,
        ).to(self.device)

        # ---- AMP timing ----
        dt_cfg = getattr(getattr(self.env, "cfg", None), "sim", None)
        dt = getattr(dt_cfg, "dt", None)
        deci_cfg = getattr(getattr(self.env, "cfg", None), "control", None)
        decimation = getattr(deci_cfg, "decimation", None) or getattr(getattr(self.env, "cfg", None), "decimation", None)
        if dt is None:
            dt = float(self.amp_cfg.get("dt", 1.0 / 60.0))
        if decimation is None:
            decimation = int(self.amp_cfg.get("decimation", 1))
        delta_t = float(dt) * int(decimation)

        # ---- AMP dims/history ----
        num_amp_obs_env = getattr(self.env, "num_amp_obs", None)
        num_amp_obs_cfg = self.amp_cfg.get("num_amp_obs", None)
        num_amp_obs = int(num_amp_obs_env if num_amp_obs_env is not None else (num_amp_obs_cfg if num_amp_obs_cfg is not None else num_actor_obs))
        history_steps = int(self.amp_cfg.get("history_steps", self.amp_cfg.get("num_amp_obs_steps", getattr(self.env, "_num_amp_obs_steps", 2))))
        history_stride = int(self.amp_cfg.get("history_stride", 1))

        # ---- MotionLib ----
        motion_file = self.amp_cfg.get("motion_file", self.amp_cfg.get("amp_data_path", None))
        if motion_file is None:
            raise RuntimeError("AMP 需要 'motion_file'（或旧键 'amp_data_path'）")

        mjcf_file = self.amp_cfg.get("mjcf_file", "/home/dy/dy/code/unitree_ti/assert/ti5/ti5_12dof.xml")
        extend_hand = bool(self.amp_cfg.get("extend_hand", False))
        extend_head = bool(self.amp_cfg.get("extend_head", False))
        expect_dof_obs_dim = int(self.amp_cfg.get("expect_dof_obs_dim", 12))

        # ✅ 优先使用名字；ids 只有在明确为 MotionLib 自身索引时才使用
        key_body_names = self.amp_cfg.get("key_body_names", None)
        key_body_ids = self.amp_cfg.get("key_body_ids", None)  # 默认 None，不做任何环境索引的猜测
        if key_body_names and not isinstance(key_body_names, (list, tuple)):
            key_body_names = list(key_body_names)
        if key_body_ids and not isinstance(key_body_ids, (list, tuple)):
            key_body_ids = list(key_body_ids)

        # 期望的 key_bodies 数，自动从 names/ids 推断；否则回退 2
        if key_body_names:
            expect_key_bodies = len(key_body_names)
        elif key_body_ids:
            expect_key_bodies = len(key_body_ids)
        else:
            expect_key_bodies = int(self.amp_cfg.get("expect_key_bodies", 2))

        bootstrap_motions = int(self.amp_cfg.get("bootstrap_motions", min(32, getattr(self.env, "num_envs", 32))))
        use_joint_mapping = bool(self.amp_cfg.get("use_joint_mapping", True))
        debug_joint_mapping = bool(self.amp_cfg.get("debug_joint_mapping", True))

        # root height flag
        use_root_h = bool(
            getattr(getattr(self.env, "cfg", None), "observations", None)
            and getattr(self.env.cfg.observations, "amp", None)
            and getattr(self.env.cfg.observations.amp, "root_height", False)
        )
        os.environ["AMP_USE_ROOT_H"] = "1" if use_root_h else "0"

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

        # 预加载（bootstrap）
        node_names = getattr(motion_lib.mesh_parsers, "model_names", None) or getattr(motion_lib.mesh_parsers, "body_names", None)
        if node_names is None:
            raise RuntimeError("MotionLib.mesh_parsers 需要提供 'model_names' 或 'body_names'。")

        class _Skel:
            __slots__ = ("node_names",)
            def __init__(self, names): self.node_names = names

        skeleton_trees = [_Skel(node_names) for _ in range(max(1, bootstrap_motions))]
        gender_betas = [torch.zeros(17) for _ in range(len(skeleton_trees))]
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

        # ---- 对 key_body_ids 做一次越界裁剪（基于刚体数），避免 CUDA 断言 ----
        if key_body_ids is not None:
            try:
                rb_count = len(node_names)  # 刚体数量
                ids = [int(i) for i in key_body_ids if 0 <= int(i) < rb_count]
                if len(ids) != len(key_body_ids):
                    bad = [int(i) for i in key_body_ids if not (0 <= int(i) < rb_count)]
                    print(f"[WARN] key_body_ids 越界已过滤: 原={key_body_ids}, 过滤后={ids}, 无效={bad}, rb_count={rb_count}")
                key_body_ids = ids if len(ids) > 0 else None
            except Exception as e:
                print(f"[WARN] key_body_ids 预处理失败，已置为 None: {e}")
                key_body_ids = None

        # ---- AMPLoader -> Adapter  (携带 names/ids + env_dof_names) ----
        self.amp_data = MotionLibAMPAdapter(
            motion_lib=motion_lib,
            dt=delta_t,
            history_steps=history_steps,
            history_stride=history_stride,
            expect_dof_obs_dim=expect_dof_obs_dim,
            expect_key_bodies=(len(key_body_names) if key_body_names else
                            (len(key_body_ids) if key_body_ids else expect_key_bodies)),
            key_body_ids=key_body_ids,                      # 可为 None
            key_body_names=key_body_names,                  # ✅ 建议传名字
            env_dof_names=self.amp_cfg.get("env_dof_names", None),  # （可选）用于打印/对齐
            use_root_h=use_root_h,
            device=self.device,
        )

        if hasattr(self.env, "set_amp_data"):
            self.env.set_amp_data(self.amp_data)

        # 初始复位
        try:
            all_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
            (self.env.reset_idx(all_ids) if hasattr(self.env, "reset_idx") else self.env.reset())
        except Exception as e:
            print(f"[Runner] 初始 reset 警告: {e}")

        # 数据概览
        num_motions = getattr(motion_lib, "num_motions", lambda: None)()
        total_len = getattr(motion_lib, "get_total_length", lambda: None)()
        sample_preview = "n/a"
        try:
            g = self.amp_data.feed_forward_generator(1, 2)
            ex_obs, ex_next = next(g)
            sample_preview = f"obs={tuple(ex_obs.shape)}, next_obs={tuple(ex_next.shape)}"
        except Exception as e:
            sample_preview = f"peek failed: {e}"

        key_desc = f"names={key_body_names}" if key_body_names else (f"ids={key_body_ids}" if key_body_ids else "none")
        self._amp_data_text = (
            f"motion_file: {motion_file}\n"
            f"mjcf_file: {mjcf_file}\n"
            f"motions_loaded: {num_motions if num_motions is not None else 'n/a'}\n"
            f"total_length(s): {total_len:.3f}" if isinstance(total_len, (int, float)) else "total_length(s): n/a"
        )
        print("[AMP DATA] === Summary ===")
        for line in self._amp_data_text.splitlines():
            print("[AMP DATA] " + line)
        print(
            f"[AMP DATA] dt={delta_t:.6f} K={int(history_steps)} S={int(history_stride)} "
            f"D_dof={expect_dof_obs_dim} key_bodies={ (len(key_body_names) if key_body_names else (len(key_body_ids) if key_body_ids else expect_key_bodies)) } ({key_desc})"
        )
        print(f"[AMP DATA] sample preview: {sample_preview}")
        print("[AMP DATA] =================")

        # ---- AMP 模块 ----
        self.amp_normalizer = Normalizer(num_amp_obs, device=self.device)
        self.discriminator = Discriminator(
            input_dim=num_amp_obs * 2,
            hidden_layer_sizes=self.discriminator_cfg.get("hidden_dims", [1024, 512]),
            reward_scale=float(self.discriminator_cfg.get("reward_scale", 1.0)),
            device=self.device,
            loss_type=self.discriminator_cfg.get("loss_type", "BCEWithLogits"),
        ).to(self.device)

        if self.amp_data.sample_item_dim != num_amp_obs:
            raise ValueError(
                f"AMP obs 维度不匹配: env={num_amp_obs}, data={self.amp_data.sample_item_dim} (K*D)."
            )
        if getattr(self.discriminator, "input_dim", 0) != 2 * num_amp_obs:
            raise ValueError("Discriminator input_dim 应为 2*num_amp_obs")

        # ---- Algorithm ----
        raw_algo_kwargs = {k: v for k, v in self.alg_cfg.items() if k != "class_name"}
        import inspect as _inspect
        allowed = set(_inspect.signature(AMP_PPO.__init__).parameters) - {"self"}
        algo_kwargs = {k: v for k, v in raw_algo_kwargs.items() if k in allowed}
        dropped = sorted(set(raw_algo_kwargs) - set(algo_kwargs))
        if dropped:
            print(f"[AMPOnPolicyRunner] 丢弃不支持的 AMP_PPO 参数: {dropped}")

        self.alg: AMP_PPO = AMP_PPO(
            actor_critic=self.actor_critic,
            discriminator=self.discriminator,
            amp_data=self.amp_data,
            amp_normalizer=self.amp_normalizer,
            device=self.device,
            **algo_kwargs,
        )

        # ---- rollout/log ----
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
        self.logger = AmpLogger(log_dir) if log_dir is not None else AmpLogger(None)

        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]


    # ---------------- training ----------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if hasattr(self.env, "reset"):
            self.env.reset()
        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf") and hasattr(self.env, "max_episode_length"):
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs, priv = unpack_env_observations(self.env)
        critic_obs = priv if isinstance(priv, torch.Tensor) else obs

        num_amp_obs = int(getattr(self.env, "num_amp_obs", self.amp_cfg.get("num_amp_obs", obs.shape[-1])))
        amp_obs = build_amp_obs_from_obs(self.env, obs, num_amp_obs)

        obs = self.obs_normalizer(obs).to(self.device)
        critic_obs = self.critic_obs_normalizer(critic_obs).to(self.device)
        amp_obs = amp_obs.to(self.device)

        self.train_mode()

        ep_infos: List[Dict[str, Any]] = []
        rewbuffer, lenbuffer = deque(maxlen=100), deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # 首次写数据概览
        self.logger.add_text_once("AMP/DataSummary", self._amp_data_text, step=0)

        for it in range(start_iter, tot_iter):
            start = time.time()
            mean_style_reward_log = 0.0
            mean_task_reward_log = 0.0

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    self.alg.act_amp(amp_obs)

                    step_ret = self.env.step(actions)
                    obs_next, priv_next, rewards, dones, infos, term_ids, term_priv = unpack_env_step(step_ret)
                    critic_next = priv_next if isinstance(priv_next, torch.Tensor) else obs_next

                    next_amp_obs = build_amp_obs_from_obs(self.env, obs_next, num_amp_obs)

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

                    # 处理终止时的 critic 替换
                    if term_ids is not None and term_priv is not None:
                        term_ids = term_ids.to(self.device)
                        term_priv = term_priv.to(self.device)
                        critic_fixed = critic_next.clone().detach()
                        critic_fixed[term_ids] = term_priv.clone().detach()
                        critic_next = critic_fixed

                    obs = obs_next
                    critic_obs = critic_next
                    amp_obs = next_amp_obs.detach()

                    # 统计 episode
                    if isinstance(infos, dict) and "episode" in infos:
                        ep_infos.append(infos["episode"])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].detach().cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs)

            # ======== 更新（AMP_PPO.update 返回 8 项）========
            update_out = self.alg.update()
            if not isinstance(update_out, (list, tuple)):
                raise RuntimeError(f"update() 返回类型异常: {type(update_out)}")

            # 映射到直观命名：Loss/actor(策略), Loss/critic(价值)
            if len(update_out) != 8:
                raise RuntimeError(f"AMP_PPO.update() 期望 8 项，得到 {len(update_out)} 项。")
            (
                loss_critic,                # value 损失 -> Loss/critic
                loss_actor,                 # surrogate 损失 -> Loss/actor
                loss_amp_disc_bce,          # 判别器 BCE
                loss_amp_grad_pen,          # 判别器梯度惩罚
                d_sigmoid_policy_mean,      # D 对 policy 的概率
                d_sigmoid_expert_mean,      # D 对 expert 的概率
                d_acc_policy_as_fake,       # policy 判为假(0)的准确率
                d_acc_expert_as_real,       # expert 判为真(1)的准确率
            ) = update_out

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # -------- logging --------
            fps = int(self.num_steps_per_env * self.env.num_envs / max(1e-6, (collection_time + learn_time)))
            try:
                mean_std = float(self.alg.actor_critic.std.mean().item())
            except Exception:
                mean_std = float("nan")

            self.logger.log_episode_stats(it, ep_infos)

            # 每步平均奖励（更可读）
            task_mean_per_step = mean_task_reward_log / max(1.0, float(self.num_steps_per_env))
            style_mean_per_step = mean_style_reward_log / max(1.0, float(self.num_steps_per_env))

            self.logger.log_scalars(it, {
                # 主损失（你想要的命名）
                "Loss/actor": loss_actor,
                "Loss/critic": loss_critic,

                # AMP 判别器
                "Loss/amp_discriminator": loss_amp_disc_bce,
                "Loss/amp_grad_penalty": loss_amp_grad_pen,
                "D/policy_sigmoid_mean": d_sigmoid_policy_mean,
                "D/expert_sigmoid_mean": d_sigmoid_expert_mean,
                "D/acc_policy_as_fake": d_acc_policy_as_fake,
                "D/acc_expert_as_real": d_acc_expert_as_real,

                # 奖励与策略分布
                "Reward/task_mean_per_step": task_mean_per_step,
                "Reward/style_mean_per_step": style_mean_per_step,
                "Policy/action_std_mean": mean_std,

                # 性能
                "Perf/fps": fps,
                "Perf/collection_s": collection_time,
                "Perf/learning_s": learn_time,
            })
            self.logger.log_train_buffers(it, rewbuffer, lenbuffer, mean_style_reward_log, mean_task_reward_log)

            # 控制台简要打印（KISS）
            if (it < start_iter + 3) or (it % 100 == 0):
                print(
                    f"[AMP] it={it} | Loss(actor/critic)={loss_actor:.4f}/{loss_critic:.4f} "
                    f"| D(loss/pen)={loss_amp_disc_bce:.4f}/{loss_amp_grad_pen:.4f} "
                    f"| D(sig p,e)=({d_sigmoid_policy_mean:.3f},{d_sigmoid_expert_mean:.3f}) "
                    f"| acc(p/e)=({d_acc_policy_as_fake:.3f}/{d_acc_expert_as_real:.3f}) "
                    f"| reward(step): task={task_mean_per_step:.4f}, style={style_mean_per_step:.4f} | fps={fps}"
                )

            # 存档
            if self.log_dir is not None and (it % self.save_interval == 0):
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), save_onnx=True)

            # 统计
            ep_infos.clear()
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += collection_time + learn_time

            # 首次保存代码快照
            if it == start_iter and self.log_dir is not None:
                try:
                    store_code_state(self.log_dir, [rsl_rl.__file__])
                except Exception:
                    pass

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"), save_onnx=True)


    # ---------------- utils ----------------
    def save(self, path, infos=None, save_onnx=False):
        obs_norm_sd = getattr(self, "obs_normalizer").state_dict() if self.empirical_normalization else None
        critic_norm_sd = getattr(self, "critic_obs_normalizer").state_dict() if self.empirical_normalization else None
        save_checkpoint(
            path,
            actor_critic=self.alg.actor_critic,
            optimizer=self.alg.optimizer,
            discriminator=self.alg.discriminator,
            amp_normalizer=self.alg.amp_normalizer,
            iteration=self.current_learning_iteration,
            obs_norm_state_dict=obs_norm_sd,
            critic_obs_norm_state_dict=critic_norm_sd,
            save_onnx=save_onnx,
        )

    def load(self, path, load_optimizer=True, weights_only=False):
        self.alg.amp_normalizer, self.current_learning_iteration, infos = load_checkpoint(
            path,
            actor_critic=self.alg.actor_critic,
            discriminator=self.alg.discriminator,
            amp_normalizer=self.alg.amp_normalizer,
            optimizer=(self.alg.optimizer if load_optimizer else None),
            device=self.device,
            empirical_normalization=self.empirical_normalization,
            obs_normalizer=(self.obs_normalizer if self.empirical_normalization else None),
            critic_obs_normalizer=(self.critic_obs_normalizer if self.empirical_normalization else None),
            weights_only=weights_only,
        )
        return infos

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
            if self.runner_cfg.get("empirical_normalization", False) and hasattr(self, "obs_normalizer"):
                self.obs_normalizer.to(device)
        if self.runner_cfg.get("empirical_normalization", False):
            return lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))
        return self.alg.actor_critic.act_inference

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
