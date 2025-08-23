# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import inspect
import torch
import torch.nn as nn
import torch.optim as optim

# External modules providing the actor-critic model, storage utilities, and AMP components.
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

from amp_rsl_rl.storage import ReplayBuffer
from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils import AMPLoader


class AMP_PPO:
    """
    AMP_PPO implements Adversarial Motion Prior (AMP) combined with Proximal Policy Optimization (PPO).

    This class integrates reinforcement learning via PPO with adversarial imitation learning using AMP.
    It improves policy training by leveraging both interactions with the environment and expert motion data.
    The discriminator is used to force the policy's behavior to align with expert demonstrations.
    """

    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic: ActorCritic,
        discriminator: Discriminator,
        amp_data: AMPLoader,
        amp_normalizer: Optional[Any],
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        amp_replay_buffer_size: int = 100000,
        use_smooth_ratio_clipping: bool = False,
        device: str = "cpu",
    ) -> None:
        # Device & lr scheduling
        self.device: str = device
        self.desired_kl: float = desired_kl
        self.schedule: str = schedule
        self.learning_rate: float = learning_rate

        # AMP components
        self.discriminator: Discriminator = discriminator.to(self.device)
        self.amp_transition: RolloutStorage.Transition = RolloutStorage.Transition()
        obs_dim: int = self.discriminator.input_dim // 2  # [s, s'] concat -> half for one state
        self.amp_storage: ReplayBuffer = ReplayBuffer(
            obs_dim=obs_dim, buffer_size=amp_replay_buffer_size, device=device
        )
        self.amp_data: AMPLoader = amp_data
        self.amp_normalizer: Optional[Any] = amp_normalizer

        # Policy & storage
        self.actor_critic = actor_critic.to(self.device)
        self.storage: Optional[RolloutStorage] = None

        # Optimizer (share one for AC + Discriminator, with trunk/head不同的wd)
        params = [
            {"params": self.actor_critic.parameters(), "name": "actor_critic"},
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 1.0e-3, "name": "amp_trunk"},
            {"params": self.discriminator.linear.parameters(), "weight_decay": 1.0e-1, "name": "amp_head"},
        ]
        self.optimizer: optim.Adam = optim.Adam(params, lr=learning_rate)

        # PPO hyperparams
        self.transition: RolloutStorage.Transition = RolloutStorage.Transition()
        self.clip_param: float = clip_param
        self.num_learning_epochs: int = num_learning_epochs
        self.num_mini_batches: int = num_mini_batches
        self.value_loss_coef: float = value_loss_coef
        self.entropy_coef: float = entropy_coef
        self.gamma: float = gamma
        self.lam: float = lam
        self.max_grad_norm: float = max_grad_norm
        self.use_clipped_value_loss: bool = use_clipped_value_loss
        self.use_smooth_ratio_clipping: bool = use_smooth_ratio_clipping

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: Tuple[int, ...],
        critic_obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
    ) -> None:
        """
        Initializes the storage for collected transitions during interactions with the environment.
        兼容不同版本的 rsl_rl.storage.RolloutStorage 初始化签名。
        """
        candidate_kwargs = {
            "training_type": "rl",  # 新版可能存在
            "num_envs": num_envs,
            "num_transitions_per_env": num_transitions_per_env,
            "obs_shape": actor_obs_shape,
            "privileged_obs_shape": critic_obs_shape,  # 有的版本可能改名或不支持
            "actions_shape": action_shape,
            "rnd_state_shape": None,  # 新版可能存在
            "device": self.device,
        }

        sig = inspect.signature(RolloutStorage.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        final_kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed}

        # 兼容别名：如果没有 privileged_obs_shape，但有 priv_obs_shape
        if "privileged_obs_shape" not in final_kwargs and "priv_obs_shape" in allowed:
            final_kwargs["priv_obs_shape"] = critic_obs_shape

        self.storage = RolloutStorage(**final_kwargs)

    def test_mode(self) -> None:
        self.actor_critic.test()

    def train_mode(self) -> None:
        self.actor_critic.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """
        Choose action and write transition fields.
        关键：同时写 privileged_observations 与 critic_observations，兼容不同实现。
        """
        if getattr(self.actor_critic, "is_recurrent", False):
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.transition.critic_observations = critic_obs  # 重要：你的 RolloutStorage 可能用这个字段名

        return self.transition.actions

    def act_amp(self, amp_obs: torch.Tensor) -> None:
        """Record AMP observation from policy side."""
        self.amp_transition.observations = amp_obs

    def process_env_step(
        self, rewards: torch.Tensor, dones: torch.Tensor, infos: Dict[str, Any]
    ) -> None:
        """
        Add env step outcome to storage and reset RNN states for done envs.
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # bootstrapping on timeouts
        if isinstance(infos, dict) and "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def process_amp_step(self, amp_obs: torch.Tensor) -> None:
        """
        Insert (s_t, s_{t+1}) from policy into AMP replay buffer.
        """
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.amp_transition.clear()

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """
        GAE returns based on critic(last_obs).
        """
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """
        Update both PPO and AMP discriminator.
        自适应支持 11/12 项 mini-batch（是否含 rnd_state_batch 均可）。
        返回 9 项统计（runner 按 9 项接收）。
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_accuracy_policy = 0.0
        mean_accuracy_expert = 0.0
        mean_accuracy_policy_elem = 0.0
        mean_accuracy_expert_elem = 0.0
        mean_kl_divergence = 0.0

        if getattr(self.actor_critic, "is_recurrent", False):
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            num_mini_batch=self.num_learning_epochs * self.num_mini_batches,
            mini_batch_size=self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
            allow_replacement=True,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
            # ---------- 自适应解包（11 或 12 项） ----------
            if len(sample) == 12:
                (
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hid_states_batch,
                    masks_batch,
                    _rnd_state_batch,
                ) = sample
            elif len(sample) == 11:
                (
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hid_states_batch,
                    masks_batch,
                ) = sample
                _rnd_state_batch = None
            else:
                raise RuntimeError(f"Unsupported minibatch sample length: {len(sample)}")

            # ---------- PPO 前向 ----------
            self.actor_critic.act(
                obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # ---------- KL 自适应 LR（可选） ----------
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + 1.0e-8) + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch) + 1.0e-8)
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    mean_kl_divergence += kl_mean.item()

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.learning_rate

            # ---------- PPO 损失 ----------
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            min_, max_ = 1.0 - self.clip_param, 1.0 + self.clip_param
            if self.use_smooth_ratio_clipping:
                clipped_ratio = (
                    1 / (1 + torch.exp((-(ratio - min_) / (max_ - min_) + 0.5) * 4))
                    * (max_ - min_) + min_
                )
            else:
                clipped_ratio = torch.clamp(ratio, min_, max_)

            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * clipped_ratio
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # ---------- AMP 判别器 ----------
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize(policy_state)
                    policy_next_state = self.amp_normalizer.normalize(policy_next_state)
                    # import ipdb;ipdb.set_trace()

                    expert_state = self.amp_normalizer.normalize(expert_state)
                    expert_next_state = self.amp_normalizer.normalize(expert_next_state)
            B_pol = policy_state.size(0)
            discriminator_input = torch.cat(
                (
                    torch.cat([policy_state, policy_next_state], dim=-1),
                    torch.cat([expert_state, expert_next_state], dim=-1),
                ),
                dim=0,
            )
            discriminator_output = self.discriminator(discriminator_input)
            policy_d, expert_d = discriminator_output[:B_pol], discriminator_output[B_pol:]

            amp_loss, grad_pen_loss = self.discriminator.compute_loss(
                policy_d, expert_d, sample_amp_expert, sample_amp_policy, lambda_=10
            )

            loss = ppo_loss + (amp_loss + grad_pen_loss)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state)
                self.amp_normalizer.update(expert_state)

            policy_d_prob = torch.sigmoid(policy_d)
            expert_d_prob = torch.sigmoid(expert_d)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d_prob.mean().item()
            mean_expert_pred += expert_d_prob.mean().item()

            mean_accuracy_policy += torch.sum(
                torch.round(policy_d_prob) == torch.zeros_like(policy_d_prob)
            ).item()
            mean_accuracy_expert += torch.sum(
                torch.round(expert_d_prob) == torch.ones_like(expert_d_prob)
            ).item()
            mean_accuracy_expert_elem += expert_d_prob.numel()
            mean_accuracy_policy_elem += policy_d_prob.numel()

        num_updates = max(1, self.num_learning_epochs * self.num_mini_batches)
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        if mean_accuracy_policy_elem > 0:
            mean_accuracy_policy /= mean_accuracy_policy_elem
        if mean_accuracy_expert_elem > 0:
            mean_accuracy_expert /= mean_accuracy_expert_elem
        mean_kl_divergence /= num_updates

        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
            mean_accuracy_policy,
            mean_accuracy_expert,
            mean_kl_divergence,
        )
