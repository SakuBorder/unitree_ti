"""AMP-PPO algorithm built on top of HIMPPO and HIMActorCritic.

This version mirrors the training pipeline used by Tiv2 tasks while adding
support for adversarial motion priors (AMP).  When the style reward weight is
set to zero the behaviour matches the original Tiv2 implementation based on
``HIMPPO`` and ``HIMActorCritic``.  When the style reward is enabled the same
policy architecture can be optimised jointly with the AMP discriminator.

The implementation is a lightweight merge of ``HIMPPO`` (for symmetry and the
transition estimator) and the previous ``AMP_PPO`` (for discriminator training).
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.him_actor_critic import HIMActorCritic
from rsl_rl.storage.him_rollout_storage import HIMRolloutStorage

from amp_rsl_rl.storage import ReplayBuffer
from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils import AMPLoader


class AMP_PPO:
    """Adversarial Motion Prior (AMP) combined with HIMPPO."""

    actor_critic: HIMActorCritic

    def __init__(
        self,
        actor_critic: HIMActorCritic,
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
        use_flip: bool = True,
        symmetry_scale: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        # device & lr scheduling
        self.device: str = device
        self.desired_kl: float = desired_kl
        self.schedule: str = schedule
        self.learning_rate: float = learning_rate

        # symmetry options (HIM specifics)
        self.use_flip: bool = use_flip
        self.symmetry_scale: float = symmetry_scale

        # AMP components
        self.discriminator: Discriminator = discriminator.to(self.device)
        self.amp_transition: HIMRolloutStorage.Transition = HIMRolloutStorage.Transition()
        obs_dim: int = self.discriminator.input_dim // 2
        self.amp_storage: ReplayBuffer = ReplayBuffer(
            obs_dim=obs_dim, buffer_size=amp_replay_buffer_size, device=device
        )
        self.amp_data: AMPLoader = amp_data
        self.amp_normalizer: Optional[Any] = amp_normalizer

        # policy & storage
        self.actor_critic = actor_critic.to(self.device)
        self.storage: Optional[HIMRolloutStorage] = None
        self.transition: HIMRolloutStorage.Transition = HIMRolloutStorage.Transition()
        self.transition_sym: HIMRolloutStorage.Transition = HIMRolloutStorage.Transition()

        # optimizer – joint for policy and discriminator
        params = [
            {"params": self.actor_critic.parameters(), "name": "actor_critic"},
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 1.0e-3, "name": "amp_trunk"},
            {"params": self.discriminator.linear.parameters(), "weight_decay": 1.0e-1, "name": "amp_head"},
        ]
        self.optimizer: optim.Adam = optim.Adam(params, lr=learning_rate)

        # PPO hyper‑parameters
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

        self.discriminator_loss = nn.BCEWithLogitsLoss()

    # ----------------------------------------------------------------------
    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: Tuple[int, ...],
        critic_obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
    ) -> None:
        self.storage = HIMRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self) -> None:
        self.actor_critic.test()

    def train_mode(self) -> None:
        self.actor_critic.train()

    # ------------------------------------------------------------------
    def discriminator_policy_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        expected = torch.zeros_like(discriminator_output, device=self.device)
        return self.discriminator_loss(discriminator_output, expected)

    def discriminator_expert_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        expected = torch.ones_like(discriminator_output, device=self.device)
        return self.discriminator_loss(discriminator_output, expected)

    # ------------------------------------------------------------------
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        if self.use_flip:
            obs_sym = self.flip_tai5_actor_obs(obs)
            critic_obs_sym = self.flip_tai5_critic_obs(critic_obs)
            self.transition_sym.actions = self.actor_critic.act(obs_sym).detach()
            self.transition_sym.values = self.actor_critic.evaluate(critic_obs_sym).detach()
            self.transition_sym.actions_log_prob = self.actor_critic.get_actions_log_prob(
                self.transition_sym.actions
            ).detach()
            self.transition_sym.action_mean = self.actor_critic.action_mean.detach()
            self.transition_sym.action_sigma = self.actor_critic.action_std.detach()
            self.transition_sym.observations = obs_sym
            self.transition_sym.critic_observations = critic_obs_sym

        return self.transition.actions

    def act_amp(self, amp_obs: torch.Tensor) -> None:
        self.amp_transition.observations = amp_obs

    # ------------------------------------------------------------------
    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: Dict[str, Any],
        next_critic_obs: torch.Tensor,
    ) -> None:
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.use_flip:
            next_critic_obs_sym = self.flip_tai5_critic_obs(next_critic_obs)
            self.transition_sym.next_critic_observations = next_critic_obs_sym.clone()
            self.transition_sym.rewards = rewards.clone()
            self.transition_sym.dones = dones

        if isinstance(infos, dict) and "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
            if self.use_flip:
                self.transition_sym.rewards += self.gamma * torch.squeeze(
                    self.transition_sym.values
                    * infos["time_outs"].unsqueeze(1).to(self.device),
                    1,
                )

        self.storage.add_transitions(self.transition)
        if self.use_flip:
            self.storage.add_transitions(self.transition_sym)
        self.transition.clear()
        if self.use_flip:
            self.transition_sym.clear()
        self.actor_critic.reset(dones)

    def process_amp_step(self, amp_obs: torch.Tensor) -> None:
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.amp_transition.clear()

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # ------------------------------------------------------------------
    def update(self) -> Tuple[float, float, float, float, float, float, float, float]:
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
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                next_critic_obs_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) = sample

            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.use_flip:
                flipped_obs_batch = self.flip_tai5_actor_obs(obs_batch)
                flipped_next_critic_obs_batch = self.flip_tai5_critic_obs(next_critic_obs_batch)
                estimator_obs = torch.cat((obs_batch, flipped_obs_batch), dim=0)
                estimator_next = torch.cat(
                    (next_critic_obs_batch, flipped_next_critic_obs_batch), dim=0
                )
            else:
                estimator_obs = obs_batch
                estimator_next = next_critic_obs_batch
            estimation_loss, swap_loss = self.actor_critic.update_estimator(
                estimator_obs, estimator_next, lr=self.learning_rate
            )

            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            min_, max_ = 1.0 - self.clip_param, 1.0 + self.clip_param
            if self.use_smooth_ratio_clipping:
                clipped_ratio = (
                    1
                    / (1 + torch.exp((-(ratio - min_) / (max_ - min_) + 0.5) * 4))
                    * (max_ - min_)
                    + min_
                )
            else:
                clipped_ratio = torch.clamp(ratio, min_, max_)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * clipped_ratio
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            if self.use_flip:
                flipped_critic_obs_batch = self.flip_tai5_critic_obs(critic_obs_batch)
                actor_sym_loss = self.symmetry_scale * torch.mean(
                    torch.sum(
                        torch.square(
                            self.actor_critic.act_inference(flipped_obs_batch)
                            - self.flip_tai5_actions(
                                self.actor_critic.act_inference(obs_batch)
                            )
                        ),
                        dim=-1,
                    )
                )
                critic_sym_loss = self.symmetry_scale * torch.mean(
                    torch.square(
                        self.actor_critic.evaluate(flipped_critic_obs_batch)
                        - self.actor_critic.evaluate(critic_obs_batch).detach()
                    )
                )
                loss = loss + actor_sym_loss + critic_sym_loss

            # ---- AMP discriminator ----
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize(policy_state)
                    policy_next_state = self.amp_normalizer.normalize(policy_next_state)
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
            discriminator_input = discriminator_input.to(self.device, non_blocking=False)
            dout = self.discriminator(discriminator_input)
            policy_d, expert_d = dout[:B_pol], dout[B_pol:]

            expert_loss = self.discriminator_expert_loss(expert_d)
            policy_loss = self.discriminator_policy_loss(policy_d)
            amp_loss = 0.5 * (expert_loss + policy_loss)

            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_states=(expert_state.to(self.device), expert_next_state.to(self.device)),
                policy_states=(policy_state.to(self.device), policy_next_state.to(self.device)),
                lambda_=10.0,
            )

            total_loss = loss + amp_loss + grad_pen_loss

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.amp_normalizer is not None:
                with torch.no_grad():
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
            mean_accuracy_policy_elem += policy_d_prob.numel()
            mean_accuracy_expert_elem += expert_d_prob.numel()

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
        )

    # ------------------------------------------------------------------
    # Flipping helpers copied from HIMPPO to maintain symmetry behaviour
    def flip_tai5_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        proprioceptive_obs = torch.clone(
            obs[:, : self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length]
        )
        proprioceptive_obs = proprioceptive_obs.view(
            -1, self.actor_critic.actor_history_length, self.actor_critic.num_one_step_obs
        )
        flipped = torch.zeros_like(proprioceptive_obs)
        flipped[:, :, 0] = -proprioceptive_obs[:, :, 0]
        flipped[:, :, 1] = proprioceptive_obs[:, :, 1]
        flipped[:, :, 2] = -proprioceptive_obs[:, :, 2]
        flipped[:, :, 3] = proprioceptive_obs[:, :, 3]
        flipped[:, :, 4] = -proprioceptive_obs[:, :, 4]
        flipped[:, :, 5] = proprioceptive_obs[:, :, 5]
        flipped[:, :, 6] = proprioceptive_obs[:, :, 6]
        flipped[:, :, 7] = -proprioceptive_obs[:, :, 7]
        flipped[:, :, 8] = -proprioceptive_obs[:, :, 8]

        flipped[:, :, 9] = -proprioceptive_obs[:, :, 15]
        flipped[:, :, 10] = proprioceptive_obs[:, :, 16]
        flipped[:, :, 11] = -proprioceptive_obs[:, :, 17]
        flipped[:, :, 12] = -proprioceptive_obs[:, :, 18]
        flipped[:, :, 13] = -proprioceptive_obs[:, :, 19]
        flipped[:, :, 14] = -proprioceptive_obs[:, :, 20]

        flipped[:, :, 15] = -proprioceptive_obs[:, :, 9]
        flipped[:, :, 16] = proprioceptive_obs[:, :, 10]
        flipped[:, :, 17] = -proprioceptive_obs[:, :, 11]
        flipped[:, :, 18] = -proprioceptive_obs[:, :, 12]
        flipped[:, :, 19] = -proprioceptive_obs[:, :, 13]
        flipped[:, :, 20] = -proprioceptive_obs[:, :, 14]

        flipped[:, :, 21] = -proprioceptive_obs[:, :, 27]
        flipped[:, :, 22] = proprioceptive_obs[:, :, 28]
        flipped[:, :, 23] = -proprioceptive_obs[:, :, 29]
        flipped[:, :, 24] = -proprioceptive_obs[:, :, 30]
        flipped[:, :, 25] = -proprioceptive_obs[:, :, 31]
        flipped[:, :, 26] = -proprioceptive_obs[:, :, 32]

        flipped[:, :, 27] = -proprioceptive_obs[:, :, 21]
        flipped[:, :, 28] = proprioceptive_obs[:, :, 22]
        flipped[:, :, 29] = -proprioceptive_obs[:, :, 23]
        flipped[:, :, 30] = -proprioceptive_obs[:, :, 24]
        flipped[:, :, 31] = -proprioceptive_obs[:, :, 25]
        flipped[:, :, 32] = -proprioceptive_obs[:, :, 26]

        flipped[:, :, 33] = -proprioceptive_obs[:, :, 39]
        flipped[:, :, 34] = proprioceptive_obs[:, :, 40]
        flipped[:, :, 35] = -proprioceptive_obs[:, :, 41]
        flipped[:, :, 36] = -proprioceptive_obs[:, :, 42]
        flipped[:, :, 37] = -proprioceptive_obs[:, :, 43]
        flipped[:, :, 38] = -proprioceptive_obs[:, :, 44]

        flipped[:, :, 39] = -proprioceptive_obs[:, :, 33]
        flipped[:, :, 40] = proprioceptive_obs[:, :, 34]
        flipped[:, :, 41] = -proprioceptive_obs[:, :, 35]
        flipped[:, :, 42] = -proprioceptive_obs[:, :, 36]
        flipped[:, :, 43] = -proprioceptive_obs[:, :, 37]
        flipped[:, :, 44] = -proprioceptive_obs[:, :, 38]

        flipped[:, :, 45] = proprioceptive_obs[:, :, 46]
        flipped[:, :, 46] = proprioceptive_obs[:, :, 45]

        return flipped.view(
            -1, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length
        ).detach()

    def flip_tai5_critic_obs(self, critic_obs: torch.Tensor) -> torch.Tensor:
        proprioceptive_obs = torch.clone(
            critic_obs[:, : self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length]
        )
        proprioceptive_obs = proprioceptive_obs.view(
            -1,
            self.actor_critic.critic_history_length,
            self.actor_critic.num_one_step_critic_obs,
        )
        flipped = torch.zeros_like(proprioceptive_obs)

        flipped[:, :, 0] = proprioceptive_obs[:, :, 0]
        flipped[:, :, 1] = -proprioceptive_obs[:, :, 1]
        flipped[:, :, 2] = proprioceptive_obs[:, :, 2]
        flipped[:, :, 3] = -proprioceptive_obs[:, :, 3]
        flipped[:, :, 4] = proprioceptive_obs[:, :, 4]
        flipped[:, :, 5] = -proprioceptive_obs[:, :, 5]
        flipped[:, :, 6] = proprioceptive_obs[:, :, 6]
        flipped[:, :, 7] = -proprioceptive_obs[:, :, 7]
        flipped[:, :, 8] = -proprioceptive_obs[:, :, 8]

        flipped[:, :, 9] = -proprioceptive_obs[:, :, 15]
        flipped[:, :, 10] = proprioceptive_obs[:, :, 16]
        flipped[:, :, 11] = -proprioceptive_obs[:, :, 17]
        flipped[:, :, 12] = -proprioceptive_obs[:, :, 18]
        flipped[:, :, 13] = -proprioceptive_obs[:, :, 19]
        flipped[:, :, 14] = -proprioceptive_obs[:, :, 20]

        flipped[:, :, 15] = -proprioceptive_obs[:, :, 9]
        flipped[:, :, 16] = proprioceptive_obs[:, :, 10]
        flipped[:, :, 17] = -proprioceptive_obs[:, :, 11]
        flipped[:, :, 18] = -proprioceptive_obs[:, :, 12]
        flipped[:, :, 19] = -proprioceptive_obs[:, :, 13]
        flipped[:, :, 20] = -proprioceptive_obs[:, :, 14]

        flipped[:, :, 21] = -proprioceptive_obs[:, :, 27]
        flipped[:, :, 22] = proprioceptive_obs[:, :, 28]
        flipped[:, :, 23] = -proprioceptive_obs[:, :, 29]
        flipped[:, :, 24] = -proprioceptive_obs[:, :, 30]
        flipped[:, :, 25] = -proprioceptive_obs[:, :, 31]
        flipped[:, :, 26] = -proprioceptive_obs[:, :, 32]

        flipped[:, :, 27] = -proprioceptive_obs[:, :, 21]
        flipped[:, :, 28] = proprioceptive_obs[:, :, 22]
        flipped[:, :, 29] = -proprioceptive_obs[:, :, 23]
        flipped[:, :, 30] = -proprioceptive_obs[:, :, 24]
        flipped[:, :, 31] = -proprioceptive_obs[:, :, 25]
        flipped[:, :, 32] = -proprioceptive_obs[:, :, 26]

        flipped[:, :, 33] = -proprioceptive_obs[:, :, 39]
        flipped[:, :, 34] = proprioceptive_obs[:, :, 40]
        flipped[:, :, 35] = -proprioceptive_obs[:, :, 41]
        flipped[:, :, 36] = -proprioceptive_obs[:, :, 42]
        flipped[:, :, 37] = -proprioceptive_obs[:, :, 43]
        flipped[:, :, 38] = -proprioceptive_obs[:, :, 44]

        flipped[:, :, 39] = -proprioceptive_obs[:, :, 33]
        flipped[:, :, 40] = proprioceptive_obs[:, :, 34]
        flipped[:, :, 41] = -proprioceptive_obs[:, :, 35]
        flipped[:, :, 42] = -proprioceptive_obs[:, :, 36]
        flipped[:, :, 43] = -proprioceptive_obs[:, :, 37]
        flipped[:, :, 44] = -proprioceptive_obs[:, :, 38]

        flipped[:, :, 45] = proprioceptive_obs[:, :, 46]
        flipped[:, :, 46] = proprioceptive_obs[:, :, 45]

        return flipped.view(
            -1,
            self.actor_critic.num_one_step_critic_obs
            * self.actor_critic.critic_history_length,
        ).detach()

    def flip_tai5_actions(self, actions: torch.Tensor) -> torch.Tensor:
        flipped_actions = torch.zeros_like(actions)
        flipped_actions[:, 0] = -actions[:, 6]
        flipped_actions[:, 1] = actions[:, 7]
        flipped_actions[:, 2] = -actions[:, 8]
        flipped_actions[:, 3] = -actions[:, 9]
        flipped_actions[:, 4] = -actions[:, 10]
        flipped_actions[:, 5] = -actions[:, 11]
        flipped_actions[:, 6] = -actions[:, 0]
        flipped_actions[:, 7] = actions[:, 1]
        flipped_actions[:, 8] = -actions[:, 2]
        flipped_actions[:, 9] = -actions[:, 3]
        flipped_actions[:, 10] = -actions[:, 4]
        flipped_actions[:, 11] = -actions[:, 5]
        return flipped_actions.detach()

