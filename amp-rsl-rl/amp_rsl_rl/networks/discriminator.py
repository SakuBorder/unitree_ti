# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import autograd, Tensor
from rsl_rl.utils import utils  # 保留（若未使用可删除）


class Discriminator(nn.Module):
    """Discriminator implements the discriminator network for the AMP algorithm.

    This network is trained to distinguish between expert and policy-generated data.
    It also provides reward signals for the policy through adversarial learning.

    Args:
        input_dim (int): Dimension of the concatenated input state (state + next state).
        hidden_layer_sizes (List[int]): List of hidden layer sizes.
        reward_scale (float): Scale factor for the computed reward.
        reward_clamp_epsilon (float): Epsilon to avoid log(0) in reward shaping.
        device (str): Device to run the model on ('cpu' or 'cuda').
        loss_type (str): 'BCEWithLogits' or 'Wasserstein'.
        eta_wgan (float): Scale used for tanh in Wasserstein-like formulation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: List[int],
        reward_scale: float,
        reward_clamp_epsilon: float = 1e-4,
        device: str = "cpu",
        loss_type: str = "BCEWithLogits",
        eta_wgan: float = 0.3,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.reward_scale = float(reward_scale)
        self.reward_clamp_epsilon = float(reward_clamp_epsilon)

        # trunk MLP
        layers: List[nn.Module] = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # output head
        self.linear = nn.Linear(curr_in_dim, 1)

        # loss config
        self.loss_type = loss_type if loss_type is not None else "BCEWithLogits"
        if self.loss_type == "BCEWithLogits":
            self.loss_fun: Optional[nn.Module] = nn.BCEWithLogitsLoss()
            self.eta_wgan: Optional[float] = None
        elif self.loss_type == "Wasserstein":
            self.loss_fun = None
            self.eta_wgan = float(eta_wgan)
            print("[Discriminator] The Wasserstein-like loss is experimental.")
        else:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}. "
                f"Supported types are 'BCEWithLogits' and 'Wasserstein'."
            )

        self.to(self.device)
        self.train()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the discriminator.

        Args:
            x: (B, input_dim)

        Returns:
            logits: (B, 1)
        """
        h = self.trunk(x)
        d = self.linear(h)
        return d

    def predict_reward(
        self,
        state: Tensor,
        next_state: Tensor,
        normalizer=None,
    ) -> Tensor:
        """Predicts reward from discriminator output (no grad).

        For BCEWithLogits:
            r = -log( 1 - sigmoid(logit) ), with clamp for numerical stability.
        For Wasserstein-like:
            r = exp( tanh(eta * logit) ) * reward_scale.

        Args:
            state: (B, S)
            next_state: (B, S)
            normalizer: Optional state normalizer with .normalize(tensor) method.

        Returns:
            reward: (B,)
        """
        with torch.no_grad():
            if normalizer is not None:
                state = normalizer.normalize(state)
                next_state = normalizer.normalize(next_state)

            logits = self.forward(torch.cat([state, next_state], dim=-1))  # (B, 1)

            if self.loss_type == "Wasserstein":
                logits = torch.tanh(self.eta_wgan * logits)
                reward = self.reward_scale * torch.exp(logits).squeeze(-1)
                return reward

            # BCEWithLogits variant
            prob = torch.sigmoid(logits)  # (B, 1)
            # -log( clamp(1 - p, eps) )
            reward = -torch.log((1.0 - prob).clamp_min(self.reward_clamp_epsilon))
            reward = (self.reward_scale * reward).squeeze(-1)
            return reward

    def policy_loss(self, discriminator_output: Tensor) -> Tensor:
        """Loss for policy-generated samples (label=0) under BCEWithLogits."""
        if self.loss_fun is None:
            raise RuntimeError("policy_loss called while loss_fun=None (Wasserstein mode).")
        expected = torch.zeros_like(discriminator_output, device=discriminator_output.device)
        return self.loss_fun(discriminator_output, expected)

    def expert_loss(self, discriminator_output: Tensor) -> Tensor:
        """Loss for expert samples (label=1) under BCEWithLogits."""
        if self.loss_fun is None:
            raise RuntimeError("expert_loss called while loss_fun=None (Wasserstein mode).")
        expected = torch.ones_like(discriminator_output, device=discriminator_output.device)
        return self.loss_fun(discriminator_output, expected)

    def compute_loss(
        self,
        policy_d: Tensor,
        expert_d: Tensor,
        sample_amp_expert: Tuple[Tensor, Tensor],
        sample_amp_policy: Tuple[Tensor, Tensor],
        lambda_: float = 10.0,
    ):
        """Compute discriminator loss + gradient penalty.

        Args:
            policy_d: logits for policy transitions, (B, 1)
            expert_d: logits for expert transitions, (B, 1)
            sample_amp_expert: (expert_state, expert_next_state)
            sample_amp_policy: (policy_state, policy_next_state)
            lambda_: grad-penalty coefficient

        Returns:
            amp_loss, grad_pen_loss
        """
        grad_pen_loss = self.compute_grad_pen(
            expert_states=sample_amp_expert,
            policy_states=sample_amp_policy,
            lambda_=lambda_,
        )

        if self.loss_type == "BCEWithLogits":
            if self.loss_fun is None:
                raise RuntimeError("BCEWithLogits path requires loss_fun not None.")
            expert_loss = self.loss_fun(expert_d, torch.ones_like(expert_d))
            policy_loss = self.loss_fun(policy_d, torch.zeros_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
        else:  # Wasserstein-like
            amp_loss = self.wgan_loss(policy_d=policy_d, expert_d=expert_d)

        return amp_loss, grad_pen_loss

    def compute_grad_pen(
        self,
        expert_states: Tuple[Tensor, Tensor],
        policy_states: Tuple[Tensor, Tensor],
        lambda_: float = 10.0,
    ) -> Tensor:
        """Gradient penalty regularizer.

        BCEWithLogits: use expert only, target grad-norm≈0 (offset=0).
        Wasserstein-like: use random interpolation of expert & policy, target grad-norm≈1.

        Args:
            expert_states: (state_e, next_state_e)
            policy_states: (state_p, next_state_p)
            lambda_: coefficient

        Returns:
            grad_pen: scalar tensor
        """
        expert = torch.cat(expert_states, dim=-1)  # (B, 2S)

        if self.loss_type == "Wasserstein":
            policy = torch.cat(policy_states, dim=-1)  # (B, 2S)
            alpha = torch.rand(expert.size(0), 1, device=expert.device)
            data = alpha * expert + (1.0 - alpha) * policy
            offset = 1.0
        elif self.loss_type == "BCEWithLogits":
            data = expert
            offset = 0.0
        else:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}. "
                f"Supported types are 'BCEWithLogits' and 'Wasserstein'."
            )

        data = data.requires_grad_(True)
        scores = self.forward(data)  # (B, 1)
        grad = autograd.grad(
            outputs=scores,
            inputs=data,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]  # (B, 2S)

        # (||grad||_2 - offset)^2
        return lambda_ * (grad.norm(2, dim=1) - offset).pow(2).mean()

    def wgan_loss(self, policy_d: Tensor, expert_d: Tensor) -> Tensor:
        """Wasserstein-like loss with tanh stabilization."""
        policy_d = torch.tanh(self.eta_wgan * policy_d)
        expert_d = torch.tanh(self.eta_wgan * expert_d)
        return policy_d.mean() - expert_d.mean()
