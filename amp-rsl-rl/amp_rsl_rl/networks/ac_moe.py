from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation


def make_act(activation: str) -> nn.Module:
    # 每次调用返回一个新的激活实例，避免模块复用
    return resolve_nn_activation(activation)


class MLPNet(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: str):
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), make_act(activation)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        super().__init__(*layers)


class ActorMoE(nn.Module):
    """
    Mixture-of-Experts actor:  [expert_1(x) … expert_K(x)] · softmax(gate(x))
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int],
        num_experts: int = 4,
        gate_hidden_dims: Optional[List[int]] = None,
        activation: str = "elu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_experts = num_experts
        self.activation = activation

        # experts
        self.experts = nn.ModuleList(
            [MLPNet(obs_dim, hidden_dims, act_dim, activation) for _ in range(num_experts)]
        )

        # gating network
        gate_layers: List[nn.Module] = []
        last_dim = obs_dim
        gate_hidden_dims = gate_hidden_dims or []
        for h in gate_hidden_dims:
            gate_layers += [nn.Linear(last_dim, h), make_act(activation)]
            last_dim = h
        gate_layers.append(nn.Linear(last_dim, num_experts))
        self.gate = nn.Sequential(*gate_layers)
        self.softmax = nn.Softmax(dim=-1)  # keep separate for ONNX clarity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expert_out: [B, A, K]
        expert_out = torch.stack([e(x) for e in self.experts], dim=-1)
        gate_logits = self.gate(x)                   # [B, K]
        weights = self.softmax(gate_logits).unsqueeze(1)  # [B, 1, K]
        return (expert_out * weights).sum(-1)        # [B, A]


class ActorCriticMoE(nn.Module):
    """Actor-critic with Mixture-of-Experts policy."""
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: List[int] = [256, 256, 256],
        critic_hidden_dims: List[int] = [256, 256, 256],
        num_experts: int = 4,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",  # 'scalar' or 'log'
        **kwargs,
    ):
        if kwargs:
            print("ActorCriticMoE.__init__ ignored unexpected arguments:", list(kwargs.keys()))
        super().__init__()

        # Actor (MoE)
        self.actor = ActorMoE(
            obs_dim=num_actor_obs,
            act_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            num_experts=num_experts,
            gate_hidden_dims=actor_hidden_dims[:-1],
            activation=activation,
        )

        # Critic
        self.critic = MLPNet(num_critic_obs, critic_hidden_dims, 1, activation)

        # noise params
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            # 用可训练原始参数 + softplus 保证正数
            self._raw_std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(torch.full((num_actions,), float(init_noise_std))))
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        self.distribution: Optional[Normal] = None
        Normal.set_default_validate_args(False)

        print(f"Actor (MoE) structure:\n{self.actor}")
        print(f"Critic MLP structure:\n{self.critic}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _std_from_params(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            # softplus 确保正数，避免 Normal 报错
            return torch.nn.functional.softplus(self._raw_std).expand_as(mean)
        else:  # "log"
            return torch.exp(self.log_std).expand_as(mean)

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(observations)
        std = self._std_from_params(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        return True
