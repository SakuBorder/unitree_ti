# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Code taken from https://github.com/isaac-sim/IsaacLab/blob/5716d5600a1a0e45345bc01342a70bd81fac7889/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py

from __future__ import annotations

import copy
import os
from typing import Optional, Any

import torch
import torch.nn as nn
from amp_rsl_rl.networks import ActorMoE


def export_policy_as_onnx(
    actor_critic: Any,
    path: str,
    normalizer: Optional[Any] = None,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export policy into an ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        path: The path to the saving directory.
        normalizer: The empirical normalizer module. If None, Identity is used.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic: Any, normalizer: Optional[Any] = None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer(
                "hidden_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            self.register_buffer(
                "cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            )
            self.forward = self.forward_lstm  # type: ignore[assignment]
            self.reset = self.reset_memory     # type: ignore[assignment]
        # copy normalizer if exists
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def forward_lstm(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self) -> None:
        pass

    def reset_memory(self) -> None:
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        target = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(target)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic: Any, normalizer: Optional[Any] = None, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm  # type: ignore[assignment]
        # copy normalizer if exists
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def forward_lstm(self, x_in: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))

    def export(self, path: str, filename: str) -> None:
        self.to("cpu")
        out_path = os.path.join(path, filename)
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            # Dry run
            _ = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                out_path,
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            # Try to infer observation dim
            if isinstance(self.actor, ActorMoE):
                obs_dim = self.actor.obs_dim
            elif hasattr(self.actor, "in_features"):  # single Linear
                obs_dim = self.actor.in_features  # type: ignore[attr-defined]
            elif isinstance(self.actor, nn.Sequential) and len(self.actor) > 0 and hasattr(self.actor[0], "in_features"):
                obs_dim = self.actor[0].in_features  # type: ignore[index]
            else:
                # Fallback
                raise RuntimeError(
                    "Cannot infer observation dimension. Please export with an Actor that exposes obs_dim or Linear[0].in_features."
                )

            obs = torch.zeros(1, obs_dim)
            torch.onnx.export(
                self,
                obs,
                out_path,
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
