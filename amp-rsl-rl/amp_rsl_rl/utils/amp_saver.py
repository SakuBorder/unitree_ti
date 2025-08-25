from __future__ import annotations
import os
import torch
from typing import Optional
from amp_rsl_rl.utils import export_policy_as_onnx


def save_checkpoint(
    path: str,
    *,
    actor_critic,
    optimizer,
    discriminator,
    amp_normalizer,
    iteration: int,
    obs_norm_state_dict: Optional[dict] = None,
    critic_obs_norm_state_dict: Optional[dict] = None,
    save_onnx: bool = False,
):
    saved = {
        "model_state_dict": actor_critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "discriminator_state_dict": discriminator.state_dict(),
        "amp_normalizer": amp_normalizer,
        "iter": iteration,
        "infos": None,
    }
    if obs_norm_state_dict is not None:
        saved["obs_norm_state_dict"] = obs_norm_state_dict
    if critic_obs_norm_state_dict is not None:
        saved["critic_obs_norm_state_dict"] = critic_obs_norm_state_dict

    torch.save(saved, path)

    if save_onnx:
        folder = os.path.dirname(path)
        try:
            iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
        except Exception:
            pass
        export_policy_as_onnx(actor_critic, normalizer=torch.nn.Identity(), path=folder, filename=f"policy_{iteration}.onnx")


def load_checkpoint(
    path: str,
    *,
    actor_critic,
    discriminator,
    amp_normalizer,
    optimizer=None,
    device="cpu",
    empirical_normalization=False,
    obs_normalizer=None,
    critic_obs_normalizer=None,
    weights_only=False,
):
    loaded = torch.load(path, map_location=device, weights_only=weights_only)
    actor_critic.load_state_dict(loaded["model_state_dict"])
    discriminator.load_state_dict(loaded["discriminator_state_dict"])
    amp_normalizer_loaded = loaded.get("amp_normalizer", None)
    if amp_normalizer_loaded is not None:
        amp_normalizer = amp_normalizer_loaded
    if empirical_normalization and obs_normalizer is not None and critic_obs_normalizer is not None:
        obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        critic_obs_normalizer.load_state_dict(loaded["critic_obs_norm_state_dict"])
    if optimizer is not None and loaded.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
    return amp_normalizer, int(loaded.get("iter", 0)), loaded.get("infos", None)
