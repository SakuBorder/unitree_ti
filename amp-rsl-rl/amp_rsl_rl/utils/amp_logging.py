from __future__ import annotations
import statistics
from typing import Optional, Dict, Any, List
from torch.utils.tensorboard import SummaryWriter


class AmpLogger:
    def __init__(self, log_dir: Optional[str] = None, flush_secs: int = 10):
        self.writer: Optional[SummaryWriter] = SummaryWriter(log_dir, flush_secs=flush_secs) if log_dir else None

    def add_text_once(self, tag: str, text: str, step: int = 0):
        if self.writer:
            self.writer.add_text(tag, f"<pre>{text}</pre>", step)

    def log_scalars(self, it: int, metrics: Dict[str, float]):
        if not self.writer:
            return
        for k, v in metrics.items():
            self.writer.add_scalar(k, float(v), it)

    def log_episode_stats(self, it: int, ep_infos: List[Dict[str, Any]]):
        if not (self.writer and ep_infos):
            return
        keys = list(ep_infos[0].keys())
        for key in keys:
            vals = []
            for ep in ep_infos:
                if key not in ep:
                    continue
                v = ep[key]
                try:
                    # 统一成 float
                    vals.append(float(v if not hasattr(v, "item") else v.item()))
                except Exception:
                    pass
            if not vals:
                continue
            mean_v = sum(vals) / len(vals)
            tag = key if "/" in key else ("Episode/" + key)
            self.writer.add_scalar(tag, mean_v, it)

    def log_train_buffers(self, it: int, rewbuffer, lenbuffer, style_reward: float, task_reward: float):
        if not self.writer:
            return
        if len(rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), it)
        self.writer.add_scalar("Train/mean_style_reward", style_reward, it)
        self.writer.add_scalar("Train/mean_task_reward", task_reward, it)
