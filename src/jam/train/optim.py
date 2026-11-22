"""Optimizer with separate world / action learning rates and the warmup-stable-decay schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class OptimConfig:
    lr: float = 1e-4
    lr_agent: float | None = None
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 500
    schedule: str = "cosine"
    min_lr_ratio: float = 0.01
    decay_steps: int = 0


def clean_name(name: str) -> str:
    for wrapper in ("_fsdp_wrapped_module.", "_orig_mod.", "_checkpoint_wrapped_module."):
        name = name.replace(wrapper, "")
    return name


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    world, agent = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (agent if clean_name(name).startswith("agent.") else world).append(p)
    groups = [{"params": params, "lr": lr, "name": name} for name, params, lr in (("world", world, cfg.lr), ("agent", agent, cfg.lr_agent or cfg.lr)) if params]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay, fused=torch.cuda.is_available())


def lr_multiplier(step: int, total: int, cfg: OptimConfig) -> float:
    if cfg.schedule not in ("cosine", "constant", "wsd"):
        raise ValueError("schedule must be cosine, constant or wsd")
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)
    if cfg.schedule == "constant":
        return 1.0
    start = max(cfg.warmup_steps, total - cfg.decay_steps) if cfg.schedule == "wsd" else cfg.warmup_steps
    if step < start:
        return 1.0
    progress = min(1.0, (step - start) / max(1, total - start))
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(opt: torch.optim.Optimizer, total_steps: int, cfg: OptimConfig):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_multiplier(s, total_steps, cfg))
