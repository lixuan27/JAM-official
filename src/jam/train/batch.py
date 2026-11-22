"""Prepared-tensor batches: one .safetensors file per batch (see docs/tensors.md)."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


def validate_batch(batch: dict, agent_cfg) -> dict:
    shapes = {key: tuple(value.shape) for key, value in batch.items()}
    if len(shapes.get("latents", ())) != 5:
        raise ValueError("latents must have shape (B, C, T, H, W)")
    size, _, frames, _, _ = shapes["latents"]
    if size < 1 or frames < 2:
        raise ValueError("a batch needs samples and future latent frames")
    expected = {"actions": (size, agent_cfg.chunk, agent_cfg.action_dim)}
    if "state" in batch:
        expected["state"] = (size, agent_cfg.state_dim)
    if agent_cfg.consequence_dim:
        expected["consequence"] = (size, agent_cfg.consequence_steps or agent_cfg.chunk, agent_cfg.consequence_dim)
    elif "consequence" in batch:
        raise ValueError("Motion Track input requires a configured Motion Track head")
    for key, shape in expected.items():
        if shapes.get(key) != shape:
            raise ValueError(f"{key} must have shape {shape}")
    text = shapes.get("text", ())
    if len(text) != 3 or text[0] != size or text[1] < 1 or text[2] != agent_cfg.ctx_dim:
        raise ValueError("text must have shape (B, L, text_dim)")
    masks = {"latent_mask": (size, frames), "text_mask": text[:2], "action_mask": shapes["actions"]}
    masks.update({key + "_mask": shapes[key] for key in ("state", "consequence") if key in batch})
    for key, shape in masks.items():
        if shapes.get(key) != shape or batch[key].dtype != torch.bool:
            raise ValueError(f"{key} must be boolean with shape {shape}")
    if not batch["latent_mask"][:, 0].all() or not batch["latent_mask"][:, 1:].any(dim=1).all():
        raise ValueError("each sample needs a valid observation and a valid future frame")
    for key, value in batch.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
    return batch


class PreparedBatches:

    def __init__(self, directory, agent_cfg, rank: int = 0, world_size: int = 1):
        files = sorted(Path(directory).glob("*.safetensors"))
        if len(files) < world_size or len(files) % world_size:
            raise ValueError("the prepared batch count must be positive and divisible by the process count")
        self.files = files[rank::world_size]
        self.agent_cfg = agent_cfg

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        return validate_batch(load_file(str(self.files[index % len(self.files)])), self.agent_cfg)

    def __iter__(self):
        return (self[i] for i in range(len(self)))
