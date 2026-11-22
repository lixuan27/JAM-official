"""Shared helpers for offline evaluation on prepared held-out batches."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from ..config import from_dict
from ..model.build import build_model
from ..train.checkpoint import load_weights, read_release


def load_model(checkpoint, device, dtype=None):
    state, raw = read_release(checkpoint)
    cfg = from_dict(raw)
    dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    model = build_model(cfg.model, dtype=dtype)
    load_weights(state, model)
    return model.to(device).eval(), cfg


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def masked_error(pred, true, mask) -> np.ndarray:
    m = mask.to(pred.dtype)
    error = ((pred - true) ** 2 * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp_min(1)
    return error.float().cpu().numpy()


def bootstrap(x, weight=None, n: int = 2000, seed: int = 0) -> dict | None:
    x = np.asarray(x, np.float64)
    w = np.ones_like(x) if weight is None else np.asarray(weight, np.float64)
    x, w = x[w > 0], w[w > 0]
    if len(x) == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    means = (x[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return {"mean": float((x * w).sum() / w.sum()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], "n": int(len(x))}


def sampler_inputs(batch: dict, clean_frames: int) -> dict:
    z0 = batch["latents"].float()
    return dict(
        obs_latent=z0[:, :, :clean_frames],
        latent_shape=tuple(z0.shape[1:]),
        text=batch["text"],
        text_mask=batch.get("text_mask"),
        state=batch.get("state"),
        state_mask=batch.get("state_mask"),
        action_shape=tuple(batch["actions"].shape[1:]),
    )


def write_report(out, report: dict, arrays: dict) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    tmp = out.with_suffix(".partial")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, out)
