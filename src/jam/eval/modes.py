"""Held-out comparison of inference modes of one checkpoint on prepared batches."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ..infer.sampler import JointSampler, SamplerConfig
from ..train.batch import PreparedBatches
from .common import autocast, bootstrap, default_device, load_model, masked_error, sampler_inputs, write_report

DEFAULT_MODES = ("joint", "action_only", "future_free", "imagine_then_act", "inverse_dynamics")
SUFFIX = "_no_tracks"


def split_mode(name: str) -> tuple[str, str]:
    return (name[: -len(SUFFIX)], "absent") if name.endswith(SUFFIX) else (name, "denoise")


@torch.no_grad()
def compare_modes(model, cfg, batches, modes=DEFAULT_MODES, *, steps: int = 10, seed: int = 0, device=None) -> tuple[dict, dict]:
    device = device or next(model.parameters()).device
    cf = cfg.model.clean_frames
    make = lambda mode: JointSampler(SamplerConfig(steps=steps, mode=mode, world_shift=cfg.objective.world_shift, action_shift=cfg.objective.action_shift, clean_frames=cf))
    samplers = {name: make(split_mode(name)[0]) for name in modes}
    acc: dict[str, list] = {name: [] for name in modes}
    acc["has_action"], acc["copy_state"] = [], []
    for bi, b in enumerate(batches):
        b = {k: v.to(device) for k, v in b.items()}
        am, common = b["action_mask"], sampler_inputs(b, cf)
        for name in modes:
            mode, tracks = split_mode(name)
            extra = {"future": b["latents"].float()} if mode == "inverse_dynamics" else {}
            generator = torch.Generator(device=device).manual_seed((seed * 1000003 + bi) * 7 + 1)
            with autocast(device):
                out = samplers[name].sample(model, consequence_mode=tracks, generator=generator, **extra, **common)
            acc[name].append(masked_error(out["actions"].float(), b["actions"].float(), am))
        acc["has_action"].append(am.flatten(1).any(1).float().cpu().numpy())
        state = b.get("state")
        if state is not None:
            acc["copy_state"].append(masked_error(state.float()[:, None, :].expand_as(b["actions"]), b["actions"].float(), am))
    A = {k: np.concatenate(v) for k, v in acc.items() if v}
    ha, ref = A["has_action"], modes[0]
    summary = {"windows": int(len(ha)), "windows_with_actions": int(ha.sum())}
    for name in [*modes, "copy_state"]:
        if name in A:
            summary[name] = bootstrap(A[name], ha)
            if name not in (ref, "copy_state"):
                summary[f"{name}_minus_{ref}"] = bootstrap(A[name] - A[ref], ha)
    return summary, A


@torch.no_grad()
def latency(model, cfg, batch, modes, *, steps: int, reps: int, device) -> dict:
    one = {k: v[:1].to(device) for k, v in batch.items()}
    common = sampler_inputs(one, cfg.model.clean_frames)
    out = {}
    for name in modes:
        mode, tracks = split_mode(name)
        sampler = JointSampler(SamplerConfig(steps=steps, mode=mode, world_shift=cfg.objective.world_shift, action_shift=cfg.objective.action_shift, clean_frames=cfg.model.clean_frames))
        extra = {"future": one["latents"].float()} if mode == "inverse_dynamics" else {}
        call = lambda: sampler.sample(model, consequence_mode=tracks, **extra, **common)
        with autocast(device):
            for _ in range(2):
                call()
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.time()
            for _ in range(reps):
                call()
            if device.type == "cuda":
                torch.cuda.synchronize()
        out[name] = (time.time() - started) / reps
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare inference modes on prepared held-out batches.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latency-reps", type=int, default=0)
    args = parser.parse_args()
    device = default_device()
    model, cfg = load_model(args.checkpoint, device)
    batches = PreparedBatches(args.batches, model.agent.cfg)
    modes = tuple(m for m in args.modes.split(",") if m)
    summary, arrays = compare_modes(model, cfg, batches, modes, steps=args.steps, seed=args.seed, device=device)
    report = {"kind": "modes", "seed": args.seed, "denoise_steps": args.steps, "modes": list(modes), "metrics": summary}
    if args.latency_reps > 0:
        report["latency_seconds"] = latency(model, cfg, batches[0], modes, steps=args.steps, reps=args.latency_reps, device=device)
    write_report(args.out, report, arrays)
    print({k: (v or {}).get("mean") if isinstance(v, dict) or v is None else v for k, v in summary.items()}, flush=True)


if __name__ == "__main__":
    main()
