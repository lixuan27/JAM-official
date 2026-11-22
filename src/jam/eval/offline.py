"""Offline held-out evaluation of one checkpoint on prepared batches."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ..infer.sampler import JointSampler, SamplerConfig
from ..model.objective import JointFlowObjective, ObjectiveConfig
from ..train.batch import PreparedBatches
from .common import autocast, bootstrap, default_device, load_model, masked_error, sampler_inputs, write_report


@torch.no_grad()
def evaluate(model, cfg, batches, *, steps: int = 10, seed: int = 0, val_draws: int = 2, device=None) -> tuple[dict, dict]:
    device = device or next(model.parameters()).device
    cf = cfg.model.clean_frames
    reference = ObjectiveConfig(lambda_world=1.0, lambda_action=1.0, lambda_consequence=1.0, world_shift=cfg.objective.world_shift,
                                action_shift=cfg.objective.action_shift, weighting=cfg.objective.weighting, consequence_max_abs=cfg.objective.consequence_max_abs)
    objective = JointFlowObjective(reference, clean_frames=cf)
    samplers = {mode: JointSampler(SamplerConfig(steps=steps, mode=mode, world_shift=reference.world_shift, action_shift=reference.action_shift, clean_frames=cf))
                for mode in ("joint", "forward_dynamics", "inverse_dynamics")}
    acc: dict[str, list] = {}
    for bi, b in enumerate(batches):
        b = {k: v.to(device) for k, v in b.items()}
        B = b["latents"].shape[0]
        if B < 2:
            continue
        z0, am = b["latents"].float(), b["action_mask"]
        lm = b["latent_mask"].clone()
        lm[:, :cf] = False
        cm = b.get("consequence_mask")
        xy = None
        if cm is not None:
            xy = cm.reshape(B, cm.shape[1], -1, 3).clone()
            xy[..., 2] = False
            xy = xy.reshape_as(cm)
        base = seed * 1000003 + bi
        gen = lambda k: torch.Generator(device=device).manual_seed(base * 7 + k)
        common = sampler_inputs(b, cf)

        def world_error(out):
            e = ((out["latents"] - z0) ** 2).mean(dim=(1, 3, 4))
            return ((e * lm).sum(1) / lm.sum(1).clamp_min(1)).float().cpu().numpy()

        def track_error(out):
            if xy is None or out["consequence"] is None:
                return np.zeros(B, np.float32)
            return masked_error(out["consequence"].float(), b["consequence"].float(), xy)

        with autocast(device):
            val_world = val_action = 0.0
            for k in range(val_draws):
                ps = objective(model, dict(b), generator=torch.Generator(device=device).manual_seed(base * 31 + k))["per_sample"]
                val_world = val_world + ps["world"].float().cpu().numpy() / val_draws
                val_action = val_action + ps["action"].float().cpu().numpy() / val_draws
            policy = samplers["joint"].sample(model, generator=gen(1), **common)
            true = samplers["forward_dynamics"].sample(model, actions=b["actions"], generator=gen(2), **common)
            swap = samplers["forward_dynamics"].sample(model, actions=torch.roll(b["actions"], 1, 0), generator=gen(2), **common)
            inverse = samplers["inverse_dynamics"].sample(model, future=z0, generator=gen(3), **common)
        state = b.get("state")
        copy = masked_error(state.float()[:, None, :].expand_as(b["actions"]), b["actions"].float(), am) if state is not None else np.zeros(B, np.float32)
        values = dict(
            has_action=am.flatten(1).any(1).float().cpu().numpy(),
            has_tracks=xy.flatten(1).any(1).float().cpu().numpy() if xy is not None else np.zeros(B, np.float32),
            val_world=val_world,
            val_action=val_action,
            policy_action=masked_error(policy["actions"].float(), b["actions"].float(), am),
            copy_state_action=copy,
            prediction=world_error(policy),
            follow_true=world_error(true),
            follow_swapped=world_error(swap),
            inverse_action=masked_error(inverse["actions"].float(), b["actions"].float(), am),
            tracks_policy=track_error(policy),
            tracks_follow_true=track_error(true),
            tracks_follow_swapped=track_error(swap),
        )
        for key, value in values.items():
            acc.setdefault(key, []).append(np.asarray(value, np.float32))
    if not acc:
        raise ValueError("offline evaluation needs batches with at least two windows")
    A = {k: np.concatenate(v) for k, v in acc.items()}
    ha, ht = A["has_action"], A["has_tracks"]
    summary = {
        "windows": int(len(ha)),
        "windows_with_actions": int(ha.sum()),
        "windows_with_tracks": int(ht.sum()),
        "val_world": bootstrap(A["val_world"]),
        "val_action": bootstrap(A["val_action"], ha),
        "policy_action_error": bootstrap(A["policy_action"], ha),
        "copy_state_action_error": bootstrap(A["copy_state_action"], ha),
        "prediction_error": bootstrap(A["prediction"]),
        "follow_true_action": bootstrap(A["follow_true"], ha),
        "follow_swapped_action": bootstrap(A["follow_swapped"], ha),
        "follow_gap": bootstrap(A["follow_swapped"] - A["follow_true"], ha),
        "inverse_dynamics_error": bootstrap(A["inverse_action"], ha),
        "track_error_policy": bootstrap(A["tracks_policy"], ht),
        "track_error_true_action": bootstrap(A["tracks_follow_true"], ht * ha),
        "track_gap": bootstrap(A["tracks_follow_swapped"] - A["tracks_follow_true"], ht * ha),
    }
    return summary, A


def main():
    parser = argparse.ArgumentParser(description="Offline held-out evaluation on prepared batches.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-draws", type=int, default=2)
    args = parser.parse_args()
    device = default_device()
    started = time.time()
    model, cfg = load_model(args.checkpoint, device)
    batches = PreparedBatches(args.batches, model.agent.cfg)
    summary, arrays = evaluate(model, cfg, batches, steps=args.steps, seed=args.seed, val_draws=args.val_draws, device=device)
    report = {"kind": "offline", "seed": args.seed, "denoise_steps": args.steps, "coupling": cfg.model.coupling.visibility,
              "seconds": time.time() - started, "metrics": summary}
    write_report(args.out, report, arrays)
    print({k: (v or {}).get("mean") if isinstance(v, dict) or v is None else v for k, v in summary.items()}, flush=True)


if __name__ == "__main__":
    main()
