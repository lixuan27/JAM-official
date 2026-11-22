"""Paired bootstrap comparison of two evaluation outputs (jam.eval.offline or jam.eval.modes)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OFFLINE = [
    ("policy_action_error", "policy_action", True),
    ("val_action", "val_action", True),
    ("inverse_dynamics_error", "inverse_action", True),
    ("val_world", "val_world", False),
    ("prediction_error", "prediction", False),
    ("follow_gap", ("follow_swapped", "follow_true"), True),
    ("copy_state_action_error", "copy_state_action", True),
]


def values(arrays, key) -> np.ndarray:
    if isinstance(key, tuple):
        return arrays[key[0]].astype(np.float64) - arrays[key[1]].astype(np.float64)
    return arrays[key].astype(np.float64)


def paired_bootstrap(xa, xb, n: int = 4000, seed: int = 0) -> dict:
    xa, xb = np.asarray(xa, np.float64), np.asarray(xb, np.float64)
    if xa.shape != xb.shape or xa.ndim != 1 or len(xa) == 0:
        raise ValueError("paired samples must be non-empty vectors of equal length")
    idx = np.random.default_rng(seed).integers(0, len(xa), (n, len(xa)))
    diff = xa[idx].mean(1) - xb[idx].mean(1)
    return {"A": float(xa.mean()), "B": float(xb.mean()), "diff": float(xa.mean() - xb.mean()),
            "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))], "n": int(len(xa))}


def compare(path_a, path_b, n: int = 4000) -> dict:
    ja, jb = json.loads(Path(path_a).read_text()), json.loads(Path(path_b).read_text())
    za, zb = np.load(Path(path_a).with_suffix(".npz")), np.load(Path(path_b).with_suffix(".npz"))
    if any(ja[k] != jb[k] for k in ("kind", "seed", "denoise_steps")):
        raise ValueError("the two outputs were produced with different evaluation settings")
    if ja["kind"] == "offline":
        metrics = OFFLINE
    else:
        metrics = [(m, m, True) for m in [*ja["modes"], "copy_state"] if m in jb["modes"] or m == "copy_state"]
    out = {}
    for name, key, action_only in metrics:
        keys = key if isinstance(key, tuple) else (key,)
        if any(k not in z.files for k in keys for z in (za, zb)):
            continue
        xa, xb = values(za, key), values(zb, key)
        wa = za["has_action"] if action_only else np.ones_like(xa)
        wb = zb["has_action"] if action_only else np.ones_like(xb)
        if len(xa) != len(xb) or not np.array_equal(wa, wb):
            raise ValueError(f"unpaired windows for {name}")
        keep = wa > 0
        if keep.any():
            out[name] = paired_bootstrap(xa[keep], xb[keep], n)
    return out


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison of two evaluation outputs.")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    out = compare(args.a, args.b, args.n)
    print("| metric | A | B | A-B | 95% CI | n |\n|---|---|---|---|---|---|")
    for name, r in out.items():
        print(f"| {name} | {r['A']:.5f} | {r['B']:.5f} | {r['diff']:+.5f} | [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] | {r['n']} |")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
