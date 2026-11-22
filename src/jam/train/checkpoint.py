"""Checkpoints: consolidated release weights with their configuration, written atomically."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

COMPLETE = "COMPLETE"


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def step_dir(root, kind: str, step: int) -> Path:
    return Path(root) / kind / f"step_{step:07d}"


def list_steps(root, kind: str) -> list[int]:
    d = Path(root) / kind
    if not d.exists():
        return []
    return sorted(int(p.name.split("_")[1]) for p in d.iterdir() if p.name.startswith("step_") and (p / COMPLETE).exists())


def _publish(tmp: Path, final: Path) -> None:
    if final.exists():
        shutil.rmtree(final)
    os.rename(tmp, final)
    (final / COMPLETE).write_text("ok")


def _fresh(final: Path) -> Path:
    tmp = final.with_name(".tmp_" + final.name)
    if _rank0():
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
    return tmp


def full_state_dict(model) -> dict | None:
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = model.state_dict()
        return state if _rank0() else None
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_release(root, step: int, model, config: dict, *, dtype: torch.dtype = torch.bfloat16) -> Path:
    state = full_state_dict(model)
    final = step_dir(root, "release", step)
    if _rank0():
        tmp = _fresh(final)
        clean = {k.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", ""): v.to(dtype).contiguous() for k, v in state.items()}
        save_file(clean, str(tmp / "model.safetensors"), metadata={"step": str(step)})
        (tmp / "config.json").write_text(json.dumps(config, indent=1))
        _publish(tmp, final)
    _barrier()
    return final


def read_release(path) -> tuple[dict, dict]:
    path = Path(path)
    if not (path / COMPLETE).is_file():
        raise ValueError("checkpoint is incomplete")
    return load_file(str(path / "model.safetensors")), json.loads((path / "config.json").read_text())


def load_weights(state: dict, model, *, strict: bool = True) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(("freqs_cos", "freqs_sin"))]
    if strict and (missing or unexpected):
        raise RuntimeError(f"weights mismatch: missing={missing[:5]} ({len(missing)}), unexpected={unexpected[:5]} ({len(unexpected)})")


def prune(root, kind: str, keep_last: int, milestones=()) -> None:
    if not _rank0():
        return
    steps = list_steps(root, kind)
    for step in steps[:-keep_last] if keep_last > 0 else steps:
        if step not in milestones:
            shutil.rmtree(step_dir(root, kind, step), ignore_errors=True)
