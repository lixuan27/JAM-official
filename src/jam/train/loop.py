"""Training loop: gradient accumulation, FSDP, per-step reproducible noise and atomic weight checkpoints."""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ..config import JamConfig, release_dict
from ..model.build import build_model, count_parameters
from ..model.objective import JointFlowObjective
from . import checkpoint as ckpt
from .batch import PreparedBatches
from .distributed import initialize, reduce_mean, wrap_model
from .optim import build_optimizer, build_scheduler


def train_step(model, objective, optimizer, batches, *, step: int, grad_accum: int = 1, grad_clip: float = 1.0, seed: int = 0, rank: int = 0, device=None, autocast: bool = False, generator=None):
    device = device or next(model.parameters()).device
    generator = generator or torch.Generator(device=device)
    optimizer.zero_grad(set_to_none=True)
    logs: dict[str, torch.Tensor] = {}
    for micro in range(grad_accum):
        batch = {k: v.to(device, non_blocking=True) for k, v in batches[step * grad_accum + micro].items()}
        generator.manual_seed(seed * 1000003 + rank * 7919 + step * 31 + micro)
        defer = hasattr(model, "no_sync") and micro < grad_accum - 1
        with model.no_sync() if defer else contextlib.nullcontext():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                result = objective(model, batch, generator=generator)
            (result["loss"] / grad_accum).backward()
        for key, value in result["logs"].items():
            logs[key] = logs.get(key, 0.0) + value.float() / grad_accum
    if hasattr(model, "clip_grad_norm_"):
        norm = model.clip_grad_norm_(grad_clip)
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if torch.isfinite(norm):
        optimizer.step()
    else:
        optimizer.zero_grad(set_to_none=True)
    return logs, norm


def train(cfg: JamConfig, batches_dir, output_dir, *, init_from=None) -> None:
    device, rank, world_size = initialize()
    output = Path(output_dir)
    torch.manual_seed(cfg.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_model(cfg.model)
    if cfg.train.freeze_world:
        model.world.requires_grad_(False)
    if init_from:
        state, _ = ckpt.read_release(init_from)
        ckpt.load_weights(state, model)
    counts = count_parameters(model)
    agent_cfg = model.agent.cfg
    model = wrap_model(model, device, bf16=cfg.train.bf16, strategy=cfg.train.fsdp_strategy)
    torch.manual_seed(cfg.train.seed + rank)

    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.train.steps, cfg.optim)
    objective = JointFlowObjective(cfg.objective, clean_frames=cfg.model.clean_frames)
    batches = PreparedBatches(batches_dir, agent_cfg, rank, world_size)

    step = 0
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"parameters": counts, "processes": world_size}), flush=True)

    model.train()
    generator = torch.Generator(device=device)
    milestones = set(cfg.train.milestones) | {cfg.train.steps}
    autocast = cfg.train.bf16 and device.type == "cuda" and not dist.is_initialized()
    started = time.time()
    while step < cfg.train.steps:
        logs, norm = train_step(
            model, objective, optimizer, batches, step=step, grad_accum=cfg.train.grad_accum, grad_clip=cfg.optim.grad_clip,
            seed=cfg.train.seed, rank=rank, device=device, autocast=autocast, generator=generator,
        )
        scheduler.step()
        step += 1
        if step % cfg.train.log_every == 0 or step == cfg.train.steps:
            record = {k: float(reduce_mean(v)) for k, v in logs.items()}
            record.update(step=step, grad_norm=float(norm), lr=optimizer.param_groups[0]["lr"], seconds=time.time() - started)
            if rank == 0:
                with open(output / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        if step % cfg.train.save_every == 0 or step in milestones:
            ckpt.save_release(output, step, model, release_dict(cfg))
            ckpt.prune(output, "release", keep_last=cfg.train.keep_last, milestones=milestones)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
