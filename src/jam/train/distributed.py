"""Process-group setup and FSDP wrapping."""

from __future__ import annotations

import functools
import os

import torch
import torch.distributed as dist


def initialize() -> tuple[torch.device, int, int]:
    if "RANK" in os.environ and not dist.is_initialized():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
        return device, dist.get_rank(), dist.get_world_size()
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 0, 1


def wrap_model(model, device, *, bf16: bool = True, strategy: str = "full"):
    if not dist.is_initialized():
        return model.to(device)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    world = dist.get_world_size()
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    if strategy == "hybrid" and world > per_node and world % per_node == 0:
        mesh = init_device_mesh(device.type, (world // per_node, per_node), mesh_dim_names=("replicate", "shard"))
        sharding = ShardingStrategy.HYBRID_SHARD
    elif strategy in ("full", "hybrid", "grad"):
        mesh = init_device_mesh(device.type, (world,))
        sharding = ShardingStrategy.SHARD_GRAD_OP if strategy == "grad" else ShardingStrategy.FULL_SHARD
    else:
        raise ValueError(f"unknown FSDP strategy {strategy!r}")
    block_types = {type(block) for block in model.world.net.blocks}
    precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16) if bf16 else None
    return FSDP(
        model,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=block_types),
        sharding_strategy=sharding,
        mixed_precision=precision,
        device_id=device,
        device_mesh=mesh,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.AVG)
    return x
