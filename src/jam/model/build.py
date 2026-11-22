"""Configuration to model."""

from __future__ import annotations

import dataclasses

import torch

from .action_stream import AgentStream
from .coupling import CouplingSpec
from .jam import JointActionWorldModel
from .world_stream import LayerHub, WorldStream, load_world_backbone


def build_model(cfg, *, dtype: torch.dtype = torch.float32) -> JointActionWorldModel:
    backbone = load_world_backbone(cfg.backbone, config=cfg.backbone_config, pretrained=cfg.pretrained, dtype=dtype)
    cfg.backbone_config = {k: v for k, v in backbone.config.items() if not k.startswith("_")}
    world = WorldStream(backbone, LayerHub())
    agent_cfg = dataclasses.replace(cfg.agent, heads=world.heads, head_dim=world.head_dim, ctx_dim=int(backbone.config.text_dim))
    cfg.agent = agent_cfg
    return JointActionWorldModel(
        world,
        AgentStream(agent_cfg).to(dtype),
        CouplingSpec(**dataclasses.asdict(cfg.coupling)),
        gradient_checkpointing=cfg.gradient_checkpointing,
        clean_frames=cfg.clean_frames,
    )


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    out = {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in ("world", "agent")}
    out["total"] = out["world"] + out["agent"]
    return out
