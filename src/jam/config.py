"""Configuration: one dataclass tree loaded from YAML files with explicit includes and key=value overrides."""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .infer.sampler import SamplerConfig
from .model.action_stream import AgentStreamConfig
from .model.objective import ObjectiveConfig
from .train.optim import OptimConfig


@dataclass
class CouplingConfig:
    visibility: str = "reciprocal"


@dataclass
class ModelConfig:
    backbone: str | None = None
    backbone_config: dict | None = None
    pretrained: bool = True
    clean_frames: int = 1
    gradient_checkpointing: bool = True
    agent: AgentStreamConfig = field(default_factory=AgentStreamConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)


@dataclass
class TrainConfig:
    steps: int = 10000
    grad_accum: int = 1
    seed: int = 0
    bf16: bool = True
    fsdp_strategy: str = "hybrid"
    freeze_world: bool = False
    log_every: int = 50
    save_every: int = 1000
    keep_last: int = 2
    milestones: list[int] = field(default_factory=list)


@dataclass
class JamConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _read(path: Path, seen: frozenset = frozenset()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular include: {path.name}")
    raw = yaml.safe_load(path.read_text()) or {}
    merged: dict = {}
    for inc in raw.pop("include", []):
        merged = _merge(merged, _read(path.parent / inc, seen | {path}))
    return _merge(merged, raw)


def _is_float(hint) -> bool:
    return hint is float or (typing.get_origin(hint) in (typing.Union, types.UnionType) and float in typing.get_args(hint))


def _build(cls, data: dict):
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            value = _build(hint, value or {})
        elif typing.get_origin(hint) is tuple and value is not None:
            value = tuple(value)
        elif _is_float(hint) and isinstance(value, (int, str)) and not isinstance(value, bool):
            value = float(value)
        kwargs[name] = value
    return cls(**kwargs)


def from_dict(data: dict) -> JamConfig:
    return _build(JamConfig, data)


def to_dict(cfg: JamConfig) -> dict:
    return dataclasses.asdict(cfg)


def release_dict(cfg: JamConfig) -> dict:
    out = to_dict(cfg)
    out["model"].update(backbone=None, pretrained=False)
    return out


def load_config(paths, overrides=()) -> JamConfig:
    merged: dict = {}
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        merged = _merge(merged, _read(Path(path)))
    for item in overrides:
        key, _, value = item.partition("=")
        node = yaml.safe_load(value)
        for part in reversed(key.split(".")):
            node = {part: node}
        merged = _merge(merged, node)
    return from_dict(merged)
