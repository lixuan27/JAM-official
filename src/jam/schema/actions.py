"""Typed action layout: named slots of the shared 80-dimensional action / state vector and embodiment bindings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Slot:
    name: str
    size: int
    kind: str = "generic"


@dataclass(frozen=True)
class ActionLayout:
    name: str
    slots: tuple[Slot, ...]

    @property
    def dim(self) -> int:
        return sum(s.size for s in self.slots)

    def index(self, slot: str) -> slice:
        offset = 0
        for s in self.slots:
            if s.name == slot:
                return slice(offset, offset + s.size)
            offset += s.size
        raise KeyError(f"layout {self.name!r} has no slot {slot!r}")

    def dims_of_kind(self, kind: str) -> np.ndarray:
        mask = np.zeros(self.dim, dtype=bool)
        for s in self.slots:
            if s.kind == kind:
                mask[self.index(s.name)] = True
        return mask


def _arm(prefix: str) -> list[Slot]:
    return [
        Slot(f"{prefix}.ee_pos", 3, "position"),
        Slot(f"{prefix}.ee_rot6d", 6, "rot6d"),
        Slot(f"{prefix}.gripper", 1, "gripper"),
        Slot(f"{prefix}.hand", 24, "joints"),
    ]


BIMANUAL_80 = ActionLayout("bimanual_80", tuple(_arm("left") + _arm("right") + [Slot("extra", 12, "generic")]))


@dataclass(frozen=True)
class Route:

    raw_start: int
    width: int
    slot: str
    slot_offset: int = 0


@dataclass(frozen=True)
class EmbodimentBinding:
    name: str
    layout: ActionLayout
    raw_dim: int
    routes: tuple[Route, ...]

    def __post_init__(self) -> None:
        raw_cover = np.zeros(self.raw_dim, dtype=np.int32)
        layout_cover = np.zeros(self.layout.dim, dtype=np.int32)
        for r in self.routes:
            sl = self.layout.index(r.slot)
            if r.raw_start < 0 or r.raw_start + r.width > self.raw_dim or r.slot_offset < 0 or r.slot_offset + r.width > sl.stop - sl.start:
                raise ValueError(f"{self.name}: route {r} is out of range")
            raw_cover[r.raw_start : r.raw_start + r.width] += 1
            layout_cover[sl.start + r.slot_offset : sl.start + r.slot_offset + r.width] += 1
        if (raw_cover > 1).any() or (layout_cover > 1).any():
            raise ValueError(f"{self.name}: overlapping routes")

    @property
    def raw_index(self) -> np.ndarray:
        return np.concatenate([np.arange(r.raw_start, r.raw_start + r.width) for r in self.routes])

    @property
    def layout_index(self) -> np.ndarray:
        return np.concatenate([np.arange(r.width) + self.layout.index(r.slot).start + r.slot_offset for r in self.routes])

    def dim_mask(self) -> np.ndarray:
        mask = np.zeros(self.layout.dim, dtype=bool)
        mask[self.layout_index] = True
        return mask

    def scatter(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw)
        if raw.shape[-1] != self.raw_dim:
            raise ValueError(f"{self.name}: expected raw width {self.raw_dim}, got {raw.shape[-1]}")
        out = np.zeros(raw.shape[:-1] + (self.layout.dim,), dtype=raw.dtype)
        out[..., self.layout_index] = raw[..., self.raw_index]
        return out

    def gather(self, unified: np.ndarray) -> np.ndarray:
        unified = np.asarray(unified)
        out = np.zeros(unified.shape[:-1] + (self.raw_dim,), dtype=unified.dtype)
        out[..., self.raw_index] = unified[..., self.layout_index]
        return out


def _pose10(side: str, start: int = 0) -> tuple[Route, ...]:
    return (Route(start, 3, f"{side}.ee_pos"), Route(start + 3, 6, f"{side}.ee_rot6d"), Route(start + 9, 1, f"{side}.gripper"))


BINDINGS = {
    "single_arm_pose10": EmbodimentBinding("single_arm_pose10", BIMANUAL_80, 10, _pose10("left")),
    "bimanual_pose20": EmbodimentBinding("bimanual_pose20", BIMANUAL_80, 20, _pose10("left") + _pose10("right", 10)),
}


def register_binding(binding: EmbodimentBinding) -> EmbodimentBinding:
    BINDINGS[binding.name] = binding
    return binding


@dataclass
class ActionScaler:

    layout: ActionLayout
    lo: np.ndarray
    hi: np.ndarray
    dim_mask: np.ndarray

    def _scale(self, x: np.ndarray, inverse: bool) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).copy()
        gripper = self.layout.dims_of_kind("gripper")
        free = ~(self.layout.dims_of_kind("rot6d") | gripper) & np.asarray(self.dim_mask, dtype=bool)
        span = np.maximum(self.hi - self.lo, 1e-6)
        if inverse:
            x[..., gripper] = (x[..., gripper] + 1.0) / 2.0
            x[..., free] = (x[..., free] + 1.0) / 2.0 * span[free] + self.lo[free]
        else:
            x[..., gripper] = 2.0 * np.clip(x[..., gripper], 0.0, 1.0) - 1.0
            x[..., free] = np.clip(2.0 * (x[..., free] - self.lo[free]) / span[free] - 1.0, -1.0, 1.0)
        return x

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return self._scale(x, inverse=False)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return self._scale(x, inverse=True)
