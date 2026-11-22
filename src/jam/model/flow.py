"""Flow-matching schedule, independent per-modality noise levels and masked errors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlowSchedule:
    shift: float = 5.0
    weighting: str = "bell"
    bell_center: float = 0.5
    bell_sharpness: float = 2.0
    _weight_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.shift <= 0:
            raise ValueError("shift must be > 0")
        if self.weighting not in ("bell", "uniform"):
            raise ValueError("weighting must be bell or uniform")
        u = (torch.arange(20000, dtype=torch.float64) + 0.5) / 20000
        self._weight_norm = float(self._raw_weight(self.warp(u)).mean())

    def warp(self, u: torch.Tensor) -> torch.Tensor:
        return self.shift * u / (1.0 + (self.shift - 1.0) * u)

    def sample(self, n: int, *, device=None, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.warp(torch.rand(n, device=device, generator=generator, dtype=torch.float32))

    def _raw_weight(self, s: torch.Tensor) -> torch.Tensor:
        if self.weighting == "uniform":
            return torch.ones_like(s)
        w = torch.exp(-self.bell_sharpness * (s - self.bell_center) ** 2)
        edge = torch.tensor(max(self.bell_center, 1 - self.bell_center), dtype=s.dtype)
        return w - torch.exp(-self.bell_sharpness * edge**2)

    def weight(self, s: torch.Tensor) -> torch.Tensor:
        return self._raw_weight(s.to(torch.float32)) / self._weight_norm

    @staticmethod
    def noise(x0: torch.Tensor, eps: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return (1.0 - s) * x0 + s * eps

    @staticmethod
    def target(x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return eps - x0

    def levels(self, u: torch.Tensor) -> torch.Tensor:
        s = self.warp(u.to(torch.float64)).float()
        s[-1] = 0.0
        return s


@dataclass
class JointNoisePlane:
    world: FlowSchedule
    action: FlowSchedule

    def draw(self, n: int, *, device=None, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        s_w = self.world.sample(n, device=device, generator=generator)
        s_a = self.action.sample(n, device=device, generator=generator)
        s_c = self.action.sample(n, device=device, generator=generator)
        return {
            "s_world": s_w,
            "s_action": s_a,
            "s_consequence": s_c,
            "w_world": self.world.weight(s_w),
            "w_action": self.action.weight(s_a),
            "w_consequence": self.action.weight(s_c),
        }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, reduce_dims: tuple[int, ...]) -> torch.Tensor:
    err = (pred.float() - target.float()) ** 2
    if mask is None:
        return err.mean(dim=reduce_dims)
    m = mask.to(err.dtype).expand_as(err)
    return (err * m).sum(dim=reduce_dims) / m.sum(dim=reduce_dims).clamp_min(1.0)
