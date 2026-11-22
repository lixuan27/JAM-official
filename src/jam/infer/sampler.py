"""Euler sampling in every inference mode of one trained model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.flow import FlowSchedule

MODES = ("joint", "action_only", "future_free", "inverse_dynamics", "forward_dynamics", "action_first", "world_first", "imagine_then_act")


@dataclass
class SamplerConfig:
    steps: int = 10
    mode: str = "joint"
    world_shift: float = 5.0
    action_shift: float = 5.0
    lag: float = 0.5
    clean_frames: int = 1
    cache_prefix: bool = True

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown inference mode {self.mode!r}; choose from {MODES}")
        if self.steps < 1:
            raise ValueError("steps must be positive")


class JointSampler:
    def __init__(self, cfg: SamplerConfig):
        self.cfg = cfg
        self.world_flow = FlowSchedule(shift=cfg.world_shift)
        self.action_flow = FlowSchedule(shift=cfg.action_shift)

    def levels(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode, lag = self.cfg.mode, self.cfg.lag
        u = torch.linspace(1.0, 0.0, self.cfg.steps + 1, dtype=torch.float64)
        u_lag = torch.clamp(u / (1 - lag), max=1.0) if lag < 1 else torch.ones_like(u)
        u_w = u_lag if mode == "action_first" else u
        u_a = u_lag if mode == "world_first" else u
        s_w, s_a = self.world_flow.levels(u_w), self.action_flow.levels(u_a)
        s_c = s_a.clone()
        if mode in ("action_only", "future_free"):
            s_w = torch.ones_like(s_w)
        elif mode in ("inverse_dynamics", "imagine_then_act"):
            s_w = torch.zeros_like(s_w)
        elif mode == "forward_dynamics":
            s_a = torch.zeros_like(s_a)
        return s_w, s_a, s_c

    @torch.no_grad()
    def sample(
        self,
        model,
        *,
        obs_latent: torch.Tensor,
        latent_shape: tuple[int, int, int, int],
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        action_shape: tuple[int, int] | None = None,
        actions: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
        consequence_mode: str = "denoise",
        action_dim_mask: torch.Tensor | None = None,
        inactive_action_clean: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict:
        mode, cf, steps = self.cfg.mode, self.cfg.clean_frames, self.cfg.steps
        if consequence_mode not in ("denoise", "absent"):
            raise ValueError("consequence_mode must be denoise or absent")
        B, dev = obs_latent.shape[0], obs_latent.device
        C, T, h, w = latent_shape
        if not 1 <= cf < T or tuple(obs_latent.shape) != (B, C, cf, h, w):
            raise ValueError("obs_latent must be (B, C, clean_frames, h, w) and leave at least one future latent frame")
        if mode == "inverse_dynamics" and (future is None or tuple(future.shape) != (B, C, T, h, w)):
            raise ValueError("inverse_dynamics needs the observed future as a full latent window")
        if mode == "forward_dynamics" and actions is None:
            raise ValueError("forward_dynamics needs clean actions")
        A, D = action_shape if action_shape is not None else actions.shape[1:]
        s_w, s_a, s_c = (s.to(dev) for s in self.levels())
        if consequence_mode == "absent":
            s_c = torch.ones_like(s_c)

        z = torch.randn((B, C, T, h, w), device=dev, generator=generator, dtype=torch.float32)
        a = torch.randn((B, A, D), device=dev, generator=generator, dtype=torch.float32)
        c = None
        if model.agent.cfg.consequence_dim > 0:
            c = torch.randn((B, model.agent.consequence_steps, model.agent.cfg.consequence_dim), device=dev, generator=generator, dtype=torch.float32)
        if mode == "inverse_dynamics":
            z = future.clone().float()
        z[:, :, :cf] = obs_latent.float()
        if mode == "future_free":
            z = z[:, :, :cf].contiguous()
        clamp_actions = mode == "forward_dynamics"
        if clamp_actions:
            a = actions.clone().float()

        active = padding = action_noise = None
        if action_dim_mask is None:
            if inactive_action_clean is not None:
                raise ValueError("inactive_action_clean requires action_dim_mask")
        else:
            if action_dim_mask.dtype != torch.bool or tuple(action_dim_mask.shape) not in ((D,), (B, D)):
                raise ValueError("action_dim_mask must be boolean (D,) or (B, D)")
            active = action_dim_mask.to(dev).reshape(-1, 1, D).expand(B, A, D)
            if inactive_action_clean is None:
                padding = torch.zeros_like(a)
            else:
                padding = inactive_action_clean.to(device=dev, dtype=torch.float32)
                if padding.ndim < 3:
                    padding = padding.reshape(-1, 1, D)
                padding = padding.expand(B, A, D)
            if clamp_actions:
                a = torch.where(active, a, padding)
            else:
                action_noise = a.clone()
                a = torch.where(active, a, (1 - s_a[0]) * padding + s_a[0] * action_noise)

        common = dict(text=text, text_mask=text_mask, state=state, state_mask=state_mask)
        if mode == "imagine_then_act":
            one = torch.ones(B, device=dev)
            s_first = self.world_flow.levels(torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)).to(dev)
            for k in range(steps):
                out = model(latents=z, actions=a, s_world=s_first[k].expand(B), s_action=one, consequence=c, s_consequence=one, **common)
                z = z + out.v_world.float() * (s_first[k + 1] - s_first[k])
                z[:, :, :cf] = obs_latent.float()

        prefix = None
        if mode == "future_free" and self.cfg.cache_prefix:
            prefix = model.encode_prefix(z, text, text_mask)
        update_world = mode in ("joint", "forward_dynamics", "action_first", "world_first")
        for k in range(steps):
            levels = dict(s_action=s_a[k].expand(B), s_consequence=s_c[k].expand(B))
            if prefix is not None:
                out = model.forward_from_prefix(prefix, a, consequence=c, **levels, **common)
            else:
                out = model(latents=z, actions=a, s_world=s_w[k].expand(B), consequence=c, **levels, **common)
            if update_world:
                z = z + out.v_world.float() * (s_w[k + 1] - s_w[k])
                z[:, :, :cf] = obs_latent.float()
            if not clamp_actions:
                a = a + out.v_action.float() * (s_a[k + 1] - s_a[k])
                if active is not None:
                    a = torch.where(active, a, (1 - s_a[k + 1]) * padding + s_a[k + 1] * action_noise)
            if c is not None and out.v_consequence is not None and consequence_mode != "absent":
                c = c + out.v_consequence.float() * (s_c[k + 1] - s_c[k])
        return {"actions": a, "latents": z, "consequence": c}
