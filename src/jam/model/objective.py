"""Training objective: per-modality flow-matching losses under independent noise levels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .flow import FlowSchedule, JointNoisePlane, masked_mse


@dataclass
class ObjectiveConfig:
    lambda_world: float = 1.0
    lambda_action: float = 1.0
    lambda_consequence: float = 1.0
    world_shift: float = 5.0
    action_shift: float = 5.0
    weighting: str = "bell"
    disable_consequence: bool = False
    consequence_max_abs: float = 8.0
    predictive_consequence_probability: float = 0.0


def present(mask: torch.Tensor | None, batch_size: int, device) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch_size, device=device)
    return mask.flatten(1).any(1).float()


def noise_absent(s: torch.Tensor, has: torch.Tensor) -> torch.Tensor:
    return torch.where(has.to(torch.bool), s, torch.ones_like(s))


def mask_implausible_consequences(c0, mask, max_abs: float):
    if max_abs <= 0:
        return c0, mask
    B, H, KC = c0.shape
    bad = (c0.reshape(B, H, KC // 3, 3)[..., :2].abs() > max_abs).any(-1)
    bad = bad.unsqueeze(-1).expand(-1, -1, -1, 3).reshape(B, H, KC)
    return c0.masked_fill(bad, 0.0), None if mask is None else mask & ~bad


def plan_prediction_slice(draw: dict, batch: dict, cfg: ObjectiveConfig, generator=None):
    n, device = len(draw["s_world"]), draw["s_world"].device
    selected = torch.zeros(n, dtype=torch.bool, device=device)
    c = batch.get("consequence")
    if cfg.predictive_consequence_probability <= 0 or c is None:
        return draw, selected
    mask = batch.get("consequence_mask", torch.ones_like(c, dtype=torch.bool)).bool()
    trip = c.reshape(n, c.shape[1], -1, 3)
    valid = mask.reshape_as(trip)[..., :2].all(-1) & torch.isfinite(trip[..., :2]).all(-1)
    if cfg.consequence_max_abs > 0:
        valid = valid & (trip[..., :2].abs() <= cfg.consequence_max_abs).all(-1)
    eligible = valid.flatten(1).any(1) & present(batch.get("action_mask"), n, device).bool()
    selected = (torch.rand(n, device=device, generator=generator) < cfg.predictive_consequence_probability) & eligible
    if not selected.any():
        return draw, selected
    out = dict(draw)
    for key in ("s_world", "s_consequence", "w_consequence"):
        out[key] = torch.where(selected, torch.ones_like(draw[key]), draw[key])
    for key in ("s_action", "w_action", "w_world"):
        out[key] = torch.where(selected, torch.zeros_like(draw[key]), draw[key])
    return out, selected


class JointFlowObjective:
    def __init__(self, cfg: ObjectiveConfig, clean_frames: int = 1):
        if not 0 <= cfg.predictive_consequence_probability <= 1:
            raise ValueError("predictive_consequence_probability must lie in [0, 1]")
        if cfg.predictive_consequence_probability > 0 and (cfg.lambda_consequence <= 0 or cfg.disable_consequence):
            raise ValueError("the predictive slice requires an enabled Motion Track loss")
        self.cfg = cfg
        self.world_flow = FlowSchedule(shift=cfg.world_shift, weighting=cfg.weighting)
        self.action_flow = FlowSchedule(shift=cfg.action_shift, weighting=cfg.weighting)
        self.plane = JointNoisePlane(self.world_flow, self.action_flow)
        self.clean_frames = clean_frames

    def __call__(self, model, batch: dict, generator: torch.Generator | None = None) -> dict:
        cfg, cf = self.cfg, self.clean_frames
        z0, a0 = batch["latents"], batch["actions"]
        B, dev = z0.shape[0], z0.device
        draw, predictive = plan_prediction_slice(self.plane.draw(B, device=dev, generator=generator), batch, cfg, generator)
        s_w, s_a, s_c = draw["s_world"], draw["s_action"], draw["s_consequence"]

        eps_w = torch.randn(z0.shape, device=dev, dtype=torch.float32, generator=generator)
        z_s = FlowSchedule.noise(z0.float(), eps_w, s_w.view(B, 1, 1, 1, 1))
        z_s[:, :, :cf] = z0[:, :, :cf].float()

        action_mask = batch.get("action_mask")
        has_action = present(action_mask, B, dev)
        s_a = noise_absent(s_a, has_action)
        eps_a = torch.randn(a0.shape, device=dev, dtype=torch.float32, generator=generator)
        a_s = FlowSchedule.noise(a0.float(), eps_a, s_a.view(B, 1, 1))

        c0, c_s, consequence_mask = batch.get("consequence"), None, batch.get("consequence_mask")
        if c0 is not None:
            if cfg.disable_consequence:
                c0, consequence_mask = torch.zeros_like(c0), torch.zeros_like(c0, dtype=torch.bool)
            c0, consequence_mask = mask_implausible_consequences(c0, consequence_mask, cfg.consequence_max_abs)
            has_c = present(consequence_mask, B, dev)
            s_c = noise_absent(s_c, has_c)
            eps_c = torch.randn(c0.shape, device=dev, dtype=torch.float32, generator=generator)
            c_s = FlowSchedule.noise(c0.float(), eps_c, s_c.view(B, 1, 1))

        out = model(
            latents=z_s.to(z0.dtype),
            actions=a_s.to(a0.dtype),
            s_world=s_w,
            s_action=s_a,
            text=batch["text"],
            text_mask=batch.get("text_mask"),
            state=batch.get("state"),
            state_mask=batch.get("state_mask"),
            consequence=None if c_s is None else c_s.to(a0.dtype),
            s_consequence=s_c,
        )

        per_frame = masked_mse(out.v_world, FlowSchedule.target(z0.float(), eps_w), None, reduce_dims=(1, 3, 4))
        frame_valid = batch.get("latent_mask")
        frame_valid = torch.ones_like(per_frame, dtype=torch.bool) if frame_valid is None else frame_valid.clone()
        frame_valid[:, :cf] = False
        world_per_sample = (per_frame * frame_valid).sum(1) / frame_valid.sum(1).clamp_min(1)
        loss_w = (world_per_sample * draw["w_world"]).mean()

        action_per_sample = masked_mse(out.v_action, FlowSchedule.target(a0.float(), eps_a), action_mask, reduce_dims=(1, 2))
        loss_a = (action_per_sample * draw["w_action"] * has_action).sum() / has_action.sum().clamp_min(1.0)

        total = cfg.lambda_world * loss_w + cfg.lambda_action * loss_a
        logs = {"loss/world": loss_w.detach(), "loss/action": loss_a.detach(), "frac/action_supervised": has_action.mean()}
        per_sample = {"world": world_per_sample.detach(), "action": action_per_sample.detach(), "has_action": has_action}

        if c0 is not None and out.v_consequence is not None:
            c_per_sample = masked_mse(out.v_consequence, FlowSchedule.target(c0.float(), eps_c), consequence_mask, reduce_dims=(1, 2))
            loss_c = (c_per_sample * draw["w_consequence"] * has_c).sum() / has_c.sum().clamp_min(1.0)
            total = total + cfg.lambda_consequence * loss_c
            logs["loss/consequence"] = loss_c.detach()
            logs["frac/consequence_supervised"] = has_c.mean()
            logs["frac/consequence_predictive"] = predictive.float().mean()
            per_sample.update(consequence=c_per_sample.detach(), has_consequence=has_c)

        logs["loss/total"] = total.detach()
        return {"loss": total, "logs": logs, "s_world": s_w, "s_action": s_a, "s_consequence": s_c, "per_sample": per_sample}
