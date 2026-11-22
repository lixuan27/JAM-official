"""Observation to action chunk, from a release checkpoint and prepared observation tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..schema.actions import BIMANUAL_80, BINDINGS, ActionScaler
from .sampler import JointSampler, SamplerConfig


@dataclass
class PolicyConfig:
    checkpoint: str
    mode: str = "joint"
    steps: int = 10
    lag: float = 0.5
    seed: int = 0
    future_frames: int = 2
    consequence_mode: str = "denoise"
    binding: str | None = None
    stats: str | None = None
    execute_steps: int | None = None


def load_scalers(path, layout) -> dict:
    raw = json.loads(Path(path).read_text())
    return {key: ActionScaler(layout, np.asarray(v["lo"], np.float32), np.asarray(v["hi"], np.float32), np.asarray(v["dim_mask"], bool))
            for key, v in raw.items() if key in ("action", "state")}


class Policy:
    def __init__(self, pcfg: PolicyConfig, model=None, cfg=None, device=None):
        from ..eval.common import default_device, load_model

        self.pcfg = pcfg
        self.device = device or default_device()
        if model is None:
            model, cfg = load_model(pcfg.checkpoint, self.device)
        self.model, self.cfg = model.to(self.device).eval(), cfg
        self.layout = BIMANUAL_80
        self.binding = BINDINGS[pcfg.binding] if pcfg.binding else None
        scalers = load_scalers(pcfg.stats, self.layout) if pcfg.stats else {}
        self.action_scaler = scalers.get("action")
        self.state_scaler = scalers.get("state", self.action_scaler)
        self.action_dim_mask = self.inactive_action_clean = None
        if self.binding is not None:
            if self.layout.dim != model.agent.cfg.action_dim:
                raise ValueError("the binding layout does not match the action width of the checkpoint")
            self.action_dim_mask = torch.from_numpy(self.binding.dim_mask()).to(self.device)
            padding = self.binding.scatter(np.zeros(self.binding.raw_dim, np.float32))
            if self.action_scaler is not None:
                padding = self.action_scaler.normalize(padding)
            self.inactive_action_clean = torch.from_numpy(padding).to(self.device)
        self.sampler = JointSampler(SamplerConfig(
            steps=pcfg.steps, mode=pcfg.mode, lag=pcfg.lag, world_shift=cfg.objective.world_shift,
            action_shift=cfg.objective.action_shift, clean_frames=cfg.model.clean_frames,
        ))
        self._calls = 0

    def _state(self, state):
        if state is None:
            return None, None
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if self.binding is not None:
            state = self.binding.scatter(state)
            mask = self.binding.dim_mask()[None]
        else:
            mask = np.ones_like(state, dtype=bool)
        if self.state_scaler is not None:
            state = self.state_scaler.normalize(state)
        return torch.from_numpy(state).to(self.device), torch.from_numpy(mask).to(self.device)

    @torch.no_grad()
    def act(self, obs_latent, text, text_mask=None, state=None, future=None) -> dict:
        obs = torch.as_tensor(obs_latent, dtype=torch.float32, device=self.device)[None]
        text = torch.as_tensor(text, dtype=torch.float32, device=self.device)[None]
        if text_mask is None:
            text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=self.device)
        else:
            text_mask = torch.as_tensor(text_mask, dtype=torch.bool, device=self.device)[None]
        state, state_mask = self._state(state)
        if future is not None:
            future = torch.as_tensor(future, dtype=torch.float32, device=self.device)[None]
        _, C, cf, h, w = obs.shape
        agent = self.model.agent.cfg
        generator = torch.Generator(device=self.device).manual_seed(self.pcfg.seed * 100003 + self._calls)
        self._calls += 1
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.sampler.sample(
                self.model,
                obs_latent=obs,
                latent_shape=(C, cf + self.pcfg.future_frames, h, w),
                text=text,
                text_mask=text_mask,
                state=state,
                state_mask=state_mask,
                action_shape=(agent.chunk, agent.action_dim),
                future=future,
                consequence_mode=self.pcfg.consequence_mode,
                action_dim_mask=self.action_dim_mask,
                inactive_action_clean=self.inactive_action_clean,
                generator=generator,
            )
        layout_actions = out["actions"][0].float().cpu().numpy()
        actions = layout_actions if self.action_scaler is None else self.action_scaler.unnormalize(layout_actions)
        if self.binding is not None:
            actions = self.binding.gather(actions)
        if self.pcfg.execute_steps:
            actions = actions[: self.pcfg.execute_steps]
        return {"actions": actions, "actions_layout": layout_actions, "latents": out["latents"][0].float().cpu().numpy()}
