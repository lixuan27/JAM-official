"""JAM: world and action streams coupled at every paired layer by multi-stream joint attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .action_stream import AgentStream, rope_1d
from .coupling import CouplingSpec, TokenBook, build_joint_mask
from .world_stream import LayerHub, WorldStream


@dataclass
class JamOutput:
    v_world: torch.Tensor | None
    v_action: torch.Tensor
    v_consequence: torch.Tensor | None


def pair_layers(n_world: int, n_agent: int) -> dict[int, int]:
    if not 1 <= n_agent <= n_world:
        raise ValueError("action-stream depth must be between one and the world-stream depth")
    if n_agent == n_world:
        return {i: i for i in range(n_world)}
    out = {}
    for j in range(n_agent):
        i = int(round((j + 0.5) * n_world / n_agent - 0.5))
        out[min(max(i, 0), n_world - 1)] = j
    return out


def attend(q, k, v, mask):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


class JointActionWorldModel(nn.Module):
    def __init__(self, world: WorldStream, agent: AgentStream, coupling: CouplingSpec | None = None, *, gradient_checkpointing: bool = False, clean_frames: int = 1):
        super().__init__()
        if agent.cfg.heads != world.heads or agent.cfg.head_dim != world.head_dim:
            raise ValueError("both streams must share one attention-head geometry")
        self.world = world
        self.agent = agent
        self.coupling = coupling or CouplingSpec()
        self.hub: LayerHub = world.hub
        self.gradient_checkpointing = gradient_checkpointing
        self.clean_frames = clean_frames
        self.layer_map = pair_layers(world.num_layers, agent.cfg.layers)

    def token_book(self, world_tokens: int, clean_tokens: int) -> TokenBook:
        return TokenBook(world_tokens, clean_tokens, 1, self.agent.cfg.chunk, self.agent.consequence_steps)

    def joint_mask(self, book: TokenBook, device) -> torch.Tensor:
        return build_joint_mask(self.coupling, book, device=device)

    def _masks(self, world_tokens: int, clean_tokens: int, device) -> dict:
        book = self.token_book(world_tokens, clean_tokens)
        full = self.joint_mask(book, device)
        world_only = full[book.world, book.world][None, None]
        return {
            "world_full": full[book.world, :][None, None],
            "world_only": None if bool(world_only.all()) else world_only,
            "agent_full": full[book.agent, :][None, None],
        }

    def _agent_entry(self, actions, s_action, text, state, state_mask, consequence, s_consequence):
        x_a = self.agent.embed(actions, state, state_mask, consequence)
        temb_a, ada, ctx_a = self.agent.conditioning(s_action, text, s_consequence)
        cos, sin = rope_1d(self.agent.positions(x_a.device), self.agent.cfg.head_dim)
        return x_a, temb_a, ada, ctx_a, cos, sin

    def _agent_layer(self, blk, x_a, qkv, mods, k_w, v_w, mask, ctx_a, text_mask):
        q_a, k_a, v_a = qkv
        gate, c_shift, c_scale, c_gate = mods
        keys = torch.cat([k_w.to(k_a.dtype), k_a], dim=1)
        values = torch.cat([v_w.to(v_a.dtype), v_a], dim=1)
        return blk.finish(x_a, attend(q_a, keys, values, mask), gate, c_shift, c_scale, c_gate, ctx_a, text_mask)

    def forward(self, latents, actions, s_world, s_action, text, text_mask=None, state=None, state_mask=None, consequence=None, s_consequence=None) -> JamOutput:
        hub = self.hub
        hub.reset()
        wst = self.world.begin(latents, s_world, text, clean_frames=self.clean_frames, text_mask=text_mask)
        x_a, temb_a, ada, ctx_a, cos, sin = self._agent_entry(actions, s_action, text, state, state_mask, consequence, s_consequence)
        masks = self._masks(wst["x"].shape[1], wst["clean_tokens"], x_a.device)

        def run_layer(i: int, x_w: torch.Tensor, x_a: torch.Tensor):
            wst["x"] = x_w
            j = self.layer_map.get(i)
            paired = j is not None
            if paired:
                blk = self.agent.blocks[j]
                shift, scale, *mods = blk.modulation(ada)
                qkv = blk.project_qkv(x_a, shift, scale, cos, sin)
                hub.agent_k, hub.agent_v = qkv[1], qkv[2]
            else:
                hub.agent_k = hub.agent_v = None
            hub.world_reads_agent = paired
            hub.world_query_mask = masks["world_full"] if paired else masks["world_only"]
            self.world.layer(i, wst)
            if paired:
                x_a = self._agent_layer(blk, x_a, qkv, mods, hub.world_k, hub.world_v, masks["agent_full"], ctx_a, text_mask)
            return wst["x"], x_a

        x_w = wst["x"]
        for i in range(self.world.num_layers):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x_w, x_a = checkpoint(run_layer, i, x_w, x_a, use_reentrant=False)
            else:
                x_w, x_a = run_layer(i, x_w, x_a)
        wst["x"] = x_w
        v_action, v_consequence = self.agent.readout(x_a, temb_a)
        return JamOutput(self.world.finish(wst), v_action, v_consequence)

    @torch.no_grad()
    def encode_prefix(self, obs_latent, text, text_mask=None) -> dict:
        hub = self.hub
        hub.reset()
        s = torch.ones(obs_latent.shape[0], device=obs_latent.device)
        wst = self.world.begin(obs_latent, s, text, clean_frames=obs_latent.shape[2], text_mask=text_mask)
        kv = {}
        for i in range(self.world.num_layers):
            hub.agent_k = hub.agent_v = None
            hub.world_reads_agent = False
            hub.world_query_mask = None
            self.world.layer(i, wst)
            if i in self.layer_map:
                kv[i] = (hub.world_k, hub.world_v)
        return {"kv": kv, "world_tokens": wst["x"].shape[1], "clean_tokens": wst["clean_tokens"]}

    @torch.no_grad()
    def forward_from_prefix(self, prefix, actions, s_action, text, text_mask=None, state=None, state_mask=None, consequence=None, s_consequence=None) -> JamOutput:
        x_a, temb_a, ada, ctx_a, cos, sin = self._agent_entry(actions, s_action, text, state, state_mask, consequence, s_consequence)
        mask = self._masks(prefix["world_tokens"], prefix["clean_tokens"], x_a.device)["agent_full"]
        for i, j in sorted(self.layer_map.items()):
            blk = self.agent.blocks[j]
            shift, scale, *mods = blk.modulation(ada)
            qkv = blk.project_qkv(x_a, shift, scale, cos, sin)
            x_a = self._agent_layer(blk, x_a, qkv, mods, *prefix["kv"][i], mask, ctx_a, text_mask)
        v_action, v_consequence = self.agent.readout(x_a, temb_a)
        return JamOutput(None, v_action, v_consequence)
