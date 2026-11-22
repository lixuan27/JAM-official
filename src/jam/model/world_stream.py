"""World stream: a pretrained video diffusion transformer whose self-attention is opened to the action stream."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LayerHub:
    agent_k: torch.Tensor | None = None
    agent_v: torch.Tensor | None = None
    world_k: torch.Tensor | None = None
    world_v: torch.Tensor | None = None
    world_query_mask: torch.Tensor | None = None
    world_reads_agent: bool = False
    text_mask: torch.Tensor | None = None

    def reset(self) -> None:
        self.agent_k = self.agent_v = None
        self.world_k = self.world_v = None
        self.world_query_mask = None
        self.world_reads_agent = False
        self.text_mask = None


def rotate(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


class HubAttnProcessor:

    def __init__(self, hub: LayerHub):
        self.hub = hub

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        if encoder_hidden_states is not None:
            raise RuntimeError("HubAttnProcessor is a self-attention processor")
        q = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        k = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        v = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            q = rotate(q, *rotary_emb)
            k = rotate(k, *rotary_emb)
        hub = self.hub
        hub.world_k, hub.world_v = k, v
        keys, values = k, v
        if hub.world_reads_agent and hub.agent_k is not None:
            keys = torch.cat([k, hub.agent_k.to(k.dtype)], dim=1)
            values = torch.cat([v, hub.agent_v.to(v.dtype)], dim=1)
        mask = hub.world_query_mask
        if mask is not None and mask.shape[-1] != keys.shape[1]:
            mask = mask[..., : keys.shape[1]]
        out = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=mask)
        out = out.transpose(1, 2).flatten(2, 3).type_as(q)
        return attn.to_out[1](attn.to_out[0](out))


class MaskedContextProcessor:
    def __init__(self, processor, hub: LayerHub):
        self.processor = processor
        self.hub = hub

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        mask = self.hub.text_mask
        if mask is None:
            mask = attention_mask
        elif attn.add_k_proj is not None:
            raise ValueError("text masking requires a text-only context")
        elif attention_mask is not None:
            mask = mask & attention_mask
        return self.processor(attn, hidden_states, encoder_hidden_states, mask, rotary_emb)


class WorldStream(nn.Module):
    def __init__(self, transformer: nn.Module, hub: LayerHub):
        super().__init__()
        if transformer.config.patch_size[0] != 1:
            raise ValueError("the clean observation prefix requires temporal patch size one")
        self.net = transformer
        self.hub = hub
        for block in self.net.blocks:
            block.attn1.set_processor(HubAttnProcessor(hub))
            block.attn2.set_processor(MaskedContextProcessor(block.attn2.processor, hub))

    @property
    def num_layers(self) -> int:
        return len(self.net.blocks)

    @property
    def heads(self) -> int:
        return int(self.net.config.num_attention_heads)

    @property
    def head_dim(self) -> int:
        return int(self.net.config.attention_head_dim)

    @property
    def width(self) -> int:
        return self.heads * self.head_dim

    def token_grid(self, latents: torch.Tensor) -> tuple[int, int, int]:
        _, _, T, H, W = latents.shape
        pt, ph, pw = self.net.config.patch_size
        return T // pt, H // ph, W // pw

    def begin(self, latents, s_world, text, clean_frames: int = 1, text_mask=None) -> dict:
        net = self.net
        if text_mask is not None:
            if text_mask.dtype != torch.bool or text_mask.shape != text.shape[:2]:
                raise ValueError("text_mask must be boolean (batch, text_tokens)")
            has_text = text_mask.any(dim=1).all()
            message = "each sample must contain at least one valid text token"
            if text_mask.device.type == "cpu":
                if not bool(has_text):
                    raise ValueError(message)
            else:
                torch._assert_async(has_text, message)
            text_mask = text_mask[:, None, None, :].to(text.device)
        B = latents.shape[0]
        T, h, w = self.token_grid(latents)
        latents = latents.to(net.patch_embedding.weight.dtype)
        text = text.to(net.condition_embedder.text_embedder.linear_1.weight.dtype)
        rotary = net.rope(latents)
        x = net.patch_embedding(latents).flatten(2).transpose(1, 2)
        per_frame = h * w
        t_tok = (s_world.float() * 1000.0)[:, None].expand(B, T).clone()
        t_tok[:, :clean_frames] = 0.0
        t_tok = t_tok.repeat_interleave(per_frame, dim=1)
        temb, tproj, ctx, _ = net.condition_embedder(t_tok.flatten(), text, None, timestep_seq_len=t_tok.shape[1])
        return {
            "x": x,
            "rotary": rotary,
            "temb": temb,
            "tproj": tproj.unflatten(2, (6, -1)),
            "ctx": ctx,
            "text_mask": text_mask,
            "grid": (T, h, w),
            "clean_tokens": clean_frames * per_frame,
        }

    def layer(self, i: int, st: dict) -> None:
        self.hub.text_mask = st["text_mask"]
        st["x"] = self.net.blocks[i](st["x"], st["ctx"], st["tproj"], st["rotary"])

    def finish(self, st: dict) -> torch.Tensor:
        net = self.net
        x, temb = st["x"], st["temb"]
        T, h, w = st["grid"]
        pt, ph, pw = net.config.patch_size
        shift, scale = (net.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        x = (net.norm_out(x.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)).type_as(x)
        x = net.proj_out(x)
        x = x.reshape(x.shape[0], T, h, w, pt, ph, pw, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def load_world_backbone(path: str | None = None, *, config: dict | None = None, pretrained: bool = True, dtype=torch.float32):
    from diffusers import WanTransformer3DModel

    if path is None:
        if config is None:
            raise ValueError("a backbone directory or a backbone configuration is required")
        model = WanTransformer3DModel(**{k: v for k, v in config.items() if not k.startswith("_")})
        return model if dtype == torch.float32 else model.to(dtype)
    if pretrained:
        return WanTransformer3DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dtype, local_files_only=True)
    loaded = WanTransformer3DModel.load_config(path, subfolder="transformer", local_files_only=True)
    model = WanTransformer3DModel.from_config(loaded)
    return model if dtype == torch.float32 else model.to(dtype)
