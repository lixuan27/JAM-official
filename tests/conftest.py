import copy

import pytest
import torch

from jam.config import JamConfig, from_dict
from jam.model.build import build_model

TINY_BACKBONE = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=16,
    in_channels=4,
    out_channels=4,
    text_dim=8,
    freq_dim=16,
    ffn_dim=64,
    num_layers=3,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=128,
)
TINY_AGENT = dict(action_dim=6, state_dim=6, chunk=8, width=16, layers=3, consequence_dim=9, consequence_steps=4, zero_init_heads=False)


def tiny_config(**model) -> JamConfig:
    base = dict(backbone_config=dict(TINY_BACKBONE), agent=dict(TINY_AGENT), gradient_checkpointing=False)
    base.update(copy.deepcopy(model))
    return from_dict({"model": base, "train": {"steps": 2, "log_every": 1, "save_every": 1, "bf16": False}, "optim": {"lr": 1e-3, "warmup_steps": 0, "schedule": "constant"}})


def tiny_model(seed: int = 17, **model):
    torch.manual_seed(seed)
    return build_model(tiny_config(**model).model)


def make_batch(seed: int = 23, size: int = 2) -> dict:
    g = torch.Generator().manual_seed(seed)
    shapes = {"latents": (size, 4, 3, 4, 4), "actions": (size, 8, 6), "state": (size, 6), "text": (size, 5, 8), "consequence": (size, 4, 9)}
    batch = {key: torch.randn(shape, generator=g) for key, shape in shapes.items()}
    batch["action_mask"] = torch.ones(size, 8, 6, dtype=torch.bool)
    batch["state_mask"] = torch.ones(size, 6, dtype=torch.bool)
    batch["consequence_mask"] = torch.ones(size, 4, 9, dtype=torch.bool)
    batch["latent_mask"] = torch.ones(size, 3, dtype=torch.bool)
    batch["text_mask"] = torch.ones(size, 5, dtype=torch.bool)
    batch["text_mask"][:, -1] = False
    return batch


def forward(model, batch, **changes):
    b = {k: v.clone() for k, v in batch.items()}
    b.update(changes)
    n = b["latents"].shape[0]
    return model(
        latents=b["latents"], actions=b["actions"], s_world=torch.full((n,), 0.4), s_action=torch.full((n,), 0.7),
        text=b["text"], text_mask=b["text_mask"], state=b["state"], state_mask=b["state_mask"],
        consequence=b["consequence"], s_consequence=torch.full((n,), 0.5),
    )


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.fixture
def model():
    return tiny_model()


@pytest.fixture
def batch():
    return make_batch()
