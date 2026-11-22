import copy

import pytest
import torch

from conftest import forward, tiny_config
from jam.model.build import build_model, count_parameters
from jam.model.jam import pair_layers


def test_shapes_and_finite_outputs(model, batch):
    out = forward(model, batch)
    assert out.v_world.shape == batch["latents"].shape
    assert out.v_action.shape == batch["actions"].shape
    assert out.v_consequence.shape == batch["consequence"].shape
    assert all(torch.isfinite(t).all() for t in (out.v_world, out.v_action, out.v_consequence))


def test_streams_keep_separate_widths(model):
    assert model.world.width == 32 and model.agent.cfg.width == 16
    assert model.agent.cfg.heads == model.world.heads and model.agent.cfg.head_dim == model.world.head_dim
    counts = count_parameters(model)
    assert counts["total"] == sum(p.numel() for p in model.parameters())


def test_layer_pairing():
    assert pair_layers(4, 4) == {0: 0, 1: 1, 2: 2, 3: 3}
    assert pair_layers(4, 2) == {0: 0, 2: 1}
    with pytest.raises(ValueError):
        pair_layers(2, 3)


def test_clean_observation_is_isolated(model, batch):
    model.eval()
    with torch.no_grad():
        ref = forward(model, batch)
        future = batch["latents"].clone()
        future[:, :, 1:] += 3
        out = forward(model, batch, latents=future, actions=batch["actions"] + 2, consequence=batch["consequence"] - 1)
    torch.testing.assert_close(ref.v_world[:, :, :1], out.v_world[:, :, :1])
    assert not torch.allclose(ref.v_action, out.v_action)


def test_text_conditions_both_streams_and_padding_is_ignored(model, batch):
    model.eval()
    with torch.no_grad():
        ref = forward(model, batch)
        text = batch["text"].clone()
        text[:, -1] = 1000
        padded = forward(model, batch, text=text)
        text = batch["text"].clone()
        text[:, 0] += 1
        changed = forward(model, batch, text=text)
    for name in ("v_world", "v_action", "v_consequence"):
        torch.testing.assert_close(getattr(ref, name), getattr(padded, name))
        assert not torch.allclose(getattr(ref, name), getattr(changed, name))


def test_empty_text_mask_is_rejected(model, batch):
    mask = batch["text_mask"].clone()
    mask[0] = False
    with pytest.raises(ValueError, match="valid text token"):
        forward(model, batch, text_mask=mask)


def test_missing_state_uses_null_token(model, batch):
    model.eval()
    with torch.no_grad():
        masked = forward(model, batch, state_mask=torch.zeros_like(batch["state_mask"]))
        other = forward(model, batch, state=batch["state"] + 5, state_mask=torch.zeros_like(batch["state_mask"]))
    torch.testing.assert_close(masked.v_action, other.v_action)


def test_gradient_checkpointing_matches(model, batch):
    other = copy.deepcopy(model)
    other.gradient_checkpointing = True
    for net in (model, other):
        out = forward(net, batch)
        (out.v_world.square().mean() + out.v_action.square().mean() + out.v_consequence.square().mean()).backward()
    for a, b in zip(model.parameters(), other.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=1e-4)


def test_no_dead_parameters(model, batch):
    out = forward(model, batch)
    (out.v_world.square().mean() + out.v_action.square().mean() + out.v_consequence.square().mean()).backward()
    assert not [n for n, p in model.named_parameters() if p.grad is None]


def test_build_records_backbone_configuration():
    cfg = tiny_config()
    build_model(cfg.model)
    assert cfg.model.backbone_config["num_layers"] == 3
    assert cfg.model.agent.ctx_dim == 8 and cfg.model.agent.heads == 2
