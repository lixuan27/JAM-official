import pytest
import torch

from conftest import tiny_model
from jam.infer.sampler import MODES, JointSampler, SamplerConfig

STEPS = 4


class Spy:
    def __init__(self, model):
        self.calls, self.prefix_calls = [], []
        full, cached = model.forward, model.forward_from_prefix
        model.forward = lambda **kw: self.calls.append(kw) or full(**kw)
        model.forward_from_prefix = lambda prefix, actions, **kw: self.prefix_calls.append(kw) or cached(prefix, actions, **kw)


def inputs(batch):
    return dict(
        obs_latent=batch["latents"][:, :, :1], latent_shape=(4, 3, 4, 4), text=batch["text"], text_mask=batch["text_mask"],
        state=batch["state"], state_mask=batch["state_mask"], action_shape=(8, 6),
    )


def sample(model, batch, mode, seed=55, steps=STEPS, **kw):
    cfg = {k: kw.pop(k) for k in ("lag", "cache_prefix") if k in kw}
    sampler = JointSampler(SamplerConfig(steps=steps, mode=mode, **cfg))
    return sampler.sample(model, generator=torch.Generator().manual_seed(seed), **inputs(batch), **kw)


def initial_noise(seed=55):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((2, 4, 3, 4, 4), generator=g), torch.randn((2, 8, 6), generator=g), torch.randn((2, 4, 9), generator=g)


def extras(batch, mode):
    return {"future": batch["latents"]} if mode == "inverse_dynamics" else {"actions": batch["actions"]} if mode == "forward_dynamics" else {}


@pytest.fixture
def model():
    return tiny_model().eval()


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_runs(model, batch, mode):
    out = sample(model, batch, mode, **extras(batch, mode))
    assert out["actions"].shape == (2, 8, 6) and out["consequence"].shape == (2, 4, 9)
    assert all(torch.isfinite(out[k]).all() for k in ("actions", "latents", "consequence"))
    assert torch.equal(out["latents"][:, :, :1], batch["latents"][:, :, :1])
    again = sample(model, batch, mode, **extras(batch, mode))
    assert torch.equal(out["actions"], again["actions"])


def test_joint_denoises_future_and_actions_together(model, batch):
    spy = Spy(model)
    out = sample(model, batch, "joint")
    z, a, _ = initial_noise()
    assert len(spy.calls) == STEPS
    assert torch.equal(spy.calls[0]["latents"][:, :, 1:], z[:, :, 1:]) and torch.equal(spy.calls[0]["actions"], a)
    assert (spy.calls[0]["s_world"] == 1).all() and (spy.calls[0]["s_action"] == 1).all()
    world_levels = [float(c["s_world"][0]) for c in spy.calls]
    assert world_levels == sorted(world_levels, reverse=True) and world_levels[-1] > 0
    assert not torch.allclose(out["latents"][:, :, 1:], z[:, :, 1:]) and not torch.allclose(out["actions"], a)


def test_action_only_keeps_future_tokens_at_noise(model, batch):
    spy = Spy(model)
    out = sample(model, batch, "action_only")
    z, a, _ = initial_noise()
    assert len(spy.calls) == STEPS
    for call in spy.calls:
        assert call["latents"].shape[2] == 3
        assert torch.equal(call["latents"][:, :, 1:], z[:, :, 1:])
        assert (call["s_world"] == 1).all()
    assert torch.equal(out["latents"][:, :, 1:], z[:, :, 1:])
    assert not torch.allclose(out["actions"], a)
    assert [float(c["s_action"][0]) for c in spy.calls][0] == 1


def test_future_free_removes_future_tokens(model, batch):
    spy = Spy(model)
    cached = sample(model, batch, "future_free")
    assert not spy.calls and len(spy.prefix_calls) == STEPS
    assert cached["latents"].shape == (2, 4, 1, 4, 4)
    uncached = sample(model, batch, "future_free", cache_prefix=False)
    assert len(spy.calls) == STEPS
    for call in spy.calls:
        assert call["latents"].shape[2] == 1 and torch.equal(call["latents"], batch["latents"][:, :, :1])
    torch.testing.assert_close(cached["actions"], uncached["actions"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cached["consequence"], uncached["consequence"], atol=1e-5, rtol=1e-5)


def test_future_free_ignores_the_future_entirely(model, batch):
    ref = sample(model, batch, "future_free")
    other = sample(model, batch, "future_free", future=batch["latents"] + 3)
    assert torch.equal(ref["actions"], other["actions"])
    assert not torch.allclose(ref["actions"], sample(model, batch, "action_only")["actions"])


def test_inverse_dynamics_clamps_the_given_future(model, batch):
    spy = Spy(model)
    out = sample(model, batch, "inverse_dynamics", future=batch["latents"])
    for call in spy.calls:
        assert torch.equal(call["latents"], batch["latents"]) and (call["s_world"] == 0).all()
    assert torch.equal(out["latents"], batch["latents"])
    shifted = batch["latents"].clone()
    shifted[:, :, 1:] += 1
    assert not torch.allclose(out["actions"], sample(model, batch, "inverse_dynamics", future=shifted)["actions"])
    with pytest.raises(ValueError):
        sample(model, batch, "inverse_dynamics")


def test_forward_dynamics_clamps_the_given_actions(model, batch):
    spy = Spy(model)
    out = sample(model, batch, "forward_dynamics", actions=batch["actions"])
    for call in spy.calls:
        assert torch.equal(call["actions"], batch["actions"]) and (call["s_action"] == 0).all()
    assert [float(c["s_consequence"][0]) for c in spy.calls][0] == 1
    assert torch.equal(out["actions"], batch["actions"])
    swapped = sample(model, batch, "forward_dynamics", actions=torch.roll(batch["actions"], 1, 0))
    assert not torch.allclose(out["latents"][:, :, 1:], swapped["latents"][:, :, 1:])
    with pytest.raises(ValueError):
        sample(model, batch, "forward_dynamics")


def test_lagged_schedules(model, batch):
    s_w, s_a, s_c = JointSampler(SamplerConfig(steps=10, mode="action_first", lag=0.3)).levels()
    assert (s_w >= s_a).all() and (s_w[1:4] == 1).all() and s_w[-1] == 0 and s_a[-1] == 0 and torch.equal(s_c, s_a)
    s_w, s_a, s_c = JointSampler(SamplerConfig(steps=10, mode="world_first", lag=0.3)).levels()
    assert (s_a >= s_w).all() and (s_a[1:4] == 1).all() and torch.equal(s_c, s_a)
    s_w, s_a, _ = JointSampler(SamplerConfig(steps=10, mode="joint")).levels()
    assert torch.equal(s_w, s_a) and s_w[0] == 1 and s_w[-1] == 0
    spy = Spy(model)
    sample(model, batch, "action_first", lag=0.5)
    assert all(float(c["s_world"][0]) >= float(c["s_action"][0]) for c in spy.calls)
    assert float(spy.calls[1]["s_world"][0]) == 1 and float(spy.calls[1]["s_action"][0]) < 1


def test_imagine_then_act_runs_two_phases(model, batch):
    spy = Spy(model)
    out = sample(model, batch, "imagine_then_act")
    assert len(spy.calls) == 2 * STEPS
    first, second = spy.calls[:STEPS], spy.calls[STEPS:]
    _, a, _ = initial_noise()
    assert all((c["s_action"] == 1).all() and torch.equal(c["actions"], a) for c in first)
    assert all((c["s_world"] == 0).all() for c in second)
    assert all(torch.equal(c["latents"], second[0]["latents"]) for c in second)
    assert torch.equal(out["latents"], second[0]["latents"])


def test_modes_share_initial_noise(model, batch):
    _, a, c = initial_noise()
    for mode in ("joint", "action_only", "future_free", "inverse_dynamics", "imagine_then_act"):
        spy = Spy(model)
        sample(model, batch, mode, cache_prefix=False, **extras(batch, mode))
        assert torch.equal(spy.calls[0]["actions"], a) and torch.equal(spy.calls[0]["consequence"], c)


def test_absent_motion_tracks_stay_at_noise(model, batch):
    spy = Spy(model)
    out = sample(model, batch, "joint", consequence_mode="absent")
    _, _, c = initial_noise()
    assert all((call["s_consequence"] == 1).all() for call in spy.calls)
    assert torch.equal(out["consequence"], c)


def test_two_step_sampling(model, batch):
    spy = Spy(model)
    out = sample(model, batch, "action_only", steps=2)
    assert len(spy.calls) == 2 and torch.isfinite(out["actions"]).all()


def test_inactive_action_dimensions_follow_their_clean_value(model, batch):
    mask = torch.tensor([True, True, True, False, False, True])
    clean = torch.full((6,), -1.0)
    out = sample(model, batch, "joint", action_dim_mask=mask, inactive_action_clean=clean)
    assert (out["actions"][..., ~mask] == -1).all()
    free = sample(model, batch, "joint")
    assert not torch.allclose(out["actions"][..., mask], free["actions"][..., mask])
    with pytest.raises(ValueError):
        sample(model, batch, "joint", inactive_action_clean=clean)


def test_invalid_requests_are_rejected(model, batch):
    with pytest.raises(ValueError):
        SamplerConfig(mode="unknown")
    with pytest.raises(ValueError):
        JointSampler(SamplerConfig()).sample(model, **{**inputs(batch), "latent_shape": (4, 1, 4, 4)})
