import pytest
import torch

from conftest import tiny_model
from jam.model.flow import FlowSchedule, JointNoisePlane, masked_mse
from jam.model.objective import JointFlowObjective, ObjectiveConfig, mask_implausible_consequences


def run(model, batch, seed=8, **cfg):
    return JointFlowObjective(ObjectiveConfig(**cfg))(model, batch, generator=torch.Generator().manual_seed(seed))


def test_flow_schedule():
    flow = FlowSchedule()
    draw = JointNoisePlane(flow, flow).draw(4096, generator=torch.Generator().manual_seed(3))
    assert not torch.equal(draw["s_world"], draw["s_action"]) and not torch.equal(draw["s_action"], draw["s_consequence"])
    for key in ("s_world", "s_action", "s_consequence"):
        assert 0 <= draw[key].min() and draw[key].max() <= 1
    assert abs(float(draw["w_world"].mean()) - 1) < 0.05 and draw["w_world"].min() >= 0
    corr = torch.corrcoef(torch.stack([draw["s_world"], draw["s_action"]]))[0, 1]
    assert abs(float(corr)) < 0.06
    levels = flow.levels(torch.linspace(1.0, 0.0, 5))
    assert levels[0] == 1 and levels[-1] == 0 and (levels[1:] < levels[:-1]).all()
    clean, noise = torch.tensor([2.0]), torch.tensor([7.0])
    torch.testing.assert_close(noise + flow.target(clean, noise) * (0.0 - 1.0), clean)
    torch.testing.assert_close(flow.noise(clean, noise, torch.tensor(0.25)), 0.75 * clean + 0.25 * noise)


def test_masked_coordinates():
    pred = torch.tensor([[[2.0, 999.0]]], requires_grad=True)
    mask = torch.tensor([[[True, False]]])
    loss = masked_mse(pred, torch.zeros_like(pred), mask, (1, 2)).sum()
    assert loss == 4
    loss.backward()
    assert pred.grad[0, 0, 1] == 0
    assert masked_mse(pred, pred, torch.zeros_like(mask), (1, 2)).item() == 0


def test_loss_is_the_weighted_sum_of_modalities(model, batch):
    out = run(model, batch, lambda_world=0.5, lambda_action=2.0, lambda_consequence=3.0)
    logs = out["logs"]
    expected = 0.5 * logs["loss/world"] + 2.0 * logs["loss/action"] + 3.0 * logs["loss/consequence"]
    torch.testing.assert_close(out["loss"].detach(), expected)
    assert not torch.equal(out["s_world"], out["s_action"]) and not torch.equal(out["s_action"], out["s_consequence"])


def test_missing_modalities_enter_as_noise_and_carry_no_loss(model, batch):
    batch["action_mask"][0] = False
    batch["consequence_mask"][1] = False
    before = run(model, batch)
    batch["actions"][0] = 7
    batch["consequence"][1] = -5
    after = run(model, batch)
    assert before["s_action"][0] == 1 and before["s_action"][1] < 1
    assert before["s_consequence"][1] == 1 and before["s_consequence"][0] < 1
    assert before["logs"]["frac/action_supervised"] == 0.5
    torch.testing.assert_close(before["logs"]["loss/world"], after["logs"]["loss/world"])
    torch.testing.assert_close(before["loss"], after["loss"])
    after["loss"].backward()
    assert model.agent.consequence_out.weight.grad.abs().sum() > 0


def test_observation_prefix_and_invalid_frames_are_excluded(model, batch):
    seen = {}
    original = model.forward
    model.forward = lambda **kw: seen.update(kw) or original(**kw)
    batch["latent_mask"][0, 2] = False
    ref = run(model, batch)
    assert torch.equal(seen["latents"][:, :, :1], batch["latents"][:, :, :1])
    assert not torch.allclose(seen["latents"][:, :, 1:], batch["latents"][:, :, 1:])

    class Shifted:
        def __init__(self, base, delta):
            self.base, self.delta = base, delta

        def __call__(self, **kw):
            out = self.base(**kw)
            out.v_world = out.v_world + self.delta
            return out

    delta = torch.zeros_like(batch["latents"])
    delta[:, :, 0] = 5
    delta[0, :, 2] = 5
    torch.testing.assert_close(run(Shifted(original, delta), batch)["logs"]["loss/world"], ref["logs"]["loss/world"])
    delta[1, :, 2] = 5
    assert run(Shifted(original, delta), batch)["logs"]["loss/world"] > ref["logs"]["loss/world"]


def test_no_world_loss_control(batch):
    model = tiny_model()
    out = run(model, batch, lambda_world=0.0)
    assert out["logs"]["loss/world"] > 0
    torch.testing.assert_close(out["loss"].detach(), out["logs"]["loss/action"] + out["logs"]["loss/consequence"])
    out["loss"].backward()
    head = model.world.net.proj_out.weight.grad
    assert head is None or head.abs().sum() == 0
    assert model.world.net.blocks[0].attn1.to_k.weight.grad.abs().sum() > 0
    reference = tiny_model()
    run(reference, batch)["loss"].backward()
    assert reference.world.net.proj_out.weight.grad.abs().sum() > 0


def test_disabled_motion_tracks(model, batch):
    out = run(model, batch, disable_consequence=True)
    assert (out["s_consequence"] == 1).all()
    assert out["logs"]["loss/consequence"] == 0 and out["logs"]["frac/consequence_supervised"] == 0
    changed = dict(batch, consequence=batch["consequence"] + 3)
    torch.testing.assert_close(out["loss"], run(model, changed, disable_consequence=True)["loss"])
    out["loss"].backward()
    grad = model.agent.consequence_out.weight.grad
    assert grad is None or grad.abs().sum() == 0


def test_implausible_track_displacements_are_unlabelled(model, batch):
    c = torch.zeros(1, 2, 6)
    c[0, 0, 0] = 9.0
    c[0, 1, 5] = 9.0
    clean, mask = mask_implausible_consequences(c, torch.ones_like(c, dtype=torch.bool), 8.0)
    assert not mask[0, 0, :3].any() and mask[0, 0, 3:].all() and mask[0, 1].all()
    assert clean[0, 0, 0] == 0 and clean[0, 1, 5] == 9
    batch["consequence"][0, 0, 0] = 100.0
    ref = run(model, batch)
    batch["consequence"][0, 0, 0] = 500.0
    torch.testing.assert_close(ref["loss"], run(model, batch)["loss"])


def test_predictive_track_slice(model, batch):
    out = run(model, batch, predictive_consequence_probability=1.0)
    assert (out["s_world"] == 1).all() and (out["s_consequence"] == 1).all() and (out["s_action"] == 0).all()
    assert out["logs"]["frac/consequence_predictive"] == 1
    assert out["logs"]["loss/world"] == 0 and out["logs"]["loss/action"] == 0 and out["logs"]["loss/consequence"] > 0
    with pytest.raises(ValueError):
        JointFlowObjective(ObjectiveConfig(predictive_consequence_probability=0.5, disable_consequence=True))


def test_objective_is_deterministic_under_a_seed(model, batch):
    torch.testing.assert_close(run(model, batch, seed=4)["loss"], run(model, batch, seed=4)["loss"])
    assert run(model, batch, seed=4)["loss"] != run(model, batch, seed=5)["loss"]
