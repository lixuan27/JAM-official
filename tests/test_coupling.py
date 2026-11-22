import pytest
import torch

from conftest import forward, tiny_model
from jam.model.coupling import VISIBILITY, CouplingSpec, TokenBook, build_joint_mask

BOOK = TokenBook(world_tokens=12, clean_world_tokens=4, agent_condition_tokens=1, action_tokens=5, consequence_tokens=3)
NW, K, CND, ACT, CSQ = BOOK.noisy_world, BOOK.clean, BOOK.condition, BOOK.action, BOOK.consequence


@pytest.mark.parametrize("visibility", list(VISIBILITY))
def test_mask_for_every_direction(visibility):
    action_reads_world, world_reads_action = VISIBILITY[visibility]
    m = build_joint_mask(CouplingSpec(visibility), BOOK)
    assert m.shape == (BOOK.total, BOOK.total)
    assert m[K, K].all() and not m[K, NW].any() and not m[K, BOOK.agent].any()
    assert m[NW, BOOK.world].all() and m[BOOK.agent, BOOK.agent].sum() > 0
    assert m[NW, CND].all() and m[ACT, K].all() and m[CSQ, K].all() and m[CND, K].all()
    assert bool(m[ACT, NW].all()) == action_reads_world and bool(m[ACT, NW].any()) == action_reads_world
    assert bool(m[CSQ, NW].all()) == action_reads_world
    assert bool(m[NW, ACT].all()) == world_reads_action and bool(m[NW, ACT].any()) == world_reads_action
    assert bool(m[NW, CSQ].all()) == world_reads_action
    assert m[ACT, ACT].all() and m[ACT, CSQ].all() and m[CSQ, ACT].all() and m[ACT, CND].all()
    reciprocal = visibility == "reciprocal"
    assert bool(m[CND, NW].any()) == reciprocal
    assert bool(m[CND, ACT].any()) == reciprocal


def test_unknown_direction_is_rejected():
    with pytest.raises(ValueError):
        CouplingSpec("sideways")


@pytest.mark.parametrize("visibility", list(VISIBILITY))
def test_information_flow_matches_the_mask(visibility, batch):
    action_reads_world, world_reads_action = VISIBILITY[visibility]
    model = tiny_model(coupling={"visibility": visibility}).eval()
    with torch.no_grad():
        ref = forward(model, batch)
        new_actions = forward(model, batch, actions=batch["actions"] + 2, consequence=batch["consequence"] - 1)
        future = batch["latents"].clone()
        future[:, :, 1:] += 3
        new_future = forward(model, batch, latents=future)
    world_changed = not torch.allclose(ref.v_world[:, :, 1:], new_actions.v_world[:, :, 1:], atol=1e-6)
    action_changed = not torch.allclose(ref.v_action, new_future.v_action, atol=1e-6)
    assert world_changed == world_reads_action
    assert action_changed == action_reads_world
    for other in (new_actions, new_future):
        torch.testing.assert_close(ref.v_world[:, :, :1], other.v_world[:, :, :1])


def test_decoupled_world_loss_never_reaches_action_tokens(batch):
    for visibility, reaches in (("decoupled", False), ("world_to_action", False), ("action_to_world", True), ("reciprocal", True)):
        model = tiny_model(coupling={"visibility": visibility})
        forward(model, batch).v_world[:, :, 1:].square().mean().backward()
        grad = model.agent.action_in.weight.grad
        assert (grad is not None and bool(grad.abs().sum() > 0)) == reaches


def test_reciprocal_gradients_cross_streams(model, batch):
    forward(model, batch).v_action.square().mean().backward()
    assert model.world.net.blocks[0].attn1.to_k.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    forward(model, batch).v_world.square().mean().backward()
    assert model.agent.blocks[0].k.weight.grad.abs().sum() > 0
