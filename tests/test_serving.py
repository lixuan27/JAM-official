import io
import json
import threading

import numpy as np
import pytest
import torch

from conftest import make_batch, tiny_config, tiny_model
from jam.infer.policy import Policy, PolicyConfig
from jam.infer.server import PolicyClient, PolicyServer, pack, read_message
from jam.schema.actions import BIMANUAL_80, BINDINGS, ActionScaler, EmbodimentBinding, Route


def test_layout_and_bindings():
    assert BIMANUAL_80.dim == 80 and BIMANUAL_80.index("right.ee_pos") == slice(34, 37)
    binding = BINDINGS["bimanual_pose20"]
    raw = np.arange(20, dtype=np.float32)
    unified = binding.scatter(raw)
    assert unified.shape == (80,) and binding.dim_mask().sum() == 20
    np.testing.assert_array_equal(binding.gather(unified), raw)
    assert unified[~binding.dim_mask()].sum() == 0
    with pytest.raises(ValueError):
        EmbodimentBinding("bad", BIMANUAL_80, 4, (Route(0, 3, "left.ee_pos"), Route(2, 1, "left.gripper")))


def test_action_scaler_round_trip():
    binding = BINDINGS["single_arm_pose10"]
    lo, hi = np.full(80, -2.0, np.float32), np.full(80, 4.0, np.float32)
    scaler = ActionScaler(BIMANUAL_80, lo, hi, binding.dim_mask())
    raw = np.array([1.0, -2.0, 4.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.25], np.float32)
    normal = scaler.normalize(binding.scatter(raw))
    np.testing.assert_allclose(normal[:3], [0.0, -1.0, 1.0])
    np.testing.assert_allclose(normal[3:9], raw[3:9])
    assert normal[9] == pytest.approx(-0.5)
    np.testing.assert_allclose(binding.gather(scaler.unnormalize(normal)), raw, atol=1e-6)


def tiny_policy(tmp_path=None, **kw):
    agent = dict(action_dim=80, state_dim=80, chunk=8, width=16, layers=3, consequence_dim=9, consequence_steps=4, zero_init_heads=False)
    cfg = tiny_config(agent=agent)
    model = tiny_model(agent=agent).eval()
    return Policy(PolicyConfig(checkpoint="", steps=2, **kw), model=model, cfg=cfg, device=torch.device("cpu"))


def observation():
    batch = make_batch()
    return dict(obs_latent=batch["latents"][0, :, :1].numpy(), text=batch["text"][0].numpy(), text_mask=batch["text_mask"][0].numpy())


def test_policy_returns_an_action_chunk(tmp_path):
    stats = tmp_path / "stats.json"
    entry = {"lo": [-1.0] * 80, "hi": [1.0] * 80, "dim_mask": BINDINGS["single_arm_pose10"].dim_mask().tolist()}
    stats.write_text(json.dumps({"action": entry}))
    policy = tiny_policy(mode="action_only", binding="single_arm_pose10", stats=str(stats), execute_steps=4)
    out = policy.act(state=np.zeros(10, np.float32), **observation())
    assert out["actions"].shape == (4, 10) and out["actions_layout"].shape == (8, 80)
    assert np.isfinite(out["actions"]).all()
    inactive = ~BINDINGS["single_arm_pose10"].dim_mask()
    inactive[BIMANUAL_80.dims_of_kind("gripper")] = False
    assert np.abs(out["actions_layout"][:, inactive]).max() == 0
    assert out["latents"].shape == (4, 3, 4, 4)
    second = policy.act(state=np.zeros(10, np.float32), **observation())
    assert not np.allclose(out["actions"], second["actions"])


def test_policy_modes(tmp_path):
    obs = observation()
    plain = tiny_policy(mode="future_free").act(**obs)
    assert plain["actions"].shape == (8, 80) and plain["latents"].shape == (4, 1, 4, 4)
    future = make_batch()["latents"][0].numpy()
    inverse = tiny_policy(mode="inverse_dynamics").act(future=future, **obs)
    np.testing.assert_allclose(inverse["latents"], future, atol=1e-6)


def test_message_round_trip():
    arrays = {"obs_latent": np.random.rand(4, 1, 4, 4).astype(np.float32), "text_mask": np.array([True, False])}
    out = read_message(io.BytesIO(pack(arrays)))
    assert out.keys() == arrays.keys() and all(np.array_equal(out[k], arrays[k]) and out[k].dtype == arrays[k].dtype for k in arrays)
    assert read_message(io.BytesIO(b"")) is None and read_message(io.BytesIO(pack(arrays)[:-3])) is None


def test_server_handles_requests():
    server = PolicyServer(tiny_policy())
    reply = server.infer(observation())
    assert reply["actions"].shape == (8, 80) and set(reply) == {"actions", "actions_layout"}
    with pytest.raises(ValueError):
        server.infer({"text": observation()["text"]})
    with pytest.raises(ValueError):
        server.infer({**observation(), "command": np.zeros(1)})


def test_client_and_server_over_loopback():
    import socket

    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", 0))
        except OSError:
            pytest.skip("loopback sockets are unavailable")
        port = probe.getsockname()[1]
    server = PolicyServer(tiny_policy())
    threading.Thread(target=server.serve, args=("127.0.0.1", port), daemon=True).start()
    client = None
    for _ in range(100):
        try:
            client = PolicyClient("127.0.0.1", port)
            break
        except OSError:
            threading.Event().wait(0.05)
    assert client is not None
    reply = client.infer(state=None, **observation())
    assert reply["actions"].shape == (8, 80)
    with pytest.raises(RuntimeError):
        client.infer(text=observation()["text"])
