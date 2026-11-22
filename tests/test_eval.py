import json

import numpy as np
import pytest
import torch

from conftest import make_batch, tiny_config, tiny_model
from jam.eval.common import bootstrap, masked_error, write_report
from jam.eval.compare import compare, paired_bootstrap
from jam.eval.modes import compare_modes, split_mode
from jam.eval.offline import evaluate


def test_bootstrap_interval():
    x = np.random.default_rng(0).normal(1.0, 0.5, 400)
    out = bootstrap(x)
    assert out["n"] == 400 and out["ci95"][0] < out["mean"] < out["ci95"][1]
    assert out["mean"] == pytest.approx(x.mean()) and out["ci95"][1] - out["ci95"][0] < 0.15
    weighted = bootstrap(x, np.r_[np.ones(200), np.zeros(200)])
    assert weighted["n"] == 200 and weighted["mean"] == pytest.approx(x[:200].mean())
    assert bootstrap(x, np.zeros(400)) is None


def test_paired_bootstrap_detects_a_paired_difference():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 5.0, 300)
    better = base - 0.2 + rng.normal(0.0, 0.05, 300)
    out = paired_bootstrap(better, base)
    assert out["diff"] == pytest.approx(-0.2, abs=0.02) and out["ci95"][1] < 0 and out["n"] == 300
    same = paired_bootstrap(base, base)
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0]
    assert paired_bootstrap(better, base, seed=3) == paired_bootstrap(better, base, seed=3)
    unpaired = paired_bootstrap(rng.normal(0.0, 5.0, 300) - 0.2, base)
    assert unpaired["ci95"][0] < 0 < unpaired["ci95"][1]
    with pytest.raises(ValueError):
        paired_bootstrap(base, base[:10])


def test_masked_error():
    pred, true = torch.tensor([[[1.0, 5.0]], [[2.0, 2.0]]]), torch.zeros(2, 1, 2)
    mask = torch.tensor([[[True, False]], [[False, False]]])
    assert masked_error(pred, true, mask).tolist() == [1.0, 0.0]


def write(tmp_path, name, kind, arrays, **meta):
    path = tmp_path / f"{name}.json"
    write_report(path, {"kind": kind, "seed": 0, "denoise_steps": 10, **meta}, arrays)
    return path


def test_compare_two_outputs(tmp_path):
    rng = np.random.default_rng(2)
    has = np.r_[np.ones(60), np.zeros(4)].astype(np.float32)
    base = {k: rng.normal(1.0, 0.3, 64).astype(np.float32) for k in ("policy_action", "val_action", "inverse_action", "val_world", "prediction", "follow_swapped", "follow_true", "copy_state_action")}
    other = {k: v.copy() for k, v in base.items()}
    other["policy_action"] = base["policy_action"] + 0.1
    other["follow_swapped"] = base["follow_swapped"] + 0.05
    a = write(tmp_path, "a", "offline", {**base, "has_action": has})
    b = write(tmp_path, "b", "offline", {**other, "has_action": has})
    out = compare(a, b)
    assert out["policy_action_error"]["n"] == 60 and out["prediction_error"]["n"] == 64
    assert out["policy_action_error"]["diff"] == pytest.approx(-0.1, abs=1e-5) and out["policy_action_error"]["ci95"][1] < 0
    assert out["follow_gap"]["diff"] == pytest.approx(-0.05, abs=1e-5)
    assert out["val_world"]["diff"] == 0
    c = write(tmp_path, "c", "offline", {**base, "has_action": np.ones(64, np.float32)})
    with pytest.raises(ValueError, match="unpaired"):
        compare(a, c)
    d = tmp_path / "d.json"
    write_report(d, {"kind": "offline", "seed": 1, "denoise_steps": 10}, {**base, "has_action": has})
    with pytest.raises(ValueError, match="different evaluation settings"):
        compare(a, d)


def test_offline_metrics_and_mode_comparison():
    cfg = tiny_config()
    model = tiny_model().eval()
    batches = [make_batch(seed=1), make_batch(seed=2)]
    summary, arrays = evaluate(model, cfg, batches, steps=2, device=torch.device("cpu"))
    assert summary["windows"] == 4 and arrays["policy_action"].shape == (4,)
    assert summary["follow_gap"]["mean"] == pytest.approx(float((arrays["follow_swapped"] - arrays["follow_true"]).mean()), rel=1e-5)
    again, _ = evaluate(model, cfg, batches, steps=2, device=torch.device("cpu"))
    assert json.dumps(summary) == json.dumps(again)
    modes = ("joint", "action_only", "future_free", "joint_no_tracks", "inverse_dynamics")
    summary, arrays = compare_modes(model, cfg, batches, modes, steps=2, device=torch.device("cpu"))
    assert set(modes) <= set(arrays) and "action_only_minus_joint" in summary and "copy_state" in summary
    assert not np.allclose(arrays["joint"], arrays["action_only"])
    assert split_mode("joint_no_tracks") == ("joint", "absent") and split_mode("future_free") == ("future_free", "denoise")
