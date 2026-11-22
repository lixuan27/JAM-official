import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from conftest import TINY_AGENT, TINY_BACKBONE, make_batch, tiny_config, tiny_model
from jam.eval.common import load_model
from jam.infer.policy import Policy, PolicyConfig
from jam.model.objective import JointFlowObjective, ObjectiveConfig
from jam.train import checkpoint as ckpt
from jam.train.batch import PreparedBatches, validate_batch
from jam.train.loop import train, train_step
from jam.train.optim import OptimConfig, build_optimizer, lr_multiplier

ROOT = Path(__file__).resolve().parents[1]


def write_batches(directory, count=2, size=2):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        save_file(make_batch(seed=100 + i, size=size), str(directory / f"batch_{i:03d}.safetensors"))
    return directory


def test_batch_contract(model, batch):
    validate_batch(batch, model.agent.cfg)
    bad = dict(batch, action_mask=batch["action_mask"].float())
    with pytest.raises(ValueError, match="action_mask must be boolean"):
        validate_batch(bad, model.agent.cfg)
    bad = dict(batch, actions=batch["actions"][:, :4])
    with pytest.raises(ValueError, match="actions must have shape"):
        validate_batch(bad, model.agent.cfg)
    bad = dict(batch, latents=torch.full_like(batch["latents"], float("nan")))
    with pytest.raises(ValueError, match="non-finite"):
        validate_batch(bad, model.agent.cfg)


def test_prepared_batches_cycle_and_partition(model, tmp_path):
    write_batches(tmp_path, count=4)
    data = PreparedBatches(tmp_path, model.agent.cfg)
    assert len(data) == 4 and torch.equal(data[5]["latents"], data[1]["latents"])
    halves = [PreparedBatches(tmp_path, model.agent.cfg, rank, 2) for rank in (0, 1)]
    assert [len(h) for h in halves] == [2, 2] and not set(halves[0].files) & set(halves[1].files)
    with pytest.raises(ValueError):
        PreparedBatches(tmp_path, model.agent.cfg, 0, 3)


def test_parameter_groups_and_schedule(model):
    optimizer = build_optimizer(model, OptimConfig(lr=1e-4, lr_agent=3e-4))
    groups = {g["name"]: g for g in optimizer.param_groups}
    assert groups["world"]["lr"] == 1e-4 and groups["agent"]["lr"] == 3e-4
    assert sum(len(g["params"]) for g in groups.values()) == len(list(model.parameters()))
    cfg = OptimConfig(warmup_steps=10, schedule="wsd", decay_steps=20, min_lr_ratio=0.05)
    assert lr_multiplier(0, 100, cfg) == pytest.approx(0.1) and lr_multiplier(50, 100, cfg) == 1.0
    assert lr_multiplier(79, 100, cfg) == 1.0 and lr_multiplier(90, 100, cfg) == pytest.approx(0.525)
    assert lr_multiplier(100, 100, cfg) == pytest.approx(0.05)
    cosine = OptimConfig(warmup_steps=0, schedule="cosine", min_lr_ratio=0.0)
    assert lr_multiplier(50, 100, cosine) == pytest.approx(0.5)


def test_step_matches_manual_accumulation(model, batch):
    other = copy.deepcopy(model)
    objective = JointFlowObjective(ObjectiveConfig())
    cfg = OptimConfig(lr=1e-3)
    optimizer, reference = build_optimizer(model, cfg), build_optimizer(other, cfg)
    batches = [batch, make_batch(seed=5)]
    logs, norm = train_step(model, objective, optimizer, batches, step=0, grad_accum=2, seed=4)
    assert torch.isfinite(norm) and "loss/total" in logs
    generator = torch.Generator()
    reference.zero_grad(set_to_none=True)
    for micro, item in enumerate(batches):
        generator.manual_seed(4 * 1000003 + micro)
        (objective(other, item, generator=generator)["loss"] / 2).backward()
    torch.nn.utils.clip_grad_norm_(other.parameters(), 1.0)
    reference.step()
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b)


def test_loss_decreases_on_a_fixed_batch(batch):
    model = tiny_model()
    objective = JointFlowObjective(ObjectiveConfig())
    optimizer = build_optimizer(model, OptimConfig(lr=2e-3))
    losses = []
    for step in range(30):
        logs, _ = train_step(model, objective, optimizer, [batch], step=0)
        losses.append(float(logs["loss/total"]))
    assert losses[-1] < 0.8 * losses[0]


def test_checkpoints_and_warm_start(tmp_path):
    batches = write_batches(tmp_path / "batches")
    cfg = tiny_config()
    cfg.train.steps = 2
    train(cfg, batches, tmp_path / "parent")
    assert ckpt.list_steps(tmp_path / "parent", "release") == [1, 2]
    first, config = ckpt.read_release(ckpt.step_dir(tmp_path / "parent", "release", 1))
    second, _ = ckpt.read_release(ckpt.step_dir(tmp_path / "parent", "release", 2))
    assert any(not torch.equal(first[k], second[k]) for k in second)
    assert config["model"]["backbone"] is None and config["model"]["pretrained"] is False
    records = [json.loads(line) for line in (tmp_path / "parent" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [1, 2] and all("loss/total" in r for r in records)
    parent = ckpt.step_dir(tmp_path / "parent", "release", 2)
    cfg = tiny_config()
    cfg.train.steps = 1
    train(cfg, batches, tmp_path / "child", init_from=parent)
    assert ckpt.list_steps(tmp_path / "child", "release") == [1]
    loaded, _ = load_model(ckpt.step_dir(tmp_path / "child", "release", 1), torch.device("cpu"))
    assert loaded.world.num_layers == 3
    policy = Policy(PolicyConfig(checkpoint=str(parent), steps=2, mode="action_only"), device=torch.device("cpu"))
    sample = make_batch()
    out = policy.act(sample["latents"][0, :, :1], sample["text"][0], sample["text_mask"][0], state=sample["state"][0])
    assert out["actions"].shape == (8, 6)


def test_incomplete_checkpoints_are_ignored(tmp_path):
    (tmp_path / "release" / "step_0000005").mkdir(parents=True)
    assert ckpt.list_steps(tmp_path, "release") == []
    with pytest.raises(ValueError):
        ckpt.read_release(tmp_path / "release" / "step_0000005")


def run_module(module, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-m", module, *map(str, args)], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_command_line_workflow(tmp_path):
    batches = write_batches(tmp_path / "batches")
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump({
        "model": {"backbone_config": {**TINY_BACKBONE, "patch_size": list(TINY_BACKBONE["patch_size"])}, "agent": TINY_AGENT, "gradient_checkpointing": False},
        "optim": {"lr": 1e-3, "warmup_steps": 0, "schedule": "constant"},
        "train": {"steps": 2, "log_every": 1, "save_every": 2, "bf16": False},
    }))
    outputs = {}
    for name, overrides in (("a", []), ("b", ["model.coupling.visibility=decoupled", "objective.lambda_world=0.0"])):
        stdout = run_module("jam.train", "--config", config, "--batches", batches, "--output", tmp_path / name, *overrides)
        assert "loss/total" in stdout
        checkpoint = ckpt.step_dir(tmp_path / name, "release", 2)
        assert (checkpoint / "COMPLETE").is_file()
        for kind in ("offline", "modes"):
            out = tmp_path / f"{name}_{kind}.json"
            run_module(f"jam.eval.{kind}", "--checkpoint", checkpoint, "--batches", batches, "--out", out, "--steps", 2)
            report = json.loads(out.read_text())
            assert report["kind"] == kind and report["metrics"]["windows"] == 4 and out.with_suffix(".npz").is_file()
            outputs[name, kind] = out
    stored = json.loads((ckpt.step_dir(tmp_path / "b", "release", 2) / "config.json").read_text())
    assert stored["model"]["coupling"]["visibility"] == "decoupled" and stored["objective"]["lambda_world"] == 0.0
    offline = json.loads(outputs["a", "offline"].read_text())["metrics"]
    for key in ("policy_action_error", "inverse_dynamics_error", "prediction_error", "follow_gap", "val_world", "val_action"):
        assert offline[key]["n"] == 4 and offline[key]["ci95"][0] <= offline[key]["mean"] <= offline[key]["ci95"][1]
    for kind in ("offline", "modes"):
        table = run_module("jam.eval.compare", outputs["a", kind], outputs["b", kind], "--json", tmp_path / f"{kind}_compare.json")
        assert "| metric | A | B | A-B | 95% CI | n |" in table
        compared = json.loads((tmp_path / f"{kind}_compare.json").read_text())
        expected = "policy_action_error" if kind == "offline" else "action_only"
        assert compared[expected]["n"] == 4
