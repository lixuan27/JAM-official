from pathlib import Path

import pytest

from jam.config import from_dict, load_config, release_dict, to_dict
from jam.model.coupling import CouplingSpec
from jam.model.objective import JointFlowObjective

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PRESETS = sorted(CONFIGS.rglob("*.yaml"))


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.stem)
def test_every_preset_loads(path):
    cfg = load_config(path)
    assert cfg.model.backbone is None
    CouplingSpec(**cfg.model.coupling.__dict__)
    JointFlowObjective(cfg.objective)
    assert from_dict(to_dict(cfg)) == cfg


def test_base_matches_the_paper_configuration():
    cfg = load_config(CONFIGS / "base.yaml")
    agent = cfg.model.agent
    assert (agent.layers, agent.width, agent.ffn_mult, agent.chunk, agent.action_dim) == (30, 1024, 4, 32, 80)
    assert agent.consequence_steps == 16 and cfg.model.coupling.visibility == "reciprocal"
    assert cfg.objective.lambda_world == 1.0 and cfg.sampler.mode == "joint"


def test_ablation_presets():
    load = lambda name: load_config(CONFIGS / "ablations" / f"{name}.yaml")
    assert load("no_motion_tracks").objective.disable_consequence
    with_world, without_world = to_dict(load("world_loss_reference")), to_dict(load("no_world_loss"))
    assert with_world["objective"]["lambda_world"] == 1.0 and without_world["objective"]["lambda_world"] == 0.0
    without_world["objective"]["lambda_world"] = 1.0
    assert with_world == without_world


def test_overrides_and_validation(tmp_path):
    cfg = load_config(CONFIGS / "base.yaml", ["model.coupling.visibility=decoupled", "optim.lr=3e-5", "train.steps=7", "sampler.steps=2"])
    assert cfg.model.coupling.visibility == "decoupled" and cfg.optim.lr == 3e-5 and cfg.train.steps == 7 and cfg.sampler.steps == 2
    assert cfg.optim.betas == (0.9, 0.95)
    with pytest.raises(ValueError, match="unknown"):
        load_config(CONFIGS / "base.yaml", ["model.depth=3"])
    loop = tmp_path / "loop.yaml"
    loop.write_text("include: [loop.yaml]\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(loop)
    stored = release_dict(load_config(CONFIGS / "base.yaml", ["model.backbone=somewhere"]))
    assert stored["model"]["backbone"] is None and stored["model"]["pretrained"] is False
