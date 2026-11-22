# JAM: Joint World-Action Modeling

JAM is a pretraining framework for world-action models. A world stream, inherited from a pretrained video diffusion transformer, and an action stream are coupled at every paired layer by multi-stream joint attention: each stream keeps its own projections and residual state, and both read the keys and values of both streams. Text conditions both streams by cross-attention, and the current observation is a clean prefix. Future video latents, native actions, and 2D Motion Tracks are corrupted with independent noise levels and trained with flow matching, so one model supports joint sampling, action-only control, future-free control, and inverse dynamics.

This repository contains the model, the training objective, all inference modes, offline evaluation, and CPU tests. Dataset preparation, data configurations, and checkpoints are not included; benchmark-specific closed-loop clients and low-rank adaptation for the real robot are not part of this release.

## Paper components and files

| Component or experiment | Where |
| --- | --- |
| World stream (video transformer with opened self-attention) | `src/jam/model/world_stream.py` |
| Action stream (state, action, and Motion Track tokens) | `src/jam/model/action_stream.py` |
| Multi-stream joint attention, layer pairing | `src/jam/model/jam.py` |
| Coupling direction: reciprocal, world-to-action, action-to-world, decoupled | `src/jam/model/coupling.py` (`model.coupling.visibility`) |
| Independent noise levels, flow matching, noise-level weight | `src/jam/model/flow.py` |
| Objective with per-modality weights and label masks | `src/jam/model/objective.py` |
| No-world-loss control: two runs from the video backbone that differ only in lambda_world | `configs/ablations/{world_loss_reference,no_world_loss}.yaml` |
| Motion Tracks removed | `configs/ablations/no_motion_tracks.yaml` |
| Inference modes, few-step sampling | `src/jam/infer/sampler.py` |
| Policy and policy server | `src/jam/infer/policy.py`, `src/jam/infer/server.py` |
| Typed 80-dimensional action layout | `src/jam/schema/actions.py` |
| Training loop, optimizer, FSDP, weight checkpoints | `src/jam/train/` |
| Offline held-out metrics, inference-mode comparison, paired bootstrap | `src/jam/eval/{offline,modes,compare}.py` |

In code, `agent` names the action stream and `consequence` names the Motion Track pathway.

## Install and test

Python 3.11 or newer.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The tests build tiny synthetic models on CPU. They need no pretrained weights, no datasets, and no network access.

## Train

Training reads prepared tensor batches, one `.safetensors` file per batch, as specified in [docs/tensors.md](docs/tensors.md). `$BACKBONE_DIR` is a local copy of the video backbone with a `transformer/` subfolder.

```bash
python -m jam.train --config configs/base.yaml --backbone "$BACKBONE_DIR" \
  --batches "$BATCH_DIR" --output "$RUN_DIR" train.steps="$STEPS"
```

Trailing `key=value` arguments override configuration entries. Under `torchrun` the batch files are partitioned across processes and the model is wrapped with hybrid-sharded FSDP (full sharding on a single node). Each save writes consolidated weights with their configuration under `release/step_XXXXXXX/`; `--init-from` starts a run from such weights.

The no-world-loss pair is trained from the video backbone with `configs/ablations/world_loss_reference.yaml` and `configs/ablations/no_world_loss.yaml`.

## Evaluate offline

Both evaluators run on prepared held-out batches and fix windows and noise by seed, so two checkpoints are paired window by window.

```bash
python -m jam.eval.offline --checkpoint "$CHECKPOINT" --batches "$HELDOUT_DIR" --out a.json
python -m jam.eval.modes   --checkpoint "$CHECKPOINT" --batches "$HELDOUT_DIR" --out a_modes.json --steps 2
python -m jam.eval.compare a.json b.json
```

`jam.eval.offline` reports policy action error (joint sampling) next to a copy-state floor, inverse-dynamics error, prediction error, and the follow gap: the change in prediction error when the recorded actions are replaced by another window's actions under the same noise. `jam.eval.modes` compares inference modes from the same initial noise; the suffix `_no_tracks` holds Motion Track tokens at pure noise. `jam.eval.compare` prints both values and their difference with a 95% paired bootstrap interval.

## Inference modes

| Mode | Behaviour |
| --- | --- |
| `joint` | Future video, actions, and Motion Tracks are denoised together |
| `action_only` | Action-only control: future video tokens are held at pure noise |
| `future_free` | Future video tokens are removed; the observation is encoded once and reused by every step |
| `inverse_dynamics` | The world stream is clamped to an observed future |
| `forward_dynamics` | Actions are clamped; the future is denoised |
| `action_first`, `world_first` | One stream leads; the other stays at noise for the first `lag` fraction of the steps (default 0.5) |
| `imagine_then_act` | The future is denoised first, then actions are denoised against it |

`--steps` sets the number of denoising steps for every mode.

## Serve

```bash
python -m jam.infer.server --checkpoint "$CHECKPOINT" --mode action_only --steps 2 --port 8000
```

A request carries the encoded observation (`obs_latent`), the encoded instruction (`text`, optional `text_mask`), and an optional `state`; the reply carries the action chunk. `PolicyClient` in `src/jam/infer/server.py` implements the client side. `--binding` maps an embodiment's raw state and action vectors to the shared layout, and `--stats` supplies normalisation ranges.

## License

MIT, see [LICENSE](LICENSE). Dependencies and separately obtained weights keep their own licenses.
