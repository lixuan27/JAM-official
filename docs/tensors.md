# Prepared tensor interface

Each `.safetensors` file contains one batch. Floating tensors must be finite. Masks are boolean and use `True` for valid entries. Use a consistent coordinate layout and normalisation across samples, and fill invalid target and state coordinates with zeros before serialisation. The same files serve training and offline evaluation; a held-out set is a separate directory.

| Key | Shape | Meaning |
| --- | --- | --- |
| `latents` | `B, C, T, H, W` | Encoded video: clean observation frame(s) followed by future frames |
| `latent_mask` | `B, T` | Valid latent frames; each sample includes a valid future frame |
| `actions` | `B, A, D` | Future action chunk in the shared layout |
| `action_mask` | `B, A, D` | Valid action coordinates |
| `text` | `B, L, E` | Encoded instruction |
| `text_mask` | `B, L` | Valid text tokens, at least one per sample |
| `state` | `B, S` | Optional proprioceptive state in the shared layout |
| `state_mask` | `B, S` | Required when `state` is supplied |
| `consequence` | `B, M, 3K` | Motion Tracks: per step, `(dx, dy, visibility)` for each of `K` points |
| `consequence_mask` | `B, M, 3K` | Valid Motion Track coordinates |

`A`, `D`, and `S` match `chunk`, `action_dim`, and `state_dim` of `model.agent`. `M` matches `consequence_steps` and `3K` matches `consequence_dim`. `C` and `E` match the video backbone. Spatial latent sizes must be divisible by its patch size; the temporal patch size is one. The number of clean observation frames is `model.clean_frames`.

With a configured Motion Track head, every batch includes `consequence` and `consequence_mask`. Samples without track labels use a zero tensor and an all-false mask; the same convention applies to samples without action labels. Such a modality enters the model as pure noise at level one and its loss is masked out. Track steps whose displacement exceeds `objective.consequence_max_abs` are treated as unlabelled. A missing state uses the learned null token.

For partially labelled modalities, the objective averages the squared error over valid coordinates within each sample, then averages the weighted per-sample loss over samples that carry the modality. The world loss averages over valid future frames of each sample; the clean observation prefix is excluded.

The action layout is typed (`src/jam/schema/actions.py`): 34 dimensions per arm (end-effector position 3, rotation 6, gripper 1, hand joints 24) and 12 embodiment-specific dimensions. An embodiment binding scatters a raw vector into this layout; unrouted dimensions are zero and masked.

## Evaluation outputs

`jam.eval.offline` and `jam.eval.modes` write a JSON summary and, next to it, an `.npz` file with per-window values. `jam.eval.compare` reads both files of two runs and requires identical seeds, step counts, and windows.

## Policy server messages

A message is an 8-byte big-endian length followed by an `.npz` archive. Request arrays: `obs_latent` `(C, clean_frames, H, W)`, `text` `(L, E)`, optional `text_mask` `(L)`, optional `state`, optional `future` `(C, T, H, W)` for inverse dynamics. Reply arrays: `actions` and `actions_layout`.
