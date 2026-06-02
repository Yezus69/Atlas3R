# Active task

Goal: Phase 4C real-data TUM RGB-D debug checkpoint for
`freiburg1_xyz`, with dependency-safe ingestion, masked RGB-D training,
checkpointing, previews, and honest run evidence.

Checklist:

- [x] Read Phase 4C prompt, compact status docs, API contracts, dataset docs,
  training docs, and loss docs.
- [x] Add data/checkpoint/run ignore rules before generating artifacts.
- [x] Add optional `train` dependencies for Torch and Pillow only.
- [x] Add minimal TUM RGB-D download, safe extract, association, and manifest
  preparation CLI.
- [x] Add lazy disk-backed TUM RGB-D PyTorch dataset with depth masks and
  scaled intrinsics.
- [x] Add masked real RGB-D depth/pose loss and keep synthetic losses intact.
- [x] Add real-data training CLI with checkpoint, metrics, summary, NPZ sample,
  and dependency-free preview artifacts.
- [x] Refactor tiny checkpoint truth-boundary validation to accept real-RGBD
  debug checkpoints without weakening mapping/accuracy gates.
- [x] Add focused optional-dependency tests and a tiny CPU smoke.
- [x] Run required format, lint, typecheck, unittest, and CLI help checks.
- [x] Commit code only, then attempt TUM download/prepare and long CUDA run if
  the environment supports it.
- [x] Update compact status/report docs with exact commands, results, gaps, and
  next task.

Stop condition: official TUM download is blocked in this environment by Python
stdlib HTTPS certificate verification failure for `cvg.cit.tum.de`; CUDA is
available, so the next unblock is certificate trust or pre-provided official TUM
files under ignored `data/tum_rgbd/`.

Exclusions: no external model repos, teacher downloads, OpenCV, PyAV,
TensorBoard, W&B, notebooks, web servers, DDP, final SMGT transformer, or
accuracy/performance claims.
