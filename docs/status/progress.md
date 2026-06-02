# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and older revisions.

## Current State

- Current phase: Phase 4C complete. The real TUM RGB-D `freiburg1_xyz` debug
  training run completed on CUDA and produced ignored checkpoint artifacts.
- Latest implementation: dependency-light TUM RGB-D `freiburg1_xyz` download,
  safe tar extraction, timestamp association, manifest preparation, lazy
  optional Torch/Pillow dataset, masked RGB-D loss, and real-data training CLI.
- Overnight run artifacts live under ignored
  `runs/tum_rgbd_freiburg1_xyz_overnight/`: `summary.json`, train/validation
  JSONL, `checkpoint_last.pt`, `checkpoint_best.pt`, `prediction_sample.npz`,
  `prediction_preview.html`, and `prediction_preview.svg`.
- Tiny checkpoint loading accepts old synthetic-only and real-RGBD debug
  checkpoints only when mapping/realtime/accuracy/performance/generalization
  truth-boundary gates remain false.

## Latest Verified Test State

- `python -m pip install -e ".[dev,train]"`: passed; Torch 2.1.0+cu121 and
  Pillow 12.2.0 were already installed.
- `python -m ruff format src tests`: passed with 92 files unchanged.
- `python -m ruff format --check src tests`: passed with 92 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 70 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: ran 146 tests and
  passed. A pre-existing `einops` import warning appeared during optional Torch
  tests.
- `python -m atlas3r train tum-rgbd-depth-pose --help`: passed.
- `git diff --check`: passed before the code commit; Git warned that changed LF
  files will be converted to CRLF in the working tree.

## Real-Data Evidence

- Official TUM download succeeded on retry after one transient stdlib HTTPS
  certificate verification failure.
- Manifest: 398 associated frames from `freiburg1_xyz` with stride 2; 358 train
  and 40 validation frames.
- CUDA: Torch `2.1.0+cu121`, 3 CUDA devices, first device
  `NVIDIA GeForce RTX 4090`.
- CUDA smoke: 2 real-data steps passed with AMP.
- Overnight run: 30,000 steps completed with `batch-size=16`, `160x120`,
  `num-workers=4`, AMP enabled, and `--max-runtime-minutes 480`.
- Best validation metrics: RMSE `0.2116048286` m, MAE `0.1216395150` m, AbsRel
  `0.0996375158`. This is training evidence, not an accuracy report.
- Real checkpoint TSDF diagnostic with RGB-only NPZ input passed and wrote
  `metric_family=not_evaluated`, 454 surface points, and no target-depth claim.

## Compact Phase Ledger

- Skeleton: package layout, CLI entry point, Makefile targets, initial tests.
- Phase 0: NumPy contracts, coordinate math, synthetic cube-room session,
  dependency-free inspection, and CPU TSDF reference artifacts.
- Phase 1: dependency-safe teacher adapter stubs, fixture adapter, summaries
  cache, optional arrays, cache inspection, and cache-to-TSDF replay.
- Phase 2: observed MeshChunk/WorldMap sidecars, TSDF output inspection,
  `DepthObservation`, mapper helpers, and deterministic runtime fixture smoke.
- Phase 3: NumPy-only student boundary, RGB frame sources, context-budget
  cleanup, and FramePacket bridges to student/teacher batch contracts.
- Phase 4A: optional PyTorch synthetic-overfit training MVP with checkpoint,
  metrics, prediction sample, and dependency-free preview.
- Phase 4B: trained-checkpoint inference bridge into `DepthObservation` and CPU
  TSDF smoke comparison artifacts.
- Phase 4C: real TUM RGB-D debug training MVP code, tests, checkpoint, and
  diagnostic checkpoint-to-TSDF smoke.

## Current Known Gaps

- No final SMGT transformer, teacher downloads, external model integration,
  video decoding, live camera runtime, object-aware mapping, GLB/PLY export, or
  benchmark accuracy/performance report.
- The TUM checkpoint is a supervised real-capture debug checkpoint only; it is
  not usable for mapping or realtime mapping.
- Phase 4D should add held-out real-checkpoint inference and mapping diagnostics
  without new datasets, external teacher models, DDP, or mesh export.
