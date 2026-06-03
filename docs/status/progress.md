# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and older revisions.

## Current State

- Current phase: Phase 5A complete; Phase 5B is next.
- Latest implementation: reusable TUM RGB-D multi-view clip cache, `atlas3r
  forge tum-rgbd-clips`, lazy Torch clip-cache dataset, `TinyTemporalMetricNetV0`,
  temporal losses/metrics, and `atlas3r train tum-rgbd-temporal`.
- Clip caches and temporal runs remain ignored under `data/` and `runs/`; no
  generated payloads, checkpoints, previews, or NPZ files are committed.
- Tiny temporal checkpoints remain debug-only: mapping, realtime, accuracy,
  performance, and generalization truth-boundary gates are false.

## Latest Verified Test State

- `python -m ruff format src tests`: passed with 114 files unchanged.
- `python -m ruff format --check src tests`: passed with 114 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 81 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: ran 157 tests and
  passed. A pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned that changed LF files will be converted
  to CRLF in the working tree.
- `make test`, `make lint`, and `make typecheck`: not run because `make` is not
  installed in this Windows shell.

## Real-Data Evidence

- Phase 4D block manifest: 398 selected TUM `freiburg1_xyz` frames, 338 train
  and 60 validation with tail-block validation.
- Phase 5A clip forge wrote ignored manifests under `data/tum_rgbd/`:
  train 330 clips, validation 56 clips, clip length 5, stride 1, `160x120`,
  `max_frame_gap_s=0.12`, pointmaps enabled, normals enabled.
- Phase 5A CUDA run wrote ignored artifacts under
  `runs/tum_rgbd_temporal_v0_5h/`, completed 12,000 steps on CUDA with AMP, and
  stopped by `completed_steps`.
- Temporal-v0 best validation diagnostic at step 11,500: RMSE `0.1631360863` m,
  MAE `0.1074985532` m, AbsRel `0.0998849113`, relative-translation mean
  `0.0180322000` m, median `0.0161373891` m, p95 `0.0362428948` m.
- Phase 4D v2 block eval remains the stronger depth diagnostic: RMSE
  `0.0754485318` m, MAE `0.0422393417` m, AbsRel `0.0396844349`, camera-center
  mean error `0.0318729403` m, and CPU TSDF diagnostic chamfer-like mean
  `0.0769099724` m.

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
- Phase 4D: real held-out checkpoint eval, predicted-vs-target CPU TSDF
  diagnostics, block validation split, single v2 model, CUDA run, v2 eval, and
  compact report.
- Phase 5A: canonical real multi-view clip cache plus tiny temporal
  center-frame depth and relative-translation training path.

## Current Known Gaps

- No final SMGT transformer, teacher downloads, external model integration,
  video decoding, live camera runtime, object-aware mapping, GLB/PLY export, or
  benchmark accuracy/performance report.
- Temporal-v0 learns center depth and relative translation only; rotation is
  explicitly not learned in Phase 5A.
- Temporal-v0 depth diagnostics are worse than Phase 4D v2, so Phase 5B should
  improve teacher-signal quality through dependency-isolated forge adapters.
