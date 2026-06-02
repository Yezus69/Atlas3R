# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and older revisions.

## Current State

- Current phase: Phase 4D complete; Phase 5A is next.
- Latest implementation: dependency-light TUM RGB-D ingestion/training plus
  `atlas3r eval tum-rgbd-checkpoint`, manifest `split_policy=block`,
  `TinyMetricDepthNetV2`, masked `log_l1` depth-loss support, and a compact
  Phase 4D diagnostic report.
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
- `python -m ruff format src tests`: passed with 104 files unchanged.
- `python -m ruff format --check src tests`: passed with 104 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 73 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: ran 152 tests and
  passed. A pre-existing `einops` import warning appeared during optional Torch
  tests.
- `git diff --check`: passed; Git warned that changed LF files will be
  converted to CRLF in the working tree.
- `make test`, `make lint`, and `make typecheck`: not run because `make` is not
  installed in this Windows shell.
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
- Phase 4C overnight run: 30,000 steps completed with `batch-size=16`, `160x120`,
  `num-workers=4`, AMP enabled, and `--max-runtime-minutes 480`.
- Phase 4C best validation metrics: RMSE `0.2116048286` m, MAE `0.1216395150` m, AbsRel
  `0.0996375158`. This is training evidence, not an accuracy report.
- Real checkpoint TSDF diagnostic with RGB-only NPZ input passed and wrote
  `metric_family=not_evaluated`, 454 surface points, and no target-depth claim.
- Phase 4D block manifest: 398 selected frames, 338 train and 60 validation.
- Phase 4D baseline block eval wrote ignored artifacts under
  `runs/phase4d_eval_baseline_block/`: depth RMSE `0.1463244370` m, MAE
  `0.0841197038` m, AbsRel `0.0746864870`, camera-center mean error
  `0.0225338522` m, and CPU TSDF diagnostic chamfer-like mean `0.1183717438` m.
- Phase 4D v2 CUDA run wrote ignored artifacts under
  `runs/tum_rgbd_freiburg1_xyz_v2_5h/`: `19009` steps, stopped at
  `max_runtime_minutes`, best validation step `17000`, RMSE `0.0737374956` m,
  MAE `0.0425126282` m, and AbsRel `0.0403562384`.
- Phase 4D v2 block eval wrote ignored artifacts under
  `runs/phase4d_eval_v2_block/`: depth RMSE `0.0754485318` m, MAE
  `0.0422393417` m, AbsRel `0.0396844349`, camera-center mean error
  `0.0318729403` m, and CPU TSDF diagnostic chamfer-like mean `0.0769099724` m.

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

## Current Known Gaps

- No final SMGT transformer, teacher downloads, external model integration,
  video decoding, live camera runtime, object-aware mapping, GLB/PLY export, or
  benchmark accuracy/performance report.
- The TUM checkpoint is a supervised real-capture debug checkpoint only; it is
  not usable for mapping or realtime mapping.
- Phase 5A should build the reusable real multi-view clip cache and temporal
  geometry training path. Do not add external model repositories or commit
  generated data/checkpoints/previews.
