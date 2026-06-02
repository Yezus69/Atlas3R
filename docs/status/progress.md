# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and older revisions.

## Current State

- Current phase: Phase 4A complete; `docs/status/next_task.md` now points to
  Phase 4B checkpoint-inference-to-`DepthObservation` and CPU TSDF smoke.
- Latest implementation: first trainable synthetic-only PyTorch MVP behind the
  optional `train` extra and dependency-safe `atlas3r.training` imports.
- Procedural training data generates deterministic in-memory RGB, analytic
  metric depth, uncertainty, confidence, object mask, intrinsics,
  `T_world_camera`, camera center, and `StudentClipInput` conversion.
- `TinyDepthPoseNet` predicts positive depth/sigma, confidence, and
  `camera_center_world_m`; supervised losses cover depth, sigma NLL,
  confidence, and camera-center error.
- `atlas3r train synthetic-overfit` writes `config.json`, `metrics.jsonl`,
  `checkpoint_last.pt`, `summary.json`, `prediction_sample.npz`,
  `prediction_preview.html`, and `prediction_preview.svg`.
- Phase 4A truth boundary is explicit: synthetic-overfit training MVP only, not
  real capture, realtime mapping, an accuracy report, or a performance report.

## Latest Verified Test State

- `python -m ruff format src tests`: ran; 4 files reformatted.
- `python -m ruff format --check src tests`: passed with 87 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 60 source files.
- `python -m unittest discover -s tests -p 'test_*.py'`: ran 136 tests and passed.
- `python -m atlas3r train synthetic-overfit --output build/smoke/synthetic_overfit --steps 5 --batch-size 2 --num-samples 8 --width 32 --height 24 --device cpu --log-every 1`: passed and wrote the Phase 4A artifact set.
- `git diff --check`: passed; Git warned that changed files will be converted
  from LF to CRLF in the working tree.
- `Get-Command make`: `make` is not recognized on PATH.
- `where.exe make`: `INFO: Could not find files for the given pattern(s).`
- `make test`, `make lint`, and `make typecheck` were not run because `make`
  is not available on PATH.

## Compact Phase Ledger

- Skeleton: package layout, CLI entry point, Makefile targets, initial tests.
- Phase 0: NumPy contracts, coordinate math, synthetic cube-room session,
  dependency-free session inspection, and CPU TSDF reference artifacts.
- Phase 1: dependency-safe teacher adapter stubs, fixture adapter, summaries
  cache, optional arrays, cache inspection, and cache-to-TSDF replay.
- Phase 2: observed MeshChunk/WorldMap sidecars, TSDF output inspection,
  `DepthObservation`, mapper helpers, and deterministic runtime fixture smoke.
- Phase 3: NumPy-only student boundary, RGB frame sources, context-budget
  cleanup, and FramePacket bridges to student/teacher batch contracts.
- Phase 4A: optional PyTorch synthetic-overfit training MVP with checkpoint,
  metrics, prediction sample, and dependency-free preview.

## Current Known Gaps

- No final SMGT transformer, real-world inference bridge, real datasets,
  teacher model downloads, or external model integration.
- The tiny trained model is synthetic-only and not usable for realtime mapping
  or real captures.
- No video decoding, live camera runtime, runtime scheduler changes, TSDF
  changes, GLB/PLY export, object-aware fusion, web server, notebook, or broad
  inspection bundle was added.
- CPU TSDF outputs, sidecars, runtime fixture artifacts, and Phase 4A previews
  remain diagnostic artifacts, not accuracy or performance reports.

## Phase 4A Update

Changed files include `pyproject.toml`, `src/atlas3r/cli.py`,
`src/atlas3r/training/*`, focused unit/synthetic tests, API contracts, and
status docs. The implementation keeps base imports Torch-free and confines
PyTorch use to optional training/model/loss modules.
