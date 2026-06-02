# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and older revisions.

## Current State

- Current phase: Phase 4C code slice implemented; real TUM RGB-D download and
  overnight CUDA run still need environment validation after the code commit.
- Latest implementation: dependency-light TUM RGB-D `freiburg1_xyz` download,
  safe tar extraction, timestamp association, and manifest preparation CLI.
- Added lazy optional Torch/Pillow TUM RGB-D dataset with depth scale
  `raw/5000.0`, valid-depth masks, resize-scaled intrinsics, pose metadata, and
  no all-images-in-memory load.
- Added `masked_rgbd_depth_pose_loss(...)`, real-data training CLI
  `atlas3r train tum-rgbd-depth-pose`, JSON/JSONL metrics, last/best tiny
  checkpoints, NPZ prediction sample, and HTML/SVG preview with valid mask.
- Tiny checkpoint loading now accepts old synthetic-only and real-RGBD debug
  checkpoints only when the truth boundary keeps mapping/realtime/accuracy/
  performance/generalization gates false.

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
- `git diff --check`: passed; Git warned that changed LF files will be
  converted to CRLF in the working tree.

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
- Phase 4C: real TUM RGB-D debug training MVP code and tests.

## Current Known Gaps

- No final SMGT transformer, teacher downloads, external model integration,
  video decoding, live camera runtime, object-aware mapping, GLB/PLY export, or
  benchmark accuracy/performance report.
- The TUM model path is a supervised real-capture debug checkpoint only; it is
  not usable for mapping or realtime mapping.
- Real TUM download, manifest preparation, CUDA availability, and overnight run
  evidence are pending after the code commit.
