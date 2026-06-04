# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and reports.

## Current State

- Current phase: Phase 6B real capture boundary and incremental mapper timing
  completed.
- Branch: `codex/phase6b-real-capture-incremental-mapper`.
- Latest implementation: `recording from-sensor-folder` imports a validated
  dependency-light measured capture folder into the existing `atlas3r_recording`
  format; `runtime fuse-recording --mode batch|incremental` preserves the
  Phase 6A batch path and adds per-keyframe incremental timing visibility for
  measured depth+pose recordings.
- Incremental mode is diagnostic only. It currently rebuilds the CPU TSDF from
  selected observations per keyframe and reports that limitation explicitly.
- Mesh extraction remains optional through `scikit-image`; local runs degrade
  to `mesh_exported=false` with an install hint because `scikit-image` is not
  installed.

## Latest Verified Test State

- `python -m ruff format src tests`: passed; 165 files left unchanged.
- `python -m ruff format --check src tests`: passed; 165 files already
  formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 124 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 213 tests. The
  pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `where.exe make` and `Get-Command make` found no `make` executable in this
  Windows shell, so `make test`, `make lint`, and `make typecheck` were not run.

## Real-Data Evidence

- Phase 6B measured incremental run:
  `python -m atlas3r runtime fuse-recording --recording runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/phase6b_incremental_fuse_recording_freiburg1_xyz_val --pose-source recording --depth-source recording --max-frames 120 --keyframe-stride 1 --voxel-size-m 0.05 --truncation-voxels 3.0 --export-point-cloud --export-mesh auto --mode incremental`.
- Result: 120 measured TUM `freiburg1_xyz` validation frames fused, frame IDs
  676-795, 3,136 surface points, `surface_points.ply` exported under ignored
  `runs/`, and 120 `per_frame_events.jsonl` records written.
- Mesh export degraded honestly with `mesh_exported=false` because optional
  `scikit-image` is unavailable.
- Diagnostic latency, milliseconds:
  - per-frame observation load p50/p95/max: 14.84 / 16.82713 / 51.109.
  - per-frame map update p50/p95/max: 1427.58275 / 3333.022785 / 3499.7332.
  - total pipeline: 195992.3747.
- Deterministic array-byte counters: observations 27,648,000; TSDF arrays
  2,457,216; surface arrays 62,720.
- Full evidence:
  `docs/status/phase6b_real_capture_incremental_mapper_report.md`.

## Compact Phase Ledger

- Skeleton through Phase 4: contracts, synthetic correctness, CPU TSDF
  diagnostics, dependency-safe adapters, NumPy student boundary, optional Torch
  training MVP, checkpoint TSDF bridge, and TUM RGB-D debug train/eval.
- Phase 5A-5H: TUM clip caches, teacher-signal caches, external teacher
  runners, temporal measured/pseudo training, student-map runtime diagnostics,
  multi-sequence measured training, and dependency-safe VGGT runner scaffolding.
- Phase 6A: stable measured `atlas3r_recording` boundary, TUM/clip-cache
  importers, recording validation, measured recording CPU TSDF fusion, PLY
  export, optional real mesh export, and diagnostic reports.
- Phase 6B: sensor-folder capture importer and incremental measured recording
  timing wrapper with per-frame events and explicit CPU TSDF rebuild bottleneck.

## Current Known Gaps

- No true realtime scheduler, live sensor adapter, bounded GPU/Metal/CUDA TSDF,
  object-aware mapping, glTF/game-engine export, or benchmark accuracy report
  exists.
- Incremental fusion is not yet a true incremental mapper; CPU TSDF rebuilds
  are the Phase 6B bottleneck.
- `student-odometry` remains too drifty for mapping-ready RGB-only use.
- VGGT is not installed/configured; set `ATLAS3R_VGGT_REPO` or install an
  importable `vggt` package before rerunning VGGT teacher generation.
