# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and reports.

## Current State

- Current phase: Phase 6C true persistent incremental TSDF backend completed.
- Branch: `codex/phase6c-true-incremental-tsdf-backend`.
- Latest implementation: `runtime fuse-recording --mode incremental` now
  defaults to `--backend cpu-persistent`, which precomputes fixed offline grid
  bounds once, allocates one dense CPU TSDF state, precomputes voxel centers
  once, and integrates only each new measured `DepthObservation`.
- `--backend cpu-rebuild` preserves the Phase 6B rebuild-per-keyframe path for
  regression comparison.
- Persistent output writes the Phase 6B artifact family plus
  `backend_comparison.json` against batch CPU TSDF.

## Latest Verified Test State

- `python -m ruff format src tests`: passed; 170 files left unchanged.
- `python -m ruff format --check src tests`: passed; 170 files already
  formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 128 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 218 tests.
  The pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `where.exe make` and `Get-Command make` found no `make` executable in this
  Windows shell, so `make test`, `make lint`, and `make typecheck` were not run.

## Real-Data Evidence

- Phase 6C measured persistent run:
  `python -m atlas3r runtime fuse-recording --recording runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/phase6c_persistent_fuse_recording_freiburg1_xyz_val --pose-source recording --depth-source recording --max-frames 120 --keyframe-stride 1 --voxel-size-m 0.05 --truncation-voxels 3.0 --export-point-cloud --export-mesh auto --mode incremental --backend cpu-persistent`.
- Result: 120 measured TUM `freiburg1_xyz` validation frames fused, frame IDs
  676-795, 3,136 surface points, `surface_points.ply` exported under ignored
  `runs/`, and 120 per-frame events written.
- Persistent map update timing, milliseconds: p50/p95/max
  `28.83565 / 31.63879 / 35.3636`; final batch comparison is outside the
  per-frame update loop and took `3407.139` ms.
- Backend comparison against batch CPU TSDF matched exactly on this run:
  max/mean common observed TSDF delta `0.0 / 0.0`, max/mean weight delta
  `0.0 / 0.0`, observed voxel delta `0`, surface and point-cloud count deltas
  `0`.
- Full evidence:
  `docs/status/phase6c_true_incremental_tsdf_backend_report.md`.

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
  timing wrapper with explicit CPU TSDF rebuild bottleneck.
- Phase 6C: persistent dense CPU TSDF backend with one-observation incremental
  updates, retained CPU rebuild backend, batch comparison, and real timing.

## Current Known Gaps

- Fixed TSDF bounds are still precomputed offline from selected observations.
- CPU dense TSDF is not the final bounded GPU/Metal/CUDA mapper.
- No live camera API, true realtime scheduler, object-aware mapping,
  glTF/game-engine export, or benchmark accuracy report exists.
- `student-odometry` remains too drifty for mapping-ready RGB-only use.
- VGGT is not installed/configured; set `ATLAS3R_VGGT_REPO` or install an
  importable `vggt` package before rerunning VGGT teacher generation.
