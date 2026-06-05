# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and reports.

## Current State

- Current phase: Phase 6D sparse block TSDF live-replay diagnostic completed.
- Branch: `codex/phase6d-sparse-block-tsdf-live-replay`.
- Latest implementation: `runtime fuse-recording --mode incremental` now accepts
  `--backend cpu-sparse`, which integrates one measured `DepthObservation` at a
  time into lazily allocated sparse TSDF blocks without fixed dense world bounds
  or precomputed global voxel centers.
- Existing `cpu-persistent` and `cpu-rebuild` incremental backends remain as
  dense baselines.
- `cpu-sparse` writes sparse state artifacts, final observed surface output,
  per-frame active block/voxel/state byte counters, sparse-vs-dense diagnostic
  comparison, and truth flags with no realtime/accuracy/performance/mapping
  readiness claim.
- Added `runtime sparse-tsdf-stress` for apartment-scale dense memory estimates
  versus sparse active-state diagnostics.

## Latest Verified Test State

- `python -m ruff format src tests`: passed; final run left 177 files unchanged.
- `python -m ruff format --check src tests`: passed; 177 files already formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 134 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 223 tests. The
  pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `make test`, `make lint`, and `make typecheck` were not run because
  `make` is not available in this Windows shell.

## Real-Data Evidence

- Phase 6D measured sparse run:
  `python -m atlas3r runtime fuse-recording --recording runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/phase6d_sparse_fuse_recording_freiburg1_xyz_val --pose-source recording --depth-source recording --max-frames 120 --keyframe-stride 1 --voxel-size-m 0.05 --truncation-voxels 3.0 --export-point-cloud --export-mesh auto --mode incremental --backend cpu-sparse`.
- Result: 120 measured TUM `freiburg1_xyz` validation frames fused, frame IDs
  676-795, 2,950 sparse surface points, 142 active blocks, 10,165 active voxels,
  approximate sparse state bytes `585,040`, and `surface_points.ply` exported
  under ignored `runs/`.
- Sparse map update timing, milliseconds p50/p95/max:
  `66.108 / 70.6655 / 73.3619`.
- Sparse-vs-dense persistent diagnostic: dense surface 3,136 points, sparse
  surface 2,950 points, sparse/dense state-memory ratio
  `0.047618117414179296`, symmetric Chamfer-like mean
  `0.019811906433573674` m, 5 cm precision/recall-like percentages
  `94.873046875 / 90.966796875`.
- Apartment stress diagnostic:
  `python -m atlas3r runtime sparse-tsdf-stress --output runs/phase6d_sparse_tsdf_stress_apartment --room-size-m 10,10,3 --voxel-size-m 0.05`.
- Stress result: dense box `[200, 200, 60]`, 2,400,000 dense voxels,
  dense persistent estimate `76,800,000` bytes, sparse active estimate
  `601,520` bytes, sparse/dense state-memory ratio `0.007832291666666666`.
- Full evidence:
  `docs/status/phase6d_sparse_block_tsdf_live_replay_report.md`.

## Compact Phase Ledger

- Skeleton through Phase 4: contracts, synthetic correctness, CPU TSDF
  diagnostics, dependency-safe adapters, NumPy student boundary, optional Torch
  training MVP, checkpoint TSDF bridge, and TUM RGB-D debug train/eval.
- Phase 5A-5H: TUM clip caches, teacher-signal caches, external teacher
  runners, temporal measured/pseudo training, student-map runtime diagnostics,
  multi-sequence measured training, and dependency-safe VGGT runner scaffolding.
- Phase 6A: stable measured `atlas3r_recording` boundary and measured recording
  CPU TSDF fusion with PLY/optional mesh/report outputs.
- Phase 6B: sensor-folder capture importer and incremental measured recording
  timing wrapper with explicit CPU TSDF rebuild bottleneck.
- Phase 6C: persistent dense CPU TSDF backend with one-observation incremental
  updates, retained CPU rebuild backend, batch comparison, and real timing.
- Phase 6D: sparse block CPU TSDF backend with online growth, sparse artifacts,
  sparse-vs-dense diagnostics, and apartment-scale memory stress estimate.

## Current Known Gaps

- Sparse CPU updates are still too slow for a realtime mapper on the measured
  run; p95 update latency is about 70.7 ms.
- Sparse output is final observed surface points and sparse state, not live
  triangle mesh chunks or game-engine streaming assets.
- No live camera API, queue/backpressure scheduler, accelerated mapper,
  object-aware mapping, loop closure, or benchmark accuracy report exists.
- `student-odometry` remains too drifty for mapping-ready RGB-only use.
- VGGT is not installed/configured; set `ATLAS3R_VGGT_REPO` or install an
  importable `vggt` package before rerunning VGGT teacher generation.
