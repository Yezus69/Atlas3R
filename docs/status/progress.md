# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Current phase: Phase 6F live observed mesh chunks completed on branch
  `codex/phase6f-live-mesh-chunks`.
- `runtime live-replay-recording` preserves the Phase 6E measured replay
  scheduler and now optionally exports observed-only sparse TSDF mesh chunk
  updates with stable `block_<x>_<y>_<z>` IDs, versioned NPZ payloads, optional
  PLY payloads, a manifest, update log, live event integration, and conservative
  truth flags.
- Sparse TSDF integration now reports changed block coordinates and per-stage
  timings: `sparse_surface_samples`, `sparse_candidate_voxel_coords`,
  `project_sparse_candidates`, `apply_sparse_updates`, and `total_integrate`.
- Existing `runtime fuse-recording --mode incremental --backend
  cpu-persistent|cpu-rebuild|cpu-sparse` and live replay without mesh flags
  remain covered by regression tests.

## Latest Verified Test State

- `python -m ruff format src tests`: passed; 189 files unchanged.
- `python -m ruff format --check src tests`: passed; 189 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 141 source files.
- Focused tests passed: `tests.unit.test_mesh_chunks`,
  `tests.unit.test_sparse_tsdf_meshing`,
  `tests.unit.test_live_replay_mesh_chunks`,
  `tests.unit.test_sparse_tsdf_mapper`,
  `tests.unit.test_live_replay_scheduler`, and `tests.unit.test_cli`.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 244 tests. The
  pre-existing optional Torch/einops import warning appeared on an earlier
  discovery run; the final rerun emitted no stdout.
- `git diff --check`: passed.
- `make test` was not run because `make` is unavailable in this Windows shell
  (`spawnSync make ENOENT`).

## Real-Data Evidence

- Command: `python -m atlas3r runtime live-replay-recording --recording runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/phase6f_live_mesh_freiburg1_xyz_val --target-fps 30 --max-frames 120 --mapper-backend cpu-sparse --map-keyframe-stride 1 --max-capture-queue 4 --max-map-queue 2 --drop-policy oldest --voxel-size-m 0.05 --truncation-voxels 3.0 --pixel-stride 8 --export-point-cloud --export-mesh-chunks --mesh-format ply`.
- Result: 120 measured TUM `freiburg1_xyz` validation frames, 120 measured pose
  updates, 120 measured depth map updates, 117 active mesh chunks, 6044 chunk
  update events, 27,944 vertices, 13,972 triangles, loadable NPZ and PLY chunk
  payloads, and observed-only truth flags.
- Optimized baseline map/mesh p95 ms: `73.695 / 154.331`; combined map+mesh
  p95 `222.709`; total pipeline `45,363.277` ms.
- Preview profile (`stride=2`, `pixel_stride=12`) map/mesh p95 ms:
  `35.041 / 136.029`; combined p95 `169.556`, so the 33 ms target was missed.
- Batched dirty-block snapshot reuse reduced baseline map+mesh p95 from the
  first Phase 6F baseline `556.001` ms to `222.709` ms.
- Full evidence: `docs/status/phase6f_live_mesh_chunks_report.md`.

## Compact Phase Ledger

- Phase 0-4: contracts, synthetic correctness, CPU TSDF diagnostics, teacher
  adapter/cache boundaries, NumPy student boundary, optional Torch training MVP,
  checkpoint bridge, and TUM RGB-D debug train/eval.
- Phase 5A-5H: TUM clip caches, teacher-signal caches, external teacher
  runners, temporal measured/pseudo training, student-map runtime diagnostics,
  multi-sequence training, and dependency-safe VGGT runner scaffolding.
- Phase 6A-6E: measured `atlas3r_recording`, sensor-folder importer, dense
  persistent/rebuild incremental TSDF, sparse block TSDF, sparse memory
  diagnostics, and bounded live replay scheduler.
- Phase 6F: observed-only sparse mesh chunk update artifacts from measured
  replay, with loadability tests and measured profile evidence.

## Current Known Gaps

- CPU sparse candidate generation and fallback mesh chunk update/export remain
  too slow for realtime or 30 FPS preview claims.
- Fallback chunk meshing is blocky and diagnostic; no marching-cubes chunk
  quality path, simplification, materials, GLB, object-aware fusion, loop
  closure, accelerated mapper, real camera hardware CI, RGB-only mapping
  readiness, or benchmark accuracy report exists.
