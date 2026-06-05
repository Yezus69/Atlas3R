# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Current phase: Phase 6G RGB teacher-assisted mapping completed on branch
  `codex/phase6g-rgb-teacher-map`.
- `runtime map-rgb-teacher` maps RGB-only recording/image/video input through
  VGGT teacher-pseudo depth/pose/intrinsics, writes pseudo recording artifacts,
  fuses pseudo observations through the existing sparse TSDF mapper, and exports
  observed mesh chunks through the Phase 6F writer.
- Mapping truth flags keep measured depth/pose false, pseudo depth/pose true,
  teacher geometry true, `metric_scale_source=teacher_scale_unverified`, and no
  RGB-only student, realtime, hidden-geometry, completion, or accuracy claim.
- `runtime live-replay-recording`, mesh chunk artifacts, sparse TSDF, and
  external VGGT teacher loading remain dependency-safe and regression-tested.

## Latest Verified Test State

- `python -m ruff format src tests`: passed; 196 files unchanged.
- `python -m ruff format --check src tests`: passed; 196 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 147 source files.
- Focused tests passed: `tests.unit.test_rgb_teacher_mapping`,
  `tests.unit.test_cli`, `tests.unit.test_live_replay_mesh_chunks`,
  `tests.unit.test_sparse_tsdf_meshing`, `tests.unit.test_sparse_tsdf_mapper`,
  and `tests.unit.test_external_teachers`.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 249 tests.
- `python -m compileall -q <new/touched Phase 6G modules>`: passed.
- `git diff --check`: passed; Git emitted CRLF normalization warnings only.

## Real-Data Evidence

- Command: `python -m atlas3r runtime map-rgb-teacher --input runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/phase6g_rgb_teacher_freiburg1_xyz_val --teacher vggt --device cuda:0 --max-frames 64 --frame-stride 2 --teacher-window-size 24 --teacher-window-overlap 8 --image-size 518 --voxel-size-m 0.05 --truncation-voxels 3.0 --pixel-stride 12 --export-mesh-chunks --mesh-format ply --rgb-only`.
- Result: VGGT `facebook/VGGT-1B` on `cuda:0`, 120 available TUM
  `freiburg1_xyz` RGB frames, 60 RGB-only frames used, 4 teacher windows, 60
  pseudo depth/pose/intrinsics observations, 73 active sparse blocks, 6,637
  active voxels, 2,224 surface points, 60 mesh chunks, 59 active chunks, 1
  deleted chunk, 1,521 chunk updates, 24,456 vertices, and 12,228 triangles.
- Diagnostic eval only: median-scale depth AbsRel `0.064570`, RMSE
  `0.255943 m`; Sim3 camera-center ATE RMSE `0.085715 m`, mean `0.076093 m`,
  max `0.128713 m`; RPE not implemented.
- Full evidence: `docs/status/phase6g_rgb_teacher_map_report.md`.

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
- Phase 6G: RGB-only teacher bridge using VGGT pseudo depth/pose/intrinsics to
  drive the sparse TSDF and observed mesh chunk path, with diagnostic eval-only
  comparison against source recording measurements.

## Current Known Gaps

- VGGT teacher inference is offline and heavy; the successful evidence run took
  `74.697 s` for teacher inference and `92.572 s` total pipeline time.
- CPU sparse candidate generation and mesh export remain too slow for realtime
  or 30 FPS preview claims.
- Fallback chunk meshing is blocky and diagnostic; no student RGB-only mapper,
  loop closure, object-aware fusion, mesh quality path, real camera hardware CI,
  benchmark accuracy report, or metric-scale proof exists.
