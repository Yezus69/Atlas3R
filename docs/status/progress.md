# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Current phase: Phase 6H teacher stitching and temporal cache export completed
  on branch `codex/phase6h-teacher-stitch-cache`.
- `runtime map-rgb-teacher` maps RGB-only recording/image/video input through
  VGGT teacher-pseudo depth/pose/intrinsics, stitches overlapping teacher
  windows with Sim3 by default, rejects inconsistent windows, fuses accepted
  pseudo observations through sparse TSDF, and emits observed mesh chunks.
- `--stitch-windows none` preserves the Phase 6G no-stitch baseline for direct
  comparison.
- `--export-teacher-cache` writes validated
  `atlas3r_teacher_temporal_cache` clips for future student training, with safe
  relative paths, finite arrays, explicit pseudo truth flags, no model weights,
  and pseudo target weights of `0.25`.
- Truth flags keep measured depth/pose false for mapping/cache labels, pseudo
  depth/pose true, teacher geometry true, `metric_scale_source=rgb_prior` /
  `teacher_scale_unverified`, and no RGB-only student, realtime, hidden
  geometry, completion, or accuracy claim.

## Latest Verified Test State

- `python -m ruff format src tests`: passed; 208 files left unchanged.
- `python -m ruff format --check src tests`: passed; 208 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 157 source files.
- Focused tests passed: `tests.unit.test_rgb_teacher_stitching`,
  `tests.unit.test_teacher_temporal_cache`,
  `tests.unit.test_rgb_teacher_mapping`, and `tests.unit.test_cli`.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 260 tests.
- `git diff --check`: passed; Git emitted CRLF normalization warnings only.

## Real-Data Evidence

- Input: `runs/phase6a_recording_freiburg1_xyz_val/recording`, TUM RGB-D
  `freiburg1_xyz`, 120 RGB frames used, measured depth/pose only for eval
  sidecars.
- No-stitch command used VGGT `facebook/VGGT-1B` on `cuda:0` with
  `--max-frames 120 --teacher-window-size 12 --teacher-window-overlap 6
  --image-size 518 --pixel-stride 12 --stitch-windows none`.
- No-stitch result: 19 pseudo submaps, boundary jump mean/p95/max
  `0.035442 / 0.076935 / 0.078969 m`, 59 mesh chunks, 28,980 vertices,
  14,490 triangles, eval-only ATE RMSE `0.118310 m`, depth AbsRel/RMSE
  `0.073069 / 0.264572 m`.
- Stitched command used the same settings with `--stitch-windows sim3-overlap
  --export-teacher-cache --cache-clip-length 8 --cache-clip-stride 4`.
- Stitched result: 18 accepted stitch edges, 0 rejected edges, 1 pseudo submap,
  boundary jump mean/p95/max `0.066620 / 0.127874 / 0.133376 m`, overlap RMSE
  mean/p95/max `0.066649 / 0.123902 / 0.130932 m`, scale min/median/max
  `0.815424 / 1.086051 / 1.260082`, 46 mesh chunks, 17,184 vertices,
  8,592 triangles, eval-only ATE RMSE `0.094728 m`, depth AbsRel/RMSE
  `0.073069 / 0.264572 m`.
- Cache inspection passed for
  `runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache`
  with 29 clips; the dataset bridge loaded clip arrays and truth/weight fields.
- Full evidence: `docs/status/phase6h_teacher_stitch_cache_report.md`.

## Compact Phase Ledger

- Phase 0-4: contracts, synthetic correctness, CPU TSDF diagnostics, teacher
  adapter/cache boundaries, NumPy student boundary, optional Torch training MVP,
  checkpoint bridge, and TUM RGB-D debug train/eval.
- Phase 5A-5H: TUM clip caches, teacher-signal caches, external teacher
  runners, temporal measured/pseudo training, student-map runtime diagnostics,
  multi-sequence training, and dependency-safe VGGT runner scaffolding.
- Phase 6A-6F: measured recording boundary, sensor-folder importer, dense and
  sparse TSDF paths, bounded live replay scheduler, and observed-only sparse
  mesh chunk update artifacts.
- Phase 6G: RGB-only teacher bridge using VGGT pseudo depth/pose/intrinsics to
  drive sparse TSDF and observed mesh chunk output.
- Phase 6H: Sim3 overlap stitching for teacher windows plus validated teacher
  temporal cache export for future SMGT training.

## Current Known Gaps

- Boundary jump metrics worsened in the Phase 6H stitched run even though
  eval-only ATE improved.
- Stitching is sequential adjacent-window Sim3, not loop closure, global bundle
  adjustment, or metric-scale proof.
- VGGT teacher inference remains offline and heavy; CPU sparse TSDF and mesh
  export are diagnostic and not realtime.
- No student was trained from the Phase 6H teacher temporal cache.
