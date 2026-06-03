# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and reports.

## Current State

- Current phase: Phase 5E streaming student map runtime implemented; Phase 5F
  external teacher data generation and mixed teacher-signal training is next.
- Branch: `codex/phase5e-streaming-student-map-runtime`.
- Base commit before Phase 5E edits: `a1fafa2`.
- Implementation commit: `a72f7f8`.
- Latest implementation: `atlas3r runtime stream-student-map`, unique
  chronological clip-cache stream builder, padded temporal windows,
  Phase 5D checkpoint-to-`DepthObservation` conversion, oracle and diagnostic
  student-relative pose modes, CPU TSDF fusion, TSDF sidecars, ASCII PLY point
  cloud export, map preview HTML, runtime events, quality reports, and latency
  reports.
- Public commands now include `atlas3r runtime stream-student-map`.

## Latest Verified Test State

- `python -m ruff format src tests`: passed with 142 files unchanged.
- `python -m ruff format --check src tests`: passed with 142 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 105 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: ran 186 tests and
  passed. A pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `where.exe make`: no `make` found in this Windows shell.

## Real-Data Evidence

- Required local Phase 5D/TUM artifacts existed:
  `runs/phase5d_teacher_temporal_v1_tum/checkpoint_best.pt`,
  `data/tum_rgbd/freiburg1_xyz_clip_cache_val`, and
  `data/tum_rgbd/freiburg1_xyz_measured_teacher_val`.
- CUDA was available with an NVIDIA GeForce RTX 4090.
- Required run completed:
  `runs/phase5e_stream_student_map_val`, max 60 unique frames, window 5, pose
  mode `both`, voxel size `0.05` m.
- Oracle diagnostics: RMSE `0.090826432` m, MAE `0.050450069` m,
  AbsRel `0.046490420`, `8622` TSDF surface points, observed coverage `0.08431`.
- Student-relative diagnostics: same depth metrics, `8732` TSDF surface points,
  observed coverage `0.08583`; pose remains diagnostic only.
- All Phase 5E reports set diagnostic/realtime/mapping/accuracy/performance
  truth flags to avoid claims.

## Compact Phase Ledger

- Skeleton through Phase 4: contracts, synthetic correctness, CPU TSDF
  diagnostics, dependency-safe adapters, NumPy student boundary, optional Torch
  training MVP, checkpoint TSDF bridge, and TUM RGB-D debug train/eval.
- Phase 5A: canonical multi-view TUM clip cache plus tiny temporal center-depth
  and relative-translation training path.
- Phase 5B/5B.1: stable teacher-signal cache, measured forge, local ingest,
  inspection, and deduped map replay.
- Phase 5C: dependency-safe external teacher runner boundary, Depth Pro
  bootstrap, VGGT local ingest scaffold, validated external signal caches.
- Phase 5D: teacher-signal temporal-v1 training, weighted losses, checkpoint
  export, and real measured TUM run/eval.
- Phase 5E: streaming checkpoint-to-map runtime diagnostic over real TUM val.

## Current Known Gaps

- No realtime scheduler, live camera loop, bounded GPU TSDF, object-aware
  mapping, triangle mesh extraction, or benchmark accuracy report exists.
- `student-relative` pose is not mapping-ready; it keeps source rotation and
  applies diagnostic predicted translation only.
- No real Depth Pro or VGGT pseudo-labels were generated in Phase 5E.
