# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and reports.

## Current State

- Current phase: Phase 5D teacher-weighted temporal student training complete;
  Phase 5E streaming runtime/mapping comparison is next.
- Implementation commit: `670aac8` on
  `codex/phase5d-teacher-weighted-temporal-mapping-training`.
- Latest implementation: lazy teacher-signal temporal dataset, weighted
  confidence/uncertainty losses, `TemporalMetricNetV1`, training CLI,
  student-checkpoint teacher-cache export, duplicate-frame-stable export for
  overlapping clips, API docs, ADR, and focused tests.
- Public commands now include `atlas3r train teacher-signals-temporal` and
  `atlas3r teachers run-student-temporal`.
- Teacher-signal caches remain the stable boundary. Student exports are
  pseudo-labels with `measured_geometry=false`; source clip poses are reused in
  exported `T_world_camera`.

## Latest Verified Test State

- `python -m ruff format src tests`: passed with 134 files unchanged.
- `python -m ruff format --check src tests`: passed with 134 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 98 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: ran 181 tests and
  passed. A pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `where.exe make`: no `make` found in this Windows shell.

## Real-Data Evidence

- Local measured caches existed:
  `data/tum_rgbd/freiburg1_xyz_measured_teacher_train` (330 clips) and
  `data/tum_rgbd/freiburg1_xyz_measured_teacher_val` (56 clips).
- CUDA was available with an NVIDIA GeForce RTX 4090.
- Measured-only run completed:
  `runs/phase5d_teacher_temporal_v1_tum`, 20,000 steps, batch 8, AMP, CUDA.
- Best validation at step 19,000: RMSE 0.087864619 m, MAE 0.051004592 m,
  AbsRel 0.046700478. These are diagnostic validation metrics, not benchmark
  accuracy claims.
- Student export wrote `runs/phase5d_student_val_teacher_cache` with 56
  pseudo-label signals. Inspect summary: RMSE 0.089514971 m, MAE 0.050030827 m,
  AbsRel 0.046260278, mean confidence 0.7613.
- Map-signals summary: 280 observations before dedupe, 60 after dedupe,
  220 duplicate frames, 8,952 surface points, voxel size 0.05 m, observed
  coverage estimate 0.08227, mean uncertainty 0.06575 m.
- `depth_pro` was importable, but no `ATLAS3R_DEPTH_PRO_CHECKPOINT` or explicit
  checkpoint URI was configured; no external pseudo-label run was faked.

## Compact Phase Ledger

- Skeleton through Phase 4: package skeleton, contracts, synthetic correctness,
  CPU TSDF diagnostics, dependency-safe adapter stubs, NumPy student boundary,
  optional Torch training MVP, checkpoint TSDF bridge, and TUM RGB-D debug
  train/eval.
- Phase 5A: canonical multi-view TUM clip cache plus tiny temporal center-depth
  and relative-translation training path.
- Phase 5B/5B.1: stable teacher-signal cache, measured forge, local ingest,
  source-clip inspection, JSON-only inspection, and deduped map replay.
- Phase 5C: dependency-safe external teacher runner boundary, Depth Pro
  bootstrap, VGGT local ingest scaffold, validated external signal caches.
- Phase 5D: teacher-signal temporal-v1 student training, weighted losses,
  checkpoint-to-teacher-cache export, and real measured TUM run/eval.

## Current Known Gaps

- No real Depth Pro or VGGT pseudo-labels were included because no configured
  checkpoint/output path was available.
- `TemporalMetricNetV1` is a small diagnostic model, not the final SMGT.
- Student exports reuse source clip poses; learned relative translations are
  diagnostic only.
- No live streaming runtime, object-aware mapping, GLB/PLY export, benchmark
  accuracy report, performance report, or realtime claim exists.
