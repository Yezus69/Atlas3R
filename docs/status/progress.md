# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and reports.

## Current State

- Current phase: Phase 5G multi-sequence measured TUM pose/depth supervision and
  learned SE(3) student-odometry runtime diagnostics completed.
- Branch: `codex/phase5g-multisequence-pose-odometry`.
- Latest implementation: multi-sequence TUM specs plus URL override validation,
  explicit teacher-signal pose metadata, `TemporalMetricNetV1` 6D relative
  rotation output, SE(3) losses/metrics, CUDA AMP-stable rotation loss, and
  `runtime stream-student-map --pose-mode student-odometry` with TUM trajectory
  and pose-quality reports.
- Public commands include `atlas3r datasets tum-rgbd download|prepare`,
  `atlas3r train teacher-signals-temporal`, and
  `atlas3r runtime stream-student-map --pose-mode oracle|student-relative|student-odometry|both`.

## Latest Verified Test State

- `python -m ruff format src tests`: passed with 147 files unchanged.
- `python -m ruff format --check src tests`: passed with 147 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 110 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 197 tests. A
  pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `where.exe make`: no `make` found in this Windows shell, so `make test`,
  `make lint`, and `make typecheck` were not run.

## Real-Data Evidence

- Optional external pose-teacher env vars were unset:
  `ATLAS3R_VGGT_REPO`, `ATLAS3R_VGGT_CHECKPOINT`, `ATLAS3R_VGGT_OUTPUT`,
  `ATLAS3R_LINGBOT_MAP_REPO`, and `ATLAS3R_LINGBOT_MAP_OUTPUT`.
- Only `freiburg1_xyz` local TUM caches were present. Additional specs for
  `freiburg1_desk`, `freiburg2_xyz`, and
  `freiburg3_long_office_household` are implemented and unit-tested, but no
  extra local sequence was trained.
- Measured train/val caches: 330 train signals and 56 validation signals, clip
  length 5, image size 160x120.
- The first Phase 5G AMP run was stopped because the rotation loss produced
  nonfinite AMP gradients and `GradScaler` skipped optimizer steps. The loss was
  fixed with float32 loss math plus a stable `atan2` geodesic rotation angle, and
  a CUDA AMP regression test now checks that weights update.
- Final training: 20,000 steps on CUDA with AMP, completed normally. Best
  checkpoint was step 19,000 with validation RMSE `0.088052159` m, MAE
  `0.054015226` m, AbsRel `0.049743026`, relative translation mean
  `0.019125509` m, relative rotation mean `0.729621351` deg, ATE-like center
  mean `0.024512752` m, RPE-like translation mean `0.013363804` m, and
  RPE-like rotation mean `0.629176323` deg.
- Runtime over 60 validation frames with `checkpoint_best.pt`:
  - Oracle pose: depth RMSE `0.090435066` m, AbsRel `0.049121839`, 8,078
    surface points, coverage `0.088784`, pose ATE mean `0.0` m.
  - Student-odometry: same depth metrics, 12,075 surface points, coverage
    `0.117054`, pose ATE mean `0.160803384` m, rotation mean `5.388367751` deg,
    RPE translation mean `0.015672370` m, RPE rotation mean `0.589294296` deg.
- Runtime latency diagnostics: oracle model inference mean/p50/p95
  `10.532/4.390/5.025` ms; student-odometry `11.383/4.440/5.576` ms. These are
  diagnostics with warmup included, not performance claims.
- Full evidence: `docs/status/phase5g_multisequence_pose_odometry_report.md`.

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
- Phase 5F: real Depth Pro pseudo-label cache generation, mixed measured+pseudo
  training, student export, and runtime comparison.
- Phase 5G: measured SE(3) pose supervision, 6D rotation head, SE(3) metrics,
  and diagnostic student-odometry runtime.

## Current Known Gaps

- No realtime scheduler, live camera loop, bounded GPU TSDF, object-aware
  mapping, triangle mesh extraction, or benchmark accuracy report exists.
- `student-odometry` is not mapping-ready; 60-frame rollout had 0.160803 m mean
  camera-center drift and 5.388 deg mean absolute rotation error.
- Multi-sequence TUM code exists, but only `freiburg1_xyz` was locally available
  for the real training/evaluation run.
