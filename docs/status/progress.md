# Progress Summary

This file is a rolling current-state summary, not an append-only transcript.
Detailed history belongs in git commits, tests, and reports.

## Current State

- Current phase: Phase 5F real Depth Pro pseudo-label generation and
  measured+pseudo teacher-signal training completed as a diagnostic vertical
  slice; Phase 5G pose-teacher work is next.
- Branch: `codex/phase5f-real-depthpro-mixed-training-runtime`.
- Base commit before Phase 5F edits: `499deb3`.
- Latest implementation: hardened real `atlas3r teachers run-depth-pro` with
  `--device`, selected-device model/input movement, installed Apple Depth Pro
  focal-length tensor compatibility, prediction resizing, unique-frame
  prediction reuse, and run-count metadata; fixed mixed teacher-signal batches
  where only some caches contain pointmaps.
- Public commands still include `atlas3r teachers run-depth-pro`,
  `atlas3r train teacher-signals-temporal`, and
  `atlas3r runtime stream-student-map`.

## Latest Verified Test State

- `python -m ruff format src tests`: passed with 145 files unchanged.
- `python -m ruff format --check src tests`: passed with 145 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 108 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 189 tests. A
  pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `where.exe make`: no `make` found in this Windows shell, so `make test`,
  `make lint`, and `make typecheck` were not run.
- CLI help passed for `teachers run-depth-pro`, `train teacher-signals-temporal`,
  and `runtime stream-student-map`.

## Real-Data Evidence

- Required local artifacts existed: TUM train/val clip caches, measured
  train/val teacher caches, and
  `runs/phase5d_teacher_temporal_v1_tum/checkpoint_best.pt`.
- Real `depth_pro` was importable from external local checkout
  `C:\Users\Asav\source\repos\slam\external\ml-depth-pro`; checkpoint used was
  that checkout's external `checkpoints/depth_pro.pt`.
- Depth Pro validation cache: 56 clips, 280 frame slots, 60 unique predictions,
  220 duplicate reuses, 0 resizes; inspection RMSE `0.167579472` m, MAE
  `0.122303924` m, AbsRel `0.102098948`, valid overlap `74.452976%`,
  confidence mean `0.5`.
- Depth Pro train cache: 256 clips, 1,280 frame slots, 264 unique predictions,
  1,016 duplicate reuses, 0 resizes.
- Mixed training completed 20,000 steps with best validation RMSE
  `0.132907202` m, MAE `0.092814473` m, AbsRel `0.087075680` over measured plus
  Depth Pro pseudo validation records.
- Student export inspection over measured validation: RMSE `0.091704673` m,
  MAE `0.053628419` m, AbsRel `0.049205437`.
- Phase 5F streaming runtime, oracle pose: RMSE `0.094295488` m, MAE
  `0.054796543` m, AbsRel `0.050120890`, `8,892` surface points, coverage
  `0.094852`.
- Phase 5F streaming runtime, student-relative pose: same depth metrics,
  `9,329` surface points, coverage `0.097579`; pose remains diagnostic only.
- Compared with Phase 5E measured-only runtime RMSE `0.090826432` m, Phase 5F
  slightly regressed measured depth quality but produced denser diagnostic maps.
- Full evidence: `docs/status/phase5f_real_depthpro_mixed_training_report.md`.

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

## Current Known Gaps

- No realtime scheduler, live camera loop, bounded GPU TSDF, object-aware
  mapping, triangle mesh extraction, or benchmark accuracy report exists.
- `student-relative` pose is not mapping-ready; it keeps source rotation and
  applies diagnostic predicted translation only.
- Phase 5F showed real Depth Pro integration is available, but single-frame
  depth pseudo-label mixing at weight `0.15` did not improve measured validation
  or runtime depth diagnostics over Phase 5E.
