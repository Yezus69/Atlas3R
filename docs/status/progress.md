# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Current branch: `codex/core-smgt-small-v2-measured-pseudo`.
- Core Phase A3 added measured temporal caches, SMGT-small-v2, mixed
  measured/pseudo training, validation-quality checkpoint selection,
  confidence/sigma calibration, and `runtime map-rgb-student-v2`.
- A3 passed the local Freiburg heldout diagnostic gates with
  `checkpoint_best.pt` selected at step 1500, 32 heldout mesh chunks, calibrated
  mapped-pixel ratio `0.3112`, depth beating constant baseline, and pose beating
  no-motion baseline.
- `SMGTTiny` remains a falsified diagnostic/toy baseline from A2. The active
  diagnostic student is now `SMGTSmallV2`, still not final SMGT or RGB-only
  readiness.
- Truth flags keep teacher geometry and measured depth/pose false during student
  mapping, `metric_scale_source=student_rgb_prior_unverified`, and no realtime,
  accuracy, millimeter, object-aware fusion, hidden-geometry, or final-SMGT claim.

## Latest Verified Test State

- Final A3 verification passed on 2026-06-05:
  `python -m ruff format src tests`, `python -m ruff format --check src tests`,
  `python -m ruff check src tests`, `python -m mypy src`,
  `python -m unittest discover -s tests -p "test_*.py"`, focused A3 unit tests,
  `git diff --check`, and the five Makefile smoke commands run directly.
- `make smoke` itself failed because `make` is not installed on this Windows host
  (`'make' is not recognized...`).

## A3 Real-Data Evidence

- Measured train caches: `runs/a3_cache_measured_freiburg1_xyz_train` and
  `runs/a3_cache_measured_freiburg2_xyz_train`, each 149 clips from 600 measured
  frames.
- Heldout measured validation cache: `runs/a3_cache_measured_freiburg1_xyz_val`,
  29 clips from 120 measured frames.
- Pseudo cache:
  `runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache`
  with 29 validated pseudo clips, target weight capped at 0.25.
- Training run:
  `runs/core_smgt_small_v2_train_freiburg1_freiburg2`, 5000 steps, 1773.03 s.
- Best validation step 1500: depth AbsRel `0.062074` vs constant baseline
  `0.183416`; pose center ratio `0.696813` vs no-motion.
- Calibration: threshold `0.955196`, max sigma `0.027155 m`, mapped ratio
  `0.409796`, mapped AbsRel lower than rejected AbsRel.
- Heldout 120-frame RGB-only map: 32 mesh chunks, 14,316 vertices, 7,158
  triangles, mapped ratio `0.311182`, active blocks/voxels `38 / 4,826`.
- Full evidence: `docs/status/core_smgt_small_v2_measured_pseudo_report.md`.

## Current Known Gaps

- A3 is local diagnostic evidence on Freiburg splits, not broad validation.
- Runtime profiling/memory is sampled externally, not first-class instrumentation.
- Metric scale remains an unverified RGB prior despite measured training labels.
- No object-aware fusion, dynamic filtering, loop closure, global optimization,
  realtime proof, benchmark accuracy report, or millimeter claim exists.
