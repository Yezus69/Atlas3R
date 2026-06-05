# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Current branch: `codex/core-smgt-tiny-student-map`.
- Core Phase A is implemented: a first learned diagnostic `SMGTTiny` student
  trains from the Phase 6H VGGT teacher temporal cache and maps real RGB-only
  input through sparse TSDF observed mesh chunks without VGGT at student
  inference.
- `atlas3r.models.smgt` contains the compact Conv/ConvGRU model, ray-channel
  geometry helpers, memory state, and checkpoint truth-boundary validation.
- `python -m atlas3r train smgt-tiny` writes config, train/val metrics,
  checkpoints, prediction previews, truth flags, and a Markdown report.
- `python -m atlas3r runtime map-rgb-student --rgb-only` loads an SMGT-tiny
  checkpoint, consumes RGB/intrinsics only, predicts depth/pose/confidence, and
  reuses sparse TSDF plus observed mesh chunk artifacts.
- Truth flags keep `teacher_geometry_used=false` during student inference,
  measured depth/pose false for mapping, `metric_scale_source` as unverified
  student RGB prior, and no final SMGT, RGB-only readiness, realtime,
  object-aware fusion, hidden geometry, accuracy, or millimeter claim.

## Latest Verified Test State

- Focused post-patch checks passed before long runs:
  `python -m ruff check src tests`, `python -m mypy src`,
  `python -m unittest tests.unit.test_smgt_tiny_losses`,
  `python -m unittest tests.unit.test_rgb_student_mapping`, and focused
  SMGT/checkpoint/dataset/CLI tests.
- Full verification is pending after the final documentation updates.

## Real-Data Evidence

- Teacher cache:
  `runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache`
  with 29 validated pseudo clips.
- Recording:
  `runs/phase6a_recording_freiburg1_xyz_val/recording`, TUM RGB-D
  `freiburg1_xyz`; measured depth/pose used only for eval sidecars.
- Overfit command: `python -m atlas3r train smgt-tiny --teacher-cache
  runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache
  --output runs/core_smgt_tiny_overfit_freiburg1_xyz_val --steps 1000
  --batch-size 2 --device cuda:0 --amp --debug-subset-clips 4
  --debug-allow-pseudo-weight 1.0`.
- Overfit result: completed 1000 steps; first/final 100-step `loss_total`
  means `0.042932 / -0.127154`; depth loss means
  `0.007812 / 0.000163`; final train depth AbsRel/RMSE
  `0.012464 / 0.028668 m`.
- Weighted command: `python -m atlas3r train smgt-tiny --teacher-cache
  runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache
  --output runs/core_smgt_tiny_weighted_freiburg1_xyz_val --steps 3000
  --batch-size 4 --device cuda:0 --amp --val-split 0.2`.
- Weighted result: completed 3000 steps; first/final 100-step `loss_total`
  means `0.174906 / -0.065941`; reported decrease `137.70%`; final train
  depth AbsRel/RMSE `0.011509 / 0.024855 m`; best checkpoint step `1000`.
- Student mapping command used the weighted `checkpoint_best.pt` on 64 RGB
  frames with `--clip-length 8 --clip-overlap 4 --image-size 120x160`.
- Student mapping result:
  `runs/core_smgt_tiny_student_map_freiburg1_xyz_val` wrote 70 observed mesh
  chunks, 35,944 vertices, 17,972 triangles, 4,596 surface points, 90 active
  sparse blocks, and 11,962 active voxels.
- Eval-only source measurements: depth AbsRel/RMSE
  `0.089889 / 0.304203 m` after median-scale alignment; constant-depth
  baseline AbsRel/RMSE `0.215698 / 0.596336 m`; Sim3 camera-center ATE RMSE
  `0.090504 m`; no-motion baseline ATE RMSE `0.146124 m`.
- Full evidence: `docs/status/core_smgt_tiny_student_map_report.md`.

## Current Known Gaps

- Training used VGGT teacher pseudo labels, not measured geometry labels.
- Metric scale is an unverified RGB prior learned from pseudo labels.
- The student checkpoint is diagnostic SMGT-tiny, not final SMGT.
- No object-aware fusion, dynamic filtering, loop closure, global optimization,
  realtime profile, long-run memory proof, or benchmark accuracy report exists.
