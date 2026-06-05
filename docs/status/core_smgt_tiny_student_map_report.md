# Core SMGT Tiny Student Map Report

## Scope

Core Phase A implemented and tested a first learned diagnostic `SMGTTiny`
student. It trains from the Phase 6H VGGT teacher temporal cache and maps real
RGB input through learned depth/pose/confidence predictions without running
VGGT at student inference.

This is not final SMGT, not realtime evidence, not object-aware fusion, not
RGB-only production readiness, and not a benchmark accuracy report.

## Inputs

- Teacher cache:
  `runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache`
- Recording:
  `runs/phase6a_recording_freiburg1_xyz_val/recording`
- Sequence: TUM RGB-D `freiburg1_xyz`
- Device: `cuda:0`
- Torch: `2.1.0+cu121`

## Training Commands

Overfit/debug:

```bash
python -m atlas3r train smgt-tiny --teacher-cache runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache --output runs/core_smgt_tiny_overfit_freiburg1_xyz_val --steps 1000 --batch-size 2 --device cuda:0 --amp --debug-subset-clips 4 --debug-allow-pseudo-weight 1.0
```

Weighted pseudo-label training:

```bash
python -m atlas3r train smgt-tiny --teacher-cache runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache --output runs/core_smgt_tiny_weighted_freiburg1_xyz_val --steps 3000 --batch-size 4 --device cuda:0 --amp --val-split 0.2
```

## Training Results

| Run | Steps | First 100 loss | Final 100 loss | Depth first/final | Final depth AbsRel/RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overfit debug | 1000 | 0.042932 | -0.127154 | 0.007812 / 0.000163 | 0.012464 / 0.028668 m |
| Weighted | 3000 | 0.174906 | -0.065941 | 0.015680 / 0.000230 | 0.011509 / 0.024855 m |

The weighted run reported `loss_total_decrease_percent=137.70057892343985`.
Its `checkpoint_best.pt` was selected at step `1000` by validation loss.

## Student Mapping Command

```bash
python -m atlas3r runtime map-rgb-student --input runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/core_smgt_tiny_student_map_freiburg1_xyz_val --checkpoint runs/core_smgt_tiny_weighted_freiburg1_xyz_val/checkpoint_best.pt --device cuda:0 --max-frames 64 --frame-stride 1 --clip-length 8 --clip-overlap 4 --image-size 120x160 --voxel-size-m 0.05 --truncation-voxels 3.0 --pixel-stride 12 --export-mesh-chunks --mesh-format ply --rgb-only
```

## Mapping Results

- Frames used: 64 RGB frames.
- Prediction windows: 15.
- Sparse map updates: 64.
- Active sparse blocks/voxels: 90 / 11,962.
- Surface points: 4,596.
- Mesh chunks: 70.
- Mesh vertices/triangles: 35,944 / 17,972.
- VGGT used during student inference: false.
- Measured depth/pose used for mapping: false / false.
- Metric scale source: `student_rgb_prior_unverified`.

Eval-only measured recording diagnostics:

- Depth AbsRel/RMSE after median-scale alignment: 0.089889 / 0.304203 m.
- Constant-depth baseline AbsRel/RMSE: 0.215698 / 0.596336 m.
- Student beats constant-depth baseline: true.
- Sim3 camera-center ATE RMSE: 0.090504 m.
- No-motion pose baseline ATE RMSE: 0.146124 m.
- Student beats no-motion pose baseline: true.

## Artifacts

- `runs/core_smgt_tiny_overfit_freiburg1_xyz_val`
- `runs/core_smgt_tiny_weighted_freiburg1_xyz_val`
- `runs/core_smgt_tiny_student_map_freiburg1_xyz_val`

Generated run artifacts and checkpoints are local evidence and are not committed.

## Known Gaps

- Training labels are VGGT pseudo labels, not measured geometry labels.
- Metric scale is learned from an unverified RGB prior.
- SMGT-tiny is a diagnostic student, not the final SMGT architecture.
- No object-aware fusion, dynamic filtering, loop closure, global bundle
  adjustment, runtime benchmark, long-run memory proof, or accuracy report
  exists.
