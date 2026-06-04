# Phase 5G Multi-Sequence Pose Odometry Report

Branch: `codex/phase5g-multisequence-pose-odometry`

Base branch: `codex/phase5f-real-depthpro-mixed-training-runtime`

## Result

Phase 5G added a full SE(3) diagnostic student-odometry path and trained it on
real measured TUM RGB-D depth plus `T_world_camera` supervision. The runtime can
now run `--pose-mode student-odometry`, anchor only the first frame to measured
source pose, roll out learned relative SE(3), and write pose/depth/map/latency
diagnostics.

The real local evidence run used only the existing `freiburg1_xyz` caches. No
second TUM sequence was present locally, and no external VGGT/LingBot pose
teacher output was configured, so multi-sequence support is implemented and
unit-tested but not yet backed by a multi-sequence local training run.

Student-odometry is useful diagnostic evidence but not mapping-ready. It reduces
oracle-pose dependence for the runtime path, but absolute rollout drift is still
large over 60 validation frames.

## Code Changes

- Added verified TUM sequence specs for `freiburg1_xyz`, `freiburg1_desk`,
  `freiburg2_xyz`, and `freiburg3_long_office_household`, plus paired
  `--archive-url` / `--groundtruth-url` override validation.
- Hardened teacher-signal pose metadata with optional `pose_source` and
  `pose_confidence`; measured TUM caches remain measured, not pseudo/external.
- Extended `TemporalMetricNetV1` with
  `relative_rotation_6d_center_from_camera B,T,6` while preserving old
  checkpoint loading for depth and `student-relative`.
- Added SE(3) losses and metrics: translation, 6D-to-SO(3) geodesic rotation,
  ATE-like rollout, and RPE-like relative transform diagnostics.
- Fixed CUDA AMP training for the rotation loss by using a stable `atan2`
  geodesic angle and float32 loss math. The first attempted AMP run skipped all
  optimizer steps; it was stopped, deleted, fixed, and rerun.
- Added `student-odometry` runtime mode and pose report artifacts:
  `pose_quality_report.json`, `per_frame_pose_quality.jsonl`,
  `trajectory_estimate_tum.txt`, and `trajectory_groundtruth_tum.txt`.
- Split runtime report helpers so source files stay below the repo size target.

## External Pose Teacher Preflight

All optional external pose-teacher locations were unavailable in the local
environment:

| Variable | Status |
| --- | --- |
| `ATLAS3R_VGGT_REPO` | unset |
| `ATLAS3R_VGGT_CHECKPOINT` | unset |
| `ATLAS3R_VGGT_OUTPUT` | unset |
| `ATLAS3R_LINGBOT_MAP_REPO` | unset |
| `ATLAS3R_LINGBOT_MAP_OUTPUT` | unset |

No external pose teacher outputs were fabricated.

## TUM Data

TUM sequence specs were checked against the official TUM RGB-D dataset download
and file-format pages. TUM files use 640x480 RGB/depth PNGs, `depth_raw/5000.0`,
and ground-truth lines `timestamp tx ty tz qx qy qz qw`; the project continues
to use ROS default intrinsics for the PNG path.

Local real-data artifacts used:

| Artifact | Count |
| --- | ---: |
| `data/tum_rgbd/freiburg1_xyz_clip_cache_train` | 330 clips |
| `data/tum_rgbd/freiburg1_xyz_clip_cache_val` | 56 clips |
| `data/tum_rgbd/freiburg1_xyz_measured_teacher_train` | 330 signals |
| `data/tum_rgbd/freiburg1_xyz_measured_teacher_val` | 56 signals |

No local `freiburg1_desk`, `freiburg2_xyz`, or
`freiburg3_long_office_household` cache was available for the real run.

## Training

Command shape:

```bash
python -m atlas3r train teacher-signals-temporal \
  --teacher-cache data/tum_rgbd/freiburg1_xyz_measured_teacher_train \
  --val-teacher-cache data/tum_rgbd/freiburg1_xyz_measured_teacher_val \
  --output runs/phase5g_se3_temporal_tum \
  --steps 20000 --batch-size 8 --device cuda --num-workers 0 \
  --learning-rate 0.00025 --log-every 50 --val-every 500 \
  --checkpoint-every 1000 --preview-every 1000 --amp \
  --max-runtime-minutes 330 --hidden-channels 24 --bottleneck-channels 32 \
  --measured-teacher-weight 1.0 --pseudo-teacher-weight 0.25 \
  --relative-translation-weight 1.0 --relative-rotation-weight 0.1 \
  --se3-pose-weight 1.0
```

Run summary:

| Metric | Value |
| --- | ---: |
| steps completed | 20,000 |
| stopped reason | `completed_steps` |
| device / AMP | `cuda` / true |
| train records | 330 measured, 0 pseudo |
| val records | 56 measured, 0 pseudo |
| best checkpoint step | 19,000 |

Best validation at step 19,000:

| Metric | Value |
| --- | ---: |
| depth RMSE / MAE / AbsRel | 0.088052 m / 0.054015 m / 0.049743 |
| relative translation mean / median / p95 | 0.019126 m / 0.016452 m / 0.036550 m |
| relative rotation mean / median / p95 | 0.729621 deg / 0.648436 deg / 1.640267 deg |
| ATE-like center mean / median / p95 | 0.024513 m / 0.023180 m / 0.053305 m |
| RPE-like translation mean / median / p95 | 0.013364 m / 0.012613 m / 0.019870 m |
| RPE-like rotation mean / median / p95 | 0.629176 deg / 0.569668 deg / 1.119271 deg |

Final validation at step 20,000 regressed slightly by depth RMSE to
0.101782 m, so runtime used `checkpoint_best.pt` from step 19,000.

## Runtime

Commands:

```bash
python -m atlas3r runtime stream-student-map \
  --checkpoint runs/phase5g_se3_temporal_tum/checkpoint_best.pt \
  --clip-cache data/tum_rgbd/freiburg1_xyz_clip_cache_val \
  --teacher-cache data/tum_rgbd/freiburg1_xyz_measured_teacher_val \
  --output runs/phase5g_stream_oracle \
  --pose-mode oracle --max-frames 60 --device cuda

python -m atlas3r runtime stream-student-map \
  --checkpoint runs/phase5g_se3_temporal_tum/checkpoint_best.pt \
  --clip-cache data/tum_rgbd/freiburg1_xyz_clip_cache_val \
  --teacher-cache data/tum_rgbd/freiburg1_xyz_measured_teacher_val \
  --output runs/phase5g_stream_student_odometry \
  --pose-mode student-odometry --max-frames 60 --device cuda
```

Both runs used the same 60 validation frames. Depth is identical because pose
mode changes the fused pose, not the per-frame depth prediction.

| Runtime mode | Depth RMSE | AbsRel | Pose ATE mean | Pose rot mean | RPE trans mean | RPE rot mean | Surface points | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | 0.090435 m | 0.049122 | 0.000000 m | 0.006384 deg | 0.000000 m | 0.007296 deg | 8,078 | 0.088784 |
| student-odometry | 0.090435 m | 0.049122 | 0.160803 m | 5.388368 deg | 0.015672 m | 0.589294 deg | 12,075 | 0.117054 |

Latency diagnostics, not a performance report:

| Runtime mode | Inference mean / p50 / p95 | Total pipeline |
| --- | ---: | ---: |
| oracle | 10.532 ms / 4.390 ms / 5.025 ms | 6.895 s total folder run |
| student-odometry | 11.383 ms / 4.440 ms / 5.576 ms | 6.943 s total folder run |

The first inference sample includes warmup, so p50/p95 are more representative
than the mean for steady-state model calls. No realtime claim is made.

## Conclusion

Phase 5G improved the evidence loop: Atlas3R now has measured TUM SE(3)
supervision, a rotation-capable temporal student, pose losses/metrics, and a
student-odometry runtime mode with TUM trajectory reports.

Training improved measured validation depth versus Phase 5F and learned useful
relative translation. Rotation learning remained weak, and student-odometry
rollout drifted to 0.160803 m mean / 0.241581 m p95 camera-center error over 60
frames. The denser student-odometry TSDF is diagnostic only and not proof of a
better map because pose drift can smear surfaces.

## Verification

Final verification commands are recorded in `docs/status/progress.md`. All
Phase 5G outputs remain diagnostic only: no benchmark accuracy, realtime,
mapping-ready, or millimeter-accuracy claim is made.

## Next Blocker

The next blocker is measured 3D scene/object dataset ingestion and mesh/object
evaluation. Another depth-only pseudo-label phase is lower leverage than adding
measured mesh/object supervision and evaluating geometry/object outputs.
