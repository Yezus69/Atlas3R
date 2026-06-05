# Core SMGT Small V2 Measured/Pseudo Report

Branch: `codex/core-smgt-small-v2-measured-pseudo`

## Verdict

A3 passed the diagnostic gates on the local TUM Freiburg evidence run. This does
not make SMGT-small-v2 final SMGT, realtime-ready, object-aware, benchmark
accurate, or millimeter accurate. Mapping remains observed-only with unverified
RGB-prior metric scale; measured depth/pose are used for training/eval only.

## Artifacts

- Measured train caches:
  `runs/a3_cache_measured_freiburg1_xyz_train` and
  `runs/a3_cache_measured_freiburg2_xyz_train`, each 149 clips from 600 measured
  RGB-D/pose train frames.
- Heldout validation cache:
  `runs/a3_cache_measured_freiburg1_xyz_val`, 29 clips from 120 measured val
  frames.
- Pseudo cache:
  `runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache`,
  29 teacher-pseudo clips, weight capped at 0.25.
- Training run: `runs/core_smgt_small_v2_train_freiburg1_freiburg2`.
- Calibration: `runs/core_smgt_small_v2_calibration`.
- Heldout map: `runs/core_smgt_small_v2_student_map_heldout_final`.
- Long-run/profile map: `runs/core_smgt_small_v2_student_map_profiled2`.

## Commands

```bash
python -m atlas3r recording from-tum --manifest data/tum_rgbd/freiburg1_xyz_phase5g1_manifest_block.json --output runs/a3_recording_freiburg1_xyz_train/recording --split train --max-frames 600 --width 224 --height 160
python -m atlas3r train build-measured-temporal-cache --recording runs/a3_recording_freiburg1_xyz_train/recording --output runs/a3_cache_measured_freiburg1_xyz_train --clip-length 8 --clip-stride 4 --image-size 160x224
python -m atlas3r train smgt-v2 --measured-cache runs/a3_cache_measured_freiburg1_xyz_train --measured-cache runs/a3_cache_measured_freiburg2_xyz_train --pseudo-cache runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache --val-source runs/a3_cache_measured_freiburg1_xyz_val --output runs/core_smgt_small_v2_train_freiburg1_freiburg2 --steps 5000 --batch-size 4 --device cuda:0 --amp --save-every 500 --num-workers 0
python -m atlas3r eval smgt-v2-calibrate-gate --checkpoint runs/core_smgt_small_v2_train_freiburg1_freiburg2/checkpoint_best.pt --cache runs/a3_cache_measured_freiburg1_xyz_val --output runs/core_smgt_small_v2_calibration --device cuda:0 --batch-size 2
python -m atlas3r runtime map-rgb-student-v2 --input runs/phase6a_recording_freiburg1_xyz_val/recording --output runs/core_smgt_small_v2_student_map_heldout_final --checkpoint runs/core_smgt_small_v2_train_freiburg1_freiburg2/checkpoint_best.pt --device cuda:0 --max-frames 120 --frame-stride 1 --clip-length 8 --clip-overlap 4 --image-size 160x224 --voxel-size-m 0.05 --truncation-voxels 3.0 --pixel-stride 12 --student-map-valid-policy confidence_sigma --export-mesh-chunks --mesh-format ply --rgb-only --calibration runs/core_smgt_small_v2_calibration/calibration.json
```

## Training Result

- Completed 5000 steps in 1773.03 s.
- Validation-selected checkpoint: `checkpoint_best.pt`, step 1500,
  selection score `0.528034`; not step 1 and not final step.
- Best validation depth AbsRel/RMSE: `0.062074 / 0.108406 m`.
- Constant-depth baseline AbsRel: `0.183416`; ratio `0.385180`.
- Best validation pose center mean: `0.010209 m`; no-motion baseline
  `0.018763 m`; ratio `0.696813`.
- Best uncalibrated mapped-pixel ratio at loss thresholds: `0.737294`; final
  runtime uses calibrated gates below.

## Calibration

- Confidence threshold: `0.955196`; max sigma: `0.027155 m`.
- Mapped-pixel ratio: `0.409796` within target `[0.10, 0.70]`.
- Mapped AbsRel `0.047359` was lower than rejected AbsRel `0.066141`.
- Selected high-error rate `0.220758` was below random high-error rate
  `0.250000`.

## Heldout Mapping

- Checkpoint step: 1500; input frames: 120; RGB-only inference, no VGGT.
- Mesh chunks/vertices/triangles: `32 / 14316 / 7158`.
- Mapped pixel ratio: `0.311182`, materially below all-positive `1.0`.
- Active blocks/voxels: `38 / 4826`; surface points: `1505`.
- Eval-only depth AbsRel/RMSE: `0.032505 / 0.101970 m`; constant-depth
  baseline AbsRel/RMSE: `0.184860 / 0.479466 m`.
- Eval-only Sim3 pose ATE RMSE: `0.024589 m`; no-motion ATE RMSE:
  `0.125669 m`.
- Truth flags keep `measured_depth_used=false`, `measured_pose_used=false`,
  `teacher_geometry_used=false`, `final_smgt=false`, `realtime_claim=false`, and
  `rgb_only_mapping_ready=false` for mapping.

## Profile Sample

- Profiled 120-frame map output: `runs/core_smgt_small_v2_student_map_profiled2`.
- Total pipeline: `47.8 s`.
- Student inference windows: 29; mean/p50/p95 `150.34 / 126.50 / 183.42 ms`.
- Map update mean/p95: `25.07 / 29.36 ms`; mesh update mean/p95:
  `53.92 / 64.56 ms`.
- External Windows sampler peak RSS: `1435.57 MiB`.
- GPU0 total-memory-used sampler baseline/peak/delta: `1666 / 1688 / 22 MiB`;
  treat this as diagnostic only, not a rigorous GPU peak report.

## Gate Table

| Gate | Result |
| --- | --- |
| Validation-selected checkpoint is not step 1 | Pass: step 1500 |
| Checkpoint produces nonzero heldout mesh chunks | Pass: 32 chunks |
| Heldout depth AbsRel <= 0.20 and 25% better than constant baseline | Pass: `0.032505` eval-only AbsRel and validation ratio `0.385180` |
| Pose beats no-motion by at least 15% | Pass: validation ratio `0.696813`; eval-only ATE also beats no-motion |
| Mapped-pixel ratio below all-positive and calibrated | Pass: `0.311182` runtime, calibration target `0.409796` |
| Mapped pixels lower error than rejected | Pass: `0.047359 < 0.066141` |
| Mesh vertices/triangles nonzero | Pass: `14316 / 7158` |
| Long-run has no explosive growth or nonfinite output | Pass on 120-frame run; 38 active blocks, 4826 active voxels |

## Next Step

Run Core Phase A4: broaden student validation and first-class profiling before
starting object/dynamic fusion.
