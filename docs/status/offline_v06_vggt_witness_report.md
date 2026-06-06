# Offline V0.6 VGGT Witness Report

Branch: `codex/offline-world-builder-v06-vggt-witness`
Commit: `pending-final-commit`

## Commands Run

```bash
python -m atlas3r offline build-world --input build/offline_v05_tiny_ppm_input --output runs/offline_v06_debug_flat_depth --max-frames 12 --keyframe-stride 2 --debug-geometry-mode flat-depth --write-ply
python -m atlas3r offline build-world --input data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb --output runs/offline_v06_real_decode_no_vggt --max-frames 60 --keyframe-stride 5 --keyframe-max-count 16 --debug-geometry-mode none --write-ply
python -m atlas3r offline build-world --input data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb --output runs/offline_v06_vggt_real_geometry --max-frames 60 --keyframe-stride 3 --keyframe-max-count 24 --enable-vggt --vggt-device cuda:0 --vggt-image-size 518 --vggt-window-size 24 --vggt-window-overlap 8 --vggt-stitch-mode overlap-sim3 --write-ply
```

## Evidence Summary

- Real input: `data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb`.
- Decoder used: Pillow for PNG frames.
- Frames decoded: 60 for real decode runs.
- Keyframes selected: 16 without VGGT, 24 with VGGT.
- VGGT status: `running`.
- VGGT model/checkpoint: `facebook/VGGT-1B`.
- VGGT device/settings: `cuda:0`, image size 518, window size 24, overlap 8.
- Proposal counts: 24 VGGT cameras, 24 VGGT depths, 1 VGGT window.
- Geometry point count: 19,800.
- PLY path: `runs/offline_v06_vggt_real_geometry/geometry/geometry_preview.ply`.
- Render diagnostics: partial projection/coverage placeholder from teacher geometry.
- Quality verdict: teacher-proposed geometry is available, not physically accurate.
- Training cache: references geometry/proposals but `usable_for_training: false`.

## Stitching Stats

- `vggt_window_count`: 1
- `stitch_mode`: `overlap-sim3`
- `accepted_edge_count`: 0
- `rejected_edge_count`: 0
- `pseudo_submap_count`: 1
- `overlap_center_rmse_m`: null
- `scale_min/median/max`: 1 / 1 / 1

## Truth Boundary

VGGT output is `teacher_pseudo` geometry with
`metric_scale_source: vggt_unanchored_metric_proposal`. It is not measured
depth, not measured pose, not physically accurate, and not training-quality.
There is no measured scale anchor, calibration target, measured depth, external
pose, optimizer result, or named evaluation report.

## Remaining Failure Points

- Other real witnesses remain unavailable: Depth Pro, MapAnything, LingBot-Map,
  SAM/DINO, CoTracker, and COLMAP/GLOMAP.
- Object permanence has no SAM/DINO masks or features.
- Render diagnostics are placeholders, not a renderer or repair optimizer.
- VGGT cache/replay files and generated runs are not committed.
