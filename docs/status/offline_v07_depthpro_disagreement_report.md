# Offline V0.7 Depth Pro Disagreement Report

Branch: `codex/offline-world-builder-v07-depthpro-disagreement`
Commit: `pending-final-commit`

## Commands Run

```bash
python -m atlas3r offline build-world \
  --input build/offline_v05_tiny_ppm_input \
  --output runs/offline_v07_debug_flat_depth \
  --max-frames 12 \
  --keyframe-stride 2 \
  --debug-geometry-mode flat-depth \
  --write-ply

python -m atlas3r offline build-world \
  --input data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb \
  --output runs/offline_v07_real_decode_no_teachers \
  --max-frames 60 \
  --keyframe-stride 5 \
  --keyframe-max-count 16 \
  --debug-geometry-mode none \
  --write-ply

python -m atlas3r offline build-world \
  --input data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb \
  --output runs/offline_v07_vggt_only \
  --max-frames 60 \
  --keyframe-stride 3 \
  --keyframe-max-count 24 \
  --enable-vggt \
  --vggt-device cuda:0 \
  --vggt-image-size 518 \
  --vggt-window-size 24 \
  --vggt-window-overlap 8 \
  --vggt-stitch-mode overlap-sim3 \
  --write-ply

python -m atlas3r offline build-world \
  --input data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb \
  --output runs/offline_v07_depthpro_only \
  --max-frames 60 \
  --keyframe-stride 3 \
  --keyframe-max-count 24 \
  --enable-depth-pro \
  --depth-pro-device cuda:0 \
  --depth-pro-checkpoint C:/Users/Asav/source/repos/homebrain/external/ml-depth-pro/checkpoints/depth_pro.pt \
  --write-ply

python -m atlas3r offline build-world \
  --input data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb \
  --output runs/offline_v07_vggt_depthpro_disagreement \
  --max-frames 60 \
  --keyframe-stride 3 \
  --keyframe-max-count 24 \
  --enable-vggt \
  --vggt-device cuda:0 \
  --vggt-image-size 518 \
  --vggt-window-size 24 \
  --vggt-window-overlap 8 \
  --vggt-stitch-mode overlap-sim3 \
  --enable-depth-pro \
  --depth-pro-device cuda:0 \
  --depth-pro-checkpoint C:/Users/Asav/source/repos/homebrain/external/ml-depth-pro/checkpoints/depth_pro.pt \
  --write-ply
```

## Evidence Summary

- Real input: `data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb`.
- Decoder used: Pillow for TUM PNG frames.
- Frames decoded: 60 for real decode and teacher runs.
- Keyframes selected: 16 without teachers, 24 for VGGT and Depth Pro runs.
- VGGT status: `running`; proposals: 24 cameras, 24 depths, 1 window.
- Depth Pro status: `running`; proposals: 24 cameras, 24 depths, 24 frames.
- Depth Pro checkpoint: external path under `homebrain/external/ml-depth-pro`;
  no weights were added to this repo.
- Disagreement status: `available`; valid overlap count: 7,372,800 pixels.
- Disagreement mean abs depth diff: 0.0996045 m; p50: 0.0739858 m;
  p95: 0.2574210 m.
- Mean relative diff: 0.0878645; p95 relative diff: 0.1971949.
- Mean log-depth diff: 0.0953817; high-disagreement ratio: 0.0313700.
- Geometry preview: 18,432 points from
  `vggt_pose_consensus_depth_diagnostic`.
- PLY produced:
  `runs/offline_v07_vggt_depthpro_disagreement/geometry/geometry_preview.ply`.
- Consensus preview: diagnostic only, not optimized.

## Evidence Runs

- A: `runs/offline_v07_debug_flat_depth` decoded 6 PPM frames with
  `stdlib_ppm`, selected 6 keyframes, wrote 288 debug points and PLY.
- B: `runs/offline_v07_real_decode_no_teachers` decoded 60 PNG frames with
  Pillow, selected 16 keyframes, and correctly wrote 0 geometry points.
- C: `runs/offline_v07_vggt_only` decoded 60 PNG frames, selected 24
  keyframes, wrote 24 VGGT camera/depth proposals, 19,800 points, and PLY.
- D: `runs/offline_v07_depthpro_only` decoded 60 PNG frames, selected 24
  keyframes, wrote 24 Depth Pro camera/depth proposals, and wrote 0 global
  geometry because Depth Pro does not provide global pose.
- E: `runs/offline_v07_vggt_depthpro_disagreement` wrote both witness streams,
  disagreement JSON/NPZ, diagnostic consensus preview, 18,432 points, and PLY.

## Truth Boundary

Physical accuracy claim: no. Training-quality claim: no.

VGGT and Depth Pro outputs are `teacher_pseudo`, unanchored, observed-only
proposal geometry. The consensus preview is diagnostic and not optimized. There
is still no measured scale anchor, calibrated capture, measured depth, external
pose truth, render-repair optimizer, consensus optimizer, or named evaluation
report.

## Remaining Failure Points

- No physical scale anchor or evaluation report exists.
- Depth Pro-only runs cannot create a global world preview without VGGT or other
  pose proposals.
- Render diagnostics remain projection placeholders, not a repair optimizer.
- Object permanence has no SAM/DINO masks, features, or object tracks.
