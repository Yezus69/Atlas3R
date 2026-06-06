# Active Task - RoomGraph Core Optimizer

## Goal

Build a real RoomGraph optimizer for the actual
`room_walk_001/frames` run, using existing VGGT and Depth Pro proposals plus
real track evidence, and export `world_map_roomgraph/` artifacts.

## Baseline Inspection

- Baseline command run: `python -m atlas3r offline build-world --input C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames --output runs/room_walk_001_roomgraph_baseline --max-frames 240 --keyframe-stride 3 --keyframe-max-count 64 --scale-mode unanchored-soft-metric --enable-vggt --vggt-device cuda:0 --vggt-image-size 384 --vggt-window-size 24 --vggt-window-overlap 8 --vggt-stitch-mode overlap-sim3 --enable-depth-pro --depth-pro-device cuda:0 --depth-pro-checkpoint C:/Users/Asav/source/repos/homebrain/external/ml-depth-pro/checkpoints/depth_pro.pt --enable-colmap --colmap-exe colmap --colmap-matcher sequential --colmap-use-gpu 1 --colmap-camera-model SIMPLE_RADIAL --colmap-max-images 96 --export-world-map --map-depth-source consensus --map-point-stride 8 --map-min-confidence 0.20 --map-max-relative-disagreement 0.35 --map-voxel-size-m 0.05 --map-write-occupancy --map-write-observed-mesh --optimize-map-consistency --optimizer-max-iterations 5 --optimizer-cross-view-pairs 3 --export-optimized-world-map --export-best-world-map --write-ply`.
- Best map exists and is inspectable: 1,904,976 points, 2,559 voxels,
  5,600 observed mesh triangles, and 88 camera poses.
- Numeric failure: trajectory bbox is only about
  0.48 m x 0.32 m x 0.47 m, trajectory length is about 1.48 m, and map bbox is
  only about 1.72 m x 0.75 m x 0.96 m. The room reconstruction is compact /
  collapsed relative to a room walk.
- Teacher disagreement improved in the existing optimizer
  (`rel_diff_mean 0.308 -> 0.061`, cross-view residual
  `0.0498 m -> 0.0197 m`), but that pass did not optimize camera poses or real
  cross-frame tracks.
- Proposal cache has no point-track teacher output. COLMAP is unavailable on
  PATH and registered zero images/points.

## Checklist

- [x] Start from V1.1 branch and create `codex/roomgraph-core-optimizer`.
- [x] Run current best room command and inspect baseline artifacts.
- [x] Check VGGT outputs for usable track/point-map correspondences.
- [x] Add a real track source path, using OpenCV tracks only if no learned
  tracker can run.
- [x] Implement RoomGraph variants: depth-only, pose-only, and joint.
- [x] Export `world_map_roomgraph/` with metrics and concise report.
- [x] Run the actual room command and pick the best variant by metrics.
- [x] Add/update focused tests and run relevant verification.
- [x] Update compact status docs and commit source/tests/docs only.

## RoomGraph Result

- Actual run used cached VGGT/Depth Pro proposals plus OpenCV LK tracks because
  the CoTracker-backed command ran too long for the full room evidence pass.
- Output directory: `runs/room_walk_001_roomgraph/world_map_roomgraph/`.
- Track source: `opencv_lk`; selected variant: `pose_only`.
- Real tracked reprojection improved from `53.11 px` to `34.16 px`
  (`+35.7%`) and inlier ratio improved from `0.165` to `0.386`.
- Collapse/scale did not improve: camera collapse score changed from
  `0.347 m` to `0.299 m`, and cross-view depth residual changed from
  `0.0203 m` to `0.0236 m`.
- Truth boundary remains unanchored teacher/optimizer geometry with no physical
  accuracy or training-quality claim.
