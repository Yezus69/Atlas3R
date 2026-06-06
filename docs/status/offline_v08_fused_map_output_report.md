# Offline V0.8 Fused Map Output Report

Branch: `codex/offline-world-builder-v08-fused-map-output`
Commit: `pending-final-commit`

## Commands Run

```bash
python -m atlas3r offline build-world \
  --input build/offline_v05_tiny_ppm_input \
  --output runs/offline_v08_debug_flat_depth_map \
  --max-frames 12 \
  --keyframe-stride 2 \
  --debug-geometry-mode flat-depth \
  --export-world-map \
  --map-write-occupancy \
  --map-write-observed-mesh \
  --write-ply

python -m atlas3r offline build-world \
  --input data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb \
  --output runs/offline_v08_vggt_world_map \
  --max-frames 60 \
  --keyframe-stride 3 \
  --keyframe-max-count 24 \
  --enable-vggt \
  --vggt-device cuda:0 \
  --vggt-image-size 518 \
  --vggt-window-size 24 \
  --vggt-window-overlap 8 \
  --vggt-stitch-mode overlap-sim3 \
  --export-world-map \
  --map-depth-source vggt \
  --map-point-stride 8 \
  --map-write-occupancy \
  --map-write-observed-mesh \
  --write-ply

python -m atlas3r offline build-world \
  --input data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb \
  --output runs/offline_v08_consensus_world_map \
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
  --export-world-map \
  --map-depth-source consensus \
  --map-point-stride 8 \
  --map-min-confidence 0.25 \
  --map-max-relative-disagreement 0.25 \
  --map-voxel-size-m 0.05 \
  --map-write-occupancy \
  --map-write-observed-mesh \
  --write-ply
```

## Evidence Summary

- Real input: `data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb`.
- Frames decoded: 60 for VGGT-only and consensus runs.
- Keyframes selected: 24 for VGGT-only and consensus runs.
- VGGT proposal counts: 24 cameras and 24 depths in B and C.
- Depth Pro proposal counts: 24 cameras and 24 depths in C.
- Depth source used in B: `vggt`.
- Depth source used in C: `consensus`.
- B fused point count: 76,440.
- B occupied voxel count: 2,313.
- B observed mesh: 21,248 vertices and 10,624 triangles.
- B camera trajectory count: 24.
- C fused point count: 111,550.
- C occupied voxel count: 3,641.
- C observed mesh: 21,432 vertices and 10,716 triangles.
- C camera trajectory count: 24.
- C bbox size: `[2.1099093, 1.7670834, 1.4453981]` m in teacher scale.
- C rejected low-confidence ratio: 0.0316840.
- C rejected high-disagreement ratio: 0.0316753.
- C mapped disagreement mean/p50/p95: 0.0769610 / 0.0715170 /
  0.1725194.
- C `inspectable_map_available`: true.

## Evidence Runs

- A: `runs/offline_v08_debug_flat_depth_map` decoded 6 PPM frames, selected 6
  keyframes, wrote 6 fused debug points, 5 occupied voxels, 96 mesh vertices,
  48 triangles, and `inspectable_map_available: true`.
- B: `runs/offline_v08_vggt_world_map` decoded 60 TUM RGB frames, selected 24
  keyframes, wrote a nonzero VGGT fused map and
  `world_map/fused_points.ply`.
- C: `runs/offline_v08_consensus_world_map` decoded 60 TUM RGB frames, selected
  24 keyframes, wrote VGGT + Depth Pro disagreement, consensus fused map,
  occupancy, observed voxel mesh, camera trajectory, and map-quality artifacts.
- D: skipped because no `.mp4` or `.mov` was found under `runs/`, `data/`,
  `datasets/`, `videos/`, or `inputs/`.

## PLY Paths

- `runs/offline_v08_vggt_world_map/world_map/fused_points.ply`
- `runs/offline_v08_vggt_world_map/world_map/observed_voxel_mesh.ply`
- `runs/offline_v08_consensus_world_map/world_map/fused_points.ply`
- `runs/offline_v08_consensus_world_map/world_map/observed_voxel_mesh.ply`

## Truth Boundary

Physical accuracy claim: no. Training-quality claim: no.

The fused map is `teacher_pseudo_fused_map`, observed-only, unanchored,
unoptimized, not measured geometry, not hidden-geometry completion, and not a
training-quality label set.

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 55 tests.
- Passed: `python -m atlas3r --help`.
- Passed: `python -m atlas3r offline --help`.
- Passed: `python -m atlas3r offline build-world --help`.
- Passed: `python -m atlas3r teachers list`.
- Passed: `python -m atlas3r smoke contracts`.
- Passed: `git diff --check` with Windows line-ending warnings only.

## Next

Offline V0.9: add the first real map consistency optimizer that adjusts
per-frame depth scale/bias and intrinsics to reduce VGGT-vs-Depth-Pro
disagreement and reprojected map inconsistency.
