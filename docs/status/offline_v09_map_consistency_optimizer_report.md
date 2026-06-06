# Offline V0.9 Map Consistency Optimizer Report

Branch: `codex/offline-world-builder-v09-map-consistency-optimizer`
Commit: `pending-final-commit`

## Commands Run

```bash
python -m atlas3r offline build-world \
  --input build/offline_v05_tiny_ppm_input \
  --output runs/offline_v09_debug_flat_depth_optimizer \
  --max-frames 12 \
  --keyframe-stride 2 \
  --debug-geometry-mode flat-depth \
  --export-world-map \
  --map-write-occupancy \
  --map-write-observed-mesh \
  --optimize-map-consistency \
  --export-optimized-world-map \
  --write-ply

python -m atlas3r offline build-world \
  --input data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb \
  --output runs/offline_v09_tum_consistency_optimizer \
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
  --optimize-map-consistency \
  --optimizer-max-iterations 5 \
  --optimizer-cross-view-pairs 3 \
  --export-optimized-world-map \
  --write-ply
```

Phone MP4 evidence command to run after placing a video at
`inputs/room_walkthrough.mp4`:

```bash
python -m atlas3r offline build-world \
  --input inputs/room_walkthrough.mp4 \
  --output runs/offline_v09_phone_mp4_consistency_optimizer \
  --max-frames 120 \
  --keyframe-stride 3 \
  --keyframe-max-count 40 \
  --enable-vggt \
  --vggt-device cuda:0 \
  --vggt-image-size 384 \
  --vggt-window-size 24 \
  --vggt-window-overlap 8 \
  --vggt-stitch-mode overlap-sim3 \
  --enable-depth-pro \
  --depth-pro-device cuda:0 \
  --export-world-map \
  --map-depth-source consensus \
  --map-point-stride 8 \
  --map-min-confidence 0.25 \
  --map-max-relative-disagreement 0.25 \
  --map-voxel-size-m 0.05 \
  --map-write-occupancy \
  --map-write-observed-mesh \
  --optimize-map-consistency \
  --optimizer-max-iterations 5 \
  --export-optimized-world-map \
  --write-ply
```

## Evidence Summary

- Real input used: `data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb`.
- Phone MP4 found: no. Search roots were `inputs/`, `videos/`, `data/`,
  `datasets/`, and `runs/` for `.mp4`, `.mov`, and `.m4v`.
- Frames decoded: 60.
- Keyframes selected: 24.
- VGGT proposal counts: 24 cameras and 24 depths.
- Depth Pro proposal counts: 24 cameras and 24 depths.
- Raw map: 111,550 fused points, 3,641 occupied voxels, 10,716 observed mesh
  triangles.
- Optimized map: 111,808 fused points, 2,435 occupied voxels, 9,848 observed
  mesh triangles.
- Retained point ratio: 1.002312864.
- Hard success target: passed.

## Before And After Metrics

- VGGT-vs-Depth-Pro relative mean: 0.087864511 -> 0.046452649.
- VGGT-vs-Depth-Pro relative p95: 0.197194909 -> 0.148134258.
- Cross-view projection residual mean m: 0.027318565 -> 0.009636894.
- Cross-view projection residual p95 m: 0.068675339 -> 0.025616455.
- Improvement ratios: relative mean 47.13%, relative p95 24.88%, projection
  mean 64.72%, projection p95 62.70%.
- Optimizer status: `improved`.
- Anti-cheat status: passed.

## Optimizer Summary

- Depth scale/bias rows: 24.
- Accepted rows: 24.
- Depth Pro scale min/mean/max: 0.727146855 / 0.865076930 / 1.008241463.
- Depth Pro bias min/mean/max m: 0.032233482 / 0.076450828 /
  0.144263002.
- Intrinsics adjustment: disabled; global focal scale 1.0.
- Optimizer artifacts: `runs/offline_v09_tum_consistency_optimizer/optimizer/`.

## Map Paths

- Raw points PLY:
  `runs/offline_v09_tum_consistency_optimizer/world_map/fused_points.ply`.
- Raw observed mesh PLY:
  `runs/offline_v09_tum_consistency_optimizer/world_map/observed_voxel_mesh.ply`.
- Optimized points PLY:
  `runs/offline_v09_tum_consistency_optimizer/world_map_optimized/fused_points.ply`.
- Optimized observed mesh PLY:
  `runs/offline_v09_tum_consistency_optimizer/world_map_optimized/observed_voxel_mesh.ply`.

## Evidence Runs

- A: `runs/offline_v09_debug_flat_depth_optimizer` wrote a raw debug map with
  6 fused points, 5 occupied voxels, and `inspectable_map_available: true`.
  The optimizer reported `insufficient_witnesses` because debug flat-depth has
  no overlapping VGGT and Depth Pro witnesses.
- B: `runs/offline_v09_tum_consistency_optimizer` wrote raw and optimized
  teacher-pseudo maps and passed the improvement target without dropping below
  the retained-point threshold.
- C: skipped because no phone MP4/MOV/M4V was found locally.

## Truth Boundary

Physical accuracy claim: no. Training-quality claim: no.

Optimized outputs are `teacher_pseudo_optimized_map`, observed-only,
unanchored, not measured geometry, not hidden-geometry completion, not
physically accurate, and not training-quality.

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 59 tests.
- Passed: `python -m atlas3r --help`.
- Passed: `python -m atlas3r offline --help`.
- Passed: `python -m atlas3r offline build-world --help`.
- Passed: `python -m atlas3r teachers list`.
- Passed: `python -m atlas3r smoke contracts`.
- Passed: `git diff --check` with CRLF warnings only.

## Next Bottleneck

Offline V1.0 should add anchored capture mode using a known-size marker/object
or manual scale measurement, so maps can have an explicit physical scale source
instead of unanchored teacher scale.
