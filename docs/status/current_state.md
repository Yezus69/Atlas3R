# Current State

Atlas3R is a reset foundation plus a connected Offline World Builder tracer
with VGGT, Depth Pro, and dependency-safe COLMAP/GLOMAP witness paths, raw and
optimized map export, and an observed-only room-walk `world_map_best/` with an
optional ground-plane/camera-height near-metric cue and local HTML viewer. It
also has a direct external ViPE import path for observed-only room maps.

## Exists

- Typed contracts for frames, cameras, poses, teacher proposals, world state,
  map artifacts, and truth boundaries.
- Dependency-safe PPM, PNG/JPG, and MP4/MOV frame decoding.
- VGGT and Depth Pro runtime/replay proposal paths.
- COLMAP/GLOMAP external executable and replay paths with safe unavailable and
  failure reports.
- COLMAP sparse text import with `T_world_camera` conversion.
- Sim3 alignment of classical camera centers to VGGT camera centers.
- Classical sparse-map comparison diagnostics.
- Optional classical-validated map export guarded by agreement and anti-collapse
  checks.
- External ViPE artifact import into the fused-map schema.
- Raw, optimized, and best map export with observed-only PLY/NPZ artifacts.
- Local `world_map_best/viewer.html` export backed by local map preview and
  metadata artifacts.
- Ground-plane/camera-height scale cue recorded in scale ledgers as
  near-metric unanchored evidence.
- Focused unit and vertical tracer tests.

## Latest Evidence

- Room input:
  `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`.
- V1.1 run: `runs/room_walk_001_v11_colmap_witness`.
- ViPE import run: `runs/room_walk_001_vipe_import`.
- Latest build-world run:
  `runs/room_walk_001_build_world_near_metric_viewer_final`.
- Decoded 240 frames and selected 64 keyframes.
- VGGT proposals: 88 cameras, 88 depths, 4 windows.
- Depth Pro proposals: 64 cameras, 64 depths.
- COLMAP status: unavailable; `colmap` was not on PATH or common checked
  Windows/repo-adjacent locations.
- COLMAP registered images: 0; sparse points: 0; common frames: 0.
- Best map: selected optimized, 1,904,976 points, 2,559 occupied voxels,
  5,600 triangles, 88 trajectory poses.
- Scale mode: `unanchored_soft_metric`; scale confidence: `medium`.
- ViPE 240-frame run exited `0` and exported 240 depth/intrinsics/pose rows.
- ViPE pose rows are nonfinite, so Atlas3R wrote 0 trajectory poses.
- ViPE dense SLAM fallback import: 168,365 points, 10,628 occupied voxels,
  42,656 observed mesh triangles, bbox about `4.60 m x 3.78 m x 5.24 m`.
- ViPE import is less collapsed than V10/V11 by bbox heuristic only; it is not
  measured geometry or an accuracy report.
- Latest build-world best map selected `raw_consensus`: 285,516 points,
  139,717 occupied voxels, 685,404 observed mesh triangles, 88 finite trajectory
  poses, bbox about `10.01 m x 5.08 m x 6.55 m`.
- Latest scale status: `near_metric_unanchored`; scale confidence: `very_low`.
- Latest viewer renders at
  `runs/room_walk_001_build_world_near_metric_viewer_final/world_map_best/viewer.html`.

## Does Not Exist

- Classical room-frame camera poses or sparse points on this machine.
- Finite ViPE camera trajectory export for `room_walk_001_main240`.
- Anchored physical scale or measured geometry from RGB-only input.
- Physical accuracy, millimeter accuracy, or RGB-only readiness proof.
- Object-aware reconstruction, hidden completion, training-quality cache export,
  student training, or runtime mapping.
- Object-size anchoring and dynamic masking for the latest room map.
