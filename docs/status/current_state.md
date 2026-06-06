# Current State

Atlas3R is a reset foundation plus a connected Offline World Builder tracer
with VGGT, Depth Pro, and dependency-safe COLMAP/GLOMAP witness paths, raw and
optimized map export, and an unanchored soft-metric room-walk `world_map_best/`.

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
- Raw, optimized, and best map export with observed-only PLY/NPZ artifacts.
- Focused unit and vertical tracer tests.

## Latest Evidence

- Room input:
  `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`.
- V1.1 run: `runs/room_walk_001_v11_colmap_witness`.
- Decoded 240 frames and selected 64 keyframes.
- VGGT proposals: 88 cameras, 88 depths, 4 windows.
- Depth Pro proposals: 64 cameras, 64 depths.
- COLMAP status: unavailable; `colmap` was not on PATH or common checked
  Windows/repo-adjacent locations.
- COLMAP registered images: 0; sparse points: 0; common frames: 0.
- Best map: selected optimized, 1,904,976 points, 2,559 occupied voxels,
  5,600 triangles, 88 trajectory poses.
- Scale mode: `unanchored_soft_metric`; scale confidence: `medium`.

## Does Not Exist

- Classical room-frame camera poses or sparse points on this machine.
- Anchored physical scale or measured geometry from RGB-only input.
- Physical accuracy, millimeter accuracy, or RGB-only readiness proof.
- Object-aware reconstruction, hidden completion, training-quality cache export,
  student training, or runtime mapping.
