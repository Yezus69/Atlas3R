# Active Task - Offline V1.1 Classical Geometry Witness

## Goal

Add a dependency-safe COLMAP/GLOMAP witness for the real room frames, align
classical sparse geometry to the current VGGT/Depth-Pro world, compare
trajectory/map consistency, and keep all outputs unanchored proposals.

## V1.0 Artifact Inspection

- Room frames folder exists: `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`
  with 350 files.
- V1.0 artifacts exist: `world_map_manifest.json`, `map_quality.md`,
  `camera_trajectory.json`, `topdown_preview.svg`, and `room_walk_001_report.md`.
- Best map: 1,904,976 points, 2,559 occupied voxels, 5,600 observed triangles.
- Best trajectory: 88 poses.
- Best bbox size: about 1.72 m x 0.75 m x 0.96 m.
- Suspicious sign: compact room-scale bbox; classical SfM should check collapse
  or scale/trajectory disagreement.

## Checklist

- [x] Start from V1.0 branch and create V1.1 branch.
- [x] Read required docs, status files, and current builder/map modules.
- [x] Add COLMAP/GLOMAP witness, sparse import, Sim3 alignment, and map comparison.
- [x] Wire classical artifacts and optional validated map through `offline build-world`.
- [x] Add focused tests that do not require COLMAP/GLOMAP.
- [x] Run verification and real room evidence command with fallback attempts if needed.
- [x] Update compact docs/status files and V1.1 report.
- [x] Commit scoped code/docs changes without generated run artifacts.
