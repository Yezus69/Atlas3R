# Current State

Atlas3R is a reset foundation plus a connected Offline World Builder tracer
with two real geometry witness vertical slices and the first inspectable fused
world-map artifact.

## Exists

- Typed contracts for frames, cameras, poses, teacher proposals, world state,
  map artifacts, and truth boundaries.
- Coordinate helpers for `T_A_B`, projection, and unprojection.
- Dependency-safe PPM, PNG/JPG, and MP4/MOV frame decoding with optional
  Pillow, imageio, or OpenCV imports only inside decoder functions.
- Frame cache, keyframes, teacher status/proposal cache, camera-scale ledger,
  consensus world state, geometry preview, fused world map, object ledger,
  render diagnostics, quality report, and training-cache manifest wired through
  `offline build-world`.
- VGGT witness runtime/replay path behind `src/atlas3r/models/adapters/`, with
  normalized camera, depth, and window proposal streams.
- Depth Pro witness runtime/replay path behind `src/atlas3r/models/adapters/`,
  with normalized per-frame camera/intrinsics and depth proposal streams.
- VGGT-vs-Depth-Pro disagreement JSON/NPZ maps and a diagnostic consensus
  preview through `offline build-world`.
- Fused teacher-pseudo map export under `world_map/`: fused points, sparse
  occupancy, observed voxel mesh, camera trajectory, manifest, and map-quality
  report.
- Minimal overlap Sim3 VGGT window stitching with pseudo-submap rejection.
- Debug flat-depth, VGGT, and diagnostic consensus geometry labeled as not
  physically accurate and not training-quality.
- Focused unit and vertical tracer tests.

## Latest Evidence

- `runs/offline_v08_vggt_world_map`: 60 TUM RGB frames, 24 keyframes, 24 VGGT
  camera/depth proposals, 76,440 fused points, 2,313 occupied voxels, and an
  inspectable observed voxel mesh.
- `runs/offline_v08_consensus_world_map`: 60 TUM RGB frames, 24 keyframes, 24
  VGGT and 24 Depth Pro depth proposals, 111,550 fused points, 3,641 occupied
  voxels, 21,432 mesh vertices, 10,716 triangles, and
  `inspectable_map_available: true`.

## Does Not Exist

- Anchored physical scale or measured geometry from RGB-only input.
- Teacher consensus optimizer.
- Real render-and-repair optimizer loop.
- Object-aware reconstruction.
- Final mesh reconstruction or hidden-geometry completion.
- Training-quality cache export.
- Student training or runtime mapping.
- Realtime, accuracy, millimeter, or RGB-only readiness proof.
