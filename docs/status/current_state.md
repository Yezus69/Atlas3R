# Current State

Atlas3R is a reset foundation plus a connected Offline World Builder tracer
with two real geometry witness vertical slices, a raw fused map exporter, and
the first diagnostic map consistency optimizer.

## Exists

- Typed contracts for frames, cameras, poses, teacher proposals, world state,
  map artifacts, and truth boundaries.
- Dependency-safe PPM, PNG/JPG, and MP4/MOV frame decoding.
- VGGT and Depth Pro witness runtime/replay paths behind optional adapters.
- VGGT-vs-Depth-Pro disagreement JSON/NPZ maps and diagnostic consensus
  previews through `offline build-world`.
- Raw fused teacher-pseudo map export under `world_map/`.
- V0.9 map consistency optimizer under `optimizer/`, with per-keyframe Depth
  Pro scale/bias, optional bounded focal scale, before/after disagreement
  metrics, and before/after projection diagnostics.
- Optimized teacher-pseudo map export under `world_map_optimized/`.
- Training-cache manifests can reference optimized maps while remaining
  `usable_for_training: false`.
- Focused unit and vertical tracer tests.

## Latest Evidence

- V0.8 VGGT-only TUM map: 60 RGB frames, 24 keyframes, 24 VGGT camera/depth
  proposals, 76,440 fused points, and 2,313 occupied voxels.
- V0.8 consensus TUM map: 111,550 fused points, 3,641 occupied voxels, 21,432
  mesh vertices, 10,716 triangles, and `inspectable_map_available: true`.
- V0.9 optimized TUM map:
  - raw map: 111,550 fused points, 3,641 occupied voxels, 10,716 triangles;
  - optimized map: 111,808 fused points, 2,435 occupied voxels, 9,848
    triangles;
  - retained point ratio: 1.002312864;
  - relative disagreement mean improved from 0.087864511 to 0.046452649;
  - projection residual mean improved from 0.027318565 m to 0.009636894 m.

## Does Not Exist

- Anchored physical scale or measured geometry from RGB-only input.
- Physical accuracy, millimeter accuracy, or RGB-only readiness proof.
- Object-aware reconstruction.
- Final mesh reconstruction or hidden-geometry completion.
- Training-quality cache export.
- Student training or runtime mapping.
