# Current State

Atlas3R is a reset foundation plus a connected Offline World Builder tracer
with VGGT and Depth Pro witness slices, raw and optimized map export, and a
V1.0 no-anchor soft-metric room-walk `world_map_best/`.

## Exists

- Typed contracts for frames, cameras, poses, teacher proposals, world state,
  map artifacts, and truth boundaries.
- Dependency-safe PPM, PNG/JPG, and MP4/MOV frame decoding.
- Per-frame metadata summary for JPG/EXIF hints with explicit missing-EXIF
  status.
- VGGT and Depth Pro witness runtime/replay paths behind optional adapters.
- VGGT-vs-Depth-Pro disagreement JSON/NPZ maps and diagnostic consensus
  previews through `offline build-world`.
- Raw fused teacher-consensus map export under `world_map/`.
- Diagnostic map consistency optimizer under `optimizer/`.
- Optimized map export under `world_map_optimized/`.
- Soft-metric scale hypotheses and ledgers under `world/`.
- Best-map selection, conservative cleanup, top-down preview, and inspection
  instructions under `world_map_best/`.
- Training-cache manifests can reference maps while remaining
  `usable_for_training: false`.
- Focused unit and vertical tracer tests.

## Latest Evidence

- Room input:
  `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`.
- V1.0 room run decoded 240 frames and selected 64 keyframes.
- VGGT proposals: 88 cameras, 88 depths, 4 windows.
- Depth Pro proposals: 64 cameras, 64 depths.
- Raw map: 1,638,747 points, 3,846 occupied voxels, 8,184 triangles.
- Optimized map: 2,000,000 points, 3,245 occupied voxels, 7,620 triangles.
- Best map: selected optimized, 1,904,976 points, 2,559 occupied voxels,
  5,600 triangles, 88 trajectory poses.
- Optimizer improved relative disagreement mean from 0.308152169 to
  0.0612177588 and projection residual mean from 0.0498260930 m to
  0.0196512938 m.
- Scale mode: `unanchored_soft_metric`; scale confidence: `medium`.

## Does Not Exist

- Anchored physical scale or measured geometry from RGB-only input.
- Physical accuracy, millimeter accuracy, or RGB-only readiness proof.
- Object-aware reconstruction.
- Final mesh reconstruction or hidden-geometry completion.
- Training-quality cache export.
- Student training or runtime mapping.
