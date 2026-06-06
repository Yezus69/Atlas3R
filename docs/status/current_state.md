# Current State

Atlas3R is a reset foundation plus a connected Offline World Builder tracer
with two real geometry witness vertical slices.

## Exists

- Typed contracts for frames, cameras, poses, teacher proposals, world state,
  map artifacts, and truth boundaries.
- Coordinate helpers for `T_A_B`, projection, and unprojection.
- Dependency-safe PPM, PNG/JPG, and MP4/MOV frame decoding with optional
  Pillow, imageio, or OpenCV imports only inside decoder functions.
- Frame cache, keyframes, teacher status/proposal cache, camera-scale ledger,
  consensus world state, geometry preview, object ledger, render diagnostics,
  quality report, and training-cache manifest wired through
  `offline build-world`.
- VGGT witness runtime/replay path behind `src/atlas3r/models/adapters/`, with
  normalized camera, depth, and window proposal streams.
- Depth Pro witness runtime/replay path behind `src/atlas3r/models/adapters/`,
  with normalized per-frame camera/intrinsics and depth proposal streams.
- VGGT-vs-Depth-Pro disagreement JSON/NPZ maps and a diagnostic consensus
  preview through `offline build-world`.
- Minimal overlap Sim3 VGGT window stitching with pseudo-submap rejection.
- Debug flat-depth, VGGT, and diagnostic consensus geometry previews labeled as
  not physically accurate and not training-quality.
- Minimal NPZ/PLY artifact writers.
- Focused unit and vertical tracer tests.

## Does Not Exist

- Anchored physical scale or measured geometry from RGB-only input.
- Teacher consensus optimizer.
- Real render-and-repair optimizer loop.
- Object-aware reconstruction.
- Final mesh/occupancy reconstruction.
- Training-quality cache export.
- Student training or runtime mapping.
- Realtime, accuracy, millimeter, or RGB-only readiness proof.
