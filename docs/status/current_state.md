# Current State

Atlas3R is a small reset foundation plus the first connected Offline World
Builder tracer.

## Exists

- Typed contracts for frames, cameras, poses, teacher proposals, world state,
  map artifacts, and truth boundaries.
- Coordinate helpers for `T_A_B`, projection, and unprojection.
- Dependency-free video input inspection and PPM sequence loading.
- Minimal recording manifest read/write helpers.
- Teacher witness registry with unavailable statuses and install hints.
- `offline build-world` vertical tracer that writes the full artifact tree.
- Dependency-free PPM frame cache, keyframes, proposal-cache skeleton,
  camera-scale ledger, consensus world state, geometry preview, object ledger,
  render diagnostics, quality report, and training-cache manifest.
- Debug-only flat-depth geometry preview labeled as not measured and not
  training-quality.
- Minimal NPZ/PLY artifact writers.
- Focused unit tests for the retained foundation and tracer.

## Does Not Exist

- MP4 decoding and PNG/JPEG image-folder decoding.
- Teacher model execution.
- Teacher consensus optimizer.
- Real render-and-repair optimizer loop.
- Real mesh/occupancy/training cache generation from optimized labels.
- Student training or runtime mapping.
- Realtime, accuracy, millimeter, or RGB-only readiness proof.
