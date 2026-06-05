# Current State

Atlas3R is a small reset foundation for Offline World Builder work.

## Exists

- Typed contracts for frames, cameras, poses, teacher proposals, world state,
  map artifacts, and truth boundaries.
- Coordinate helpers for `T_A_B`, projection, and unprojection.
- Dependency-free video input inspection and PPM sequence loading.
- Minimal recording manifest read/write helpers.
- Teacher witness registry with unavailable statuses and install hints.
- Offline quality report skeleton for input inspection.
- Minimal NPZ/PLY artifact writers.
- Focused unit tests for the retained foundation.

## Does Not Exist

- MP4 decoding and keyframe extraction.
- Teacher model execution.
- Teacher proposal cache writer.
- Teacher consensus optimizer.
- Render-and-repair loop.
- Real mesh/occupancy/training cache generation from video.
- Student training or runtime mapping.
- Realtime, accuracy, millimeter, or RGB-only readiness proof.
