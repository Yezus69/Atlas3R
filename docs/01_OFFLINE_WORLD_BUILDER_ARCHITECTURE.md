# 01 - Offline World Builder Architecture

The architecture is:

```text
MP4/RGB input
  -> frames and keyframes
  -> teacher proposals
  -> consensus world state
  -> render-and-repair optimizer
  -> mesh / occupancy / training cache
```

## Stages

1. Ingestion reads MP4/RGB input, estimates or records camera metadata, and
   stores frames/keyframes with explicit source paths and timestamps.
2. Teacher witnesses propose depth, intrinsics, poses, point tracks, object
   masks, and uncertainty. Witnesses do not define truth.
3. Consensus optimization reconciles teachers, classical geometry, scale
   anchors, calibration, and frame consistency into a world state.
4. Render-and-repair compares rendered world predictions back to source frames
   and updates uncertain poses, depth, objects, and map surfaces.
5. Artifact export writes observed and predicted geometry separately, with
   confidence, uncertainty, scale source, and source frame IDs.

## Reset Implementation Boundary

The current codebase stops at contracts, input inspection, teacher status, and
artifact writer helpers. It does not run teacher models or optimize geometry.
