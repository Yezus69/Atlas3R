# 00 - Project Objective

Atlas3R's immediate objective is to build an Offline World Builder:

```text
phone MP4 / RGB camera stream
  -> estimate intrinsics, per-frame pose, depth, objects, uncertainty
  -> build one physically scaled persistent 3D world over time
  -> output inspectable mesh / voxel / occupancy map
  -> produce high-quality offline training labels for a future realtime checkpoint
```

The realtime neural checkpoint is downstream of the offline label factory, not
the next immediate foundation.

## Non-Goals For The Reset Foundation

- no new SMGT-tiny or SMGT-small-v2 training;
- no student runtime mapper;
- no measured live replay scheduler;
- no benchmark-looking report from weak toy paths;
- no hidden geometry marked as measured;
- no millimeter, realtime, or RGB-only readiness claim.

## Active Foundation

The repository keeps only the pieces needed to start Offline V0:

- coordinate, frame, pose, teacher proposal, world state, artifact, and truth
  contracts;
- dependency-safe video/recording primitives;
- teacher witness registry with unavailable statuses and install hints;
- minimal NPZ/PLY artifact writers for future map inspection;
- concise docs and focused tests.
