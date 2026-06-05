# Atlas3R

Atlas3R is being reset around one foundation:

```text
MP4/RGB video -> offline optimized 3D world -> inspectable mesh/occupancy -> training cache
```

The current repository is a reset foundation, not a working mapper. It keeps
only dependency-safe contracts, input primitives, teacher-adapter boundaries,
and small map artifact helpers that future Offline World Builder work can build
on.

## Boundaries

- No realtime mapping claim exists.
- No millimeter accuracy claim exists.
- No RGB-only production readiness claim exists.
- No final neural checkpoint exists.
- Teacher models are witnesses and proposal generators, not truth.
- Physical accuracy requires anchors, calibration, measured depth, LiDAR/ARKit,
  or known-scale objects.
- Unanchored MP4 input produces pseudo labels only.
- Hidden or completed geometry must be marked predicted/uncertain, not measured.
- Every geometry or map artifact must carry confidence or uncertainty metadata.

## Direction

The Offline World Builder path is:

```text
MP4/RGB input
  -> frame and keyframe cache
  -> teacher proposals from Depth Pro, VGGT, MapAnything, LingBot-Map,
     SAM/DINO, CoTracker, COLMAP/GLOMAP
  -> global teacher-consensus optimizer
  -> render-and-repair consistency loop
  -> optimized camera trajectory, intrinsics, depth, static map, object tracks
  -> inspectable mesh/occupancy/training cache
```

The realtime neural checkpoint is downstream of the offline label factory, not
the next immediate foundation.

## Current CLI

```bash
python -m atlas3r --help
python -m atlas3r offline --help
python -m atlas3r offline inspect-video --input <mp4-or-ppm-folder> --output <run>
python -m atlas3r teachers list
python -m atlas3r smoke contracts
```

Removed command families include old SMGT training, student runtime mapping,
measured live replay, teacher temporal cache training, TUM training/eval, and
old phase smoke commands.

## Repository Map

- `docs/`: reset objective, architecture, truth boundary, contracts, quality.
- `src/atlas3r/contracts/`: coordinate, frame, pose, proposal, world, artifact,
  and truth-boundary contracts.
- `src/atlas3r/input/`: dependency-safe PPM/video inspection and recording
  manifest primitives.
- `src/atlas3r/teachers/`: dependency-safe teacher witness registry.
- `src/atlas3r/offline/`: skeleton quality report for Offline World Builder V0.
- `src/atlas3r/mapping/`: minimal NPZ/PLY artifact writers for inspection.
- `tests/unit/`: focused tests for the kept foundation only.
