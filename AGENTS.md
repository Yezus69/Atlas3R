# AGENTS.md - Codex Operating Rules for Atlas3R

This file defines how Codex should work in Atlas3R.

## Context Load Order

Before non-trivial coding or architecture changes, load:

1. `README.md`
2. `ARCHITECTURE.md`
3. `AGENTS.md`

On resumed work or compacted context, reload the same files and continue from
the earliest incomplete milestone in `README.md`.

## Mission

Atlas3R builds a Scale-Aware Monocular Reconstruction Teacher for RGB videos.
The teacher filters videos, reconstructs static geometry from strong evidence,
fuses rays into TSDF/occupancy maps, and emits metric pseudo-labels only through
a scale posterior and validation gate.

Core pipeline:

```text
RGB video
  -> reconstructability gate
  -> keyframe selector
  -> ViPE/DA3 geometry backbone
  -> ray/depth/pose representation
  -> visibility graph
  -> SAM2 mask grouping
  -> scale-aware robust optimizer
  -> static/dynamic inference
  -> ray-based TSDF and occupancy fusion
  -> validation and metric acceptance gate
```

## Operating Model

Codex should work continuously from the architecture roadmap, not as isolated
one-off turns. Each implementation turn should:

1. identify the earliest incomplete milestone in `README.md`;
2. build the smallest coherent slice that advances that milestone;
3. keep changes tied to the contracts and math in `ARCHITECTURE.md`;
4. add tests that verify real behavior or boundary contracts;
5. update `README.md` only when current state or next priority changes.

## Architecture Invariants

- Internal camera geometry is ray-map first: `r_i(u,v)`, radial depth, camera
  pose, confidence, global scale, and static probability are the core atom.
- Units are meters unless a field explicitly says otherwise.
- Use explicit coordinate frame names such as `T_world_camera`.
- Global scale `s` is a state variable with posterior uncertainty and source
  metadata.
- Unknown, free, occupied, dynamic, predicted, and measured states remain
  separate through mapping and export.
- Dynamic pixels are excluded before TSDF/occupancy fusion.
- Metric output requires `ScalePosterior` and `ValidationReport`.
- Third-party model repositories and weights remain external to the repo.
