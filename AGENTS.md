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
2. choose the highest-value coherent slice inside that milestone;
3. keep changes tied to the contracts and math in `ARCHITECTURE.md`;
4. verify real behavior or boundary contracts;
5. update `README.md` only when current state, milestone completion, or next
   priority materially changes.

## README Budget

`README.md` is a bounded state snapshot, not a session log. Edits should replace
stale state instead of appending history.

Keep it compact:

- `Current State`: at most 5 bullets.
- `Current Priority`: one short paragraph.
- `Milestones`: stable roadmap; update acceptance only when the architecture or
  implementation plan actually changes.
- No per-turn logs, command transcripts, chat summaries, or minor fix notes.

## Momentum Rule

Codex should avoid local minima where a turn is spent polishing scaffolding,
chasing tiny incidental errors. Small fixes are valuable when they unlock the next architecture slice;
otherwise prefer work that moves one of these core surfaces forward:

- architecture contracts and typed boundaries;
- reconstructability and keyframe selection;
- ViPE/DA3 adapter boundary;
- SAM2 mask-track boundary;
- visibility graph and optimizer variables;
- ray-fused TSDF/occupancy mapping;
- held-out validation and metric gate;
- accepted dataset export.

If the same class of issue repeats, step back to the milestone objective,
identify the root dependency or missing abstraction, and implement the smallest
slice that restores forward progress.

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
