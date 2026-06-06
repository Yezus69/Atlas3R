# PLANS.md - Offline World Builder Plan

This plan is the active source of direction after the repository reset.

## Reset 0 - Repo Cleanup And Architecture Reset

Keep only contracts, dependency-safe input primitives, teacher boundaries,
minimal artifact inspection helpers, focused tests, and concise docs.

Acceptance:

- stale SMGT/student/training/runtime code is deleted;
- old command surfaces are removed;
- reset docs and reports state what exists and what does not;
- focused unit tests pass.

## Offline V0 - MP4 Ingestion And Proposal Cache Skeleton

Implement MP4 ingestion, keyframe extraction, teacher proposal cache schemas,
and a quality report skeleton.

Acceptance:

- dependency-safe input path reports missing video dependencies clearly;
- keyframe metadata and camera intrinsics placeholders validate;
- proposal cache schema carries truth boundaries and uncertainty;
- no teacher model is imported at package import time.

## Offline V0.5 - Parallel World Builder Tracer

Build the first end-to-end offline spine across ingestion, keyframes, teacher
status/proposal cache, camera-scale ledger, consensus state, geometry preview,
object ledger, render diagnostics, quality report, and training-cache manifest.

Acceptance:

- one `offline build-world` command writes the full artifact tree;
- missing decoders and teachers become explicit failure points;
- debug flat-depth geometry can produce inspectable NPZ/PLY previews without
  claiming measured geometry or training quality;
- future work must continue by vertical slices through the tracer.

## Offline V0.6 - VGGT Geometry Witness

Wire real RGB decoding and VGGT teacher proposals through the full
`offline build-world` artifact tree.

Acceptance:

- PPM, PNG/JPG folders, single PNG/JPG files, and MP4/MOV inputs decode through
  dependency-safe optional decoders or explicit failure points;
- VGGT can run from an external package/repo or replay a normalized proposal
  cache without importing heavy dependencies at `atlas3r` import time;
- VGGT cameras/depths/windows feed proposal cache, camera/scale ledgers,
  consensus state, geometry preview, diagnostics, quality report, and
  training-cache manifest;
- VGGT geometry is labeled `teacher_pseudo`, unanchored, not measured, not
  physically accurate, and not training-quality.

## Offline V0.7 - Depth Pro Disagreement Witness

Add Depth Pro as an independent per-frame depth/intrinsics witness and compare
it with VGGT through the full `offline build-world` path.

Acceptance:

- Depth Pro can run from an external package/repo or replay normalized proposal
  caches without import-time heavy dependencies;
- proposal cache writes Depth Pro camera, depth, and per-frame streams;
- camera/scale ledgers and world state record both witness sources while still
  blocking physical accuracy claims;
- `diagnostics/teacher_disagreement.json`,
  `diagnostics/disagreement_maps.npz`, and
  `diagnostics/consensus_preview.npz` summarize VGGT-vs-Depth-Pro agreement;
- geometry preview may use VGGT pose plus diagnostic consensus depth, but Depth
  Pro alone does not create global world geometry;
- all outputs remain teacher-pseudo, unanchored, not optimized, and not
  training-quality.

## Offline V1 - Additional Teacher Witnesses

Run additional teacher witnesses where available: Depth Pro, COLMAP/GLOMAP,
MapAnything, LingBot-Map, SAM/DINO, and CoTracker. Each witness is a proposal
source, not truth.

Acceptance:

- adapters are isolated under dependency-safe boundaries;
- missing dependencies produce explicit unavailable status and install hints;
- proposal payloads validate against the API contracts.

## Offline V2 - Teacher Consensus World State

Build the first consensus world state: intrinsics, camera poses, depth, and a
static map from teacher proposals and geometric constraints.

Acceptance:

- disagreement and confidence are recorded per frame/keyframe;
- scale source is explicit;
- unanchored MP4 remains pseudo-labeled.

## Offline V3 - Render-And-Repair Optimizer Loop

Add render-vs-frame consistency checks and iterative repair of depth, poses,
intrinsics, masks, and map artifacts.

Acceptance:

- reports include reprojection error, render mismatch, scale source, surface
  confidence, and failure modes;
- predicted or completed geometry is separated from observed geometry.

## Offline V4 - Object Permanence

Use SAM/DINO tracks and long-lived point tracks to build object permanence and
object canonical volumes.

Acceptance:

- object tracks carry uncertainty and source provenance;
- dynamic objects do not contaminate static map artifacts.

## Offline V5 - Anchored Capture Mode

Support physically anchored capture using measured depth, LiDAR/ARKit,
calibration targets, known-scale objects, or external poses.

Acceptance:

- anchored labels are separated from unanchored pseudo labels;
- physical scale claims cite the scale source.

## Dataset V0 - Training Cache Export

Export training caches from optimized offline worlds only after quality reports
show usable proposal consistency.

Acceptance:

- cache records include truth boundaries, uncertainty, source frame IDs, and
  map artifact provenance;
- caches do not include model weights or external repos.

## Runtime V0 - Future Realtime Checkpoint

Train a realtime checkpoint only after Offline Builder labels are good enough.

Acceptance:

- training begins from optimized offline labels, not weak toy paths;
- runtime claims require measured profile and evaluation reports.
