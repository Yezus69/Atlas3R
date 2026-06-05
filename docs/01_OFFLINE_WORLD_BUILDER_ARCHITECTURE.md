# 01 - Offline World Builder Architecture

Atlas3R milestones are vertical slices through all modules, not isolated
implementation of one stage.

```text
MP4/RGB input
  -> frame cache and keyframes
  -> teacher witness proposals and proposal cache
  -> camera / intrinsics / scale ledger
  -> consensus world state
  -> geometry lifter and object permanence ledger
  -> render-and-repair diagnostics
  -> artifacts, quality report, and training-cache manifest
```

## Modules

1. Run Orchestrator owns one world-building run, artifact paths, provenance,
   module order, and explicit failure states. It writes the run manifest and
   calls every architecture module in one command.
2. Frame Cache converts MP4 or image sequences into stable frame records with
   timestamps, source paths, quality scores, and truth-boundary metadata.
3. Keyframe Selector picks useful frames by stride, blur/exposure quality, and
   visual-change heuristics. It must not assume pose exists.
4. Teacher Witness Layer runs or records proposals from Depth Pro, VGGT,
   MapAnything, LingBot-Map, SAM/DINO, CoTracker, and COLMAP/GLOMAP. Every
   witness output is a proposal, not truth.
5. Proposal Cache stores teacher outputs in normalized records: depth,
   intrinsics, pose, point tracks, masks, features, uncertainty, coordinate
   convention, and source.
6. Camera / Intrinsics / Scale Ledger records what is known, guessed,
   unanchored, anchored, or measured. It blocks fake physical accuracy claims.
7. Consensus World State combines proposals into one candidate world containing
   cameras, frame poses, depth hypotheses, static surfaces, object tracks, and
   uncertainties. Missing or unresolved fields stay explicit.
8. Geometry Lifter converts 2D pixel, depth, and camera evidence into 3D
   points/surfaces using camera rays, intrinsics, depth, and `T_world_camera`.
9. Object Permanence Ledger tracks masks/features across frames as persistent
   object candidates and keeps them separate from static-scene evidence.
10. Render-And-Repair Diagnostics projects or renders the current world back
    into frames, measures mismatch, and records pose/depth/object/scale repairs
    needed before optimization can claim success.
11. Artifact Exporter writes point cloud, mesh/voxel placeholders, object
    ledgers, world state, and future training-cache metadata with truth flags.
12. Quality Report explains teacher availability/disagreement, scale source,
    confidence, render mismatch, observed/predicted separation, and failure
    points.

## Reset Implementation Boundary

The reset branch kept contracts, dependency-safe input inspection, teacher
status, and artifact helpers. Offline V0.5 adds the first connected tracer; it
does not run heavy teacher models or optimize geometry.
