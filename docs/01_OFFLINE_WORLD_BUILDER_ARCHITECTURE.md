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

## V0.8 Implementation Boundary

Offline V0.8 keeps normal RGB decoding, VGGT geometry, and Depth Pro
disagreement from V0.7, then exports the first inspectable fused world map.
`offline build-world --export-world-map` can decode PPM, PNG/JPG, and MP4/MOV
when optional decoders are available, normalize frames into the frame cache, run
or replay VGGT and Depth Pro, compare their depth proposals, and fuse sampled
observed depth into `world_map/`.

VGGT writes normalized streams:

```text
proposals/vggt_cameras.jsonl
proposals/vggt_depths.npz
proposals/vggt_windows.jsonl
proposals/depth_pro_cameras.jsonl
proposals/depth_pro_depths.npz
proposals/depth_pro_frames.jsonl
proposals/proposal_manifest.json
diagnostics/teacher_disagreement.json
diagnostics/disagreement_maps.npz
diagnostics/consensus_preview.npz
world_map/world_map_manifest.json
world_map/camera_trajectory.json
world_map/fused_points.npz
world_map/fused_points.ply
world_map/occupancy_grid.npz
world_map/occupancy_grid_metadata.json
world_map/observed_voxel_mesh.ply
world_map/map_quality.json
world_map/map_quality.md
```

Multiple VGGT windows are stitched only by a minimal overlap Sim3 estimate. If
overlap is insufficient or inconsistent, the later window becomes a separate
`pseudo_submap_id`; no global optimization is implied.

VGGT output is `teacher_pseudo`, unanchored, observed-only proposal geometry.
It is not measured geometry, not physically accurate, and not training-quality.

Depth Pro output is also `teacher_pseudo`, unanchored, and observed-only. It
proposes per-frame depth and intrinsics/focal length but no global trajectory, so
Depth Pro alone cannot create a global fused map. When both witnesses exist,
V0.8 prefers diagnostic consensus depth lifted with VGGT poses; when only VGGT
exists, it falls back to VGGT depth with VGGT poses. The sparse occupancy grid
only marks observed voxels from fused points, and the observed voxel mesh only
emits boundary faces of occupied voxels. This is not an optimized consensus,
hidden-geometry completion, physical accuracy report, or final mesh
reconstruction.

## V0.9 Implementation Boundary

Offline V0.9 keeps the same witnesses and adds a diagnostic consistency
optimizer inside `offline build-world`. It fits bounded per-keyframe Depth Pro
scale/bias to VGGT depth on common valid pixels, optionally searches a bounded
global focal scale, lowers consensus confidence where teachers still disagree,
and records before/after projection residuals.

The optimizer writes `optimizer/` artifacts and, when enabled, exports
`world_map_optimized/` in the same inspectable map schema as `world_map/`.
The optimized map is still observed-only teacher-pseudo geometry. It is not an
anchored physical world, measured depth, hidden completion, training-quality
cache, or final mesh reconstruction.
