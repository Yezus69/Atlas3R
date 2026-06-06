# 03 - Teacher Consensus And Optimization

Teacher models are witnesses. They provide proposals that the Offline World
Builder can compare, weight, reject, or refine. No witness output may silently
become measured geometry.

## Witness Roles

- Depth Pro proposes per-frame metric-ish depth and focal length.
- VGGT proposes multi-view camera geometry, depth, intrinsics, point maps, and
  tracks.
- MapAnything cross-checks or refines geometry from images and optional
  geometry inputs.
- LingBot-Map proposes long-sequence streaming-style trajectory and map memory
  behavior.
- SAM/DINO propose object masks, visual features, and object identity links.
- CoTracker proposes long-lived 2D tracks.
- COLMAP/GLOMAP proposes classical geometric constraints when texture and
  parallax allow it.

## Disagreement Is Signal

Teachers will disagree on depth, focal length, pose, object masks, tracks,
scale, and surface coverage. The pipeline records disagreement rather than
averaging it away. Disagreement can indicate blur, dynamic objects, weak
parallax, poor lighting, missing calibration, bad masks, or teacher failure.

## Consensus Loop

1. Normalize proposals into proposal-cache records with truth boundaries,
   uncertainty, coordinate convention, and source metadata.
2. Check multi-view consistency, reprojection behavior, scale compatibility,
   mask/track agreement, and render-vs-frame mismatch.
3. Build a candidate world state with explicit pose/depth/map statuses.
4. Fuse only observed evidence into map artifacts and keep predicted completion
   separate.
5. Render or project the candidate world back into frames and record needed
   repairs.

Consensus is an optimization target, not an assumption. Until the optimizer
runs and an evaluation report supports it, outputs are proposals or debug
artifacts, not physical accuracy claims.

## V0.6 VGGT Witness

VGGT is the first wired geometry witness. The runtime adapter is dependency
safe: `atlas3r` import and CLI help do not import Torch or VGGT. A build can
either run an external VGGT package/repo or replay a previous normalized
proposal cache.

VGGT contributes camera, intrinsics, depth, uncertainty/confidence, and window
metadata. The camera/scale ledger marks the scale source as
`vggt_unanchored_metric_proposal`. The consensus state may mark pose and depth
as `proposed`, and map status as `preview`, but optimization remains `not_run`.

V0.6 only estimates adjacent Sim3 transforms from overlapping window camera
centers. Rejected overlaps create separate pseudo-submaps instead of silent
fusion.

## V0.7 Depth Pro Disagreement

Depth Pro is the second wired geometry witness. The runtime adapter is
dependency safe: `atlas3r` import and CLI help do not import Torch or Depth Pro.
A build can either run an external `depth_pro` package/repo or replay a previous
normalized proposal cache.

Depth Pro contributes per-frame depth, derived confidence/uncertainty, and
intrinsics/focal-length proposals. It does not contribute global camera pose.
`offline build-world` records VGGT-vs-Depth-Pro absolute, relative, and
log-depth disagreement where frame proposals overlap.

The V0.7 consensus preview is diagnostic only. It marks agreeing pixels as
higher confidence and strong disagreements as lower confidence, but it is not an
optimizer result and must not be used as training-quality labels.

## V0.9 Map Consistency Optimizer

V0.9 adds the first repair pass over existing VGGT and Depth Pro proposals. It
does not add teachers or train a model.

The optimizer estimates:

- per-keyframe Depth Pro scale and bias;
- optional bounded global focal scale;
- per-frame consensus confidence after alignment.

The objective measures robust VGGT-vs-Depth-Pro log-depth agreement and
cross-view projection residuals, while regularizing scale, bias, and focal
adjustments to conservative bounds. Disagreement is still recorded and lowers
confidence; it is not silently converted into truth.

Optimized outputs use `label_type: teacher_pseudo_optimized_map` and
`optimized_world_state: diagnostic_depth_consistency_only`. They remain
unanchored, observed-only, not measured geometry, not physically accurate, and
not training-quality.
