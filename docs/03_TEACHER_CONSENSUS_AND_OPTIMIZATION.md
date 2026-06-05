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
