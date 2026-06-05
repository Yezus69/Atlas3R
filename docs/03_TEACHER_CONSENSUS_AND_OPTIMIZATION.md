# 03 - Teacher Consensus And Optimization

Teacher models are witnesses. They provide proposals that the Offline World
Builder can compare, weight, reject, or refine.

## Witness Roles

- Depth Pro: per-frame metric depth and focal-length proposal.
- VGGT: multi-view pose, depth, pointmap, and intrinsics proposal.
- MapAnything: geometry cross-check and refinement proposal.
- LingBot-Map: long-sequence streaming reconstruction proposal.
- SAM/DINO: object masks, features, and tracks.
- CoTracker: long-lived point tracks.
- COLMAP/GLOMAP: classical geometric constraints and sparse reconstruction.

## Consensus Loop

1. Normalize proposals into `TeacherProposal` records.
2. Compare teacher disagreement and confidence per frame/keyframe.
3. Estimate or constrain intrinsics, pose trajectory, depth, and scale.
4. Fuse observed surfaces into a map with uncertainty.
5. Render the current world back into frames.
6. Repair inconsistent poses, surfaces, masks, and object tracks.

No witness output is allowed to silently become measured geometry.
