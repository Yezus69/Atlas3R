# 05 - Evaluation And Quality

Quality reports must measure or explicitly mark unknown:

- teacher disagreement;
- reprojection error;
- render-vs-frame mismatch;
- scale source;
- surface confidence;
- object track consistency;
- observed vs predicted geometry;
- map completeness;
- known failure modes.

## Offline V0.6 Report Requirements

The tracer quality report must summarize:

- teacher availability and missing install hints;
- whether disagreement can be measured yet;
- scale source and whether physical accuracy claims are blocked;
- geometry preview point count and truth boundary;
- VGGT proposal counts and whether geometry is teacher-proposed;
- VGGT window stitching counts, rejected windows, pseudo-submap count, residuals,
  and scale range when VGGT is used;
- object tracking status and whether objects were invented;
- render/repair diagnostic status;
- observed-only vs predicted-completion flags;
- training-cache usability;
- failure points with missing inputs, missing dependencies, and future modules.

For VGGT runs, the report must state that the output is unanchored
`teacher_pseudo` geometry. It must also list missing anchors, missing optimizer,
missing render repair, and missing evaluation report as blockers for physical
accuracy or training-quality claims.

## Offline V0.7 Report Requirements

Reports with Depth Pro must summarize:

- Depth Pro runtime/replay availability, checkpoint/source, and proposal counts;
- whether Depth Pro lacks global pose and therefore cannot create a global
  preview by itself;
- VGGT-vs-Depth-Pro valid overlap count and disagreement statistics;
- diagnostic consensus preview status and source mask semantics;
- geometry preview path used: VGGT pose plus diagnostic consensus depth, VGGT
  depth, debug geometry, or empty geometry with a failure reason;
- unchanged truth boundary: teacher-pseudo, not measured, not physically
  accurate, and not training-quality.

## Offline V0.8 Report Requirements

Reports with fused world-map export must summarize:

- input source, frames decoded, and keyframes selected;
- VGGT and Depth Pro proposal counts;
- depth source used for the map: diagnostic consensus, VGGT, or explicit
  failure without VGGT pose;
- fused point count, occupied voxel count, observed mesh vertex/triangle count,
  camera trajectory count, and bounding-box size;
- valid depth ratio, low-confidence rejection ratio, high-disagreement
  rejection ratio, and disagreement mean/p50/p95 for mapped pixels when
  available;
- fused-map paths under `world_map/`;
- `inspectable_map_available`, which means only that the requested artifacts
  exist with nonzero point and voxel counts;
- unchanged truth boundary: teacher-pseudo fused map, not measured, not
  physically accurate, not optimized, and not training-quality.

## Offline V0.9 Report Requirements

Reports with map consistency optimization must summarize:

- optimizer status and whether hard improvement target passed;
- raw and optimized fused point, voxel, and observed mesh triangle counts;
- retained point ratio and anti-cheat status;
- before/after VGGT-vs-Depth-Pro absolute, relative, and log-depth metrics;
- before/after cross-view projection residual mean and p95;
- accepted/rejected Depth Pro scale/bias rows;
- intrinsics adjustment summary when enabled;
- raw and optimized PLY paths;
- unchanged truth boundary: unanchored teacher-consensus map, observed-only,
  soft-metric unanchored, not measured, not physically accurate, and not
  training-quality.

## Offline V1.0 Report Requirements

Reports with soft-metric best-map export must summarize:

- actual input path, frames decoded, frame dimensions, and keyframes selected;
- VGGT proposal count, window count, stitching mode, accepted/rejected overlaps,
  and overlap residuals;
- Depth Pro proposal count, checkpoint/source, and availability;
- EXIF/camera metadata availability and whether intrinsics are proposal-only;
- raw, optimized, and selected best-map point/voxel/observed-triangle counts;
- optimizer before/after teacher disagreement and cross-view projection
  residual mean/p95;
- selected best-map source, cleanup retained ratio, connected components, and
  bounding-box size;
- artifact paths for PLYs, top-down preview, trajectory, map quality, and
  inspection instructions;
- scale mode, scale confidence, scale reasons, and `scale_status:
  soft_metric_unanchored`;
- physical accuracy claim: false;
- training-quality claim: false.

## Offline V1.1 Report Requirements

Reports with classical geometry enabled must summarize:

- COLMAP/GLOMAP availability, executable path or replay source, and exact
  commands attempted;
- stage results and failure reason when unavailable or failed;
- registered image count and sparse point count when reconstruction succeeds;
- common frame count with VGGT, Sim3 scale/rotation/translation, and
  camera-center RMSE/p50/p95;
- sparse point agreement against raw, optimized, and best maps;
- whether classical validation changed best-map selection or was skipped;
- `world_map_best/` point, voxel, triangle, and trajectory counts;
- unchanged truth boundary: classical SfM is an unanchored proposal, not
  measured geometry, physical accuracy, or training-quality labels.

## Claim Rules

- A smoke test is not an accuracy report.
- A teacher proposal is not measured truth.
- Unanchored MP4 produces pseudo labels only.
- Physical scale claims require anchors, calibration, measured depth,
  LiDAR/ARKit, known-scale objects, or external poses.
- Runtime claims require measured profiling on named hardware.
