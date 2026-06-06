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

## Claim Rules

- A smoke test is not an accuracy report.
- A teacher proposal is not measured truth.
- Unanchored MP4 produces pseudo labels only.
- Physical scale claims require anchors, calibration, measured depth,
  LiDAR/ARKit, known-scale objects, or external poses.
- Runtime claims require measured profiling on named hardware.
