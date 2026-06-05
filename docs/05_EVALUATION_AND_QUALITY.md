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

## Claim Rules

- A smoke test is not an accuracy report.
- A teacher proposal is not measured truth.
- Unanchored MP4 produces pseudo labels only.
- Physical scale claims require anchors, calibration, measured depth,
  LiDAR/ARKit, known-scale objects, or external poses.
- Runtime claims require measured profiling on named hardware.
