# 02 - Data And Truth Boundary

Atlas3R separates label sources because they are not interchangeable.

## Label Types

- Measured GT: direct measured geometry or poses from calibrated sensors or a
  named benchmark. This can support accuracy claims only with an evaluation
  report.
- Anchored capture labels: labels scaled by anchors such as LiDAR/ARKit,
  calibration targets, known-scale objects, or external poses.
- Synthetic GT: generated geometry where the renderer defines truth.
- CAD-aligned approximate labels: labels aligned to CAD or survey data with
  known approximation limits.
- Teacher pseudo labels: outputs from Depth Pro, VGGT, MapAnything,
  LingBot-Map, SAM/DINO, CoTracker, COLMAP/GLOMAP, or similar witnesses.
- Unanchored MP4 pseudo labels: monocular RGB labels without a physical scale
  anchor.
- Unanchored soft-metric teacher-consensus maps: observed RGB geometry using
  Depth Pro and VGGT metric priors without an anchor, calibration target,
  measured depth, or evaluation report.

## Rules

- Teacher pseudo labels are proposals, not truth.
- Unanchored MP4 output cannot claim physical accuracy.
- Observed geometry and predicted completion must be stored separately.
- Every geometry artifact carries uncertainty and a scale source.
- Accuracy claims require a named evaluation report.
- `scale_status: soft_metric_unanchored` can support inspection and debugging,
  but it cannot support physical accuracy or training-quality claims.
