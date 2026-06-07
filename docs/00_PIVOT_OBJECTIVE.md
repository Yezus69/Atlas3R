# Pivot Objective

Atlas3R is pivoting to a Scale-Aware Monocular Reconstruction Teacher for RGB
videos.

The teacher is not a promise that arbitrary monocular video produces measured
metric ground truth. It is a filter and reconstructor:

```text
RGB video -> accept metric pseudo-labels only when scale is identifiable enough
RGB video -> otherwise reject or mark output as non-metric pseudo-label data
```

## Central Truth Boundary

Unanchored monocular RGB has an unknown global scale unless the system receives
or infers a strong metric cue. The pivot therefore separates outputs into:

- measured metric data: requires LiDAR, RGB-D, ARKit/ARCore depth or pose,
  laser scan, measured marker, known trajectory, or a precise known object;
- metric pseudo-label data: uses a learned metric prior, scene anchors,
  geometric consistency, validation, and a tight scale posterior;
- non-metric pseudo-label data: visually useful reconstruction without a strong
  scale posterior.

Only the first category is measured ground truth. The second may be useful for
training but must keep confidence, scale uncertainty, and provenance attached.

## Target Teacher

The intended teacher stack is:

```text
Video quality and reconstructability gate
  -> keyframe selector
  -> ViPE/DA3 geometry backbone
  -> canonical ray/depth/pose representation
  -> visibility graph
  -> SAM2 mask grouping
  -> scale-aware robust optimizer
  -> static/dynamic inference
  -> ray-based TSDF and occupancy fusion
  -> floor-aligned robot grid
  -> held-out validation
  -> metric acceptance or rejection
```

## Non-Goals For This Baseline

- no runtime implementation in this cleanup session;
- no old VGGT/Depth-Pro witness stack;
- no old offline-world-builder phase history;
- no training cache export until validation exists;
- no physical accuracy claim without benchmark or calibration evidence.

## Operating Principle

The model proposes geometry. The optimizer refines it. The ray mapper proves
free space. The scale posterior decides whether the output is metric. The
validator decides whether the video belongs in the dataset.
