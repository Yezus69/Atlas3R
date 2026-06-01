# 00 — Feasibility and Truth Boundary

## The user goal

Input: any RGB video or live camera feed.

Output: camera center in `(x, y, z)` relative to scene objects, all visible 3D objects as triangle/polygon meshes, and a continuously built environment map in real time.

Desired: 30 FPS, mostly neural, GPU/Apple M support, physically accurate to millimeter level.

## What is physically possible

Atlas3R can be built as a strong real-time RGB mapping system, but the system must not promise impossible guarantees. RGB-only reconstruction is not a metrology sensor. It estimates geometry from projection, learned priors, and multi-view consistency.

### Why “any RGB video + mm accuracy” is impossible as a blanket guarantee

1. **Scale ambiguity.** With a monocular camera, many differently scaled 3D worlds can produce the same image sequence unless some metric information is present or inferred from priors.
2. **Hidden surfaces.** The back side of an object not seen by the camera cannot be measured. It can only be completed by a prior.
3. **Textureless/glossy/transparent surfaces.** They may produce weak or misleading correspondences.
4. **Motion blur and rolling shutter.** A video frame may not correspond to a single pinhole camera pose.
5. **Dynamic objects.** Moving objects break static-scene multi-view assumptions.
6. **Unknown intrinsics/lens distortion.** Cropped/resized/compressed video can destroy the original camera model.
7. **Dataset bias.** Metric monocular depth models learn scale priors from data; they do not create missing information.

## Accuracy modes

Atlas3R must expose modes honestly.

### Mode A — RGB-only best effort

- Input: arbitrary RGB.
- Expected: useful dense 3D map, pose tracking, object meshes with uncertainty.
- Typical claim allowed: qualitative mapping and relative/metric-by-prior estimates.
- Claim not allowed: universal mm-level physical accuracy.

### Mode B — calibrated RGB multi-view

- Input: known intrinsics, distortion, timestamping, enough parallax, mostly static scene.
- Expected: much better metric consistency.
- Claim allowed only after benchmark: local millimeter-to-centimeter error depending on resolution, baseline, range, texture, and calibration.

### Mode C — RGB with known scale anchors

- Input: calibrated RGB plus known scale source such as AprilTag board, fiducial object, measured camera motion, ARKit/IMU scale, or occasional depth/LiDAR calibration during training or setup.
- Expected: strongest metric accuracy.
- Claim allowed only within validated regions.

## Required uncertainty outputs

Every geometry output must expose uncertainty:

- per-frame pose covariance or 6-vector standard deviation;
- per-pixel depth/point uncertainty;
- per-voxel TSDF weight and uncertainty;
- per-mesh-chunk uncertainty percentiles;
- per-object observed coverage and completion confidence.

## Required user-visible flags

The runtime API must distinguish:

- `observed_surface`: surface directly supported by multi-view evidence;
- `single_view_prior`: surface supported mainly by monocular prediction;
- `completed_surface`: inferred hidden geometry;
- `dynamic_surface`: object moved during capture;
- `low_confidence`: insufficient evidence.

## Practical design implication

The correct architecture is not “one big neural net outputs a perfect mesh.” The correct architecture is:

1. neural streaming geometry to predict pose/depth/pointmaps/confidence;
2. neural object segmentation/tracking;
3. geometry back-end for consistency, scale correction, and loop closure;
4. probabilistic TSDF/surfel fusion for measured surfaces;
5. optional neural completion, clearly marked as predicted.

