# Atlas3R Architecture

This document is the technical source of truth for Atlas3R. The README points to work; this file defines what the work must mean.

Atlas3R is a scale-aware monocular reconstruction teacher for real RGB videos. It should eventually create physically useful 3D maps and floor-cleaner occupancy labels, but it must not pretend that unanchored monocular RGB is measured metric ground truth.

## Truth Boundary

A monocular RGB video can constrain scene shape, camera motion, visibility, and free space, but global metric scale is not guaranteed unless scale evidence exists.

Therefore Atlas3R separates:

```text
measured_metric        real metric evidence exists and validation passes
metric_pseudo_label    learned/scene/object scale evidence is tight and validation passes
non_metric_pseudo_label geometry may be useful but metric scale is not accepted
rejected               reconstruction or validation evidence is too weak
```

The teacher is a dataset filter plus reconstructor. Rejecting bad videos is part of the product.

## Canonical Data Strategy

Atlas3R development must stay grounded in two canonical video tracks.

### Track A: `reference_metric`

A public indoor RGB sequence with metric evidence. The preferred first target is a small TUM RGB-D handheld indoor sequence. Acceptable alternatives include ARKitScenes, ScanNet++, or another public indoor sequence with registered RGB plus measured depth, measured pose, laser scan, known marker, or benchmark ground truth.

This track answers:

```text
When metric evidence exists, can Atlas3R load it, compare against it, and prevent false acceptance?
```

### Track B: `phone_room`

A user-captured phone RGB video of a real room. It starts as unanchored unless measured evidence is added.

This track answers:

```text
Can Atlas3R handle the target input while preserving uncertainty and refusing unearned metric claims?
```

No milestone should use synthetic toy scenes as its primary evidence. If canonical assets are missing, the correct behavior is an explicit missing-asset status.

## System Overview

The cohesive teacher stack is:

```text
canonical video asset
-> video inspection and keyframe proposal
-> metric reference adapter when measured evidence exists
-> ViPE/DA3 geometry artifact adapter
-> canonical ray/depth/pose packets
-> visibility and residual graph
-> optional SAM2 mask grouping
-> scale posterior and robust global refinement
-> static/dynamic inference
-> ray-fused TSDF and occupancy
-> floor-aligned robot grid
-> validation and metric acceptance gate
```

Only the geometry and mask proposals are learned components. The teacher value is in the representation, evidence accounting, optimization, mapping, and acceptance logic.

## Coordinate And Unit Invariants

These are non-negotiable:

```text
units: meters unless the field says otherwise
pose: T_world_camera is 4x4 camera-to-world
camera rays: unit vectors in camera coordinates
internal depth: radial range along the unit ray
scale: global scale variable with posterior uncertainty
state separation: unknown/free/occupied_static/movable_static/dynamic/predicted/measured stay distinct
```

Pixel lifting:

```text
X_camera(u,v) = radial_depth_m(u,v) * ray_camera(u,v)
X_world(u,v)  = R_world_camera * X_camera(u,v) + t_world_camera
X_metric      = scale * X_world when geometry is soft-metric
```

Depth alone is not the map. Pose alone is not the map. The core atom is:

```text
ray + radial depth + T_world_camera + confidence + scale state + static probability
```

## Mathematical Objective

Atlas3R optimizes for a static, scale-aware 3D world that explains the RGB video evidence while preserving uncertainty, separating dynamic content, and refusing unearned metric claims.

The teacher is not optimizing for a pretty mesh. It is optimizing for a robot-useful world model:

- static surfaces are geometrically consistent across views;
- camera motion explains observed parallax;
- free space observed by rays is not contradicted by static occupancy;
- dynamic and movable objects do not contaminate the static map;
- metric scale is accepted only when scale evidence and validation justify it.

### Variables

For keyframe `i` and pixel `u = (x, y)`:

$$
T_i = T_{\text{world}\leftarrow\text{camera},i} = (R_i, t_i) \in SE(3)
$$

$$
r_i(u) \in S^2
$$

$$
d_i(u) > 0
$$

$$
q_i(u) \in [0,1]
$$

$$
m_i(u) \in [0,1]
$$

$$
s > 0
$$

Where:

- `T_i` is the camera-to-world pose.
- `r_i(u)` is the unit camera ray.
- `d_i(u)` is radial depth along the ray.
- `q_i(u)` is geometry confidence.
- `m_i(u)` is static probability.
- `s` is global metric scale.

The lifted 3D point is:

$$
X_{c_i}(u) = d_i(u)\,r_i(u)
$$

$$
X_w(u) = R_i X_{c_i}(u) + t_i
$$

$$
X_m(u) = s\,X_w(u)
$$

Where:

- `X_c_i` is in camera coordinates.
- `X_w` is in reconstruction/world coordinates.
- `X_m` is in metric coordinates after applying global scale.

### Cross-View Projection

To compare a point observed in frame `i` against frame `j`:

$$
X_{c_j}(u) = R_j^\top \left(X_w(u) - t_j\right)
$$

$$
v = \pi_j\left(X_{c_j}(u)\right)
$$

$$
\hat d_j(v) = \left\|X_{c_j}(u)\right\|
$$

Where:

- `v` is the projected pixel in frame `j`.
- `pi_j` is the frame-`j` camera projection function.
- `hat d_j` is the predicted radial depth in frame `j`.

This comparison happens in reconstruction units because monocular reprojection is invariant to global scale.

### Depth Correction

Backbone depth is evidence, not truth. The optimizer may correct it, but corrections must stay low-dimensional and smooth unless multi-view evidence justifies change.

A refined depth field may be represented as:

$$
d_i(u) =
\exp\left(
\alpha_i \log d_i^0(u) + \beta_i + \delta_i(u)
\right)
$$

Where:

- `d_i^0(u)` is the backbone depth proposal.
- `alpha_i` is a per-frame depth scale correction.
- `beta_i` is a per-frame log-depth bias correction.
- `delta_i(u)` is a smooth residual correction field.

The system must not replace depth with unconstrained per-pixel hallucination.

### Core Residuals

Multi-view depth consistency:

$$
e_{\text{depth},ij}(u)
=
\log d_j(v) - \log \hat d_j(v)
$$

Image or feature consistency when reliable:

$$
e_{\text{image},ij}(u)
=
\phi_i(u) - \phi_j(v)
$$

Backbone depth prior:

$$
e_{\text{prior},i}(u)
=
\log d_i(u) - \log d_i^0(u)
$$

Scale evidence for a known or estimated metric length:

$$
e_{\text{scale},k}
=
\frac{s L^{\text{recon}}_k - L^{\text{evidence}}_k}{\sigma_k}
$$

Free-space consistency:

$$
0 < \lambda < d_i(u) - \epsilon
\Rightarrow
t_i + R_i(\lambda r_i(u))
\text{ is observed free space}
$$

A trusted static occupied voxel must not lie in space that trusted rays observed as free.

### Robust Objective

Real videos contain blur, compression, exposure changes, reflections, rolling shutter, moving objects, bad masks, and bad depth. Atlas3R must not optimize plain L2 over all observations.

The teacher minimizes a robust objective:

$$
\min_{\{T_i, d_i, r_i, m_i, s, V\}}
E
=
\lambda_d E_{\text{depth}}
+
\lambda_f E_{\text{image}}
+
\lambda_p E_{\text{prior}}
+
\lambda_{\text{free}} E_{\text{free-space}}
+
\lambda_s E_{\text{scale}}
+
\lambda_r E_{\text{room}}
+
\lambda_{\text{smooth}} E_{\text{smooth}}
$$

Each residual family must use robust losses or explicit outlier rejection. Valid choices include Huber, Cauchy, Tukey, or another documented robust loss.

Static fusion weight is:

$$
w_i(u) = q_i(u)\,m_i(u)
$$

Pixels with high dynamic probability contribute to dynamic evidence, not static occupancy.

### Robot-Useful Accuracy

For floor-cleaning robots, the teacher optimizes the map toward conservative traversability, not visual completeness.

The map must preserve:

- observed free space;
- observed static occupancy;
- movable-static occupancy;
- dynamic occupancy;
- unknown space;
- scale uncertainty;
- map confidence.

Unknown space is not free.  
Dynamic occupancy is not static occupancy.  
Movable-static occupancy is not free space.  
Hallucinated completion is not ground truth.

### Acceptance Is Separate From Optimization

A low objective value is not enough to claim metric ground truth.

Metric acceptance requires:

- `ScalePosterior`;
- `ValidationReport`;
- compatible scale evidence;
- acceptable free-space contradiction rate;
- acceptable held-out view consistency;
- acceptable dynamic leakage score.

The final status must be one of:

- `measured_metric`;
- `metric_pseudo_label`;
- `non_metric_pseudo_label`;
- `rejected`.

A good-looking reconstruction without accepted scale evidence is not metric GT.

## API Contracts

These contracts are binding once implemented. Code changes that alter a field, unit, status, or coordinate convention must update this section first.

### VideoAsset

Purpose: identify a canonical video track or external artifact source.

Required fields:

```text
asset_id
track_type: reference_metric | phone_room | external_candidate
source_uri_or_path
expected_modalities[]
status: available | missing_asset | corrupt | unsupported | unchecked
metadata
```

### VideoInspectionReport

Purpose: inspect real video files before reconstruction.

Required fields:

```text
asset_id
frame_count
fps_or_frame_timestamps
width_px
height_px
duration_s
codec_or_container_optional
sampled_frame_ids[]
blur_summary
exposure_summary
motion_summary
scene_change_summary
usable_frame_ratio
hard_rejection_reasons[]
soft_risk_flags[]
confidence
```

This report may reject corrupt/unreadable/extremely low-quality video. It must not decide metric acceptance.

### KeyframeProposal

Purpose: choose frames for geometry while keeping all frames available for validation and dense fusion.

Required fields:

```text
asset_id
selected_frame_ids[]
timestamps_s[]
selection_reasons[]
sharpness_scores[]
scene_change_scores[]
motion_or_baseline_proxy_scores[]
coverage_or_overlap_proxy_scores[]
risk_flags[]
```

M1 keyframes are proposals from image/video evidence. They may be revised after geometry is available.

### Camera Projection Contract

Every `FrameRayPacket.camera_model` must support:

```text
unproject(pixel_uv, radial_depth_m) -> X_camera[3]
project(X_camera[3]) -> pixel_uv, radial_depth_m, valid
```

Rules:

- `unproject` uses unit camera rays and radial depth: `X_camera = radial_depth_m * ray_camera`.
- `project` maps a camera-space 3D point back to pixel coordinates.
- Pinhole and fisheye models should use analytic projection.
- Dense ray-map-only models may use approximate inverse projection, such as nearest-ray angular lookup.
- Geometry packets without projection support cannot participate in visibility graph, reprojection factors, or cross-view depth factors.

### Depth Convention Conversion

Internal `radial_depth_m` is always distance along the unit camera ray.

Backbone adapters must convert source depth conventions:

```text
if source gives radial/range depth:
  radial_depth_m = source_depth_m

if source gives optical-axis z-depth:
  radial_depth_m = z_depth_m / max(ray_camera_z, epsilon)
```

Required metadata:

```text
source_depth_convention: radial_range | optical_z | inverse_depth | disparity | unknown
```

`unknown` depth convention is not allowed for accepted geometry packets.

### FrameRayPacket

Purpose: canonical per-frame geometry from measured reference data, a geometry backbone, or an optimizer.

Required fields:

```text
asset_id
frame_id
T_world_camera[4,4]
rays_camera[H,W,3]
radial_depth_m[H,W]
confidence[H,W]
camera_model
source
source_depth_convention
uncertainty
provenance
```

Optional fields:

```text
intrinsics
rolling_shutter_model
depth_residual_field
camera_confidence
```

### ScaleEvidence

Purpose: represent one source of metric scale information.

Required fields:

```text
evidence_id
evidence_type
measured
source
frame_ids[]
confidence
provenance
```

Optional fields:

```text
object_or_region_id
length_mean_m
length_std_m
scale_mean
scale_std
residual_after_optimization
```

Suggested `evidence_type` values:

```text
measured_depth
measured_pose
manual_distance
known_marker
benchmark_gt
object_size_prior
architecture_prior
learned_metric_depth_prior
scene_layout_prior
```

Rules:

- `measured=true` only for real metric measurements: LiDAR, RGB-D, ARKit/ARCore depth or pose, measured markers, benchmark GT, known measured trajectory, or manually supplied measured distances.
- Learned metric priors, object-size priors, and architecture priors are scale evidence, not measured truth.
- `measured_metric` requires at least one compatible measured evidence source and validation pass.
- Learned and prior-based evidence can support `metric_pseudo_label`, not `measured_metric`.

### ScalePosterior

Purpose: make metric scale explicit.

Required fields:

```text
scale_mean
scale_std
relative_scale_uncertainty
scale_evidence_ids[]
anchor_residuals[]
metric_acceptance_status: measured_metric | metric_pseudo_label | non_metric_pseudo_label | rejected
rejection_reasons[]
```

Operating bands before calibration:

```text
excellent: relative_scale_uncertainty < 0.03
usable:    relative_scale_uncertainty < 0.07
weak:      relative_scale_uncertainty < 0.15
reject:    relative_scale_uncertainty >= 0.15
```

These bands are defaults, not laws. Calibrate them against `reference_metric`.

### VisibilityGraph

Purpose: connect frames that likely observe common static surfaces.

Required fields:

```text
asset_id
nodes: keyframe IDs
temporal_edges
overlap_edges
loop_edges
scale_edges
edge_measurements
```

Edge measurements should include:

```text
sample_count
depth_consistency_summary
reprojection_or_image_consistency_summary
free_space_compatibility_summary
confidence_weight
risk_flags[]
```

### MaskTrackSet

Purpose: represent SAM2 or Grounded-SAM2 mask groups.

Required fields:

```text
asset_id
track_id
frame_ids[]
mask_rle_or_bitmap
mask_confidence
prompt_or_source
semantic_label_optional
provenance
```

Mask grouping does not itself mark static or dynamic.

### StaticDynamicState

Purpose: record geometry-led static/dynamic inference.

Required fields:

```text
asset_id
frame_id
static_probability[H,W]
dynamic_probability[H,W]
unknown_probability[H,W]
movable_static_probability_optional[H,W]
mask_track_decisions[]
residual_summary
```

### VoxelMapState

Purpose: hold volumetric TSDF, occupancy, and uncertainty.

Required fields:

```text
asset_id
voxel_size_m
coordinate_frame
tsdf_value
tsdf_weight
occupancy_log_odds
free_space_count
surface_count
dynamic_count
uncertainty
```

### OccupancyGrid2D

Purpose: floor-aligned robot grid for downstream training or planning.

Required fields:

```text
grid_frame
resolution_m
origin_world
P_free[x,y]
P_occupied_static[x,y]
P_movable_static[x,y]
P_dynamic[x,y]
P_unknown[x,y]
height_min_m[x,y]
height_max_m[x,y]
scale_uncertainty
map_confidence
```

Rules:

- `P_free` means observed free space, not absence of observed obstacles.
- `P_occupied_static` means structural or stable static occupancy.
- `P_movable_static` means geometrically static but likely non-structural movable obstacles.
- `P_dynamic` means moving or temporally inconsistent occupancy.
- `P_unknown` means insufficient ray evidence.
- Unknown is not free. Dynamic is not static. Movable-static is not free.

### ValidationReport

Purpose: decide whether outputs are accepted for metric training.

Required fields:

```text
asset_id
held_out_render_error
free_space_contradiction_rate
scale_posterior
floor_wall_consistency
dynamic_leakage_score
accepted_for_metric_training
acceptance_category: measured_metric | metric_pseudo_label | non_metric_pseudo_label | rejected
rejection_reasons[]
```

Metric acceptance requires this report plus a compatible `ScalePosterior`.

## Core Modules

### Module 1: Canonical Asset Registry

Purpose: keep work grounded in the two canonical tracks.

Inputs:

```text
local paths or URIs for reference_metric and phone_room
optional metadata for calibration, measured distances, or external artifacts
```

Outputs:

```text
VideoAsset records
asset availability report
missing/corrupt/unsupported statuses
```

Rules:

- Missing assets are not replaced with synthetic data.
- A missing asset is useful information and should be reported clearly.
- The phone track is not metric unless scale evidence is supplied later.

### Module 2: Video Inspection And Keyframe Proposal

Purpose: inspect real RGB videos and choose initial frames for geometry.

Inputs:

```text
VideoAsset with readable RGB video or decoded frame sequence
```

Outputs:

```text
VideoInspectionReport
KeyframeProposal
```

Must compute from real frames when available:

```text
frame count, resolution, fps or timestamps
blur/sharpness summary
exposure/clipping summary
motion/scene-change proxy
duplicate/near-static-frame ratio
usable frame ratio
keyframe candidates with reasons
```

Must not claim:

```text
metric scale
static/dynamic segmentation
true parallax from geometry
3D reconstruction success
```

Pure rotation, stabilization, zoom, and weak overlap can be flagged as risks before geometry, but strong rejection for those belongs after geometry evidence exists.

### Module 3: Metric Reference Adapter

Purpose: load measured data for `reference_metric` into the same evidence system used by monocular outputs.

Inputs:

```text
RGB frames
measured depth or RGB-D
camera intrinsics/calibration
measured trajectory or benchmark ground truth
optional laser/mesh/scan evidence
```

Outputs:

```text
FrameRayPacket from measured depth when available
ScaleEvidence with measured=true
reference validation targets
```

Rules:

- Measured reference data is for evaluation/calibration, not for cheating inside the monocular teacher path.
- The same coordinate/depth conventions must be used as the monocular path.

### Module 4: Geometry Backbone Artifact Adapter

Default external backbone:

```text
ViPE with DA3 pipeline
```

Fallback:

```text
MegaSaM only when explicitly selected for hard dynamic or weak-parallax videos
```

Purpose: normalize external geometry proposals into Atlas3R packets.

Inputs:

```text
external model artifact directory or manifest
selected frames/keyframes
```

Outputs:

```text
FrameRayPacket[]
geometry confidence
camera confidence
provenance
clear unavailable/missing_artifact status
```

Rules:

- Do not vendor model repositories or weights.
- Do not import heavy model dependencies at package import time.
- Convert all depth to radial depth before acceptance.
- Provide a projection-capable camera model or mark the packet unusable for visibility/cross-view factors.

### Module 5: Mask Grouping Artifact Adapter

Purpose: ingest object/group masks from SAM2 or Grounded-SAM2 artifacts.

Inputs:

```text
external mask artifact directory or manifest
```

Outputs:

```text
MaskTrackSet
```

Rules:

- SAM2 groups pixels over time.
- Geometry decides static/dynamic/unknown.
- Grounded-SAM2 prompts are optional and mainly for scale-anchor candidates.

### Module 6: Visibility And Residual Graph

Purpose: stitch observations by evidence, not by post-hoc mesh merging.

For a pixel `u` in frame `i`:

```text
X_world = R_i * d_i(u) * r_i(u) + t_i
X_camera_j = R_j^T * (X_world - t_j)
v = project_j(X_camera_j)
```

If `v` is valid and the target depth agrees, frames likely overlap.

Required residual families:

```text
depth consistency
reprojection/image consistency when available
free-space compatibility
confidence-weighted edge quality
```

Rules:

- A visibility edge is evidence, not proof.
- Low-confidence/dynamic/occluded pixels should not dominate graph edges.

### Module 7: Scale Posterior And Metric Gate

Purpose: decide whether a reconstruction can be used as metric data.

Evidence classes:

```text
measured evidence: depth, pose, marker, benchmark, known trajectory, manual measured distance
soft evidence: learned metric depth, object-size prior, architecture prior, scene-layout prior
```

Acceptance rules:

```text
measured_metric requires measured evidence + tight posterior + validation pass
metric_pseudo_label allows soft evidence + tight posterior + validation pass
non_metric_pseudo_label allows useful geometry without accepted metric scale
rejected means the evidence is too weak or contradictory
```

A nice-looking mesh is not scale evidence.

### Module 8: Robust Global Refinement And Static/Dynamic Inference

Purpose: refine geometry while preventing dynamic objects from contaminating the static map.

Variables:

```text
T_i                 refined camera pose
d_i                 refined depth through constrained correction
r_i or camera model  refined rays/camera state where allowed
s                   global scale
m_i                 static probability
```

Depth correction form:

```text
d_i(u) = exp(alpha_i * log(d_i_initial(u)) + beta_i + delta_i(u))
```

Residual terms:

```text
multi-view depth consistency
image/feature consistency where reliable
backbone prior
free-space consistency
scale evidence
room/floor/wall regularization where supported
smoothness of corrections
```

Use robust losses. Do not let one bad model prediction, moving person, reflection, or blur patch dominate the solution.

Static/dynamic rule:

```text
SAM2 says: these pixels belong together
geometry says: static, dynamic, movable_static, or unknown
```

Dynamic or uncertain pixels are excluded before static map fusion.

### Module 9: Ray-Fused Static Mapping

Purpose: convert accepted static rays into TSDF, log-odds occupancy, and uncertainty.

For each trusted static ray:

```text
camera origin -> before surface: free evidence
near surface: surface/TSDF/occupied evidence
behind surface: unknown
```

Rules:

- Fuse rays, not just points.
- Unknown is never converted to free.
- Dynamic pixels are skipped or counted separately before fusion.
- Mesh quality is not occupancy quality.

### Module 10: Floor-Aligned Robot Occupancy

Purpose: produce the robot-relevant 2D grid from 3D evidence.

Inputs:

```text
VoxelMapState
floor plane estimate
robot height/collision band
scale posterior
```

Cell classes:

```text
free: observed free-space rays and no obstacle in collision band
occupied_static: structural/stable obstacle in collision band
movable_static: static during video but likely movable obstacle
dynamic: moving or temporally inconsistent object occupied the cell
unknown: insufficient ray evidence
```

The primary output is multichannel, not binary.

### Module 11: Validation And Acceptance

Purpose: decide whether the output can enter a training dataset.

Validation evidence:

```text
held-out render/depth consistency
free-space contradiction rate
scale posterior uncertainty
floor/wall plausibility when available
dynamic leakage score
reference metric error when measured evidence exists
```

Final status:

```text
measured_metric
metric_pseudo_label
non_metric_pseudo_label
rejected
```

For `phone_room`, metric acceptance requires strong evidence. Otherwise the correct result is non-metric pseudo-label or rejection.

## Verification Policy

Atlas3R should be verified on the canonical tracks, not on toy synthetic scenes.

Preferred verification artifacts:

```text
asset manifest report
video inspection report
keyframe proposal report
reference metric comparison report
backbone artifact normalization report
visibility residual report
scale posterior report
map/occupancy validation report
```

Do not add broad unit-test scaffolding or synthetic scene pipelines to create the illusion of progress. Small invariant checks inside code are allowed when they prevent dangerous contract violations, but the project is judged by data-grounded reports on `reference_metric` and `phone_room`.

## External Technical References

- ViPE: https://github.com/nv-tlabs/vipe
- Depth Anything 3: https://github.com/bytedance-seed/depth-anything-3
- MegaSaM: https://arxiv.org/html/2412.04463v1
- SAM2: https://github.com/facebookresearch/sam2
- Grounded-SAM2: https://github.com/IDEA-Research/Grounded-SAM-2
- TUM RGB-D benchmark: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- ARKitScenes: https://machinelearning.apple.com/research/arkitscenes
- ScanNet++: https://kaldir.vc.in.tum.de/scannetpp/
- Open3D TSDF integration: https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html
- nvblox: https://arxiv.org/html/2311.00626v2
