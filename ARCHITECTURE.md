# Atlas3R Architecture

This document is the technical source of truth for Atlas3R. The README points to work; this file defines what the work must mean.

Atlas3R is the offline TEACHER in a teacher→student robotics stack: it turns real RGB video into scale-aware 3D occupancy labels behind a GT-free verification gate strict enough that accepted labels can train a real-time student (ego-centric collision-band occupancy on embedded SoCs) for commercial robots. There is no robot fleet; the gate is the fleet substitute, and verified yield is the product. The teacher must not pretend that unanchored monocular RGB is measured metric ground truth — and the gate is strengthened only by evidence, never loosened to pass a scene.

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

Atlas3R development must stay grounded in canonical video tracks: measured-metric
reference scenes spanning a DIFFICULTY SPREAD (so the acceptance gate reflects
realistic motion, not a gentle best case) plus the unanchored phone target.

### Track A: `reference_metric` scenes (measured, multi-scene gate)

Public indoor RGB sequences with metric evidence, registered as a difficulty spread so
the gate cannot be over-fit to an easy case. Current canonical scenes (TUM RGB-D, which
gives RGB + measured depth + mocap-accurate ground-truth trajectory):

```text
reference_metric        = freiburg1_xyz   (GENTLE, low-rotation handheld sweep)
reference_metric_desk   = freiburg1_desk  (HARDER desk-orbit; exposes pose degradation
                                           under realistic motion -- see the band
                                           evidence Phase 4: the xyz-only gate
                                           over-states real-world performance)
reference_metric_room   = freiburg1_room  (HARDEST full room loop; exposes keyframe
                                           under-sampling and drift)
```

Acceptable alternatives/additions include ARKitScenes, ScanNet++, or another public
indoor sequence with registered RGB plus measured depth/pose/laser/marker/benchmark GT.
A new measured scene is registered by adding a `track_type: reference_metric` entry to
`config/canonical_assets.json` (shared intrinsics via the per-scene metadata sidecar)
and to `evaluate.CANONICAL_TRACKS`.

These scenes answer:

```text
When metric evidence exists, can Atlas3R load it, compare against it, prevent false
acceptance, AND do so consistently across EASY and HARD camera motion?
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
-> geometry backbone artifact adapter (MapAnything default, DA3 fallback)
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
\begin{aligned}
d_i(u) &=
\exp\left(
\alpha_i \log d_i^0(u) + \beta_i + \delta_i(u)
\right)
\end{aligned}
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
\begin{aligned}
e_{\text{depth},ij}(u)
&=
\log d_j(v) - \log \hat d_j(v)
\end{aligned}
$$

Image or feature consistency when reliable:

$$
\begin{aligned}
e_{\text{image},ij}(u)
&=
\phi_i(u) - \phi_j(v)
\end{aligned}
$$

Backbone depth prior:

$$
\begin{aligned}
e_{\text{prior},i}(u)
&=
\log d_i(u) - \log d_i^0(u)
\end{aligned}
$$

Scale evidence for a known or estimated metric length:

$$
\begin{aligned}
e_{\text{scale},k}
&=
\frac{s L^{\text{recon}}_k - L^{\text{evidence}}_k}{\sigma_k}
\end{aligned}
$$

Free-space consistency:

$$
\begin{aligned}
0 < \lambda < d_i(u) - \epsilon
&\Rightarrow \\
&t_i + R_i(\lambda r_i(u))
\text{ is observed free space}
\end{aligned}
$$

A trusted static occupied voxel must not lie in space that trusted rays observed as free.

### Robust Objective

Real videos contain blur, compression, exposure changes, reflections, rolling shutter, moving objects, bad masks, and bad depth. Atlas3R must not optimize plain L2 over all observations.

The teacher minimizes a robust objective:

$$
\begin{aligned}
\min_{\{T_i, d_i, r_i, m_i, s, V\}}
E
&=
\lambda_d E_{\text{depth}} \\
&\quad+
\lambda_f E_{\text{image}} \\
&\quad+
\lambda_p E_{\text{prior}} \\
&\quad+
\lambda_{\text{free}} E_{\text{free-space}} \\
&\quad+
\lambda_s E_{\text{scale}} \\
&\quad+
\lambda_r E_{\text{room}} \\
&\quad+
\lambda_{\text{smooth}} E_{\text{smooth}}
\end{aligned}
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

The final status must be one of the four Truth Boundary categories.

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

### VoxelOccupancyGrid3D

Purpose: PRIMARY robot-facing output. A per-voxel multichannel occupancy field
bounded to the robot's vertical collision envelope, in the floor-aligned metric
(or reconstruction) frame. Units are meters.

Required fields:

```text
grid_frame
voxel_size_m
origin_world[3]                       # world coord of the (0,0,0) voxel corner
floor_axis                            # 0|1|2: the world axis cropped to the band
band_min_m, band_max_m                # band extent along floor_axis (world coords)
P_free[A0,A1,B]
P_occupied_static[A0,A1,B]
P_movable_static[A0,A1,B]
P_dynamic[A0,A1,B]
P_unknown[A0,A1,B]
map_confidence[A0,A1,B]
scale_uncertainty                     # scene-level
acceptance_category                   # measured_metric | metric_pseudo_label | ...
```

Where `B` is the number of band slices along `floor_axis`. The band is
`[floor_plane, floor_plane + robot_collision_height + margin]`, CONFIG-DRIVEN via
`configs/robot_envelope.json` (`RobotEnvelopeConfig`). Resolution is spent only
inside the collision envelope; the ceiling / full room volume is never modelled.

`grid_frame` is FLOOR-ALIGNED, per reconstruction. The fuser derives an
up-alignment rotation `R_up` that maps the estimated floor NORMAL onto `floor_axis`
and rotates the whole reconstruction (surfaces, camera origins, ray directions) by
it before fusion, so the axis-aligned band crop is a genuine floor-parallel slab
(not an oblique cut through a tilted floor). `grid_frame` is then
`<metric|reconstruction>_world_floor_aligned`. Candidate and measured tracks are
aligned to their OWN floors independently; a track's alignment is never reused for
the other, and measured data never aligns the candidate. When the floor RANSAC is
too weak to trust the up vector (low `inlier_ratio`), NO alignment is fabricated:
`R_up` is identity, `grid_frame` is
`<metric|reconstruction>_world_axis_aligned_band_not_floor_aligned`, and a loud
`floor_normal_unreliable_band_not_floor_aligned` blocker is recorded. The residual
floor tilt (normal vs `floor_axis`) BEFORE and AFTER alignment is reported so the
evaluation harness surfaces it.

Rules (per-voxel, non-negotiable):

- Probability convention mirrors `OccupancyGrid2D`: each channel is a probability
  in `[0,1]` with PAIRWISE non-collapse (NOT a strict sum-to-one simplex).
- Unknown is never free (`P_free + P_unknown <= 1`). Dynamic is never static
  (`P_dynamic + P_occupied_static <= 1`). Movable-static is never free
  (`P_movable_static + P_free <= 1`).
- Free space comes from RAY TRAVERSAL only. Space behind a surface along a ray
  stays unknown. Dynamic surfaces are painted into `P_dynamic` only -- never into
  static occupancy and never free-carving.

### OccupancyGrid2D

Purpose: floor-aligned robot grid for downstream training or planning. It is the
PURE top-down projection of `VoxelOccupancyGrid3D` (single source of truth = the
3D field). It must not be fused independently.

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

Projection rule (per column over the band, contract-invariant by construction):

```text
P_occupied_static = max over band column
P_movable_static  = max over band column
P_dynamic         = min(max_dynamic, 1 - P_occupied_static)
P_free            = max_free * (1 - max(occupied, movable, dynamic))
P_unknown         = 1 where the column is unobserved, else a small residual
```

It drops the height-within-band at which an obstacle occurs; that height is kept
in `height_min_m` / `height_max_m`.

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
band3d_agreement (optional)
```

Metric acceptance requires this report plus a compatible `ScalePosterior`.

`band3d_agreement` (optional) holds the per-voxel agreement of the monocular 3D
field vs the measured 3D field inside the collision band: per-class agreement,
`occupied_static_iou`, `free_space_contradiction_rate` (candidate calls free where
the measured GT sees an obstacle -- robot-critical), `dynamic_leakage_rate`,
`coverage_of_measured_band`, and the Sim(3) alignment used. It is REPORTAGE: it
never gates `accepted_for_metric_training` (the category is driven by the scale
posterior). It is absent / `missing_measured_3d_reference` when no measured 3D
reference exists (e.g. `phone_room`); a too-small overlap yields an explicit
`insufficient_overlap_for_sim3_band_comparison` status, never a fabricated number.

### GT-Free Acceptance Cascade (Stage 0 / Stage 1)

A reconstruction with no measured evidence carries the burden of proving its own
trustworthiness from internal evidence. The acceptance gate is a precedence
cascade built on three principles:

```text
independence      : auditors must use different evidence than the builder optimized
evidence mass     : a consistency score over near-zero co-observation is vacuous,
                    not reassuring -- low evidence mass rejects regardless of scores
measured authority: a signal only counts where it has been shown it WOULD have
                    detected an error (detection-limit calibration, see Module 11)
```

Stages, evaluated before the classic consistency checks; ALL failing stages
contribute rejection reasons (no early-exit -- every defect is reported):

```text
Stage 0 - evidence mass (visibility graph, already computed, previously unread):
  median_reprojection_inbounds_ratio >= 0.30
  edges_with_depth_residual / edge_count >= 0.70
  mean_confidence_weight >= 0.30
  -> failure reasons: evidence_mass_median_inbounds_ratio_too_low:<v>
                      evidence_mass_depth_residual_edge_fraction_too_low:<v>
                      evidence_mass_mean_confidence_weight_too_low:<v>
Stage 1 - gravity alignment (floor estimate, already computed, previously unread):
  up_alignment_applied must be true (floor RANSAC reliable, band floor-aligned)
  -> failure reason:  gravity_alignment_unverified_band_not_floor_aligned_inlier:<v>
Stage 2 - classic consistency (existing): held-out render error, free-space
  contradiction rate, dynamic leakage.
```

Scope rule: the cascade applies ONLY to paths whose `ScalePosterior` has no
measured evidence (`metric_pseudo_label` candidates). A measured baseline's
authority comes from instruments, not internal consistency; it is the yardstick,
not the examinee. The cascade outcome is surfaced in a `gate_cascade` block on
the validation report (per-stage inputs, thresholds, verdict, and whether the
stage applied).

Threshold honesty: the Stage 0 thresholds sit inside a measured chasm on the
canonical scenes (inbounds ratio: bad scenes 0.000/0.035 vs good scenes
0.605/0.648; depth-residual edge fraction: 0.434/0.537 vs 1.000/1.000; mean
confidence weight: 0.116/0.187 vs 0.532/0.541). Values inside the chasm are
provisional and carry no calibrated authority between the clusters; they must be
re-derived from injected-corruption response curves (Module 11 detection-limit
calibration) before any claim is made about intermediate-quality scenes.

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
MapAnything (facebook/map-anything-apache) via tools/run_mapanything_backbone.py
fallback: Depth Anything 3 via tools/run_da3_backbone.py
(adoption evidence: docs/sota_backbone_research.md)
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

Variables, depth-correction form, and residual families are defined ONCE in the
Mathematical Objective section above; Module 8 implements them. Use robust losses. Do not let one bad model prediction, moving person, reflection, or blur patch dominate the solution.

Static/dynamic rule:

```text
SAM2 says: these pixels belong together
geometry says: static, dynamic, movable_static, or unknown
```

Dynamic or uncertain pixels are excluded before static map fusion.

### Module 9: Ray-Fused Static Mapping

Purpose: convert accepted rays into TSDF, log-odds occupancy, per-voxel class
counts, and uncertainty.

For each trusted ray:

```text
camera origin -> before surface: free evidence (static/movable rays only)
near surface: surface/TSDF/occupied evidence, tagged by class
behind surface: unknown
```

Rules:

- Fuse rays, not just points.
- Unknown is never converted to free.
- Each surface sample is tagged occupied_static / movable_static / dynamic. Static
  and movable rays carve free in front and mark their class at the surface;
  dynamic rays mark ONLY the dynamic channel (never static, never free-carving).
- Mesh quality is not occupancy quality.

#### Candidate Occupancy-Estimation Policy

The raw ray fusion above is the GT-grade yardstick. The monocular CANDIDATE map may
additionally apply a generic, embodiment-agnostic occupancy-estimation policy
(config-driven via `RobotEnvelopeConfig`; all levers default OFF so an unconfigured
run reproduces the raw fuser byte-for-byte). The policy is applied ONLY to the
candidate; the measured 3D GT field is always fused raw, so a band-agreement gain
is unambiguously a better candidate, never an easier yardstick.

```text
free_carve_margin_m         DIRECTIONAL truncation: free is retracted only in the
                            column directly BELOW a CONFIDENT fused surface (toward
                            the floor) within this distance -- the grazing-ray flood
                            that masks an obstacle's support column. Lateral free
                            (beside the obstacle) is preserved. Retracted free becomes
                            UNKNOWN, never occupied.
occupancy_support_height_m  gravity/support prior: a detected obstacle rests on the
                            floor, so occupancy is propagated DOWNWARD within the
                            collision band by up to this height. By default it fills
                            ONLY UNKNOWN voxels below a CONFIDENT obstacle.
occupancy_support_overrides_free  when set, support ALSO fills observed-FREE base
                            voxels below a confident obstacle. It lowers band_fsc but
                            CLAIMS occupied over ray-traversal-observed free space -- a
                            FABRICATION barred by the free-space rule below ("free
                            comes from RAY TRAVERSAL only"). Left OFF on purpose
                            (honest labels over metric scores); the honest truncation
                            lever (free -> UNKNOWN, not occupied) is the alternative.
occupancy_support_min_count minimum fused occupied-hit count for a voxel to act as a
                            truncation/support SOURCE, so single-hit depth noise high
                            in the band cannot conjure occupancy or retract floor.
occupancy_close_voxels      in-plane morphological closing radius that bridges small
                            gaps enclosed by occupancy without expanding outward.
```

A "confident obstacle" is a voxel with occupied/movable hit count >=
`occupancy_support_min_count`. These levers ADD occupancy into UNKNOWN space or
RETRACT free to UNKNOWN only; they never convert unknown to free, never override an
observed-free voxel with occupancy laterally, never paint dynamic into static, and
the downstream per-voxel probability construction preserves the pairwise non-collapse
invariants by design. The measured baseline (`measured_metric`) never receives the
policy, so a band-agreement gain is always a better candidate, never an easier
yardstick.

### Module 10: Floor-Aligned Robot Occupancy

Purpose: produce the robot-relevant occupancy from 3D evidence. The PRIMARY output
is the `VoxelOccupancyGrid3D` -- a per-voxel multichannel field bounded to the
robot collision band. The `OccupancyGrid2D` is its pure top-down projection.

Inputs:

```text
VoxelMapState / per-voxel class counts
floor plane estimate
robot collision envelope (config-driven: collision_height + margin)
scale posterior
```

Cell/voxel classes:

```text
free: observed free-space rays and no obstacle
occupied_static: structural/stable obstacle
movable_static: static during video but likely movable obstacle
dynamic: moving or temporally inconsistent object
unknown: insufficient ray evidence
```

The primary output is multichannel and 3D, not binary. The grid is bounded to the
collision envelope; resolution is not spent on the ceiling or full room volume.

### Module 11: Validation And Acceptance

Purpose: decide whether the output can enter a training dataset.

Validation evidence:

```text
evidence-mass statistics from the visibility graph (Stage 0 of the cascade)
gravity/floor alignment reliability (Stage 1 of the cascade)
held-out render/depth consistency
free-space contradiction rate
scale posterior uncertainty
floor/wall plausibility when available
dynamic leakage score
reference metric error when measured evidence exists
per-voxel band agreement vs the measured 3D field (occupied IoU, free-space
  contradiction, dynamic leakage, coverage) when a measured reference exists
```

Detection-limit calibration (planned, the "measured authority" principle): inject
known corruptions into a finished candidate reconstruction (corruption families
chosen OUTSIDE the refiner's parametric span, so the refiner cannot simply repair
them), re-score every GT-free signal, and record per scene which injected
magnitudes each signal detects. A signal that cannot detect an injected
corruption has no authority on that scene and its clean reading is reported as
`no_authority`, never as evidence of correctness. Detection limits are reported
in scene-relative units (fraction of trajectory span) because absolute scale is
gauge-free without an anchor.

Final status: one of the four Truth Boundary categories.

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

- Depth Anything 3: https://github.com/bytedance-seed/depth-anything-3
- SAM2: https://github.com/facebookresearch/sam2
- Grounded-SAM2: https://github.com/IDEA-Research/Grounded-SAM-2
- TUM RGB-D benchmark: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- ARKitScenes: https://machinelearning.apple.com/research/arkitscenes
- ScanNet++: https://kaldir.vc.in.tum.de/scannetpp/
- Open3D TSDF integration: https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html
- nvblox: https://arxiv.org/html/2311.00626v2
