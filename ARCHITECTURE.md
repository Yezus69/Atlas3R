# Atlas3R Architecture

This file defines the target architecture, mathematical model, data contracts,
and acceptance logic for Atlas3R.

Atlas3R is a Scale-Aware Monocular Reconstruction Teacher for RGB videos:

```text
RGB video
  -> one video geometry backbone
  -> scale-aware global refinement
  -> static/dynamic separation
  -> ray-based TSDF + occupancy fusion
  -> metric-quality gate
```

Monocular multi-view geometry has an unobservable global scale until anchored
by an external metric cue, learned metric prior, known object/scene anchor,
measured trajectory, RGB-D, LiDAR, or similar signal. Atlas3R therefore treats
scale as a state variable with posterior uncertainty:

```text
estimate geometry
estimate scale posterior
validate static map and free space
accept metric pseudo-labels only when the posterior and validation pass
```

For web videos, the teacher is a dataset filter plus reconstructor, not just a
reconstructor.

## Minimal Learned Stack

The core teacher uses a small learned stack and puts project value in the
optimizer, mapper, validator, and acceptance gate.

### Component 1: Video Geometry Backbone

Default:

```text
ViPE with DA3 pipeline
```

Rationale: the target input is raw monocular video with unknown calibration.
ViPE is intended for camera intrinsics, camera motion, and dense near-metric
depth from unconstrained raw video, including pinhole, wide-angle, and panorama
footage. DA3 is relevant because it uses a depth-ray representation and supports
multi-view depth, pose-conditioned depth, pose estimation, intrinsics
estimation, and metric-scale variants.

Backbone contract:

```text
VideoGeometryBackbone(video) ->
  rays_i(u, v)
  depth_i(u, v)
  pose_i
  confidence_i(u, v)
  intrinsics_i or camera_model_i
```

Expected output:

```text
camera intrinsics or ray maps
camera poses
dense depth / range maps
depth confidence
camera confidence
```

Fallback:

```text
MegaSaM for selected hard dynamic / weak-parallax videos
```

MegaSaM is an alternative backend, not a parallel default. Running multiple
geometry backbones is reserved for explicit diagnosis or high-value cases.

### Component 2: Segmentation / Object Tracking

Use SAM2 for video object masks and mask propagation. First-version SAM2 usage
is mask grouping, not semantics.

SAM2 gives grouped masks such as:

```text
person-like blob
chair-like blob
dog-like blob
moving shadow-like blob
unknown object blob
```

Geometry decides whether a mask is static:

```text
Does this mask move consistently with the static world?
```

If no, mark it dynamic or uncertain and exclude it from static-map fusion.

Grounded-SAM2 is optional and only for scale-anchor prompts such as:

```text
door
outlet
stairs
refrigerator
kitchen cabinet
toilet
bathtub
bed
table
```

Open-vocabulary semantics must not be mandatory for all videos because it adds
failure modes.

### Component 3: Optimizer And Mapper

This is the actual teacher. Learned models propose geometry. Atlas3R enforces:

```text
multi-view geometric consistency
static-scene consistency
free-space consistency
global scale consistency
robot occupancy consistency
```

The optimizer, validator, and ray mapper decide whether the model proposal is
usable.

## Canonical Internal Representation

The system is ray-map first rather than pinhole-`K`-centric. Ray maps keep
unknown intrinsics, wide-angle lenses, cropped/stabilized video, frame-varying
camera models, and panorama-like content representable.

For each frame `i`, store:

$$
I_i
$$

$$
T_{wc_i} =
\begin{bmatrix}
R_i & t_i \\
0 & 1
\end{bmatrix}
$$

$$
r_i(u, v) \in S^2
$$

$$
d_i(u, v)
$$

$$
q_i(u, v)
$$

where:

```text
I_i       = RGB frame
T_wc_i    = camera-to-world pose
r_i(u,v)  = unit ray for pixel (u,v), in camera coordinates
d_i(u,v)  = radial depth/range along the ray
q_i(u,v)  = confidence
```

One pixel becomes one 3D point:

$$
X_{c_i}(u, v) = d_i(u, v) r_i(u, v)
$$

$$
X_w(u, v) = R_i X_{c_i}(u, v) + t_i
$$

If the geometry backbone is soft-metric, introduce a global scale variable `s`:

$$
X_w^\text{metric}(u, v)
= s \left(R_i d_i(u, v) r_i(u, v) + t_i\right)
$$

Depth alone is not the map. Pose alone is not the map. Intrinsics alone are not
the map. The useful atom is:

```text
ray + depth + pose + scale + static probability
```

## Data Flow

### 1. RGB Video Enters

Input:

```text
30 fps RGB video
unknown camera
unknown intrinsics
unknown scale
possible rolling shutter
possible video stabilization
possible dynamic objects
possible motion blur
possible compression
```

The first step is reconstructability scoring, not reconstruction.

Important signals:

```text
enough translation/parallax
not pure rotation
not mostly dynamic foreground
not severe digital zoom
not severe stabilization crop changes
not too much blur
enough visible static floor/wall/object structure
enough overlap between views
```

For web video, many videos should be rejected.

For floor-cleaner data, prefer:

```text
visible floor
slow walking motion
sideways translation, not only panning
loop-like path or repeated views
limited zoom
limited motion blur
limited moving people/pets
```

Pure pan has weak reconstruction evidence. Translational motion through rooms,
circling objects, and returning to a previously seen doorway or room provide
better parallax and loop evidence.

### 2. Frame Selection

Global geometry optimization uses selected keyframes. All frames remain
available for validation and dense fusion.

A good keyframe has:

```text
sharp image
enough texture or structure
enough baseline from previous keyframes
sufficient overlap with existing map
low dynamic-object ratio
visible floor/room structure if indoor
```

The keyframe set should maximize information, not frame count:

```text
30 fps video
  -> 1-5 geometry keyframes per second
  -> extra frames around turns, loops, and high-parallax moments
  -> all frames retained for final render validation
```

### 3. Run Geometry Backbone

Run ViPE/DA3 on selected frames or chunks.

Expected output for each keyframe `i`:

```text
camera pose T_i
camera model / ray map r_i
dense depth d_i
confidence q_i
```

Initial reconstruction:

$$
P_i = \left\{ R_i d_i(u) r_i(u) + t_i \right\}_{u \in \Omega_i}
$$

This is initialization. At this point there are per-frame point clouds that
approximately live in one world coordinate system.

### 4. Build Visibility Graph

The teacher asks:

```text
Which frames observe the same surfaces?
```

For frame pair `(i, j)`, project frame `i` points into frame `j`:

$$
X_w = R_i d_i(u) r_i(u) + t_i
$$

$$
X_{c_j} = R_j^\top (X_w - t_j)
$$

$$
v = \pi_j(X_{c_j})
$$

If `v` lands inside frame `j` and predicted depth agrees with `d_j(v)`, the
frames overlap.

Graph:

```text
nodes:
  keyframes

edges:
  frame pairs that see the same static surface

edge measurements:
  reprojection consistency
  depth consistency
  feature/color consistency
  free-space consistency
```

Stitching is implemented through global optimization over poses, depths, scale,
camera/ray corrections, and static masks, not through post-hoc mesh stitching or
point-cloud averaging.

## Global Optimization

The geometry backbone gives:

$$
T_i^0,\quad d_i^0,\quad r_i^0
$$

The optimizer solves for:

$$
T_i,\quad d_i,\quad r_i,\quad s,\quad m_i
$$

where:

```text
T_i = refined camera pose
d_i = refined depth
r_i = refined ray/camera model
s   = global metric scale
m_i = static probability mask
```

Depth correction is represented by constrained per-frame and smooth residual
parameters rather than independent free variables per pixel:

$$
d_i(u) =
\exp\left(
\alpha_i \log d_i^0(u) + \beta_i + \delta_i(u)
\right)
$$

where:

```text
alpha_i = per-frame depth scale correction
beta_i  = per-frame log-depth bias correction
delta_i = low-resolution smooth residual field
```

This can fix systematic depth bias without inventing arbitrary geometry.

### Multi-View Depth Consistency

For a pixel `u` in frame `i`:

$$
X_w = R_i d_i(u) r_i(u) + t_i
$$

Project into frame `j`:

$$
v = \pi_j(R_j^\top (X_w - t_j))
$$

Predicted depth in camera `j`:

$$
\hat d_j(v) = \left| R_j^\top (X_w - t_j) \right|
$$

Residual:

$$
r_d = \log d_j(v) - \log \hat d_j(v)
$$

Cost:

$$
E_\text{depth}
=
\sum_{(i,j)}
\sum_u
m_i(u)m_j(v)q_i(u)q_j(v)
\rho(r_d^2)
$$

Use log-depth because it behaves better over near/far ranges.

### Reprojection / Image Consistency

For static geometry, projected points should land on the same visual structure:

$$
r_\text{img}
=
\phi_i(u) - \phi_j(v)
$$

where `phi` is a robust patch descriptor, gradient descriptor, or feature
provided by the geometry backbone.

Cost:

$$
E_\text{img}
=
\sum
m_i(u)m_j(v)
\rho(\left| \phi_i(u)-\phi_j(v) \right|^2)
$$

Keep this weaker than depth consistency because phone/web videos include
exposure changes, blur, compression, specular surfaces, screens, and shadows.

### Backbone Prior

Backbone priors keep refined depth and pose near model predictions unless
multi-view evidence strongly disagrees.

$$
E_\text{prior-depth}
=
\sum_{i,u}
q_i(u)
\rho\left(
\log d_i(u)-\log d_i^0(u)
\right)^2
$$

$$
E_\text{prior-pose}
=
\sum_i
\rho\left(
\left|
\log\left((T_i^0)^{-1}T_i\right)
\right|^2_{\Sigma_i^{-1}}
\right)
$$

The confidence `q_i` matters. Low-confidence depth should be easy to move.
High-confidence depth should resist movement.

### Free-Space Consistency

If frame `i` sees a surface at depth `d_i(u)`, then all points along the ray
before that surface are free:

$$
X(\lambda) = R_i(\lambda r_i(u)) + t_i,\quad 0 < \lambda < d_i(u)-\epsilon
$$

No other frame should place a static occupied surface there.

Cost:

$$
E_\text{free}
=
\sum_\text{rays}
\sum_{\lambda < d_i(u)}
\rho\left(
\text{occupied}(X(\lambda))
\right)
$$

This prevents ghost walls, floating surfaces, duplicated furniture, and
dynamic-object smear. A mesh alone cannot give free space. A ray-based map can.

### Static / Dynamic Gating

Initialize:

$$
m_i(u)=1
$$

After projecting, fusing, and rendering, compute residuals. A pixel or mask
becomes dynamic or uncertain if:

```text
its projected depth disagrees across frames
its image/feature residual is high
its motion is coherent inside an object mask
it violates free-space evidence
it appears/disappears inconsistently
```

For a SAM2 mask `M_k^i`, aggregate residuals:

$$
R_k =
\operatorname{median}_{u \in M_k^i}
\left(
|r_d(u)| + \lambda |r_\text{img}(u)|
\right)
$$

If `R_k` stays high across time:

$$
m_i(u) \downarrow \quad \forall u \in M_k^i
$$

Those pixels stop contributing to the static map.

Division of labor:

```text
SAM2 says: these pixels belong together
optimizer says: this object is static, dynamic, or uncertain
```

### Room-Structure Regularization

Indoor home data has useful structure. Use geometry, not mandatory semantics:

```text
large horizontal plane -> floor
vertical planes -> walls, cabinets, doors
Manhattan directions -> common indoor layout
```

Fit planes to stable 3D points:

$$
n^\top X + h = 0
$$

Floor term:

$$
E_\text{floor}
=
\sum_{X \in \text{floor}}
\rho((n_f^\top X + h_f)^2)
$$

Wall term:

$$
E_\text{wall}
=
\sum_{X \in \text{wall}}
\rho((n_w^\top X + h_w)^2)
$$

Orthogonality:

$$
E_\text{ortho}
=
(n_f^\top n_w)^2
$$

This improves geometry, but it does not solve metric scale. A room can be
scaled up or down and still satisfy floor/wall constraints.

### Metric Scale Optimization

Let `s` be global scale. Scale is a first-class variable, not an afterthought.

#### Learned Metric Depth Prior

If ViPE/DA3 gives near-metric or metric depth, treat it as a soft measurement:

$$
E_\text{scale-prior}
=
\sum_{i,u}
w_i(u)
\rho\left(
\log(s d_i(u)) - \log d_i^\text{metric}(u)
\right)^2
$$

This term contributes a metric scale prior. Acceptance still depends on the
scale posterior and validation reports.

#### Object / Architecture Anchors

If the video has known-size objects or architectural elements:

$$
L_a \sim \mathcal{N}(\mu_a,\sigma_a^2)
$$

Examples:

```text
door height
stair riser
outlet plate
kitchen counter
cabinet height
appliance dimensions
tile size if regular
```

If reconstructed distance for anchor `a` is `L_a^0`:

$$
E_\text{anchor}
=
\sum_a
\rho\left(
\frac{sL_a^0-\mu_a}{\sigma_a}
\right)^2
$$

Use distributions, not constants. Real homes vary. Strong anchors have small
`sigma`; weak anchors have large `sigma`.

#### Scale Posterior

The optimizer outputs:

$$
p(s \mid \text{video})
$$

At minimum:

```text
s_mean
s_std
relative_scale_uncertainty = s_std / s_mean
```

Call reconstruction metric only if:

```text
scale uncertainty is low
metric depth prior is stable across views
anchor residuals agree
free-space consistency passes
held-out render consistency passes
```

Operating thresholds:

```text
excellent:  < 3% relative scale uncertainty
usable:     3-7%
weak:       7-15%
reject:     > 15%
```

These are operating thresholds and should be calibrated against evaluation data.

## Unknown Scale Anchors At Web Scale

Most web videos do not have AprilTags, LiDAR, ARKit, or measured trajectories.
The scalable strategy is:

```text
learned metric prior
+ scene-scale priors
+ geometric self-consistency
+ hard rejection
```

Accept as metric pseudo-label data only if:

```text
ViPE/DA3 metric depths produce consistent scale across time
local sub-scenes agree on one global scale
floor/wall/object geometry is plausible
at least one strong or multiple weak scale anchors agree
held-out frames render correctly
occupancy free space is not contradictory
dynamic regions are not fused into static map
```

Otherwise reject for metric training.

Evaluate and calibrate on public metric indoor datasets such as ARKitScenes and
ScanNet++ or on controlled calibration captures. Without a metric benchmark,
the teacher can prove self-consistency but not physical scale accuracy.

## Stitching Long Videos

Stitching is not point-cloud ICP after the fact. Correct stitching is:

```text
1. every frame produces rays + depths
2. rays + depths + poses produce 3D points
3. overlapping observations create consistency constraints
4. global optimizer adjusts poses/depths/scale
5. static depths are fused into one volumetric map
```

For long video, frame comparisons are sparse and graph-structured:

```text
temporal edges:
  nearby frames

overlap edges:
  frames with common visible surfaces

loop edges:
  later frames that re-observe earlier places

scale edges:
  sub-scenes that should agree in meters
```

Variables:

$$
\{T_i\}_{i=1}^N
$$

$$
\{d_i\}_{i=1}^N
$$

$$
\{r_i\}_{i=1}^N
$$

$$
s
$$

$$
\{m_i\}_{i=1}^N
$$

Final optimization:

$$
\min E =
E_\text{depth}
+
\lambda_\text{img}E_\text{img}
+
\lambda_\text{free}E_\text{free}
+
\lambda_\text{prior}E_\text{prior}
+
\lambda_\text{scale}E_\text{scale}
+
\lambda_\text{room}E_\text{room}
+
\lambda_\text{smooth}E_\text{smooth}
$$

Use robust losses everywhere:

```text
Huber
Cauchy
Geman-McClure
Tukey
```

Raw residuals use robust losses because real videos contain frequent outliers.

The solver can alternate:

```text
estimate poses/scale with depths mostly fixed
estimate depth corrections with poses mostly fixed
estimate static/dynamic masks
fuse provisional map
render map back into frames
penalize residuals
repeat
```

This is an EM-like loop:

```text
E step:
  infer which pixels are static, dynamic, uncertain

M step:
  optimize geometry using static/high-confidence pixels
```

## Rolling Shutter And Video Stabilization

Phone video often has:

```text
rolling shutter
electronic image stabilization
frame-dependent crop
autofocus changes
small focal length drift
```

If ignored, walls bend and maps warp.

Represent camera trajectory as continuous:

$$
T(t)
$$

For frame `i`, row `y` was captured at:

$$
t_{i,y} = t_i + \tau_i \frac{y}{H}
$$

Projection uses:

$$
T(t_{i,y})
$$

Operational policy:

```text
default:
  one pose per frame

if high angular velocity / rolling-shutter residual detected:
  use row-time rolling-shutter correction

if focal/crop drift detected:
  allow smooth time-varying focal length or reject the video
```

For clean web data, reject severe stabilization/zoom cases. For high-value
data, model them.

## Static Map Construction

Once optimized geometry exists, mapping is ray fusion.

For every static pixel:

$$
X_w = R_i d_i(u)r_i(u) + t_i
$$

Ray origin:

$$
C_i = t_i
$$

Ray direction:

$$
\hat r_w = R_i r_i(u)
$$

Surface point:

$$
X_w = C_i + d_i(u)\hat r_w
$$

For each ray:

```text
from camera to just before surface:
  mark free space

near surface:
  update TSDF / occupancy

behind surface:
  leave unknown
```

A depth image gives both:

```text
surface exists at depth d
free space exists before depth d
```

Robots need the free-space evidence, not just the surface.

Use TSDF for mesh reconstruction and occupancy/log-odds for robot planning.

Free update:

$$
L_v \leftarrow L_v + \log\frac{P_\text{free}}{1-P_\text{free}}
$$

Occupied update:

$$
L_v \leftarrow L_v + \log\frac{P_\text{occ}}{1-P_\text{occ}}
$$

where `L_v` is log-odds occupancy.

TSDF update:

$$
\phi_v
\leftarrow
\frac{
W_v \phi_v + w\,\operatorname{clip}\left(\frac{z_\text{surface}-z_v}{\mu}, -1, 1\right)
}{
W_v+w
}
$$

$$
W_v \leftarrow W_v+w
$$

Store per voxel:

```text
TSDF value
TSDF weight
occupancy log-odds
free-space count
surface count
dynamic count
semantic/object label if available
uncertainty
```

ESDF is also valuable for robot planners because they need distance-to-obstacle
fields. TSDF/ESDF-style maps with explicit free-space reasoning are the right
planning substrate.

## Floor-Cleaner Occupancy Grid

The occupancy grid is derived from:

```text
optimized rays
optimized depth
static masks
free-space evidence
surface evidence
floor plane
scale uncertainty
```

Estimate floor plane:

$$
n_f^\top X + h_f = 0
$$

Transform world points into a floor coordinate system:

```text
x, y = floor plane coordinates
z    = height above floor
```

Height bands for a floor-cleaner robot:

```text
floor band:
  z near 0

low obstacle band:
  0.02 m to 0.35 m

body collision band:
  0.02 m to robot height

overhang band:
  above robot height, usually not blocking

unknown:
  no ray evidence
```

Cell classification:

```text
occupied if:
  static occupied voxel exists in collision band

free if:
  floor was observed
  enough rays passed through the cell
  no occupied voxel exists in collision band

unknown if:
  insufficient ray evidence

dynamic if:
  only dynamic objects occupied it

movable if:
  object is static during video but likely movable, e.g. chair/bin/shoe
```

Grid channels:

```text
P_free(x,y)
P_occupied_static(x,y)
P_movable_static(x,y)
P_dynamic(x,y)
P_unknown(x,y)
height_min(x,y)
height_max(x,y)
scale_uncertainty
map_confidence
```

Keep the grid multichannel through export. A student network can learn more
from free / occupied / dynamic / unknown / confidence than from a binary map.

## Mesh Quality Is Not Occupancy Quality

A pretty mesh can still be bad for a robot:

```text
chair leg is missing in mesh
floor behind person is hallucinated as free
glass table disappears
black rug becomes a hole
motion-blurred wall becomes curved
```

Robot teacher rules:

```text
unknown is not free
dynamic is not static
low-confidence depth is not obstacle GT
hallucinated completion is not GT
unseen space remains unknown
```

A floor-cleaning robot teacher should be conservative.

## Cohesive Teacher Stack

```text
RGB video
  -> Video quality + reconstructability gate
  -> Keyframe selector
  -> ViPE/DA3 geometry backbone
  -> Canonical ray/depth/pose representation
  -> Visibility graph construction
  -> SAM2 object mask grouping
  -> Scale-aware robust global optimizer
  -> Static/dynamic inference loop
  -> Ray-based TSDF + occupancy fusion
  -> Floor-plane extraction
  -> 2D robot occupancy grid projection
  -> Held-out render + free-space validation
  -> Metric acceptance / rejection
```

Only heavy learned pieces:

```text
ViPE/DA3
SAM2 or Grounded-SAM2
```

Everything else is geometry, optimization, validation, and mapping.

## Critical Implementation Rules

Use ray maps, not only pinhole intrinsics:

$$
r_i(u,v)
$$

Treat backbone depth as:

```text
initial evidence
metric prior
surface proposal
```

Treat scale as first-class state:

$$
s
$$

Every output must know:

```text
global scale
scale uncertainty
scale source
scale acceptance status
```

Fuse rays, not point clouds:

```text
point clouds say: surface here
rays say: free until here, surface here, unknown behind
```

Use static masks before mapping so dynamic objects do not create ghost obstacles
or false walls in the fused map.

Validate by rendering held-out frames:

```text
does rendered depth match predicted/observed depth?
do silhouettes align?
do floor/wall boundaries align?
do free-space rays contradict the map?
does the map explain multiple viewpoints?
```

## Acceptable Web Videos

Acceptable for metric pseudo-GT only if:

```text
camera has real translation, not just rotation
geometry backbone has stable intrinsics/rays
depth scale is stable over time
visible floor plane exists
static structure dominates dynamic objects
room surfaces close consistently
held-out views render correctly
scale posterior is tight
occupancy has low contradiction
```

Reject if:

```text
pure panning
heavy zoom
heavy electronic stabilization
mostly people/pets
only close-up objects
no visible floor
too much motion blur
reflective/glass-dominant scene
large textureless white walls with weak parallax
scale posterior wide
```

Quality from web data comes from processing many videos and keeping little.

## Metric Categories

### Category A: Real Metric GT

Requires:

```text
LiDAR
RGB-D
ARKit/ARCore depth/pose
laser scan
measured marker
known camera trajectory
known object with precise measurement
```

This is measured metric data.

### Category B: Metric Pseudo-GT

Uses:

```text
ViPE/DA3 metric prior
scene anchors
object priors
global consistency
strict filtering
```

Useful for training, with scale uncertainty and provenance attached.

### Category C: Unanchored Reconstruction

Uses:

```text
RGB-only geometry
learned priors
no stable scale anchor
```

Visual quality alone is insufficient for metric training labels.

For web-video training, most accepted data will be Category B.

Practical data strategy:

```text
use web videos for scale-aware pseudo-supervision
use public RGB-D / laser-scan datasets for calibration and evaluation
eventually collect robot/phone videos with real anchors for final supervision
```

## Final Build Contract

```text
Input:
  RGB video

Core learned model:
  ViPE with DA3 pipeline

Optional learned mask model:
  SAM2
  Grounded-SAM2 only for scale-anchor detection

Internal state:
  ray map per frame
  dense depth per frame
  camera pose per frame
  global scale variable
  per-frame depth correction
  static/dynamic probability
  uncertainty

Optimizer:
  robust factor graph over pose, depth correction, scale, camera/ray
  correction, static masks

Map:
  ray-fused TSDF
  log-odds occupancy
  floor-aligned 2D robot grid
  unknown/free/occupied/dynamic channels

Validator:
  held-out view rendering
  free-space contradiction
  scale posterior
  floor/wall consistency
  dynamic leakage check

Output:
  metric map only if scale posterior passes threshold
  otherwise reject or mark as non-metric pseudo-label
```

Core decision:

```text
ViPE/DA3 initializes geometry. The optimizer, mapper, posterior, and validator
decide acceptance.
```

The model proposes. The factor graph refines. The ray mapper proves free space.
The scale posterior determines metric status. The validator determines dataset
acceptance.

## API Contracts

These contracts are intended interfaces. They become binding once implemented.
Any coordinate, shape, unit, or schema change must update this section before code changes are declared done.

### Coordinate And Unit Rules

- Units are meters unless a field explicitly says otherwise.
- `T_world_camera` is a 4x4 camera-to-world transform.
- Camera rays are unit vectors in camera coordinates.
- Radial depth is distance along a ray.
- Unknown, free, occupied, dynamic, predicted, and measured states must remain
  distinct.
- Metric output requires `ScalePosterior` and `ValidationReport`.

### Camera Projection Contract

Every `FrameRayPacket.camera_model` must support:

```text
unproject(pixel_uv, radial_depth_m) -> X_camera[3]
project(X_camera[3]) -> pixel_uv, radial_depth_m, valid
```

Rules:

- `unproject` uses unit camera rays and radial depth:
  `X_camera = radial_depth_m * ray_camera`.
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

### VideoInput

Purpose: identify a source RGB video or decoded frame sequence.

Required fields:

```text
source_uri
frame_count
fps
width_px
height_px
timestamp_s[]
metadata
```

Failure modes:

```text
missing source
unsupported codec
variable frame timing not represented
severe corruption
```

### ReconstructabilityReport

Purpose: decide whether a video should enter reconstruction.

Required fields:

```text
accepted_for_reconstruction
rejection_reasons[]
parallax_score
blur_score
dynamic_foreground_ratio
zoom_or_stabilization_score
static_structure_score
confidence
```

This report does not decide metric acceptance.

### KeyframeSet

Purpose: choose frames for geometry while retaining all frames for validation.

Required fields:

```text
selected_frame_ids[]
selection_reasons[]
baseline_scores[]
overlap_scores[]
sharpness_scores[]
dynamic_ratio_scores[]
timestamps_s[]
```

### FrameRayPacket

Purpose: canonical per-frame geometry from a backbone or optimizer.

Required fields:

```text
frame_id
T_world_camera[4,4]
rays_camera[H,W,3]
radial_depth_m[H,W]
confidence[H,W]
camera_model
source
uncertainty
```

Optional fields:

```text
intrinsics
rolling_shutter_model
depth_residual_field
```

### VideoGeometryBackbone

Purpose: normalize external geometry engines.

Conceptual interface:

```text
predict(video_or_keyframes) -> GeometryBackbonePrediction
```

Output contract:

```text
frame ray packets
camera confidence
depth confidence
dependency and artifact provenance
clear unavailable status when external dependencies are missing
```

Planned backbones:

```text
default: ViPE with DA3
fallback: MegaSaM for selected hard videos
```

### MaskTrackSet

Purpose: represent SAM2 or Grounded-SAM2 mask groups.

Required fields:

```text
track_id
frame_ids[]
mask_rle_or_bitmap
mask_confidence
prompt_or_source
semantic_label_optional
```

Mask grouping does not itself mark static or dynamic.

### StaticDynamicState

Purpose: record geometry-led static/dynamic inference.

Required fields:

```text
frame_id
static_probability[H,W]
dynamic_probability[H,W]
unknown_probability[H,W]
mask_track_decisions[]
residual_summary
```

### VisibilityGraph

Purpose: connect frames that likely observe common static surfaces.

Required fields:

```text
nodes: keyframe IDs
temporal_edges
overlap_edges
loop_edges
scale_edges
edge_measurements
```

Edge measurements should include depth consistency, reprojection/image
consistency, feature/color consistency when available, and free-space
consistency.

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
- `metric_acceptance_status=measured_metric` requires at least one compatible measured evidence source.
- Learned and prior-based evidence can support `metric_pseudo_label`, not `measured_metric`.

### ScalePosterior

Purpose: make metric scale explicit.

Required fields:

```text
scale_mean
scale_std
relative_scale_uncertainty
scale_sources[]
anchor_residuals[]
metric_acceptance_status
```

Suggested statuses:

```text
measured_metric
metric_pseudo_label
non_metric_pseudo_label
rejected
```

### OptimizedSceneState

Purpose: hold refined geometry before mapping.

Required fields:

```text
frame_ray_packets[]
scale_posterior
static_dynamic_state[]
visibility_graph
optimizer_trace
validation_inputs
```

### VoxelMapState

Purpose: hold volumetric TSDF, occupancy, and uncertainty.

Required fields:

```text
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

### MeshChunkMetadata

Purpose: keep geometry provenance attached to mesh outputs.

Required fields:

```text
chunk_id
source_frame_ids[]
observed_coverage_estimate
voxel_size_m
coordinate_frame
metric_scale_source
mean_uncertainty_m
p50_uncertainty_m
p95_uncertainty_m
observed_only
predicted_completion
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

- `P_free` means observed free space, not merely absence of observed obstacles.
- `P_occupied_static` means structural or stable static occupancy.
- `P_movable_static` means objects that appear static during the video but are likely non-structural movable obstacles, such as chairs, bins, shoes, or small furniture.
- `P_dynamic` means moving or temporally inconsistent occupancy.
- `P_unknown` means insufficient ray evidence.
- Unknown is not free.
- Dynamic is not static.
- Movable-static is not free.
- Keep this multichannel until an explicit downstream export requires a derived binary product.

### ValidationReport

Purpose: decide whether outputs are accepted for metric training.

Required fields:

```text
held_out_render_error
free_space_contradiction_rate
scale_posterior
floor_wall_consistency
dynamic_leakage_score
accepted_for_metric_training
rejection_reasons[]
```

Metric acceptance requires this report plus a tight scale posterior.

## External Technical References

These references are implementation context for adapters, evaluation, and
mapping choices.

- ViPE: https://github.com/nv-tlabs/vipe
- Depth Anything 3: https://github.com/bytedance-seed/depth-anything-3
- MegaSaM: https://arxiv.org/html/2412.04463v1
- SAM2: https://github.com/facebookresearch/sam2
- Grounded-SAM2: https://github.com/IDEA-Research/Grounded-SAM-2
- ARKitScenes: https://machinelearning.apple.com/research/arkitscenes
- Open3D TSDF integration:
  https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html
- nvblox: https://arxiv.org/html/2311.00626v2
- Depth Anything 3 depth-ray paper: https://arxiv.org/html/2511.10647v1
