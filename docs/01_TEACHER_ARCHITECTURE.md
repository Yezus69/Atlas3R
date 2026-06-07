# Teacher Architecture

## Learned Components

Keep the learned stack small:

- ViPE with DA3 as the primary video geometry backbone.
- MegaSaM as an optional fallback for hard dynamic or weak-parallax videos.
- SAM2 for video mask grouping and propagation.
- Grounded-SAM2 only for optional scale-anchor prompts.

Do not run many geometry teachers in parallel by default. The project value is
the optimizer, mapper, validator, and acceptance gate.

## Canonical Representation

Internal geometry is ray-map first:

```text
I_i             RGB frame
T_world_camera  camera-to-world transform
r_i(u, v)       unit ray in camera coordinates
d_i(u, v)       radial depth or range along the ray
q_i(u, v)       confidence
s               global scale variable
m_i(u, v)       static probability
```

One pixel becomes:

```text
X_camera = d_i(u, v) * r_i(u, v)
X_world = R_i * X_camera + t_i
X_metric = s * X_world, when scale is not already metric
```

Pinhole intrinsics can be represented when available, but downstream mapping and
optimization should consume rays.

## Optimizer

The optimizer refines:

```text
camera poses
depth scale, bias, and smooth residual fields
camera/ray corrections
global scale
static/dynamic probabilities
```

Costs should include:

- multi-view depth consistency;
- weak image or feature consistency;
- backbone priors;
- free-space consistency;
- room-structure regularization where appropriate;
- scale priors and anchor residuals;
- smoothness and robust outlier handling.

Use robust losses and confidence weights. Do not let optimization invent
unconstrained geometry.

## Static And Dynamic Separation

SAM2 groups pixels into masks. Geometry decides whether those masks are static,
dynamic, or uncertain.

Dynamic pixels must be excluded before TSDF and occupancy fusion. Removing them
after mapping leaves ghost obstacles and false free space.

## Mapping

Fuse rays, not just point clouds.

Each valid static ray contributes:

- free-space evidence before the surface;
- surface evidence near the observed depth;
- unknown space behind the surface.

Outputs should include:

- TSDF surface state;
- occupancy log odds;
- free, occupied, dynamic, and unknown channels;
- floor-aligned 2D robot grid;
- mesh chunks with source and uncertainty metadata.

## Metric Gate

The system may call an output metric pseudo-label data only when:

- scale posterior uncertainty is below threshold;
- learned metric depth priors are stable across views;
- anchors agree when present;
- free-space contradictions are low;
- held-out render validation passes;
- dynamic regions are not fused into the static map.

Otherwise the video is rejected for metric training or marked as non-metric
pseudo-label data.
