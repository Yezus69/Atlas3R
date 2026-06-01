# 03 — Datasets and Preprocessing

## Unified dataset schema

Every dataset adapter must produce the same logical schema:

```text
SceneRecord
  scene_id
  units_meters: bool
  metric_scale_source: enum
  frames[]
    rgb_path
    timestamp_ns
    K
    distortion
    T_world_camera
    depth_path optional
    normal_path optional
    semantic_mask optional
    instance_mask optional
  mesh_gt optional
  pointcloud_gt optional
  object_boxes_3d optional
  object_meshes optional
  split: train/val/test
  license metadata
```

Do not mix coordinate conventions silently. Convert all datasets into Atlas3R convention:

- right-handed world;
- +X right, +Y down or gravity convention must be explicitly recorded;
- camera frame: +X right, +Y down, +Z forward unless a downstream renderer requires conversion;
- units: meters.

## Dataset categories

### High-fidelity metric indoor geometry

Use for metric geometry and mesh losses.

- ScanNet++: high-fidelity indoor scenes, sub-millimeter laser scans, high-res DSLR images, iPhone RGB-D streams.
- ARKitScenes: RGB-D captures from Apple LiDAR devices, camera poses, surface reconstruction, object boxes.
- ScanNet: large RGB-D video dataset with camera poses, reconstructed surfaces, semantic/instance labels.
- Replica / ReplicaCAD: clean dense indoor meshes, textures, semantic/instance data.
- Matterport3D / HM3D: building-scale textured meshes and indoor diversity.

### Synthetic controllable geometry

Use for exact supervision, rare camera/lens cases, dynamic data, and overfit tests.

- Hypersim: photorealistic indoor synthetic images with complete geometry, camera info, semantic labels, materials, lighting.
- TartanAir V2: simulation with RGB, depth, segmentation, optical flow, camera poses, LiDAR, weather/lighting/dynamic cases.
- Blender/Habitat custom generator: create cube rooms, reflective failure cases, low-texture walls, known object meshes.

### Outdoor / large-scale / in-the-wild

Use for generalization and long-trajectory robustness.

- ETH3D, Tanks and Temples, KITTI, nuScenes/Waymo if licenses allow, MegaDepth, CO3D, RealEstate10K/DL3DV-style video data.
- For datasets without metric GT, use teacher pseudo-labels and self-supervised multi-view losses.

### Object priors

Use for object completion as **predicted** geometry only, not measured geometry.

- Objaverse / Objaverse-XL, ShapeNet, OmniObject3D, ABO, and curated high-quality subsets.
- Render multi-view synthetic videos with known object meshes, materials, and scales.

### Dynamic scenes

Use for static/dynamic separation and movable object handling.

- TartanAir dynamic environments.
- MonST3R-style dynamic video datasets and pseudo-labels.
- DAVIS/YTVOS/MOSE-style videos for mask/tracking pretraining.
- Driving datasets for moving vehicles/pedestrians if target includes outdoor scenes.

## Preprocessing pipeline

### Step 1: ingest raw data

- Download/verify checksums.
- Store raw data read-only.
- Create processed shards under `data/processed/<dataset>/<version>/`.
- Record license and citation metadata.

### Step 2: normalize camera models

- Convert intrinsics to pixel units.
- Convert distortion to a standard model:
  - pinhole;
  - Brown-Conrady radial/tangential;
  - fisheye if needed.
- Record resize/crop transforms.
- Generate undistorted images only when necessary; otherwise train with distortion tokens.

### Step 3: generate clips

Clip sampling policy:

- short clips: 2, 4, 8 frames for geometry bootstrapping;
- medium clips: 16, 32, 64 frames for streaming training;
- long clips: 128–512 frames for memory/loop/drift training, sampled sparsely on 2×4090;
- negative clips: little/no parallax, blur, dynamic foreground, repeated texture, reflective surfaces.

### Step 4: precompute teacher caches

Store each teacher under:

```text
data/teacher_cache/<teacher>/<version>/<dataset>/<scene_id>/<clip_id>.npz
```

Cache must include:

- prediction tensors;
- teacher version/hash;
- input frame IDs and resize transforms;
- confidence scores;
- runtime/config metadata;
- license tag.

### Step 5: augmentations

Geometry-preserving augmentations:

- color jitter;
- exposure/gamma;
- noise/compression;
- motion blur;
- crop/resize with exact K update;
- lens distortion simulation;
- rolling-shutter simulation;
- random occluders.

Metric-sensitive augmentations:

- never arbitrarily scale metric ground-truth scenes unless all depths/poses/meshes are scaled consistently and the scale label is updated;
- train scale ambiguity explicitly by using `metric_known` and `metric_unknown` flags.

## Data loader outputs

A training batch should include:

```text
rgb:              B×T×3×H×W
K:                B×T×3×3
T_world_camera:   B×T×4×4 optional
metric_mask:      B bool
valid_depth:      B×T×H×W bool optional
depth_m:          B×T×H×W optional
normal:           B×T×H×W×3 optional
instance_masks:   ragged/object list optional
teacher:          dict of teacher outputs optional
clip_metadata:    scene IDs, frame IDs, scale source, licenses
```

