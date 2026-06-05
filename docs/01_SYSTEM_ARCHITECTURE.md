# 01 — System Architecture

## System name

**Atlas3R**: object-centric real-time RGB neural metric reconstruction.

## High-level dataflow

```text
FramePacket
  -> PreprocessPipeline
  -> Streaming Metric Geometry Transformer (SMGT)
  -> PoseGraph / Tracker
  -> ObjectTracker3D
  -> ProbabilisticTSDF / SurfelMap
  -> IncrementalMesher
  -> RuntimeAPI / Exporters
```

## Module 1: Frame IO and preprocessing

### Inputs

- video file, image folder, webcam, RTSP, AVFoundation, OpenCV, or platform camera API;
- timestamp in nanoseconds if available;
- optional camera metadata: intrinsics, focal length, sensor size, distortion, exposure, rolling shutter info;
- optional inertial/ARKit pose is not required but can be used as scale prior if available.

### Output contract

`FramePacket`:

- `frame_id: int64`
- `timestamp_ns: int64`
- `rgb_u8: H×W×3`
- `rgb_linear: 3×H'×W' float16/float32`
- `K_original: Optional[3×3]`
- `K_model: 3×3` after resize/crop
- `distortion: Optional[CameraDistortion]`
- `resize_transform: 3×3`
- `valid_mask: H'×W' bool`

### Required preprocessing

- preserve aspect ratio or record exact crop/resize transform;
- normalize using model-specific mean/std;
- undistort when calibration is known;
- if calibration unknown, pass `intrinsics_unknown=True` and let SMGT predict intrinsics;
- optionally estimate blur/exposure and feed quality tokens to the model.

## Module 2: Streaming Metric Geometry Transformer (SMGT)

SMGT is the neural core. It is trained as a compact student model using teacher supervision from geometry foundation models.
Teacher models and external runners are offline supervision/adaptation paths;
the runtime target is the compact SMGT student plus geometric checks, not the
large teacher models running in the live loop.

### Runtime target variants

- `smgt_b`: balanced, ~70–120M parameters, NVIDIA target.
- `smgt_s`: small, ~25–60M parameters, high-end Apple M and laptop GPU target.
- `smgt_tiny`: ~8–20M parameters, low-power or high-FPS pose-only mode.

### Inputs

- current RGB frame tokens;
- optional intrinsics/camera tokens;
- anchor context from first/key canonical frames;
- local pose-reference window tokens;
- trajectory memory tokens;
- optional previous map render tokens.

### Architecture

1. **Image encoder**
   - Initialize from DINOv3-S/B or DINOv3 ConvNeXt-small when license and runtime allow.
   - Produce multi-scale features at 1/4, 1/8, 1/16, 1/32.
   - Use patch size 14/16 for transformer variants or ConvNeXt-style pyramid for Apple-friendly variants.

2. **Token construction**
   - frame tokens: dense visual tokens from the current frame;
   - camera tokens: normalized focal, principal point, aspect, resize metadata, camera-known bit;
   - quality tokens: blur, exposure, motion magnitude, compression estimate;
   - map tokens: sparse map anchors, if available.

3. **Geometric context attention**
   - **Anchor context:** first 4–8 reliable keyframes define canonical world frame and scale prior.
   - **Local pose-reference window:** last 8–16 keyframes retain dense tokens for fine matching.
   - **Trajectory memory:** 6–16 compact tokens per historical keyframe encode pose, scale, and uncertainty.
   - **Loop memory:** sparse retrieved keyframes injected only when image retrieval/dense matching is confident.

4. **Attention implementation**
   - NVIDIA: FlashAttention/FlashInfer or PyTorch SDPA fallback.
   - Apple: use attention patterns exportable to Core ML where possible; otherwise use MPS path for research.
   - Bounded memory is mandatory; do not let attention cost grow with all past frames.

5. **Heads**
   - `DepthHead`: metric inverse depth, depth confidence, edge-aware residual.
   - `NormalHead`: surface normal in camera frame and confidence.
   - `PointMapHead`: local camera pointmap and world pointmap.
   - `PoseHead`: `T_anchor_camera`, `T_prev_camera`, pose covariance.
   - `IntrinsicsHead`: focal, principal point, optional radial/tangential distortion, rolling shutter row time.
   - `DenseMatchHead`: fine correspondences to local/loop keyframes.
   - `ObjectHead`: per-pixel object embedding, mask logits, track embedding.
   - `DynamicHead`: static/dynamic mask and optional object motion SE(3).
   - `UncertaintyHead`: calibrated aleatoric uncertainty for all geometric outputs.

### Primary outputs per frame

- `T_world_camera: 4×4`
- `K: 3×3`
- `depth_m: H×W`
- `normal_camera: H×W×3`
- `point_world: H×W×3`
- `confidence: H×W`
- `static_mask: H×W`
- `object_logits/embeddings`
- `dense_matches`
- `uncertainty maps`

## Module 3: Pose tracker and pose graph

Even with a mostly neural front end, a small geometric consistency layer is required for high reliability.

### Tracker stream

Runs every frame.

- Use SMGT pose as initial estimate.
- Refine pose using map reprojection/dense correspondence residuals when cheap enough.
- Output a pose estimate within the frame budget even if mapper lags.

### Keyframe selector

Create a keyframe when any condition holds:

- translation baseline exceeds threshold relative to scene depth;
- rotation exceeds threshold;
- scene overlap drops;
- uncertainty grows;
- new object/region appears;
- loop-closure candidate is detected.

### Pose graph

Nodes:

- keyframe SE(3) pose;
- optional scale/intrinsics nodes for uncalibrated clips;
- object poses for movable rigid objects.

Edges:

- neural relative-pose edges;
- dense correspondence edges;
- map reprojection edges;
- loop closure edges;
- scale-anchor edges when known;
- gravity/manhattan priors when detected.

Optimization:

- local window optimization every keyframe;
- global low-priority optimization when loop closures occur;
- robust loss on all residuals;
- update map chunk transforms after pose correction.

## Module 4: Object tracker and 3D instance fusion

The system must not just create a single mesh. It must segment objects.

### 2D instance sources

- SAM3/SAM3.1 teacher or runtime adapter;
- Atlas3R `ObjectHead` after distillation;
- optional text prompts for open-vocabulary object filtering.

### 3D fusion

- project per-frame masks into the map using depth/pose;
- maintain object tracks with 2D mask IoU, 3D overlap, appearance embedding, DINO feature similarity, and motion consistency;
- assign each voxel/surfel an object-id distribution;
- resolve conflicts with Bayesian updates and temporal smoothing;
- create new object IDs only after multi-frame confirmation.

### Dynamic handling

- static scene TSDF excludes dynamic pixels;
- moving rigid objects get per-object TSDFs and object pose trajectories;
- non-rigid/uncertain dynamic regions are tracked as object point clouds unless enough evidence exists for a stable mesh.

## Module 5: Probabilistic TSDF / surfel fusion

### Why TSDF

Neural pointmaps and depth are noisy. A TSDF gives stable surfaces, easy mesh extraction, incremental updates, and compatibility with game engines.

### Voxel state

Each voxel stores:

- TSDF value `s`;
- weight `w`;
- color mean and variance;
- normal mean;
- semantic/object logits;
- uncertainty;
- last update timestamp;
- source keyframe bitset or compact Bloom filter.

### Integration rule

For each valid static pixel:

1. backproject depth to 3D using `K` and `T_world_camera`;
2. raycast through voxel blocks up to truncation distance;
3. compute signed distance `d = z_surface - z_voxel_along_ray`;
4. truncate `d` to `[−mu, mu]`;
5. update with confidence weight:
   `w_pixel = confidence / (sigma_depth^2 + sigma_pose^2 + epsilon)`;
6. fuse color and object probabilities.

### Resolution

Use adaptive blocks:

- near-field precision mode: 1–2 mm voxels for desktop-scale captures;
- indoor balanced mode: 5–10 mm voxels;
- large-room/long-video mode: 2–5 cm voxels plus local high-resolution blocks around objects.

The system must record voxel size in mesh metadata. It cannot claim errors below voxel resolution.

## Module 6: Meshing and game-engine export

### Mesh extraction

- incremental marching cubes or dual contouring over dirty voxel blocks;
- vertex positions in meters, world frame;
- vertex normals from TSDF gradient;
- per-vertex color and object ID;
- optional texture atlas from selected keyframes;
- mesh chunking by spatial block and object ID.

### Export formats

- GLB/glTF 2.0 for game engines;
- OBJ/MTL for simple interchange;
- PLY for geometry debugging;
- USDZ for Apple ecosystem;
- optional `navmesh` or collision mesh output after simplification.

### Object meshes

For each `ObjectInstance`, export:

- observed mesh;
- optional completed mesh, clearly flagged;
- oriented bounding box;
- transform `T_world_object`;
- observed coverage ratio;
- confidence and uncertainty.

## Module 7: Runtime scheduler

Use bounded asynchronous streams with different rates:

```text
Thread/stream A: capture/decode/preprocess at input FPS
Thread/stream B: every-frame pose stream at camera FPS
Thread/stream C: selected-keyframe depth/pointmap/map update stream
Thread/stream D: lower-rate object stream on keyframes or changed regions
Thread/stream E: asynchronous pose graph and loop closure
Thread/stream F: asynchronous TSDF integration, mesh extraction, and export
```

The API should emit pose at camera FPS even when map/object/mesh streams lag.
Mesh updates can be chunked and delayed, but queues must be bounded and drops
must be explicit. Persistent map state is an object-aware sparse TSDF/surfel map
with source-frame/uncertainty metadata, not hidden transformer memory alone.

## Module 8: Deployment backends

### NVIDIA

- PyTorch eager for research;
- Torch compile where stable;
- ONNX export for fixed-shape inference;
- TensorRT for production;
- CUDA/nvblox or custom kernels for TSDF/meshing.

### Apple Silicon

- PyTorch MPS for research;
- Core ML export for production;
- separate `smgt_s`/`smgt_tiny` architecture with Core ML-friendly ops;
- object segmentation and loop closure at reduced rate.
