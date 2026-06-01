# 08 — API Contracts and Coordinate Conventions

## Coordinate conventions

Internal Atlas3R convention:

- units: meters;
- camera frame: `x` right, `y` down, `z` forward;
- world frame: initialized from anchor keyframes unless external frame is supplied;
- transform naming: `T_A_B` maps homogeneous points from frame `B` into frame `A`.

Example:

```text
p_world = T_world_camera @ p_camera
camera_center_world = T_world_camera[:3, 3]
```

Never use ambiguous variable names like `pose` at public boundaries.

## FramePacket

```python
@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp_ns: int
    rgb_u8: NDArray[np.uint8]          # H,W,3 original frame
    rgb_model: Tensor                  # 3,Hm,Wm normalized
    K_original: NDArray[np.float32] | None
    K_model: NDArray[np.float32]       # adjusted for model image
    distortion: CameraDistortion | None
    resize_transform: NDArray[np.float32]
    camera_metadata: dict[str, Any]
```

## CameraModel

```python
@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    K: NDArray[np.float32]             # 3,3
    distortion_model: str              # none|brown_conrady|fisheye
    distortion_params: NDArray[np.float32] | None
    rolling_shutter_row_time_s: float | None
    confidence: float
    source: str                        # metadata|predicted|calibrated|external
```

## PoseEstimate

```python
@dataclass(frozen=True)
class PoseEstimate:
    frame_id: int
    timestamp_ns: int
    T_world_camera: NDArray[np.float32] # 4,4
    q_world_camera_xyzw: NDArray[np.float32]
    camera_center_world_m: NDArray[np.float32] # 3
    covariance_6x6: NDArray[np.float32] | None
    confidence: float
    tracking_state: str                # OK|LOW_CONFIDENCE|RELOCALIZING|LOST|NEW_SUBMAP
    scale_source: str                  # rgb_prior|calibrated_rgb|known_anchor|external_pose
    diagnostics: dict[str, Any]
```

## DenseMatchSet

Minimal Phase 0A placeholder used by `FramePrediction`. Later teacher/student model
work may extend this schema, but the core frame-to-frame correspondence contract is:

```python
@dataclass(frozen=True)
class DenseMatchSet:
    source_frame_id: int
    target_frame_id: int
    source_pixels_uv: NDArray[np.float32] # N,2
    target_pixels_uv: NDArray[np.float32] # N,2
    confidence: NDArray[np.float32]       # N values in [0,1]
```

## FramePrediction

```python
@dataclass
class FramePrediction:
    pose: PoseEstimate
    camera: CameraModel
    depth_m: Tensor                    # H,W float32/float16
    depth_sigma_m: Tensor              # H,W
    normal_camera: Tensor              # H,W,3
    point_world: Tensor                # H,W,3
    confidence: Tensor                 # H,W
    static_mask: Tensor                # H,W bool or probability
    object_embeddings: Tensor | None
    object_mask_logits: Tensor | None
    dense_matches: DenseMatchSet | None
```

## ObjectInstance

```python
@dataclass
class ObjectInstance:
    object_id: int
    label_candidates: list[tuple[str, float]]
    T_world_object: NDArray[np.float32] # 4,4
    oriented_bbox_center_m: NDArray[np.float32]
    oriented_bbox_axes: NDArray[np.float32]      # 3,3
    oriented_bbox_extents_m: NDArray[np.float32] # 3
    mesh_chunk_ids: list[str]
    is_dynamic: bool
    observed_coverage_ratio: float
    confidence: float
    uncertainty_m: float
    first_seen_frame_id: int
    last_seen_frame_id: int
    metadata: dict[str, Any]
```

## MeshChunk

```python
@dataclass
class MeshChunk:
    chunk_id: str
    version: int
    T_world_chunk: NDArray[np.float32]
    vertices_m: NDArray[np.float32]     # N,3
    faces: NDArray[np.int32]            # M,3
    normals: NDArray[np.float32] | None # N,3
    colors: NDArray[np.uint8] | None    # N,3/4
    uvs: NDArray[np.float32] | None
    object_id_per_face: NDArray[np.int32] | None
    surface_source_per_face: NDArray[np.int8] | None
    voxel_size_m: float
    mean_uncertainty_m: float
    p95_uncertainty_m: float
    source_frame_ids: list[int]
    scale_source: str
    flags: list[str]
```

Surface source enum:

```text
0 observed_surface
1 single_view_prior
2 completed_surface
3 dynamic_surface
4 low_confidence
```

## WorldMap

```python
@dataclass
class WorldMap:
    map_id: str
    world_frame_name: str
    created_at_ns: int
    mesh_chunks: dict[str, MeshChunk]
    objects: dict[int, ObjectInstance]
    keyframes: dict[int, PoseEstimate]
    scale_source: str
    global_confidence: float
    metadata: dict[str, Any]
```

## Live API events

The runtime should stream events:

```text
PoseUpdate(frame_id, PoseEstimate)
DepthUpdate(frame_id, optional compressed depth/confidence)
ObjectUpdate(object_id, ObjectInstance)
MeshChunkAdded(chunk_id, version)
MeshChunkUpdated(chunk_id, version)
MeshChunkRemoved(chunk_id, version)
TrackingStateChanged(state)
BenchmarkMetric(name, value)
```

## File formats

### `.atlas3r` session folder

```text
session.atlas3r/
  metadata.json
  poses.jsonl
  cameras.jsonl
  objects.jsonl
  mesh_chunks/
    chunk_<id>_v<version>.glb
    chunk_<id>_v<version>.json Phase 0B metadata/full synthetic mesh sidecar
  depth/
    frame_<id>.npz optional
  logs/
    runtime_profile.json
```

Phase 0C supports a minimal reader for this Phase 0B sidecar format. The reader
reconstructs `PoseEstimate`, `CameraModel`, `ObjectInstance`, and `MeshChunk`
records from JSON/JSONL sidecars and records sorted `depth/frame_<id>.npz` paths
without loading every depth array by default.

### Session inspection preview

`atlas3r inspect session --input <session.atlas3r> --output <preview_dir>` writes
deterministic dependency-free HTML/SVG files:

```text
preview_dir/
  index.html
  top_down.svg
  depth_frame_000000.svg
  object_mask_frame_000000.svg
```

The preview is diagnostic only. It must not be used as an accuracy report, and
hidden or completed geometry must not be presented as measured geometry.

### Metadata requirements

Every export must include:

- coordinate convention;
- unit scale;
- scale source;
- camera metadata source;
- model checkpoint hash;
- voxel size;
- accuracy report path or `null`;
- warnings if RGB-only best effort.
