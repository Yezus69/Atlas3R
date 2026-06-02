# 08 - API Contracts and Coordinate Conventions

This is the concise source-of-truth index for public Atlas3R contracts. It is
not a phase log. Public field names, tensor shapes, coordinate conventions, and
CLI names must not change silently.

Diagnostic smoke, sidecar, and inspection outputs are contract-plumbing
artifacts. They are not accuracy reports, not performance reports, and must not
present hidden or completed geometry as measured geometry.

## Coordinate Conventions

- Units are meters unless a field explicitly says otherwise.
- Camera frame: `x` right, `y` down, `z` forward.
- World frame: initialized from anchor keyframes unless an external frame is supplied.
- Transform names use `T_A_B`, mapping homogeneous points from frame `B` into frame `A`.
- Public transform fields must use explicit names such as `T_world_camera`, not `pose`.

```text
p_world = T_world_camera @ p_camera
camera_center_world = T_world_camera[:3, 3]
```

## Core Frame And Geometry Contracts

### FramePacket

```python
@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp_ns: int
    rgb_u8: NDArray[np.uint8]          # H,W,3 original frame
    rgb_model: Tensor                  # 3,Hm,Wm normalized
    K_original: NDArray[np.float32] | None
    K_model: NDArray[np.float32]       # 3,3 adjusted for model image
    distortion: CameraDistortion | None
    resize_transform: NDArray[np.float32]
    camera_metadata: dict[str, Any]
```

### RGB Frame-Source Boundary

`atlas3r.data.frame_source` provides dependency-free local fixture ingestion.
It must not import video decoders, image libraries, model packages, runtime
schedulers, mappers, or exporters.

```python
class RGBFrameSource(Protocol):
    def frames(self) -> Iterator[FramePacket]: ...
```

Public loaders: `NPZFrameSource(path)`, `load_npz_clip_frames(path)`,
`PPMSequenceFrameSource(directory)`, `load_ppm_sequence_frames(directory)`, and
`write_frame_source_smoke_fixture(output_dir)`.

NPZ clips require `rgb_u8` shaped `T,H,W,3` and `K` shaped `3,3` or `T,3,3`.
PPM sequences load binary `P6` `.ppm` files sorted by filename and use explicit
`K` or `intrinsics.npz` with key `K`.

Each emitted `FramePacket` uses deterministic `frame_id` values, placeholder
`timestamp_ns=0`, channel-first float32 `rgb_model` normalized to `[0, 1]`,
validated `K_original`/`K_model`, identity `resize_transform`, and
`camera_metadata.source_format` of `npz` or `ppm_sequence`. Invalid paths,
RGB layouts, PPM headers, or intrinsics raise path-named `ValueError`s.

### CameraModel

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

### PoseEstimate

```python
@dataclass(frozen=True)
class PoseEstimate:
    frame_id: int
    timestamp_ns: int
    T_world_camera: NDArray[np.float32]       # 4,4
    q_world_camera_xyzw: NDArray[np.float32]  # 4
    camera_center_world_m: NDArray[np.float32] # 3
    covariance_6x6: NDArray[np.float32] | None
    confidence: float
    tracking_state: str      # OK|LOW_CONFIDENCE|RELOCALIZING|LOST|NEW_SUBMAP
    scale_source: str        # rgb_prior|calibrated_rgb|known_anchor|external_pose
    diagnostics: dict[str, Any]
```

### DenseMatchSet

```python
@dataclass(frozen=True)
class DenseMatchSet:
    source_frame_id: int
    target_frame_id: int
    source_pixels_uv: NDArray[np.float32] # N,2
    target_pixels_uv: NDArray[np.float32] # N,2
    confidence: NDArray[np.float32]       # N values in [0,1]
```

### FramePrediction

```python
@dataclass
class FramePrediction:
    pose: PoseEstimate
    camera: CameraModel
    depth_m: Tensor                    # H,W float32/float16
    depth_sigma_m: Tensor              # H,W non-negative
    normal_camera: Tensor              # H,W,3
    point_world: Tensor                # H,W,3
    confidence: Tensor                 # H,W values in [0,1]
    static_mask: Tensor                # H,W bool or probability
    object_embeddings: Tensor | None
    object_mask_logits: Tensor | None
    dense_matches: DenseMatchSet | None
```

### DepthObservation

Mapper inputs use `atlas3r.mapping.observations.DepthObservation`. CPU TSDF
fusion and teacher-cache replay convert source data to this public boundary
before integration.

```python
@dataclass(frozen=True)
class DepthObservation:
    frame_id: int
    camera: CameraModel
    pose: PoseEstimate
    depth_m: NDArray[np.float32]       # H,W finite, non-negative meters
    depth_sigma_m: NDArray[np.float32] # H,W finite, non-negative meters
    confidence: NDArray[np.float32]    # H,W finite values in [0,1]
    static_mask: NDArray[np.bool_] | NDArray[np.float32] | None = None
    object_id: NDArray[np.int32] | None = None
    rgb_u8: NDArray[np.uint8] | None = None
    source: str = "unknown"
```

Validation requires non-negative `frame_id`, `CameraModel`, `PoseEstimate`,
HxW depth/sigma/confidence matching camera size, finite non-negative depth and
sigma, confidence in `[0, 1]`, optional HxW bool/probability `static_mask`,
optional HxW integer `object_id`, optional HxWx3 uint8 `rgb_u8`, and non-empty
`source`.

Related public helpers: `atlas3r.data.synthetic_observations.depth_observation_from_synthetic_frame(frame)`,
`atlas3r.mapping.tsdf_grid.compute_tsdf_grid_shape(...)`, and
`atlas3r.mapping.tsdf_grid.voxel_centers_world(...)`.

## Teacher Adapter And Cache Contracts

Third-party models stay external and are isolated behind dependency-safe
adapters under `atlas3r.models.adapters`.

```python
class GeometryTeacherAdapter(Protocol):
    def predict(self, frames: FrameBatch) -> TeacherPrediction: ...

@dataclass(frozen=True)
class FrameBatch:
    frames: tuple[FramePacket, ...]     # non-empty, unique frame_id values
    batch_id: str
    metadata: Mapping[str, Any]

@dataclass(frozen=True)
class TeacherPrediction:
    adapter_name: str
    frame_predictions: tuple[FramePrediction, ...]
    capabilities: AdapterCapabilities
    metadata: Mapping[str, Any]
```

`AdapterCapabilities` fields are `predicts_camera`, `predicts_pose`,
`predicts_depth`, `predicts_normals`, `predicts_points`,
`predicts_dense_matches`, `predicts_objects`, `supports_batch`,
`supports_streaming`, and `notes`.

`AdapterStatus` fields are `name`, `display_name`, `availability`
(`available|unavailable|stub-only`), `capabilities`, `install_hint`, and
`reason`.

Known stubs include `VGGTAdapter` and `DepthProAdapter`; missing optional
dependencies raise `AdapterDependencyError` at construction or prediction time.
`fixture-cube-room` is an available synthetic-only adapter for plumbing tests
and cache writing.

`atlas3r.data.teacher_frame_batch_from_frame_packets(frames, ...)` and
`teacher_frame_batch_from_rgb_source(source, ...)` build ordered `FrameBatch`
records from existing `FramePacket` / `RGBFrameSource` inputs. They preserve
input order, reject empty/non-packet/duplicate-frame-id inputs, keep compact
deterministic metadata, and do not run inference or touch mapper/runtime/TSDF
paths.
Teacher cache layout: `metadata.json`, `frame_summaries.jsonl`, and optional
`arrays/` per-frame `.npz` payloads.

`metadata.json` records `format_name=atlas3r_teacher_prediction_cache`,
`format_version=1`, adapter status/capabilities, prediction metadata,
coordinate frame/convention, frame count/IDs, scale sources, summary path, and
array storage state. `frame_summaries.jsonl` is sorted by `frame_id` and records
frame/timestamp, coordinate frame, scale source, camera fields, pose fields,
confidence summaries, uncertainty summaries, tensor shape/dtype summaries, dense
match summary, and `arrays_path`.

Optional payloads are written only with `--store-arrays` or `store_arrays=True`
as `arrays/frame_<frame_id:06d>.npz`. Required keys are `depth_m`,
`depth_sigma_m`, `normal_camera`, `point_world`, `confidence`, and
`static_mask`; optional keys are `object_embeddings` and `object_mask_logits`.
Payload validation checks shapes, dtypes, finite values, probability ranges, and
non-negative depth/uncertainty.

Public commands: `atlas3r adapters list`; `atlas3r adapters run --adapter <name>
--input <session.atlas3r> --output <cache_dir> [--store-arrays]`; `atlas3r inspect teacher-cache --input <cache_dir>`.

## TSDF, MeshChunk, And WorldMap Diagnostic Outputs

`atlas3r smoke tsdf-cube-room --output <folder>` writes deterministic NumPy CPU
TSDF reference artifacts:

```text
<folder>/
  synthetic_cube_room.atlas3r/
  tsdf_grid.npz        tsdf, weight, grid_min_corner_world_m, voxel_size_m
  surface_points.npz   points_world_m, confidence, uncertainty_m, voxel_indices_xyz
  metadata.json        source frames, coordinate frame, voxel size, coverage, uncertainty
  metrics.json         conservative synthetic fixture metrics or not-evaluated data
```

`atlas3r smoke teacher-cache-tsdf --input <cache_dir> --output <folder>`
requires a full-array cache, converts replay frames through `DepthObservation`,
and writes the same TSDF artifact family.

Optional sidecar flags:

```bash
atlas3r smoke tsdf-cube-room --output <folder> --write-mesh-sidecar|--write-world-map-sidecar
atlas3r smoke teacher-cache-tsdf --input <cache_dir> --output <folder> --write-mesh-sidecar|--write-world-map-sidecar
```

`mesh_chunk_sidecar.json` has `format_name=atlas3r_tsdf_surface_mesh_chunk_sidecar`,
`format_version=1`, a validated `MeshChunk`, sidecar/source metadata, and
sample confidence/uncertainty arrays. It uses world-frame vertices with
`T_world_chunk=identity`, low-fidelity marker triangles, `surface_source_per_face=0`,
and `object_id_per_face=-1` until object-aware fusion exists.

`world_map_sidecar.json` has `format_name=atlas3r_tsdf_world_map_sidecar`,
`format_version=1`, one validated observed `MeshChunk` inside a validated
`WorldMap`, empty `objects` and `keyframes`, deterministic `created_at_ns=0`,
and source metadata.

Inspection commands: `atlas3r inspect world-map --input <folder>/world_map_sidecar.json`;
`atlas3r inspect tsdf-output --input <folder> [--mode surface|mesh|world-map|complete]`.

TSDF output inspection validates required artifacts, sidecars when required,
metrics when present, and cross-checks coordinate frame, source frame IDs,
voxel size, scale source, observed coverage, confidence, and mean/p95 uncertainty.

## Runtime Fixture Contracts

`atlas3r smoke runtime-fixture --output <folder>` is a single-threaded
deterministic scheduler skeleton over the synthetic cube-room fixture. It writes:

```text
<folder>/
  runtime_events.jsonl
  runtime_summary.json
  synthetic_cube_room.atlas3r/
  teacher_cache/
  teacher_cache_tsdf/
```

Runtime event records use `format_name=atlas3r_runtime_fixture_event_log`,
`format_version=1`, monotonic `event_index`, known `stage_name`, optional
`frame_id`, deterministic placeholder `timestamp_ns` and `latency_ns`,
`dropped_frame`, scheduler-owned `memory_counters`, relative `paths`, and
deterministic `metadata`.

Known stage names: `runtime_start`, `session_write`, `source_frame`,
`adapter_cache_write`, `adapter_cache_frame`, `tsdf_replay`,
`tsdf_replay_frame`, `tsdf_output_write`, and `runtime_complete`.

`runtime_summary.json` uses `format_name=atlas3r_runtime_fixture_smoke_summary`
and records runtime paths, frame IDs, event-log path, bounded-memory counters,
artifact paths, and a truth boundary. `atlas3r inspect runtime-fixture --input
<folder>` validates the event log, summary, generated session, full-array
teacher cache, and nested complete TSDF output inspection.

## Student Model Boundary
`atlas3r.models.student` is a dependency-safe NumPy-only boundary for future
Streaming Metric Geometry Transformer work. It is not a mapper input contract.

```python
@dataclass(frozen=True)
class StudentClipInput:
    frame_ids: tuple[int, ...]           # length T
    images_rgb: NDArray                  # B,T,3,H,W uint8|float32|float64
    intrinsics: NDArray                  # B,T,3,3 or shared 3,3
    T_world_camera_prior: NDArray | None # optional B,T,4,4 context
    coordinate_frame: str = "x_right_y_down_z_forward"
    metadata: dict[str, object]
```

Validation requires positive `B,T,H,W`, `len(frame_ids)==T`, accepted image
dtypes, finite floating images, valid intrinsics, optional clip-shaped valid
transforms, and coordinate frame `x_right_y_down_z_forward`.

```python
@dataclass(frozen=True)
class StudentForwardOutput:
    frame_ids: tuple[int, ...]       # length T
    depth_m: NDArray                 # B,T,H,W non-negative
    depth_sigma_m: NDArray           # B,T,H,W non-negative
    confidence: NDArray              # B,T,H,W values in [0,1]
    dynamic_probability: NDArray     # B,T,H,W values in [0,1]
    normals_camera: NDArray          # B,T,3,H,W
    pointmap_camera_m: NDArray       # B,T,3,H,W
    T_world_camera: NDArray          # B,T,4,4
    intrinsics: NDArray              # B,T,3,3
    truth_boundary: dict[str, object]
    coordinate_frame: str = "x_right_y_down_z_forward"
```

`truth_boundary` must include boolean `shape_only`, `learned_inference`,
`usable_for_mapping`, `accuracy_report`, and `performance_report`. The
`ShapeOnlyStudentModel.forward(input)` stub returns deterministic placeholder
arrays with `learned_inference=false`, `usable_for_mapping=false`,
`accuracy_report=false`, and `performance_report=false`.

### FramePacket -> StudentClipInput Bridge
`atlas3r.data.student_clip_from_frame_packets(frames, batch_id=...)` converts a
non-empty ordered `FramePacket` sequence into one `StudentClipInput` with
`images_rgb` shaped `1,T,3,H,W`, `intrinsics` shaped `1,T,3,3`, preserved
`frame_ids`, compact source metadata, and no inference or mapper/TSDF use. It
rejects non-packets, duplicate frame IDs, mismatched/non-`3,H,W` `rgb_model`
shapes, and invalid intrinsics.

## Training MVP Contracts
`atlas3r train synthetic-overfit --output <run_dir>` is an optional PyTorch MVP
over deterministic procedural synthetic RGB/depth. Base imports,
`atlas3r.training`, and dataset generation must not import Torch eagerly;
missing Torch exits CLI training with code 2 and the train-extra install hint.

`SyntheticDepthSample` fields: `sample_id`, `frame_id`, `rgb_u8 H,W,3`,
`rgb_model 3,H,W`, `depth_m H,W`, `depth_sigma_m H,W`, `confidence H,W`,
`object_mask H,W`, `K 3,3`, `T_world_camera 4,4`, `camera_center_world_m 3`,
and metadata with `synthetic_only=true`, seed, and scene bounds.
`sample_to_student_clip(sample)` returns shapes `1,1,3,H,W`, `1,1,3,3`, and
`T_world_camera_prior 1,1,4,4`.

`checkpoint_last.pt`: `format_name=atlas3r_tiny_depth_pose_checkpoint`,
`format_version=1`, `step`, `model_state_dict`, `optimizer_state_dict`,
`config`, `metrics`, and `truth_boundary`. Runs write `config.json`,
`metrics.jsonl`, `summary.json`, `prediction_sample.npz`,
`prediction_preview.html`, and `prediction_preview.svg`. Truth boundary fields:
`training_mvp=true`, `synthetic_only=true`, `real_capture_model=false`,
`usable_for_realtime_mapping=false`, `accuracy_report=false`,
`performance_report=false`; synthetic-overfit only, not realtime or real-capture.

## Map Object Contracts
```python
@dataclass
class ObjectInstance:
    object_id: int
    label_candidates: list[tuple[str, float]]
    T_world_object: NDArray[np.float32]        # 4,4
    oriented_bbox_center_m: NDArray[np.float32] # 3
    oriented_bbox_axes: NDArray[np.float32]    # 3,3
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

```python
@dataclass
class MeshChunk:
    chunk_id: str
    version: int
    T_world_chunk: NDArray[np.float32]      # 4,4
    vertices_m: NDArray[np.float32]         # N,3
    faces: NDArray[np.int32]                # M,3
    normals: NDArray[np.float32] | None     # N,3
    colors: NDArray[np.uint8] | None        # N,3/4
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

Surface source enum: `0 observed_surface`, `1 single_view_prior`,
`2 completed_surface`, `3 dynamic_surface`, `4 low_confidence`.

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

## Live API Events
Runtime event names: `PoseUpdate(frame_id, PoseEstimate)`,
`DepthUpdate(frame_id, optional compressed depth/confidence)`,
`ObjectUpdate(object_id, ObjectInstance)`, `MeshChunkAdded(chunk_id, version)`,
`MeshChunkUpdated(chunk_id, version)`, `MeshChunkRemoved(chunk_id, version)`,
`TrackingStateChanged(state)`, and `BenchmarkMetric(name, value)`.

## File Formats
### `.atlas3r` Session Folder
Folder layout: `metadata.json`, `poses.jsonl`, `cameras.jsonl`,
`objects.jsonl`, `mesh_chunks/chunk_<id>_v<version>.json`, optional GLB files,
optional `depth/frame_<id>.npz`, and `logs/runtime_profile.json`.

The Phase 0C reader reconstructs `PoseEstimate`, `CameraModel`,
`ObjectInstance`, and `MeshChunk` records from JSON/JSONL sidecars and records
sorted `depth/frame_<id>.npz` paths without loading every depth array by default.

`atlas3r inspect session --input <session.atlas3r> --output <preview_dir>`
writes deterministic `index.html`, `top_down.svg`, `depth_frame_000000.svg`,
and `object_mask_frame_000000.svg` preview files.

Every export or sidecar must include or preserve coordinate convention, unit
scale, scale source, camera metadata source when known, model checkpoint hash or
`null`, voxel size when relevant, accuracy report path or `null`, and warnings
for RGB-only best effort.
