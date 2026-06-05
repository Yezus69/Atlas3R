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

Related helpers: `depth_observation_from_synthetic_frame(frame)`,
`compute_tsdf_grid_shape(...)`, and `voxel_centers_world(...)`.

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

`AdapterCapabilities` records modalities plus batch/streaming support.
`AdapterStatus` records name/display name, availability, capabilities, install
hint, and reason; known stubs include `VGGTAdapter` and `DepthProAdapter`.
Missing optional dependencies raise `AdapterDependencyError`.

`teacher_frame_batch_from_frame_packets(...)` and
`teacher_frame_batch_from_rgb_source(...)` build ordered `FrameBatch` records
without running inference or touching mapper/runtime/TSDF paths. Teacher
prediction caches are `metadata.json`, `frame_summaries.jsonl`, and optional
`arrays/frame_<frame_id:06d>.npz` payloads enabled by `--store-arrays`.

Teacher-signal caches use `atlas3r_teacher_signal_manifest.json` plus
`signals/clip_<id>.npz`; manifests record source clip cache, dataset/sequence,
split, clip size, teacher identity/source type, relative payloads, frame IDs,
timestamps, and truth flags (`diagnostic_only=true`, no accuracy/performance
report, measured-vs-pseudo source). Payload required arrays are `depth_m`,
`depth_sigma_m`, `confidence`, `valid_mask`, `K`, `T_world_camera`,
`frame_ids`, and `timestamps_s`; optional arrays include pointmaps, normals,
object IDs/confidence, and dynamic probability. Validation checks finite
arrays, non-negative depth/sigma, probabilities, valid intrinsics/transforms,
safe relative paths, and matching source clip metadata. External teacher
runners expose `ExternalTeacherStatus`, `ExternalTeacherRunConfig`, and
`ExternalTeacherRunner`, import no external model packages at module import
time, and keep Depth Pro/VGGT outputs pseudo-labels unless source-measured.
Commands include `adapters list|run`, `inspect teacher-cache`, `teachers
forge-measured-tum|ingest-local|run-depth-pro|run-vggt|ingest-vggt-local|
run-student-temporal|inspect-signals|map-signals`.
## TSDF, MeshChunk, And WorldMap Diagnostic Outputs
`smoke tsdf-cube-room` writes `tsdf_grid.npz` (`tsdf`, `weight`, bounds, voxel
size), `surface_points.npz` (`points_world_m`, confidence, uncertainty,
voxel indices), `metadata.json`, and `metrics.json`. `smoke teacher-cache-tsdf`
requires full arrays, converts replay frames through `DepthObservation`, and
writes the same family. Optional `--write-mesh-sidecar` and
`--write-world-map-sidecar` add observed-only sidecars:
`mesh_chunk_sidecar.json` wraps a validated `MeshChunk` with
`T_world_chunk=identity`, marker triangles, observed surface source, and
`object_id_per_face=-1`; `world_map_sidecar.json` wraps that chunk in a
validated `WorldMap` with empty objects/keyframes and deterministic
`created_at_ns=0`. Inspection commands: `inspect world-map` and `inspect
tsdf-output --mode surface|mesh|world-map|complete`, which cross-check
coordinate frame, source frame IDs, voxel size, scale source, coverage,
confidence, and mean/p95 uncertainty.

## Runtime Contracts

`atlas3r smoke runtime-fixture --output <folder>` writes deterministic
`runtime_events.jsonl`, `runtime_summary.json`, a synthetic session, teacher
cache, and complete TSDF replay. Events use format
`atlas3r_runtime_fixture_event_log`, monotonic indexes, known stages, bounded
memory counters, relative paths, and deterministic timings. `inspect
runtime-fixture` validates the folder. `runtime stream-student-map` emits
diagnostic checkpoint `DepthObservation`s; pose modes are `oracle`,
`student-relative`, `student-odometry`, or `both`, and truth claims stay false.

### RGB Teacher Mapping
`runtime map-rgb-teacher --input <recording-or-rgb-folder-or-video> --output <run> --teacher vggt --rgb-only` converts VGGT teacher-pseudo depth/pose/intrinsics into validated `DepthObservation`s, then reuses sparse TSDF and observed mesh chunks. Inputs are RGB-only `atlas3r_recording`, NPZ/PPM, image folders, or videos; measured recording depth/pose are mapping-ignored and eval-only. Fixture teacher mode is test-only; optional deps raise explicit errors. VGGT camera-from-world extrinsics are inverted to `T_world_camera`; missing teacher pose is rejected. Teacher scale is `rgb_prior` or `teacher_scale_unverified`.

Multi-window runs default to `--stitch-windows sim3-overlap`; `none` preserves the Phase 6G baseline. Stitching estimates local-window to pseudo-global Sim3 from duplicate overlap camera centers and optional teacher pointmaps. Scale is applied to camera centers, depth, and depth sigma; intrinsics are unchanged. Insufficient overlap, inliers, scale ratio, or center RMSE rejects a window.

Outputs: `rgb_teacher_frames/`, `rgb_teacher_predictions/`, `rgb_teacher_recording/`, `rgb_teacher_summary.json`, `rgb_teacher_report.md`, live replay JSON/Markdown, `sparse_tsdf/`, `mesh_chunks/`, optional point cloud/eval sidecars, and optional `teacher_temporal_cache/`. Summary format `atlas3r_rgb_teacher_mapping_summary` records teacher metadata, counts, sparse/mesh counters, latency, artifacts, limitations, stitch diagnostics, and `truth_boundary`.

Stitch fields: `stitch_mode`, window/edge accepted/rejected counts, `stitch_submap_count`, overlap RMSE, scale min/median/max, boundary jumps, and `rejected_windows`. Mesh metadata inherits `stitch_mode`, `pseudo_submap_id`, `metric_scale_source`, and `teacher_geometry_used=true`.

Teacher temporal caches use `teacher_temporal_cache/` with manifest `atlas3r_teacher_temporal_cache.json`, `clips/clip_<id>.npz`, `frame_index.jsonl`, and inspect reports. Clip arrays: `frame_ids`, `timestamps_ns`, `rgb_u8`, `K`, `T_world_camera`, `depth_m`, `depth_sigma_m`, `confidence`, `valid_mask`, `teacher_valid`, and `pseudo_submap_id`. Manifests record teacher source/device/checkpoint, stitch mode, metric scale source, clip length/stride, truth boundary, array schema, safe relative paths, and `no_model_weights_in_cache=true`. Inspect with `atlas3r inspect teacher-temporal-cache --cache <cache_dir>`. Default pseudo depth and pose target weights are `0.25`.

Truth flags keep pseudo/teacher/observed diagnostic flags true and measured mapping, hidden geometry, completion, student RGB-only, accuracy, and realtime flags false.
### Capture Adapter Boundary
Runtime capture adapters live under `atlas3r.runtime`, import no optional
camera deps at module import time, and expose `CaptureAdapterStatus` fields
`name`, `display_name`, `available`, optional `reason`/`install_hint`,
`capabilities`, and dependency versions. `CaptureFrame` has `frame_id`,
`timestamp_ns`, optional `FramePacket`, metadata, measured-depth/pose booleans,
and optional measured depth path. Public command:
`atlas3r runtime capture-adapters list`.

### Live Replay Recording
`atlas3r runtime live-replay-recording --recording <recording_dir> --output
<run_dir> --target-fps N --mapper-backend cpu-sparse` replays a measured
recording through bounded capture/map queues. Default pacing is deterministic
simulation; `--wall-clock-pacing` is opt-in. Pose output is separate from map
updates. Mapping only uses `DepthObservation`s built from measured recording
depth and measured `T_world_camera`; missing depth/pose skips mapping.
`live_replay_events.jsonl` uses `format_name=atlas3r_live_replay_event`,
`format_version=1`, monotonic `event_index`, `stage_name`, optional `frame_id`,
queue depths, latency, `drop_reason`, `keyframe_selected`, and
`keyframe_reason`. Drop reasons are `capture_queue_full`, `map_queue_full`,
`missing_depth`, `missing_pose`, `duplicate_frame`, `not_keyframe`,
`scheduler_shutdown`, or `other`. Keyframe reasons are `first_frame`, `stride`,
`translation_threshold`, `rotation_threshold`, `uncertainty_threshold`,
`forced`, or `not_selected`.
`live_replay_summary.json` uses `format_name=atlas3r_live_replay_summary` and records source metadata, target FPS, simulated pacing, frame/pose/keyframe/map/drop/queue counts, latency p50/p95/max, mapper backend, voxel size, sparse/dirty/mesh counters, memory counters, artifacts, and `truth_boundary` flags: `diagnostic_only=true`, `accuracy_report=false`, `performance_report=false`, `realtime_claim=false`, `rgb_only_mapping_ready=false`, `hidden_geometry_measured=false`, `observed_only=true`, `predicted_completion=false`, measured depth/pose booleans, and `student_rgb_only_used=false`. Outputs include `live_replay_report.md`, `sparse_tsdf/`, optional `surface_points.ply`, and optional observed mesh chunks.

### Observed Mesh Chunk Artifacts
`runtime live-replay-recording --export-mesh-chunks` writes `mesh_chunks/mesh_chunk_manifest.json`, `mesh_chunks/mesh_chunk_updates.jsonl`, `mesh_chunks/chunks/*.npz`, and `*.ply` when `--mesh-format ply|both` is requested. NPZ is the canonical array payload; PLY is a loadability/debug export. Stable chunk IDs are `block_<x>_<y>_<z>`.
`MeshChunk` metadata uses `format_name=atlas3r_observed_mesh_chunk`, `format_version=1`, `chunk_id`, `chunk_coord_xyz [int,int,int]`, `version`, `update_type upsert|delete`, `coordinate_frame`, `T_world_chunk 4x4`, `voxel_size_m`, `truncation_distance_m`, `block_size_voxels`, `bbox_world_min_m/max_m`, `source_frame_ids`, `first_source_frame_id`, `last_source_frame_id`, `active_voxel_count`, `surface_voxel_count`, `vertex_count`, `triangle_count`, `uncertainty_summary_m mean/p50/p95/max`, `confidence_summary mean/p50/p95/max`, `observed_coverage_estimate`, `metric_scale_source`, `mesher_backend`, and truth flags `observed_only=true`, `predicted_completion=false`, `hidden_geometry_measured=false`, `rgb_only_mapping_ready=false`, `student_rgb_only_used=false`, `accuracy_report=false`, `realtime_claim=false`. Payload arrays: `vertices_world_m float32[N,3]`, `normals_world float32[N,3]`, `triangles uint32[M,3]`.
`MeshChunkUpdateEvent` records monotonic `event_index`, `frame_id`, `timestamp_ns`, `chunk_id`, `chunk_coord_xyz`, `version`, `update_type`, relative `payload_npz`, optional relative `payload_ply`, `vertex_count`, `triangle_count`, `dirty_reason`, `mesh_latency_ns`, capture/map queue depths, and the same truth flags. `MeshChunkManifest` uses `format_name=atlas3r_observed_mesh_chunk_manifest`, `format_version=1`, `chunk_count`, `active_chunk_count`, `deleted_chunk_count`, `total_vertex_count`, `total_triangle_count`, `source_frame_ids_mapped`, `coordinate_frame`, `voxel_size_m`, `mapper_backend`, `mesher_backend`, `truth_boundary`, and a `chunks` index of active latest payloads.
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
`atlas3r.data.student_clip_from_frame_packets(frames, batch_id=...)` converts
ordered `FramePacket`s into one `StudentClipInput` with `images_rgb 1,T,3,H,W`,
`intrinsics 1,T,3,3`, preserved frame IDs, compact metadata, and no inference or
mapper use. It rejects non-packets, duplicate IDs, bad RGB shapes, and invalid K.

## Training And Checkpoint-Inference MVP Contracts
Optional Torch/Pillow paths must not load through base `import atlas3r`; missing
deps exit CLI training/checkpoint commands with code 2 and the train-extra hint.

Synthetic MVP: `atlas3r train synthetic-overfit --output <run_dir>` uses
`SyntheticDepthSample` fields `sample_id`, `frame_id`, `rgb_u8 H,W,3`,
`rgb_model 3,H,W`, `depth_m/depth_sigma_m/confidence/object_mask H,W`, `K 3,3`,
`T_world_camera 4,4`, `camera_center_world_m 3`, and synthetic-only metadata.
`sample_to_student_clip` returns `1,1,3,H,W`, `1,1,3,3`, and
`T_world_camera_prior 1,1,4,4`.

Tiny checkpoints use `format_name=atlas3r_tiny_depth_pose_checkpoint`,
`format_version=1`, `step`, `model_state_dict`, `optimizer_state_dict`,
`config`, `metrics`, `model_config`, and `truth_boundary`. Required gates are
`training_mvp=true`, `learned_inference=true`, `usable_for_mapping=false`,
`usable_for_realtime_mapping=false`, `accuracy_report=false`,
`performance_report=false`, and `generalizes_to_real_world=false`.

`load_tiny_depth_pose_checkpoint(path)` validates truth flags and runs
`TinyDepthPoseNet` for student-clip or frame-packet prediction.
`depth_observations_from_tiny_prediction(...)` emits validated
`DepthObservation` records with predicted depth/sigma/confidence, RGB-prior
scale, coordinate frame, and truth-boundary diagnostics. `atlas3r smoke
checkpoint-tsdf --checkpoint <checkpoint.pt> --output <folder> [--input
<clip.npz>]` writes predicted TSDF artifacts and `prediction_sample.npz`;
synthetic mode adds target comparison, while NPZ clips are not target-evaluated.

### TUM RGB-D Real-Data Debug Training, Eval, And Temporal Clips
Commands include `datasets tum-rgbd download|prepare`, `train
tum-rgbd-depth-pose --model tiny-v1|tiny-v2`, `eval tum-rgbd-checkpoint
[--write-tsdf]`, `forge tum-rgbd-clips`, `train tum-rgbd-temporal`, and
`train teacher-signals-temporal`. Download uses safe tar extraction; supported
sequence specs include `freiburg1_xyz`, `freiburg1_desk`, `freiburg2_xyz`, and
`freiburg3_long_office_household`.

The TUM manifest is `format_name=atlas3r_tum_rgbd_manifest`,
`format_version=1`, and records RGB/depth metadata, ROS default `K`,
`depth_raw/5000.0`, split metadata, frame IDs, timestamps, paths,
`T_world_camera`, camera center, and truth boundary. Clip caches use
`atlas3r_clip_cache_manifest.json` plus `clips/clip_<id>.npz` payloads with
RGB/depth/mask/K/`T_world_camera`/frame/timestamp arrays, `center_index`, and
optional pointmap/normal arrays.

`TeacherSignalTemporalDataset` aligns validated teacher-signal caches to source
clip RGB, intrinsics, `T_world_camera`, frame IDs, timestamps, depth/sigma/
confidence/valid-mask targets, measured flags, and optional pointmaps. Mixed
caches may span datasets/sequences but must share split, clip length, and image
size. `TemporalMetricNetV1` predicts depth/sigma/confidence and relative
translation/6D rotation; `student-odometry` requires trained rotation-head
weights. Teacher-signal losses use confidence/sigma weighting, measured teacher
weight `1.0`, pseudo teacher weight `0.25`, SE(3) pose losses, and diagnostic
depth/pose metrics. `run-student-temporal` writes pseudo-label caches and uses
source clip-cache `T_world_camera` unless a later phase upgrades pose export.

## Map Object Contracts
`ObjectInstance` fields: `object_id`, `label_candidates`, `T_world_object 4,4`,
`oriented_bbox_center_m 3`, `oriented_bbox_axes 3,3`,
`oriented_bbox_extents_m 3`, `mesh_chunk_ids`, `is_dynamic`,
`observed_coverage_ratio`, `confidence`, `uncertainty_m`,
`first_seen_frame_id`, `last_seen_frame_id`, and `metadata`.

`MeshChunk` fields: `chunk_id`, `version`, `T_world_chunk 4,4`,
`vertices_m N,3`, `faces M,3`, optional `normals N,3`, optional
`colors N,3/4`, optional `uvs`, optional `object_id_per_face`, optional
`surface_source_per_face`, `voxel_size_m`, `mean_uncertainty_m`,
`p95_uncertainty_m`, `source_frame_ids`, `scale_source`, and `flags`. Surface
source enum: `0 observed_surface`, `1 single_view_prior`,
`2 completed_surface`, `3 dynamic_surface`, `4 low_confidence`.

`WorldMap` fields: `map_id`, `world_frame_name`, `created_at_ns`,
`mesh_chunks: dict[str, MeshChunk]`, `objects: dict[int, ObjectInstance]`,
`keyframes: dict[int, PoseEstimate]`, `scale_source`, `global_confidence`, and
`metadata`.

## Live API Events
Runtime event names: `PoseUpdate`, `DepthUpdate`, `ObjectUpdate`,
`MeshChunkAdded/Updated/Removed`, `TrackingStateChanged`, and
`BenchmarkMetric`; payloads use the public contracts above.

## File Formats
### Atlas3R Recording Folder
Layout is `atlas3r_recording.json`, `frames.jsonl`, optional local
`rgb/<frame_id>.<ext>`, and optional measured `depth/<frame_id>.npz|png`.
Manifest fields: `format_name=atlas3r_recording`, `format_version=1`,
`coordinate_frame=x_right_y_down_z_forward`, `frame_count`, `width`, `height`,
`source_dataset`, `source_sequence`, `capture_metadata`,
`known_calibration_metadata`, `depth_present`, `pose_present`,
`truth_boundary`, optional `external_roots`; truth flags keep
`diagnostic_only=true`, `accuracy_report=false`, `performance_report=false`,
and `hidden_geometry_measured=false`. Frame fields: `frame_id`, chronological
`timestamp_s`, safe relative `rgb_path`, optional `depth_path`, optional PNG
`depth_scale`, `K`, optional `T_world_camera`, optional
`camera_center_world_m`, and `source_metadata`; absolute paths and `..` are
rejected. Import commands:
```bash
atlas3r recording from-tum --manifest <tum_manifest.json> --output <recording_dir> --split val --max-frames N --width W --height H
atlas3r recording from-clip-cache --clip-cache <manifest-or-folder> --output <recording_dir> --dedupe-frame-id
atlas3r recording from-sensor-folder --input <sensor_capture_dir> --output <recording_dir>
```
`from-tum` references external TUM RGB/depth and measured poses;
`from-clip-cache` writes local NPZ RGB/depth payloads. `from-sensor-folder`
references an external `sensor_capture.json`/`frames.jsonl` folder with Atlas3R
coordinate frame, calibration metadata, safe relative RGB/depth paths, finite
K/poses/timestamps, image dimensions, positive PNG depth scale, and no-claim
truth flags including `realtime_claim=false`.
### Recording Fusion Runtime
`runtime fuse-recording --recording <recording_dir> --output <run_dir>` streams measured depth+pose to `DepthObservation`, events/reports, final surface
output, mesh status, and preview HTML. Incremental accepts
`--backend cpu-persistent|cpu-rebuild|cpu-sparse`; dense backends write `tsdf/`,
while `cpu-sparse` writes `sparse_tsdf/sparse_tsdf_state.npz`, active state
counters, final-only surface output, and non-exact sparse-vs-dense
`backend_comparison.json`. All reports keep diagnostic/no-accuracy/no-performance/
no-realtime/no-mapping-ready flags. `runtime sparse-tsdf-stress` writes dense
apartment-box memory estimates and sparse active-state estimates.
### `.atlas3r` Session Folder
Layout: `metadata.json`, `poses.jsonl`, `cameras.jsonl`, `objects.jsonl`,
mesh chunk sidecars, optional GLB/depth NPZs, and `logs/runtime_profile.json`.
Readers reconstruct contracts from sidecars; exports preserve coordinate
convention, units, scale source, camera metadata, checkpoint hash, voxel size,
accuracy report path, and RGB-only warnings.
