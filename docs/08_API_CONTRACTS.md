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

`AdapterCapabilities` records predicted modalities plus batch/streaming support.
`AdapterStatus` records name/display name, availability
(`available|unavailable|stub-only`), capabilities, install hint, and reason.
Known stubs include `VGGTAdapter` and `DepthProAdapter`; missing optional
dependencies raise `AdapterDependencyError`.

`atlas3r.data.teacher_frame_batch_from_frame_packets(frames, ...)` and
`teacher_frame_batch_from_rgb_source(source, ...)` build ordered `FrameBatch`
records from existing `FramePacket` / `RGBFrameSource` inputs. They preserve
input order, reject empty/non-packet/duplicate-frame-id inputs, keep compact
deterministic metadata, and do not run inference or touch mapper/runtime/TSDF
paths.
Teacher prediction caches remain `metadata.json`, `frame_summaries.jsonl`, and
optional `arrays/frame_<frame_id:06d>.npz` payloads enabled by `--store-arrays`.
Metadata records format/version, adapter status/capabilities, coordinate frame,
frame IDs, scale sources, summaries path, and array storage state.

Teacher-signal caches use
`atlas3r_teacher_signal_manifest.json` plus `signals/clip_<id>.npz`. Manifest
fields include `format_name=atlas3r_teacher_signal_cache`, `format_version=1`,
source clip-cache manifest path, dataset/sequence, split, clip length, width,
height, teacher name/version/source type, `signal_count`, relative payload
paths, per-signal source clip IDs, frame IDs, timestamps, and truth boundary
flags: `diagnostic_only=true`, `accuracy_report=false`,
`performance_report=false`, `teacher_source`, `measured_geometry`, and
`pseudo_label`.

Teacher-signal payload required arrays are `depth_m`, `depth_sigma_m`,
`confidence`, `valid_mask`, `K`, `T_world_camera`, `frame_ids`, and
`timestamps_s` with shapes `T,H,W`, `T,3,3`, `T,4,4`, `T`, and `T`. Optional
arrays are `pointmap_camera_m`, `pointmap_world_m`, `normal_camera`,
`object_mask_ids`, `object_confidence`, and `dynamic_probability`. Validation
checks finite arrays, non-negative depth/sigma, positive sigma on valid pixels,
probabilities in `[0,1]`, valid intrinsics/transforms, safe relative paths, and
matching source clip metadata.
`source_metadata.pose_source` is optional but must be a non-empty string when
present; `source_metadata.pose_confidence` is optional but must be in `[0,1]`
when present. Measured TUM teacher caches mark measured geometry true and
pseudo-label false; pseudo/external caches do the inverse and must not relabel
measured TUM pose as an external teacher.
Raw `teachers ingest-local` NPZ inputs must be named `clip_<source_clip_id:06d>.npz` or `source_clip_<source_clip_id:06d>.npz`; parsed IDs select source clips, and duplicate/out-of-range IDs, bad filenames, or frame ID/timestamp mismatches are rejected.
`teachers inspect-signals` writes only `summary.json` and `per_clip_metrics.jsonl`. `teachers map-signals` deduplicates by `frame_id` before CPU TSDF integration; first occurrence wins in signal order then frame offset, duplicates must match depth/K/`T_world_camera`, and `map_summary.json` records before/after counts, duplicate count, and policy.
External teacher runners under `atlas3r.teachers.external` expose `ExternalTeacherStatus`, `ExternalTeacherRunConfig`, and `ExternalTeacherRunner`; must not import external model packages at module import time; `status()` reports availability/install/input/output/local-run capability; and `run(...)` writes a validated signal cache or raises an explicit error.
Depth Pro uses source RGB plus clip-cache `T_world_camera`, accepts `--device auto|cuda|mps|cpu`, records `teacher_source=depth_pro`, and sets measured false/pseudo-label true; it predicts each unique `frame_id` once, validates duplicate RGB/K before reuse, resizes depth/confidence/sigma to clip-cache resolution, and records clip/frame-slot/unique/reuse/resize counts in source metadata. VGGT local ingest maps `clip_<source_clip_id:06d>.npz` arrays with `teacher_source_type=local_external_geometry_teacher`.
VGGT real execution uses `atlas3r teachers run-vggt --clip-cache ... --output ... --device auto|cuda|mps|cpu [--max-clips N] [--vggt-repo PATH] [--checkpoint PATH_OR_URI] [--align-to-source-pose diagnostic_sim3|diagnostic_se3|none]`. The runner accepts an importable `vggt` package or `ATLAS3R_VGGT_REPO`, keeps VGGT code/weights external, extracts depth/confidence, decoded camera pose, optional intrinsics, and optional pointmaps when exposed, then writes standard pseudo-label teacher signals with `teacher_name=vggt`, `teacher_source_type=external_multiview_geometry_teacher`, measured false, and pseudo-label true. `diagnostic_*` alignment modes may align VGGT local poses/pointmaps to source clip poses for TUM evaluation; alignment metadata must state that aligned pseudo-labels are not measured geometry. Post-run VGGT evaluation writes `summary.json`, `per_clip_metrics.jsonl`, and `report.md` with depth, ATE-like center, RPE-like relative pose, optional pointmap, coverage, and confidence metrics.
Public commands: `atlas3r adapters list`; `atlas3r adapters run ...`; `atlas3r inspect teacher-cache ...`; `atlas3r teachers forge-measured-tum`; `atlas3r teachers ingest-local`; `atlas3r teachers run-depth-pro`; `atlas3r teachers run-vggt`; `atlas3r teachers ingest-vggt-local`; `atlas3r teachers run-student-temporal`; `atlas3r teachers inspect-signals`; and `atlas3r teachers map-signals`.
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

`atlas3r runtime stream-student-map --checkpoint ... --clip-cache ... --teacher-cache ... --output ...`
runs a temporal checkpoint over unique frames, emits `DepthObservation`s, fuses
CPU TSDF, and writes per-mode reports, TSDF sidecars, PLY, and preview HTML.
Pose modes are `oracle`, diagnostic-only `student-relative`,
`student-odometry`, or `both`; `student-odometry` anchors only the first frame
to source pose, then rolls out learned relative SE(3) from each window's
previous-frame slot. Per-mode outputs include `quality_report.json`,
`per_frame_quality.jsonl`, `pose_quality_report.json`,
`per_frame_pose_quality.jsonl`, TUM estimate/ground-truth trajectories,
`latency_report.json`, `tsdf/`, and `point_cloud.ply`; truth-claim flags stay false.
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
Commands: `atlas3r datasets tum-rgbd download|prepare`, `atlas3r train tum-rgbd-depth-pose --model tiny-v1|tiny-v2`, `atlas3r eval tum-rgbd-checkpoint [--write-tsdf]`, `atlas3r forge tum-rgbd-clips`, `atlas3r train tum-rgbd-temporal`, and `atlas3r train teacher-signals-temporal`. Download uses stdlib networking and safe tar extraction. Supported sequence specs include `freiburg1_xyz`, `freiburg1_desk`, `freiburg2_xyz`, and `freiburg3_long_office_household`; `download` accepts paired `--archive-url` and `--groundtruth-url` overrides for verified mirrors or URL changes.

The TUM manifest is `format_name=atlas3r_tum_rgbd_manifest`, `format_version=1`,
and records RGB/depth metadata, ROS default `K`, `depth_raw/5000.0`, split
metadata, frame IDs, timestamps, paths, `T_world_camera`, camera center, and
truth boundary. `every10` preserves the original split; `block` uses the
selected-frame tail as validation.

Single-frame real-RGBD datasets return `images_rgb 3,H,W`, scaled `intrinsics`,
depth/mask/confidence/camera-center/`T_world_camera` targets, and metadata.
`TinyDepthPoseNet` is RGB-only; `TinyMetricDepthNetV2` adds intrinsics rays.

Clip caches use `atlas3r_clip_cache_manifest.json` with format/version, split,
clip/image sizes, frame IDs, timestamps, payload paths, source metadata, and
diagnostic TUM sensor depth/pose truth flags. Payloads are `clips/clip_<id>.npz`
with RGB/depth/mask/K/`T_world_camera`/frame/timestamp arrays, `center_index`,
and optional pointmap/normal arrays.

`TumRgbdClipCacheDataset` returns `images_rgb T,3,H,W`, `intrinsics T,3,3`, `T_world_camera T,4,4`, center depth/mask/confidence targets, and `relative_T_center_camera T,4,4` where `T_center_camera_i = inverse(T_world_camera_center) @ T_world_camera_i`. `TinyTemporalMetricNetV0` predicts center depth/sigma/confidence and `relative_translation_center_from_camera B,T,3`; rotation is not learned in Phase 5A. Temporal runs write config, train/validation JSONL, summary, last/best checkpoints, NPZ sample, and HTML/SVG preview. All metrics are diagnostic.

`TeacherSignalTemporalDataset` lazily aligns one or more validated teacher-signal caches to source clip RGB, intrinsics, `T_world_camera`, frame IDs, and timestamps. Samples expose `images_rgb T,3,H,W`, `intrinsics T,3,3`, `T_world_camera T,4,4`, `frame_ids T`, `timestamps_s T`, targets `depth_m/depth_sigma_m/confidence/valid_mask T,1,H,W`, `teacher_is_measured`, teacher metadata, and `pointmap_camera_m T,3,H,W` plus `pointmap_camera_valid` (false with zero pointmaps when absent); mixed caches may span datasets/sequences but must share split, clip length, and image size. Training config and summaries record aggregate counts plus per-cache and per-sequence selected record counts; validation JSONL records aggregate metrics plus a per-sequence diagnostic breakdown.

`TemporalMetricNetV1` is the Phase 5D/5G trainable teacher-signal model: RGB plus ray channels, shared 2D encoder, small ConvGRU bottleneck, `depth_m/depth_sigma_m/confidence B,T,1,H,W`, `relative_translation_center_from_camera B,T,3`, and `relative_rotation_6d_center_from_camera B,T,6` using the first two columns of `T_center_camera_i = inverse(T_world_camera_center) @ T_world_camera_i`. Older Phase 5D/5F checkpoints without rotation-head weights still load for depth and `student-relative` diagnostics, but `student-odometry` requires trained rotation-head weights. `atlas3r train teacher-signals-temporal` writes config/metrics/validation/summary/last+best checkpoints and a compact preview; checkpoints use `format_name=atlas3r_teacher_signal_temporal_checkpoint` and include model/loss config, teacher cache lists, step, metrics, optimizer state, and truth flags with mapping/realtime/accuracy/performance/final-SMGT false.

Teacher-signal losses weight valid pixels by `confidence / clamp(depth_sigma_m^2, min_sigma^2, max_sigma^2)`, clamp and normalize weights per batch, and default to measured teacher weight `1.0` and pseudo teacher weight `0.25`. Pose losses include relative translation SmoothL1 in meters, 6D-to-SO(3) geodesic rotation loss in radians, and an optional `se3_pose_weight`. Metrics include depth RMSE/MAE/AbsRel, relative translation mean/median/p95, relative rotation mean/median/p95 in degrees, ATE-like camera-center rollout error, and RPE-like consecutive transform error. `atlas3r teachers run-student-temporal` loads a temporal checkpoint and writes `teacher_name=atlas3r_temporal_v1_student` pseudo-label caches; exported `T_world_camera` comes from the source clip cache unless a later phase upgrades pose export.

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
### `.atlas3r` Session Folder
Folder layout: `metadata.json`, `poses.jsonl`, `cameras.jsonl`, `objects.jsonl`,
`mesh_chunks/chunk_<id>_v<version>.json`, optional GLB files, optional
`depth/frame_<id>.npz`, and `logs/runtime_profile.json`.

The Phase 0C reader reconstructs `PoseEstimate`, `CameraModel`, `ObjectInstance`,
and `MeshChunk` from sidecars and records sorted depth NPZ paths without loading
every depth array by default.

`atlas3r inspect session --input <session.atlas3r> --output <preview_dir>` writes
deterministic HTML/SVG preview files.

Every export/sidecar must preserve coordinate convention, unit scale, scale
source, camera metadata source when known, model checkpoint hash or `null`, voxel
size when relevant, accuracy report path or `null`, and RGB-only warnings.
