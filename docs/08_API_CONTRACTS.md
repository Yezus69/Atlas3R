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

## Mapping observations

Phase 2C makes mapper inputs explicit through `DepthObservation` under
`atlas3r.mapping.observations`. CPU TSDF fusion and teacher-cache TSDF replay
must convert their source frames to this public contract before integration;
mapping code should not rely on synthetic fixture frame classes or private
helper type suppressions.

```python
@dataclass(frozen=True)
class DepthObservation:
    frame_id: int
    camera: CameraModel
    pose: PoseEstimate
    depth_m: NDArray[np.float32]          # H,W finite, non-negative meters
    depth_sigma_m: NDArray[np.float32]    # H,W finite, non-negative meters
    confidence: NDArray[np.float32]       # H,W finite values in [0, 1]
    static_mask: NDArray[np.bool_] | NDArray[np.float32] | None = None
    object_id: NDArray[np.int32] | None = None
    rgb_u8: NDArray[np.uint8] | None = None
    source: str = "unknown"
```

Validation requirements:

- `frame_id` is non-negative;
- `camera` is a `CameraModel` and `pose` is a `PoseEstimate`;
- `depth_m`, `depth_sigma_m`, and `confidence` match
  `camera.height x camera.width`;
- depth and sigma arrays are floating point, finite, and non-negative;
- confidence arrays are floating point, finite, and in `[0, 1]`;
- `static_mask`, when present, is HxW bool or numeric values in `[0, 1]`;
- `object_id`, when present, is an HxW integer array;
- `rgb_u8`, when present, is HxWx3 `uint8`;
- `source` is non-empty.

`atlas3r.data.synthetic_observations.depth_observation_from_synthetic_frame(frame)`
converts Phase 0B synthetic cube-room frames and is exported from
`atlas3r.data`. `atlas3r.mapping.observations` remains the generic mapper
observation contract and must not import synthetic fixture classes.
Teacher-cache replay performs its replay-frame conversion in
`atlas3r.mapping.teacher_cache_replay`.

Shared TSDF grid geometry helpers live in `atlas3r.mapping.tsdf_grid`:
`compute_tsdf_grid_shape(...)` computes deterministic XYZ voxel grid shape from
world bounds and voxel size, and `voxel_centers_world(...)` returns Nx3
world-frame voxel centers. TSDF replay paths should use these public helpers
instead of importing private implementation details from `cpu_tsdf.py`.

## Teacher adapter contracts

Phase 0E introduces dependency-safe teacher adapter contracts under
`atlas3r.models.adapters`. Third-party model code and weights remain external to
Atlas3R; adapter modules must import without optional teacher packages installed.

```python
class GeometryTeacherAdapter(Protocol):
    def predict(self, frames: FrameBatch) -> TeacherPrediction: ...
```

```python
@dataclass(frozen=True)
class FrameBatch:
    frames: tuple[FramePacket, ...]         # non-empty, unique frame_id values
    batch_id: str
    metadata: Mapping[str, Any]
```

```python
@dataclass(frozen=True)
class TeacherPrediction:
    adapter_name: str
    frame_predictions: tuple[FramePrediction, ...]
    capabilities: AdapterCapabilities
    metadata: Mapping[str, Any]
```

`TeacherPrediction.frame_predictions` reuses `FramePrediction`, so every teacher
output keeps the same `CameraModel`, `PoseEstimate`, dense geometry,
confidence, and uncertainty conventions as Atlas3R runtime outputs.

Adapter discovery reports:

```python
@dataclass(frozen=True)
class AdapterCapabilities:
    predicts_camera: bool
    predicts_pose: bool
    predicts_depth: bool
    predicts_normals: bool
    predicts_points: bool
    predicts_dense_matches: bool
    predicts_objects: bool
    supports_batch: bool
    supports_streaming: bool
    notes: tuple[str, ...]

@dataclass(frozen=True)
class AdapterStatus:
    name: str
    display_name: str
    availability: str    # available|unavailable|stub-only
    capabilities: AdapterCapabilities
    install_hint: str | None
    reason: str | None
```

Known Phase 0E stubs are `VGGTAdapter` and `DepthProAdapter`. Missing optional
dependencies must raise `AdapterDependencyError` from adapter construction or
prediction with the adapter name and installation hint in the message.

Phase 1B adds `fixture-cube-room`, an `available` dependency-free fixture
adapter for exercising runner plumbing and cache writing only. It accepts Phase
0B synthetic cube-room `.atlas3r` sessions, reconstructs `TeacherPrediction`
records from analytic depth/session sidecars, and marks prediction metadata with
`coordinate_frame: synthetic_world`, `fixture: true`, and a synthetic-only truth
boundary. It is not an external teacher model, does not run neural inference, and
must not be treated as measured geometry for real captures.

## TeacherPrediction cache

Phase 1A introduces a dependency-light cache for serialized teacher prediction
metadata and per-frame contract summaries. Phase 1C adds an explicit opt-in full
array payload path. The default cache still stores summaries, not full model
tensors:

```text
teacher_cache/
  metadata.json
  frame_summaries.jsonl
  arrays/ optional tensor payloads
```

`metadata.json` is deterministic JSON with:

- `format_name`: `atlas3r_teacher_prediction_cache`;
- `format_version`: `1`;
- `adapter`: adapter name, display name, availability, install hint, reason,
  and full `AdapterCapabilities`;
- `prediction_metadata`: JSON-serializable `TeacherPrediction.metadata`;
- `coordinate_frame` and coordinate convention;
- `frame_count`, sorted `frame_ids`, and observed `scale_sources`;
- `frame_summaries_path`;
- `arrays`: whether arrays are stored, the array directory when enabled, and
  the documented `.npz` keys.

`frame_summaries.jsonl` contains one deterministic JSON object per frame, sorted
by `frame_id`. Each summary must preserve:

- `frame_id` and `timestamp_ns`;
- `coordinate_frame` and `scale_source`;
- camera confidence/source and pose confidence/tracking state/scale source;
- replay-needed camera fields: `K`, `distortion_model`, `distortion_params`,
  and `rolling_shutter_row_time_s`;
- replay-needed pose fields: `T_world_camera`, `q_world_camera_xyzw`,
  `camera_center_world_m`, `covariance_6x6`, and `diagnostics`;
- pose covariance presence plus a small covariance-derived uncertainty summary;
- dense confidence summary from `FramePrediction.confidence`;
- depth uncertainty summary from `FramePrediction.depth_sigma_m`;
- depth value summary and tensor shape/dtype summaries;
- dense-match count and confidence summary when present;
- `arrays_path`, `null` for summaries-only caches or a relative payload path
  when full arrays are explicitly stored.

When full arrays are explicitly requested, the cache writer stores NumPy `.npz`
payloads under `arrays/frame_<frame_id:06d>.npz`. Required payload keys are:

```text
depth_m
depth_sigma_m
normal_camera
point_world
confidence
static_mask
```

Optional payload keys are stored only when the corresponding `FramePrediction`
field is not `None`:

```text
object_embeddings
object_mask_logits
```

Payload validation must check shapes against frame tensor summaries, dtype
matches, finite numeric values, confidence/probability ranges in `[0, 1]`, and
non-negative depth/uncertainty arrays. Missing or corrupt `.npz` payloads must
raise explicit path-named errors.

Cache readers must validate metadata, adapter capabilities/status, frame
summaries, payload references when declared, confidence ranges, non-negative
uncertainty summaries, frame counts, frame IDs, and coordinate-frame consistency
with explicit path-named errors. The cache is not an accuracy report and must
not present predicted or completed geometry as measured geometry.

`atlas3r adapters run --adapter fixture-cube-room --input <session.atlas3r>
--output <cache_dir>` writes this cache for valid synthetic cube-room sessions.
The default runner behavior is summaries-only. `--store-arrays` explicitly opts
in to full tensor payloads under `arrays/`. The runner must reject non-synthetic
or malformed sessions with explicit errors. External adapter stubs such as
`vggt` and `depth-pro` must continue to fail gracefully with adapter name,
availability status, reason, and guidance until future phases implement real
adapter prediction paths.

`atlas3r inspect teacher-cache --input <cache_dir>` validates a teacher cache and
prints deterministic JSON with adapter name/status, frame IDs, coordinate frame,
scale sources, array storage state, confidence summaries, and uncertainty
summaries. It must state that the cache inspection is not an accuracy report.

### Phase 1D teacher-cache TSDF replay output

`atlas3r smoke teacher-cache-tsdf --input <cache_dir> --output <folder>` replays
a validated teacher cache with full array payloads into the dependency-free CPU
TSDF reference path. Replay requires `metadata.json` to declare
`arrays.stored=true`; summaries-only caches are rejected with a path-named error.
Each frame payload is loaded through the Phase 1C `.npz` validation path, and
replay reconstructs only the fields needed by CPU TSDF integration:

- `CameraModel` width, height, intrinsics, distortion fields, confidence, and source;
- `PoseEstimate` `T_world_camera`, quaternion, camera center, covariance,
  confidence, tracking state, scale source, and diagnostics;
- payload `depth_m`, `depth_sigma_m`, `confidence`, and `point_world` arrays.

Replay writes deterministic artifacts:

```text
<folder>/
  tsdf_grid.npz        tsdf, weight, grid_min_corner_world_m, voxel_size_m
  surface_points.npz   points_world_m, confidence, uncertainty_m, voxel_indices_xyz
  metadata.json        replay/source metadata and confidence/uncertainty summaries
  metrics.json         synthetic fixture metrics or an explicit not-evaluated record
```

`metadata.json` preserves source frame IDs, coordinate frame, metric scale
source(s), voxel size, observed coverage estimate, adapter name, cache array
state, and input confidence/uncertainty summaries. The output is a deterministic
smoke artifact and not an accuracy report.

`metrics.json` contains synthetic cube-room fixture metrics only when cache
metadata proves the source is the `fixture-cube-room` synthetic fixture with
`fixture=true`, `fixture_session_type=synthetic_cube_room`, and
`coordinate_frame=synthetic_world`. All other caches write
`metric_family: not_evaluated` and must not report geometric accuracy.

### Phase 1E TSDF MeshChunk sidecar

CPU TSDF smoke commands can optionally write a dependency-free MeshChunk JSON
sidecar from observed TSDF surface samples:

```bash
atlas3r smoke tsdf-cube-room --output <folder> --write-mesh-sidecar
atlas3r smoke teacher-cache-tsdf --input <cache_dir> --output <folder> --write-mesh-sidecar
```

The flag preserves all Phase 0D/1D artifacts and adds:

```text
<folder>/
  mesh_chunk_sidecar.json
```

`mesh_chunk_sidecar.json` is deterministic JSON:

```text
{
  "format_name": "atlas3r_tsdf_surface_mesh_chunk_sidecar",
  "format_version": 1,
  "mesh_chunk": { ... MeshChunk fields ... },
  "metadata": { ... sidecar/source TSDF metadata ... },
  "sample_attributes": { ... emitted confidence/uncertainty arrays ... }
}
```

The `mesh_chunk` object validates against the existing `MeshChunk` contract.
It uses `T_world_chunk=identity` and world-frame vertices because the TSDF
surface samples are already in the output coordinate frame. Faces are
low-fidelity marker triangles around observed voxel-center surface samples; they
are reference geometry for early pipeline testing, not marching-cubes output and
not a GLB/PLY/game-engine final asset. `surface_source_per_face` is always
`0 observed_surface`; `object_id_per_face` is `-1` until object-aware fusion
exists.

The sidecar metadata preserves:

- source surface artifact type and full source surface metadata;
- coordinate frame, unit scale, metric scale source, and source frame IDs;
- voxel size, observed coverage estimate, and surface coverage estimate;
- source and emitted sample counts plus the deterministic max-sample cap;
- confidence summary and mean/p95 uncertainty;
- `accuracy_report_path: null` and an explicit not-an-accuracy-report note.

MeshChunk flags must include `low_fidelity_reference_only`,
`observed_surface_samples`, `not_completed_surface`, and
`not_accuracy_report`. The sidecar must not claim hidden/completed geometry as
measured geometry. Missing or malformed `surface_points.npz` or `metadata.json`
inputs must raise path-named errors.

### Phase 2A CPU TSDF WorldMap sidecar

CPU TSDF smoke commands can optionally assemble a dependency-free WorldMap JSON
sidecar from the observed Phase 1E MeshChunk sidecar:

```bash
atlas3r smoke tsdf-cube-room --output <folder> --write-world-map-sidecar
atlas3r smoke teacher-cache-tsdf --input <cache_dir> --output <folder> --write-world-map-sidecar
```

The flag preserves all Phase 0D/1D/1E artifacts. If the MeshChunk sidecar has
not also been requested, the command writes and validates
`mesh_chunk_sidecar.json` first, then adds:

```text
<folder>/
  world_map_sidecar.json
```

`world_map_sidecar.json` is deterministic JSON:

```text
{
  "format_name": "atlas3r_tsdf_world_map_sidecar",
  "format_version": 1,
  "world_map": { ... WorldMap fields ... },
  "metadata": { ... sidecar/source MeshChunk metadata ... }
}
```

The `world_map` object validates against the existing `WorldMap` contract and
contains exactly one validated observed `MeshChunk`. `objects` and `keyframes`
are empty in Phase 2A, `created_at_ns` is deterministically `0`, and no object
meshes or completed hidden surfaces are invented.

The sidecar metadata preserves:

- source MeshChunk sidecar format and full source MeshChunk metadata;
- coordinate frame/world frame name, unit scale, metric scale source, and source
  frame IDs;
- voxel size, observed coverage estimate, and surface coverage estimate when
  present;
- mesh chunk IDs/count, object count `0`, and keyframe count `0`;
- confidence summary, global confidence, mean uncertainty, and p95 uncertainty;
- `accuracy_report_path: null` and an explicit not-an-accuracy-report note.

WorldMap sidecar flags include `low_fidelity_reference_only`,
`observed_surface_samples`, `not_completed_surface`, and
`not_accuracy_report`. Missing or malformed MeshChunk sidecars must raise
path-named errors.

```bash
atlas3r inspect world-map --input <folder>/world_map_sidecar.json
```

The inspect command validates the WorldMap sidecar and prints deterministic JSON
summarizing the map ID, frame name, mesh chunk IDs, source frame IDs, coordinate
frame, scale source, confidence, uncertainty, truth-boundary flags, and the fact
that the sidecar is not an accuracy report.

### Phase 2B CPU TSDF output folder inspection

Complete CPU TSDF smoke output folders can be inspected without GLB/PLY/trimesh,
marching-cubes, model, or cloud dependencies:

```bash
atlas3r inspect tsdf-output --input <folder> [--mode surface|mesh|world-map|complete]
```

The command validates `metadata.json` and `surface_points.npz` in every mode,
validates `metrics.json` when present, validates `mesh_chunk_sidecar.json` when
present or required, and validates `world_map_sidecar.json` when present or
required. The default `complete` mode requires both sidecars. `surface` permits
surface-only Phase 0D/1D outputs, `mesh` requires the Phase 1E MeshChunk
sidecar, and `world-map`/`complete` require both the MeshChunk and WorldMap
sidecars so the folder can be checked as one mapper pipeline output.

Inspection prints deterministic JSON:

```text
{
  "format_name": "atlas3r_cpu_tsdf_output_folder_inspection",
  "format_version": 1,
  "inspect_mode": "complete",
  "artifacts": { ... required and optional artifact presence ... },
  "surface": { ... surface artifact summary ... },
  "mesh_chunk": { ... MeshChunk sidecar summary or null ... },
  "world_map": { ... WorldMap sidecar summary or null ... },
  "cross_checks": { ... shared metadata checks, "passed": true ... },
  "truth_boundary": { ... explicit not-accuracy-report flags ... }
}
```

The inspector cross-checks coordinate frame, source frame IDs, voxel size,
metric scale source, observed coverage estimate, confidence summary, and
mean/p95 uncertainty across the surface arrays/metadata, MeshChunk sidecar, and
WorldMap sidecar. Missing required artifacts and metadata mismatches must raise
path-named errors. The inspection JSON is a mapper pipeline diagnostic only; it
is low-fidelity/reference-only, observed-only, not completed geometry, and not
an accuracy report.

### Phase 2D runtime fixture scheduler smoke

`atlas3r smoke runtime-fixture --output <folder>` runs a single-threaded,
deterministic scheduler skeleton over the synthetic cube-room fixture. It writes
a Phase 0B session, runs `fixture-cube-room` through the existing teacher-cache
path with `store_arrays=True`, and replays that full-array cache into CPU TSDF
mapping through the public `DepthObservation` mapper input path. It does not
start threads, asyncio workers, GPU work, neural inference, video decoding, or
mesh export.

The output layout is:

```text
<folder>/
  runtime_events.jsonl
  runtime_summary.json
  synthetic_cube_room.atlas3r/
  teacher_cache/
    metadata.json
    frame_summaries.jsonl
    arrays/frame_000000.npz
    arrays/frame_000001.npz
    arrays/frame_000002.npz
  teacher_cache_tsdf/
    tsdf_grid.npz
    surface_points.npz
    metadata.json
    metrics.json
    mesh_chunk_sidecar.json
    world_map_sidecar.json
```

`runtime_events.jsonl` contains deterministic JSON Lines records. Paths inside
event records are relative to `<folder>` so two runs in different output folders
produce byte-identical event logs. Each event has:

```text
{
  "format_name": "atlas3r_runtime_fixture_event_log",
  "format_version": 1,
  "event_index": 0,
  "stage_name": "runtime_start",
  "frame_id": null,
  "timestamp_ns": 0,
  "latency_ns": 0,
  "dropped_frame": false,
  "memory_counters": {
    "configured_frame_array_bound": 1,
    "frame_arrays_in_memory": 0,
    "peak_frame_arrays_in_memory": 0,
    "processed_frame_count": 0,
    "dropped_frame_count": 0,
    "queued_frame_count": 0
  },
  "paths": { ... relative source/cache/output paths ... },
  "metadata": { ... deterministic stage metadata ... }
}
```

Known Phase 2D stage names are `runtime_start`, `session_write`,
`source_frame`, `adapter_cache_write`, `adapter_cache_frame`, `tsdf_replay`,
`tsdf_replay_frame`, `tsdf_output_write`, and `runtime_complete`.
`timestamp_ns` and `latency_ns` are deterministic placeholders, not wall-clock
measurements. `memory_counters` are scheduler-owned bounded-memory counters for
this fixture skeleton; they are used to verify the runtime plumbing does not
retain all fixture frame array payloads in scheduler state. They are not a
process memory profile.

`runtime_summary.json` contains:

```text
{
  "format_name": "atlas3r_runtime_fixture_smoke_summary",
  "format_version": 1,
  "runtime": { ... adapter/cache/DepthObservation path summary ... },
  "frame_ids": [0, 1, 2],
  "event_log": { ... event-log format and path ... },
  "bounded_memory": {
    "configured_frame_array_bound": 1,
    "peak_frame_arrays_in_memory": 1,
    "all_frame_arrays_accumulated": false,
    "bounded_memory_check_passed": true,
    ...
  },
  "artifacts": { ... relative output paths ... },
  "truth_boundary": {
    "accuracy_report": false,
    "note": "Runtime fixture smoke uses synthetic analytic cube-room data and is not an accuracy report."
  }
}
```

The runtime fixture output is a deterministic plumbing smoke artifact only. It
must not be presented as real-time performance, a geometric accuracy report, or
measured real-capture geometry.

### Phase 2E runtime fixture output inspection

Runtime fixture smoke output folders can be inspected without model, video,
GPU/CUDA, GLB/PLY, marching-cubes, web, or notebook dependencies:

```bash
atlas3r inspect runtime-fixture --input <folder>
```

The command validates the Phase 2D output folder as a diagnostic artifact:

- `runtime_events.jsonl` format name/version, event index ordering, known stage
  sequence, frame IDs, deterministic timestamp and latency placeholders,
  dropped-frame flags, bounded-memory counters, and forward-slash relative paths;
- `runtime_summary.json` format name/version and cross-checks against the event
  log, bounded-memory counters, frame IDs, truth boundary, and artifact list;
- required generated `synthetic_cube_room.atlas3r` session files;
- required `teacher_cache` metadata, frame summaries, `arrays.stored=true`, and
  full `.npz` array payloads;
- required `teacher_cache_tsdf` artifacts by running the existing
  `atlas3r inspect tsdf-output --mode complete` validation path.

Inspection prints deterministic JSON:

```text
{
  "format_name": "atlas3r_runtime_fixture_output_inspection",
  "format_version": 1,
  "artifacts": { ... required artifact presence and relative paths ... },
  "event_log": { ... stage counts, frame IDs, timing placeholders, memory counters ... },
  "summary": { ... runtime summary cross-check fields ... },
  "session": { ... generated synthetic session summary ... },
  "teacher_cache": { ... fixture adapter, frame IDs, full array payload paths ... },
  "teacher_cache_tsdf": { ... complete TSDF output inspection summary ... },
  "cross_checks": { ... "passed": true ... },
  "diagnostic_boundary": {
    "diagnostic_only": true,
    "performance_report": false,
    "accuracy_report": false,
    "note": "Runtime fixture inspection is a diagnostic only; it is not a performance report and not an accuracy report."
  }
}
```

Missing or malformed paths must produce path-named errors without tracebacks
through the CLI. This inspection JSON is a runtime plumbing diagnostic only. It
is not a performance report, not an accuracy report, and does not make claims
about real-time throughput, metric reconstruction quality, or measured
real-capture geometry.

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

### Phase 0D CPU TSDF smoke output

`atlas3r smoke tsdf-cube-room --output <folder>` writes deterministic
pure-NumPy reference artifacts:

```text
<folder>/
  synthetic_cube_room.atlas3r/  Phase 0B analytic input session
  tsdf_grid.npz                 tsdf, weight, grid_min_corner_world_m, voxel_size_m
  surface_points.npz            points_world_m, confidence, uncertainty_m, voxel_indices_xyz
  metadata.json                 surface metadata and uncertainty summary
  metrics.json                  conservative synthetic fixture metrics
```

The Phase 0D surface is a voxel-center point cloud extracted from observed
near-zero TSDF voxels, not a game-engine mesh. `metadata.json` must include
source frame IDs, voxel size, coordinate frame, metric scale source, observed
coverage estimate, and mean/p50/p95/max uncertainty. `metrics.json` compares
the points to the synthetic cube-room ground-truth box mesh with voxel-scale
fixture checks and must include known limitations. It is not an accuracy report
and must not claim millimeter-level accuracy.

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
