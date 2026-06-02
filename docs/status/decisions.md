# Architecture decisions

Record decisions that change interfaces, coordinate systems, tensor shapes, training stages, dependencies, or accuracy claims.

```text
Decision ID:
Date:
Context:
Decision:
Alternatives considered:
Consequences:
Docs/tests updated:
```

```text
Decision ID: D-0001
Date: 2026-06-01
Context: docs/08_API_CONTRACTS.md referenced DenseMatchSet in FramePrediction but did not define its fields. Phase 0A needs a validated placeholder without introducing model-specific matcher details.
Decision: Define DenseMatchSet as source/target frame IDs, Nx2 source and target pixel arrays, and an N-length confidence array in [0, 1].
Alternatives considered: Leave DenseMatchSet unimplemented; add richer descriptors or track IDs now.
Consequences: FramePrediction can validate dense_matches today while future teacher/student work can extend the schema deliberately.
Docs/tests updated: docs/08_API_CONTRACTS.md and tests/unit/test_contracts.py.
```

```text
Decision ID: D-0002
Date: 2026-06-01
Context: Phase 0B needs the smoke command to write mesh chunk metadata before a
GLB exporter exists.
Decision: Write `mesh_chunks/chunk_<id>_v<version>.json` as a Phase 0B
metadata/full synthetic mesh sidecar inside `.atlas3r` sessions while preserving
the future GLB path in the documented folder layout.
Alternatives considered: Add a GLB dependency now; omit mesh vertices from the
session; write an undocumented test-only file.
Consequences: The synthetic smoke session is dependency-light and contains a
contract-validatable ground-truth mesh payload. Later exporter work can add GLB
without breaking the JSON sidecar.
Docs/tests updated: docs/08_API_CONTRACTS.md and tests/synthetic/test_synthetic_cube_room.py.
```

```text
Decision ID: D-0003
Date: 2026-06-01
Context: Phase 0C needs humans and Codex to inspect synthetic geometry before
neural models, TSDF fusion, GLB export, or visualization dependencies exist.
Decision: Add a narrow Phase 0B `.atlas3r` sidecar reader and deterministic
HTML/SVG preview output under `atlas3r inspect session`, using only stdlib and
NumPy and leaving generated previews outside version control.
Alternatives considered: Add matplotlib/Pillow/trimesh; emit PNG or GLB
previews; generalize the session format now.
Consequences: Early geometry can be inspected in any browser while runtime
dependencies stay unchanged. Future export/viewer work can add richer formats
without changing the Phase 0B reader contract.
Docs/tests updated: docs/08_API_CONTRACTS.md and tests/synthetic/test_session_inspect.py.
```

```text
Decision ID: D-0004
Date: 2026-06-01
Context: Phase 0D needs a deterministic CPU TSDF correctness reference and
smoke artifacts before mapper/exporter dependencies or game-engine mesh export
exist.
Decision: Add `atlas3r smoke tsdf-cube-room --output <folder>` with a narrow
artifact layout: the Phase 0B input session, `tsdf_grid.npz`,
`surface_points.npz`, `metadata.json`, and `metrics.json`.
Alternatives considered: Emit a GLB/PLY mesh now; hide artifacts behind tests
only; add a generic map export format before a mapper API exists.
Consequences: Synthetic fusion can be verified with dependency-light,
voxel-scale outputs while preserving explicit confidence/uncertainty metadata
and avoiding premature accuracy claims.
Docs/tests updated: docs/08_API_CONTRACTS.md and tests/synthetic/test_cpu_tsdf.py.
```

```text
Decision ID: D-0005
Date: 2026-06-01
Context: Phase 0E needs Phase 1 teacher integrations to share one dependency-safe
contract without importing optional third-party model packages or creating
parallel geometry schemas.
Decision: Define `FrameBatch`, `TeacherPrediction`, `GeometryTeacherAdapter`,
adapter capability metadata, and adapter availability status under
`atlas3r.models.adapters`, with `TeacherPrediction` containing existing
`FramePrediction` records and runtime-only dependency errors for stubs.
Alternatives considered: Put teacher outputs in `atlas3r.api`; let each adapter
define its own prediction schema; import third-party packages at module import
time.
Consequences: Phase 1 adapters can be discovered and tested before dependencies
or weights are installed while preserving the existing pose, camera,
confidence, and uncertainty conventions.
Docs/tests updated: docs/08_API_CONTRACTS.md and tests/unit/test_adapters.py.
```

```text
Decision ID: D-0006
Date: 2026-06-01
Context: Phase 1A needs a persistent teacher prediction artifact before any
external teacher model inference is wired, and the format must keep adapter
status, capabilities, confidence, uncertainty, coordinate frame, and scale
source metadata without introducing heavyweight dependencies.
Decision: Add a deterministic summaries-first `TeacherPrediction` cache under
`teacher_cache/metadata.json` and `teacher_cache/frame_summaries.jsonl`.
Full tensor payloads are optional future `.npz` files with documented keys.
Alternatives considered: Serialize every tensor immediately; reuse `.atlas3r`
sessions for teacher outputs; leave cache format private to tests.
Consequences: Phase 1 runner and future adapters have a stable, dependency-light
artifact to validate while avoiding large tensor files and premature inference
claims.
Docs/tests updated: docs/08_API_CONTRACTS.md and tests/unit/test_teacher_cache.py.
```

```text
Decision ID: D-0007
Date: 2026-06-01
Context: Phase 1B needs the adapter runner to produce a real TeacherPrediction
cache without downloading model weights, vendoring third-party repositories,
calling cloud APIs, running neural inference, or claiming real capture
measurements.
Decision: Add `fixture-cube-room` as an available dependency-free fixture
adapter that only accepts Phase 0B synthetic cube-room `.atlas3r` sessions and
reconstructs teacher predictions from analytic depth/session sidecars.
Alternatives considered: Make VGGT or Depth Pro produce fake outputs; add a
generic session-to-cache converter outside the adapter registry; serialize
synthetic fixture predictions as prebuilt files.
Consequences: Runner/cache plumbing can be tested end to end while real external
adapters remain honest stubs until model-specific integration exists. The
fixture name and metadata make the synthetic-only truth boundary explicit.
Docs/tests updated: docs/08_API_CONTRACTS.md, tests/unit/test_adapters.py, and
tests/unit/test_teacher_cache.py.
```

```text
Decision ID: D-0008
Date: 2026-06-01
Context: Phase 1C needs teacher caches to optionally preserve full tensor arrays
for synthetic fixture data while keeping summaries-only caches as the default
and avoiding heavyweight serialization dependencies.
Decision: Keep `metadata.json` and `frame_summaries.jsonl` as the deterministic
cache source of truth, and add an explicit `--store-arrays` / `store_arrays=True`
path that writes per-frame NumPy payloads under `arrays/frame_<frame_id:06d>.npz`.
Cache inspection emits deterministic JSON and explicitly states that cache
inspection is not an accuracy report.
Alternatives considered: Always write arrays; put every tensor into JSON; add a
binary archive or visualization/export dependency now.
Consequences: Existing summaries-only caches remain backward compatible, while
fixture payloads can round-trip and validate shapes, dtypes, confidence ranges,
and uncertainty values with path-named errors for missing or corrupt payloads.
Docs/tests updated: docs/08_API_CONTRACTS.md and tests/unit/test_teacher_cache.py.
```

```text
Decision ID: D-0009
Date: 2026-06-01
Context: Phase 1D needs full-array teacher caches to replay into the CPU TSDF
reference path, but Phase 1C summaries did not include camera intrinsics or the
full T_world_camera pose needed for projection-based TSDF integration.
Decision: Keep the Phase 1C `.npz` tensor payload keys stable and add
replay-needed camera/pose fields to newly written `frame_summaries.jsonl`
records. Add `atlas3r smoke teacher-cache-tsdf` to require `arrays.stored=true`,
load payloads through the cache validation path, and compute synthetic fixture
metrics only when cache metadata proves the synthetic cube-room source.
Alternatives considered: Put camera/pose matrices into every `.npz` payload;
derive TSDF directly from cached point_world without depth/pose projection;
force summaries-only caches to become replayable.
Consequences: Existing summaries-only caches remain inspectable but are
explicitly rejected for TSDF replay. New payload caches carry enough deterministic
metadata for dependency-free replay without adding model or visualization
dependencies.
Docs/tests updated: docs/08_API_CONTRACTS.md,
tests/unit/test_teacher_cache.py, and
tests/synthetic/test_teacher_cache_replay.py.
```

```text
Decision ID: D-0010
Date: 2026-06-01
Context: Phase 1E needs CPU TSDF replay outputs to exercise the public
MeshChunk contract before marching cubes, GLB/PLY export, object-aware fusion,
or heavyweight mesh dependencies exist.
Decision: Add an opt-in `mesh_chunk_sidecar.json` wrapper with a
contract-valid `MeshChunk`, source TSDF surface metadata, and emitted
confidence/uncertainty arrays. The mesh uses deterministic low-fidelity marker
triangles around observed voxel-center samples and flags the artifact as
reference-only, observed-only, not completed, and not an accuracy report.
Alternatives considered: Add GLB/PLY/trimesh/marching-cubes dependencies now;
connect neighboring surface points into larger inferred surfaces; add new fields
to the MeshChunk dataclass for coordinate frame and coverage.
Consequences: Smoke runs can validate MeshChunk plumbing without presenting
completed or hidden geometry as measured. Coordinate frame, coverage, and
per-sample confidence/uncertainty stay in the sidecar wrapper metadata until a
future MeshChunk schema revision deliberately promotes them into the contract.
Docs/tests updated: docs/08_API_CONTRACTS.md and
tests/synthetic/test_tsdf_mesh_sidecar.py.
```

```text
Decision ID: D-0011
Date: 2026-06-01
Context: Phase 2A needs CPU TSDF replay outputs to exercise the public WorldMap
contract before object-aware mapping, keyframe map storage, GLB/PLY export, or
heavyweight mesh dependencies exist.
Decision: Add an opt-in `world_map_sidecar.json` wrapper that validates the
Phase 1E `mesh_chunk_sidecar.json`, wraps the observed MeshChunk in a
contract-valid `WorldMap`, leaves `objects` and `keyframes` empty, uses
deterministic `created_at_ns=0`, and preserves coordinate frame, source frames,
coverage, confidence, scale source, and mean/p95 uncertainty metadata.
Alternatives considered: Add object/keyframe records now; write a session-level
map export; make WorldMap sidecars infer completed object meshes from the
surface samples.
Consequences: Early mapper pipeline tests can validate WorldMap plumbing without
inventing hidden geometry or adding export dependencies. The sidecar remains
reference-only, observed-only, not completed, and not an accuracy report.
Docs/tests updated: docs/08_API_CONTRACTS.md and
tests/synthetic/test_world_map_sidecar.py.
```

```text
Decision ID: D-0012
Date: 2026-06-01
Context: Phase 2B needs one dependency-free inspection path for complete CPU
TSDF output folders instead of inspecting surface artifacts, MeshChunk sidecars,
and WorldMap sidecars separately.
Decision: Add `atlas3r inspect tsdf-output --input <folder>` with deterministic
JSON output and explicit `surface`, `mesh`, `world-map`, and `complete` modes.
The default `complete` mode requires both sidecars and cross-checks shared
coordinate frame, source frame IDs, voxel size, scale source, observed coverage,
confidence, and uncertainty metadata across the folder artifacts.
Alternatives considered: Only extend `inspect world-map`; always require all
sidecars with no partial mode; add a generated manifest file to every smoke run.
Consequences: Early mapper pipeline checks can validate a complete output folder
without changing Phase 0D/1D/1E/2A artifacts or adding export dependencies.
Partial surface-only outputs remain inspectable when explicitly requested.
Docs/tests updated: docs/08_API_CONTRACTS.md and
tests/synthetic/test_tsdf_output_inspection.py.
```

```text
Decision ID: D-0013
Date: 2026-06-01
Context: Phase 2C revised needs the mapper to consume a public observation
contract instead of accepting synthetic fixture frames by accident or requiring
teacher-cache replay to import private TSDF helpers with a type suppression.
Decision: Add `atlas3r.mapping.observations.DepthObservation` as the validated
depth/confidence/uncertainty mapper input contract. Synthetic cube-room fusion
and teacher-cache TSDF replay now convert their source frames to
DepthObservation before CPU TSDF integration.
Alternatives considered: Keep `_integrate_frame` typed to the synthetic fixture
and suppress replay type errors; create separate mapper paths for fixture and
teacher-cache replay; move the observation contract into the broad public API
package before runtime requirements are known.
Consequences: Mapper fusion has a stable public boundary that can later be fed
by video adapters, neural teachers, or streaming runtime code without changing
TSDF integration internals. Optional object/rgb/static-mask fields are validated
now but object-aware fusion remains future work.
Docs/tests updated: docs/08_API_CONTRACTS.md,
tests/unit/test_mapping_observations.py, and
tests/synthetic/test_teacher_cache_replay.py.
```

```text
Decision ID: D-0014
Date: 2026-06-01
Context: Phase 2C.1 found two mapper boundary leaks before Phase 2D: the
generic observation module imported the synthetic cube-room fixture, and
teacher-cache replay imported private TSDF grid helpers from cpu_tsdf.py.
Decision: Keep DepthObservation in atlas3r.mapping.observations, move the
synthetic frame converter to atlas3r.data.synthetic_observations and export it
from atlas3r.data, and add public TSDF grid helpers in
atlas3r.mapping.tsdf_grid for grid shape and voxel-center computation.
Alternatives considered: Keep deprecated mapper re-exports for the synthetic
converter; leave private helper wrappers in cpu_tsdf.py; duplicate grid math in
teacher-cache replay.
Consequences: Mapper observations remain generic and synthetic-free, TSDF
replay uses public mapper helpers, and Phase 2D can build runtime scheduling on
explicit public boundaries without changing TSDF numerical behavior.
Docs/tests updated: docs/08_API_CONTRACTS.md,
tests/unit/test_mapping_observations.py, and
tests/synthetic/test_teacher_cache_replay.py.
```

```text
Decision ID: D-0015
Date: 2026-06-01
Context: Phase 2D needs a deterministic runtime scheduler skeleton that
exercises the synthetic fixture adapter cache path into TSDF mapping without
introducing threads, asyncio, GPU work, neural inference, video decoding, or new
heavy dependencies.
Decision: Add `atlas3r smoke runtime-fixture --output <folder>` and a public
runtime fixture event-log format. The smoke output writes relative-path
`runtime_events.jsonl`, `runtime_summary.json`, a generated synthetic
cube-room session, a full-array `fixture-cube-room` teacher cache, and the
existing teacher-cache TSDF artifacts including MeshChunk and WorldMap sidecars.
Event timestamps and latencies are deterministic placeholders, and bounded
memory counters describe scheduler-owned frame-array payload state only.
Alternatives considered: Build a threaded or asyncio scheduler now; reuse
wall-clock runtime logs; add a separate runtime output manifest without event
records; skip MeshChunk/WorldMap sidecars in the runtime smoke output.
Consequences: Runtime plumbing can be tested deterministically and inspected
through existing cache/TSDF artifact paths while avoiding premature performance
or accuracy claims. Future real runtime work must replace placeholder timing
with measured profiling and broaden the memory accounting deliberately.
Docs/tests updated: docs/08_API_CONTRACTS.md and
tests/synthetic/test_runtime_fixture_scheduler.py.
```

```text
Decision ID: D-0016
Date: 2026-06-01
Context: Phase 2E needs a dependency-free way to validate complete Phase 2D
runtime fixture smoke output folders after generation, including the event log,
summary, teacher cache, generated session, and nested TSDF mapper artifacts.
Decision: Add `atlas3r inspect runtime-fixture --input <folder>` with a new
deterministic JSON format named `atlas3r_runtime_fixture_output_inspection`.
The inspector validates event ordering, deterministic timing placeholders,
bounded-memory counters, relative paths, summary cross-checks, required
full-array teacher-cache payloads, and runs the existing complete TSDF output
folder inspector for `teacher_cache_tsdf`.
Alternatives considered: Extend the TSDF output inspector to understand runtime
folders; add a manifest to the runtime smoke output; rely on test-only helper
validation without a public command.
Consequences: Runtime fixture folders can be checked as one diagnostic artifact
without adding threads, GPU/model/video/export dependencies, or performance and
accuracy claims. The JSON explicitly states that inspection is not a performance
report and not an accuracy report.
Docs/tests updated: docs/08_API_CONTRACTS.md and
tests/synthetic/test_runtime_fixture_inspection.py.
```

```text
Decision ID: D-0017
Date: 2026-06-01
Context: Phase 3A needs future Streaming Metric Geometry Transformer work to
share one small, dependency-safe student-model input/output boundary before any
training code, weights, tensor library, or mapper bridge exists.
Decision: Add `atlas3r.models.student` with NumPy-only `StudentClipInput`,
`StudentForwardOutput`, and `ShapeOnlyStudentModel`. The output keeps batched
versions of existing Atlas3R geometry names such as `depth_m`,
`depth_sigma_m`, `confidence`, `T_world_camera`, and `intrinsics`, validates
them through existing helpers, and requires truth-boundary flags that mark the
shape-only stub as not learned inference, not mapping-ready, not a performance
report, and not an accuracy report.
Alternatives considered: Add a real tensor/model dependency now; emit
`FramePrediction` or `DepthObservation` directly from the stub; create an
unvalidated free-form dictionary boundary.
Consequences: Future student implementations have a stable shape contract while
mapper/runtime paths remain untouched. Any later bridge to existing public
geometry contracts must explicitly reject outputs where
`truth_boundary.usable_for_mapping` is false.
Docs/tests updated: docs/08_API_CONTRACTS.md and
tests/unit/test_student_model_contracts.py.
```

```text
Decision ID: D-0018
Date: 2026-06-01
Context: Phase 3B needs future video/live ingestion to share one small RGB
frame-source boundary before adding video decoder dependencies, runtime
scheduling, or student-model inference.
Decision: Add `atlas3r.data.frame_source.RGBFrameSource` plus dependency-free
NPZ and simple binary PPM sequence loaders that emit existing `FramePacket`
records with deterministic IDs, placeholder timestamps, validated intrinsics,
identity resize metadata, and source-format metadata.
Alternatives considered: Add OpenCV/PyAV/imageio/Pillow now; create a parallel
frame API for ingestion; wire the source directly into the student model or
runtime scheduler.
Consequences: Ingestion contract plumbing can be tested from local fixtures
using only stdlib and NumPy while keeping model/runtime imports out of the data
path. Future video/live adapters must convert into this boundary or explicitly
extend it.
Docs/tests updated: docs/08_API_CONTRACTS.md and tests/unit/test_frame_source.py.
```
