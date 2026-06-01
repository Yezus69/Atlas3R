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
