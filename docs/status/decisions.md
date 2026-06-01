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
