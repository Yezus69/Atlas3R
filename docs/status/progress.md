# Progress log

Codex should append concise entries here after each task.

Template:

```text
YYYY-MM-DD HH:MM local
Task:
Changed files:
Commands run:
Results:
Known gaps:
```

```text
2026-05-31 19:48 local
Task: Initialize the Atlas3R repository skeleton only.
Changed files:
- Added pyproject.toml with src-layout package metadata and atlas3r console entry point.
- Added Makefile targets: format, lint, typecheck, test, smoke, profile.
- Added src/atlas3r package directories from README.md, including models/adapters.
- Added tests/unit, tests/integration, tests/synthetic with one unit test for package import and CLI help.
- Updated docs/status/active_task.md for the current task.
Commands run:
- make test
- python -m unittest discover -s tests/unit -p "test_*.py"
Results:
- make is not available on PATH in this environment.
- Direct unittest fallback ran 1 test and passed.
Known gaps:
- Neural models, geometry contracts, and synthetic geometry smoke tests are intentionally not implemented in this skeleton slice.
```

```text
2026-05-31 20:29 local
Task: Phase 0A - core NumPy contracts and coordinate math foundation.
Changed files:
- Added NumPy as a base dependency in pyproject.toml.
- Added validation helpers in src/atlas3r/api/validation.py.
- Added public contracts/enums in src/atlas3r/api/contracts.py and exported them from src/atlas3r/api/__init__.py.
- Added pinhole projection utilities in src/atlas3r/camera/pinhole.py and transform utilities in src/atlas3r/pose/transforms.py.
- Added unit tests for contracts, enum values, validation failures, projection/unprojection, quaternions, transforms, and CLI import/help.
- Documented the minimal DenseMatchSet placeholder in docs/08_API_CONTRACTS.md and recorded decision D-0001.
- Updated Makefile test discovery to run all tests under tests/.
- Added docs/status/next_task.md with the Phase 0B continuation prompt.
Commands run:
- python -m pip install -e ".[dev]"
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p test_*.py
- make test
- make lint
- make typecheck
Results:
- Editable dev install succeeded; NumPy was already installed and Ruff/mypy were installed.
- Ruff format check passed.
- Ruff lint passed.
- mypy passed with no issues in 19 source files.
- unittest discovery ran 18 tests and passed.
- make commands could not run because `make` is not available on PATH in this environment.
Known gaps:
- Phase 0A intentionally does not implement neural models, teacher adapters, TSDF fusion, mesh extraction, runtime video capture, or the synthetic cube-room generator.
```

```text
2026-05-31 20:44 local
Task: Phase 0B - deterministic synthetic cube-room generator and smoke command.
Changed files:
- Added src/atlas3r/data/synthetic_cube_room.py with fixed intrinsics, three
  known poses, analytic ray-box depth, object masks, object records, and a
  contract-valid ground-truth MeshChunk/WorldMap.
- Exported the synthetic fixture from src/atlas3r/data/__init__.py.
- Implemented `atlas3r smoke synthetic-cube-room --output <folder>`.
- Updated Makefile smoke to run the synthetic cube-room writer.
- Documented the Phase 0B mesh chunk JSON sidecar in docs/08_API_CONTRACTS.md
  and recorded decision D-0002.
- Added tests/synthetic/test_synthetic_cube_room.py for contracts,
  projection/unprojection agreement, object mask alignment, mesh/world
  validation, session writing, and CLI smoke output.
Commands run:
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p test_*.py
- make test
- make lint
- make typecheck
- make smoke
Results:
- Ruff format check passed.
- Ruff lint passed.
- mypy passed with no issues in 20 source files.
- unittest discovery ran 24 tests and passed.
- make commands could not run because `make` is not available on PATH in this
  environment.
Known gaps:
- Phase 0B intentionally does not implement neural models, teacher adapters,
  TSDF fusion, learned mesh extraction, runtime video capture, or GLB export.
- The smoke session writes a JSON mesh sidecar for the ground-truth synthetic
  mesh; GLB export remains future mapper/exporter work.
```

```text
2026-05-31 21:17 local
Task: Phase 0C - session inspector and dependency-free synthetic previews.
Changed files:
- Added src/atlas3r/io/session.py and src/atlas3r/io/__init__.py with a
  Phase 0B `.atlas3r` reader/validator, contract reconstruction, sorted depth
  file discovery, and one-depth-file load helper.
- Added src/atlas3r/visualization/session_preview.py and
  src/atlas3r/visualization/__init__.py with deterministic HTML/SVG previews
  for top-down scene layout, first depth frame, and first object-id mask.
- Added `atlas3r inspect session --input <session.atlas3r> --output <preview>`
  and a Makefile `inspect` target.
- Added .gitignore coverage for build outputs, `.atlas3r` folders, caches, and
  egg-info directories.
- Documented the Phase 0C reader/preview outputs in docs/08_API_CONTRACTS.md
  and recorded decision D-0003.
- Replaced docs/status/next_task.md with the Phase 0D CPU TSDF reference
  integrator prompt and updated docs/status/active_task.md.
- Added tests/synthetic/test_session_inspect.py for load/validate failures,
  depth ordering, preview outputs and determinism, CLI inspect, and the Phase
  0D status handoff.
Commands run:
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- make --version
Results:
- Ruff format check passed.
- Ruff lint passed.
- mypy passed with no issues in 24 source files.
- unittest discovery ran 33 tests and passed.
- `make --version` reported: `make` is not recognized as the name of a cmdlet,
  function, script file, or operable program. Therefore `make test`,
  `make lint`, `make typecheck`, `make smoke`, and `make inspect` were not run.
Known gaps:
- Phase 0C intentionally does not implement neural models, teacher adapters,
  TSDF fusion, learned mesh extraction, OpenGL, web servers, notebooks,
  matplotlib, Pillow, trimesh, or GLB export.
- SVG/HTML previews are diagnostic inspection artifacts only, not accuracy
  reports.
```

```text
2026-05-31 21:37 local
Task: Phase 0D - deterministic pure-NumPy CPU TSDF reference smoke for the
synthetic cube-room fixture.
Changed files:
- Added src/atlas3r/mapping/cpu_tsdf.py with a small TSDF grid integrator,
  voxel-center surface extraction, uncertainty/confidence metadata, synthetic
  box-mesh metrics, and deterministic smoke artifact writing.
- Exported the TSDF reference helpers from src/atlas3r/mapping/__init__.py.
- Added `atlas3r smoke tsdf-cube-room --output <folder>` and updated `make
  smoke` to exercise the TSDF smoke path.
- Documented the Phase 0D TSDF smoke output format in docs/08_API_CONTRACTS.md
  and recorded decision D-0004.
- Added tests/synthetic/test_cpu_tsdf.py for determinism, metadata, voxel-scale
  synthetic bounds overlap, metrics, and CLI smoke output.
- Updated tests/synthetic/test_session_inspect.py for the Phase 0E handoff.
- Replaced docs/status/next_task.md with the Phase 0E teacher-adapter contract
  prompt and updated docs/status/active_task.md.
Commands run:
- python -m unittest tests.synthetic.test_cpu_tsdf
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- make --version
- Get-Command make
Results:
- Focused TSDF unittest ran 4 tests and passed.
- Ruff format check passed.
- Ruff lint passed.
- mypy passed with no issues in 25 source files.
- Full unittest discovery ran 37 tests and passed.
- `make --version` / `Get-Command make` reported: `make` is not recognized as
  the name of a cmdlet, function, script file, or operable program. Therefore
  `make test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect`
  were not run.
Known gaps:
- Phase 0D intentionally does not implement CUDA, neural models, teacher
  adapters, nvblox integration, OpenGL, web servers, notebooks, trimesh, or GLB
  export.
- The TSDF smoke output is an observed voxel-center point cloud with
  voxel-scale synthetic fixture metrics; it is not an accuracy report and makes
  no millimeter-level claim.
```

```text
2026-05-31 21:58 local
Task: Phase 0E - dependency-safe geometry teacher adapter contracts and
discovery stubs.
Changed files:
- Added adapter contracts, capability/status metadata, dependency errors, and
  registry helpers under src/atlas3r/models/adapters/.
- Added dependency-safe VGGT and Depth Pro stubs that avoid optional imports at
  module import time.
- Added `atlas3r adapters list` for known adapter availability discovery.
- Documented the teacher adapter public contracts in docs/08_API_CONTRACTS.md
  and recorded decision D-0005.
- Added tests/unit/test_adapters.py for contracts, stub imports, dependency
  errors, registry status, and CLI listing.
- Updated the status handoff test and replaced docs/status/next_task.md with
  the Phase 1A teacher prediction cache prompt.
Commands run:
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- python -m atlas3r adapters list
- Get-Command make
- make --version
Results:
- Ruff format check passed.
- Ruff lint passed.
- mypy passed with no issues in 30 source files.
- unittest discovery ran 43 tests and passed.
- Direct adapter listing printed known adapter statuses.
- `Get-Command make` / `make --version` reported: `make` is not recognized as
  the name of a cmdlet, function, script file, or operable program. Therefore
  `make test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect`
  were not run.
Known gaps:
- Phase 0E intentionally does not implement neural inference, cloud APIs, model
  downloads, CUDA, or vendored third-party model code/weights.
- VGGT and Depth Pro are discovery/contract stubs only; with dependencies
  installed they still report `stub-only` until Phase 1 wiring adds inference.
```

```text
2026-05-31 22:16 local
Task: Phase 1A - dependency-light TeacherPrediction cache format and adapter
runner skeleton.
Changed files:
- Added src/atlas3r/io/teacher_cache.py with deterministic metadata/frame
  summary cache writing and public loading helpers.
- Added src/atlas3r/io/_teacher_cache_validation.py and
  src/atlas3r/io/teacher_cache_schema.py for on-disk validation and shared
  cache constants.
- Exported teacher cache helpers from src/atlas3r/io/__init__.py.
- Added src/atlas3r/models/adapters/runner.py with session validation and
  clear unavailable/stub-only adapter run errors.
- Added `atlas3r adapters run --adapter <name> --input <session.atlas3r>
  --output <cache_dir>`.
- Documented the Phase 1A teacher cache format in docs/08_API_CONTRACTS.md and
  recorded decision D-0006.
- Added tests/unit/test_teacher_cache.py for deterministic cache output,
  reader validation, and CLI runner errors.
- Updated the status handoff test and replaced docs/status/next_task.md with
  the Phase 1B fixture teacher adapter prompt.
Commands run:
- python -m unittest tests.unit.test_teacher_cache
- python -m unittest tests.unit.test_adapters
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- Get-Command make
Results:
- Focused teacher cache unittest ran 4 tests and passed.
- Focused adapter unittest ran 6 tests and passed.
- Ruff format left 46 files unchanged after earlier formatting passes.
- Ruff format check passed.
- Ruff lint passed.
- mypy passed with no issues in 34 source files.
- Full unittest discovery ran 47 tests and passed.
- `Get-Command make` reported: `make` is not recognized as the name of a
  cmdlet, function, script file, or operable program. Therefore `make test`,
  `make lint`, `make typecheck`, `make smoke`, and `make inspect` were not run.
Known gaps:
- Phase 1A intentionally does not implement neural inference, external model
  execution, cloud APIs, model downloads, CUDA, or vendored third-party code or
  weights.
- `atlas3r adapters run` validates input sessions and reports clear adapter
  status/guidance, but known external adapters still do not write caches until
  future adapter implementations produce `TeacherPrediction` records.
```

```text
2026-05-31 23:02 local
Task: Phase 1B - dependency-free synthetic fixture teacher adapter and first
cache-producing adapter runner path.
Changed files:
- Added src/atlas3r/models/adapters/fixture_teacher_adapter.py with the
  available `fixture-cube-room` adapter, synthetic-session validation, analytic
  depth sidecar loading, deterministic point/normal reconstruction, object mask
  logits, confidence, and depth uncertainty outputs.
- Wired fixture discovery/export through src/atlas3r/models/adapters/registry.py
  and src/atlas3r/models/adapters/__init__.py.
- Updated src/atlas3r/models/adapters/runner.py so
  `atlas3r adapters run --adapter fixture-cube-room --input <session.atlas3r>
  --output <cache_dir>` writes a validated Phase 1A cache while VGGT/Depth Pro
  continue to fail gracefully as unavailable or stub-only adapters.
- Documented the fixture adapter and runner cache behavior in
  docs/08_API_CONTRACTS.md and recorded decision D-0007.
- Updated tests/unit/test_adapters.py, tests/unit/test_teacher_cache.py, and
  tests/synthetic/test_session_inspect.py for fixture availability,
  deterministic cache output, summary preservation, malformed-session errors,
  CLI cache writing, and the Phase 1C handoff.
- Rewrote docs/status/active_task.md and replaced docs/status/next_task.md with
  the Phase 1C prompt.
Commands run:
- python -m ruff format src tests
- python -m unittest tests.unit.test_adapters
- python -m unittest tests.unit.test_teacher_cache
- python -m ruff check src/atlas3r/models/adapters/runner.py --fix
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- Get-Command make
- git diff --check
Results:
- Ruff format left 47 files unchanged after the import cleanup.
- Ruff format check passed.
- Ruff lint passed.
- mypy passed with no issues in 35 source files.
- Full unittest discovery ran 52 tests and passed.
- `Get-Command make` reported: `The term 'make' is not recognized as the name
  of a cmdlet, function, script file, or operable program.` Therefore `make
  test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect` were
  not run.
- `git diff --check` reported no whitespace errors; Git warned that files will
  be converted from LF to CRLF in the working tree.
Known gaps:
- Phase 1B intentionally does not implement neural inference, external model
  execution, cloud APIs, model downloads, CUDA, heavyweight visualization/export
  dependencies, or real capture geometry claims.
- The fixture adapter only accepts Phase 0B synthetic cube-room sessions and
  writes summaries-only teacher caches; optional full tensor payloads are left
  for Phase 1C.
```

```text
2026-06-01 08:48 local
Task: Phase 1C - optional teacher cache tensor payloads and deterministic cache
inspection.
Changed files:
- Extended src/atlas3r/io/teacher_cache.py,
  src/atlas3r/io/_teacher_cache_validation.py, and
  src/atlas3r/io/teacher_cache_schema.py with opt-in `.npz` payload writing,
  loading, and validation for shapes, dtypes, confidence ranges, and
  non-negative depth/uncertainty arrays.
- Added src/atlas3r/io/teacher_cache_inspection.py and
  `atlas3r inspect teacher-cache --input <cache_dir>` deterministic JSON
  inspection output.
- Added `--store-arrays` to `atlas3r adapters run` and propagated
  `store_arrays=True` through the fixture adapter runner path while preserving
  summaries-only defaults and graceful external adapter stub failures.
- Updated Makefile `inspect` to exercise teacher-cache inspection when make is
  available.
- Updated docs/08_API_CONTRACTS.md, docs/status/active_task.md,
  docs/status/decisions.md, and docs/status/next_task.md with the Phase 1C
  contract and Phase 1D handoff.
- Updated tests/unit/test_teacher_cache.py and
  tests/synthetic/test_session_inspect.py for payload round-trip, missing/corrupt
  payload errors, deterministic CLI inspection, summaries-only compatibility,
  and Phase 1D handoff.
Commands run:
- python -m unittest tests.unit.test_teacher_cache
- python -m unittest tests.unit.test_adapters
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- Get-Command make
- git diff --check
Results:
- Focused teacher cache unittest ran 12 tests and passed.
- Focused adapter unittest ran 7 tests and passed.
- Ruff format left 48 files unchanged on the final run.
- Ruff format check passed with 48 files already formatted.
- Ruff lint passed.
- mypy passed with no issues in 36 source files.
- Full unittest discovery ran 56 tests and passed.
- `Get-Command make` reported: `The term 'make' is not recognized as the name
  of a cmdlet, function, script file, or operable program.` Therefore `make
  test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect` were
  not run.
- `git diff --check` reported no whitespace errors; Git warned that files will
  be converted from LF to CRLF in the working tree.
Known gaps:
- Phase 1C intentionally does not implement neural inference, external model
  execution, cloud APIs, model downloads, CUDA, heavyweight visualization/export
  dependencies, cache replay into mapping, or real capture geometry claims.
- Teacher cache inspection validates metadata and payload state only; it is not
  an accuracy report.
```

```text
2026-06-01 12:18 local
Task: Phase 1D - teacher cache full-array replay into the dependency-free CPU
TSDF reference path.
Changed files:
- Extended new teacher cache frame summaries with replay-needed camera intrinsics
  and T_world_camera pose metadata while keeping Phase 1C `.npz` payload keys
  stable.
- Added src/atlas3r/mapping/teacher_cache_replay.py with summaries-only cache
  rejection, validated payload loading, TSDF replay, deterministic artifacts,
  and synthetic-only fixture metrics.
- Added `atlas3r smoke teacher-cache-tsdf --input <cache_dir> --output <folder>`
  and wired Makefile smoke/inspect to exercise fixture cache replay when make is
  available.
- Updated docs/08_API_CONTRACTS.md, docs/status/active_task.md,
  docs/status/decisions.md, and docs/status/next_task.md with the Phase 1D
  contract and Phase 1E handoff.
- Added tests/synthetic/test_teacher_cache_replay.py and updated focused cache
  and handoff tests.
Commands run:
- python -m unittest tests.unit.test_teacher_cache
- python -m unittest tests.synthetic.test_teacher_cache_replay
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- Get-Command make
- git diff --check
Results:
- Focused teacher cache unittest ran 12 tests and passed.
- Focused teacher cache replay unittest ran 6 tests and passed.
- Ruff format reformatted 1 file on the first run and left 50 files unchanged
  on the final run.
- Ruff format check passed with 50 files already formatted.
- Ruff lint passed.
- mypy passed with no issues in 37 source files.
- Full unittest discovery ran 62 tests and passed.
- `Get-Command make` reported: `The term 'make' is not recognized as the name
  of a cmdlet, function, script file, or operable program.` Therefore `make
  test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect` were
  not run.
- `git diff --check` reported no whitespace errors; Git warned that files will
  be converted from LF to CRLF in the working tree.
Known gaps:
- Phase 1D intentionally does not implement neural inference, external model
  execution, cloud APIs, model downloads, CUDA, heavyweight visualization/export
  dependencies, GLB/PLY export, marching cubes, or real capture geometry claims.
- Teacher-cache TSDF replay writes observed voxel-center surface points and
  conservative synthetic fixture metrics only when cache metadata proves the
  synthetic cube-room fixture source. Other caches get an explicit
  `not_evaluated` metrics record, not an accuracy report.
```

```text
2026-06-01 12:37 local
Task: Phase 1E - dependency-free MeshChunk sidecar writer for CPU TSDF replay
outputs.
Changed files:
- Added src/atlas3r/mapping/mesh_sidecar.py with TSDF surface artifact loading,
  contract-valid MeshChunk construction, deterministic sidecar JSON writing,
  sidecar loading/validation, and path-named missing/malformed artifact errors.
- Exported MeshChunk sidecar helpers from src/atlas3r/mapping/__init__.py.
- Extended `write_tsdf_cube_room_smoke` and `write_teacher_cache_tsdf_replay`
  with opt-in MeshChunk sidecar writing while preserving existing artifacts.
- Added `--write-mesh-sidecar` to `atlas3r smoke tsdf-cube-room` and
  `atlas3r smoke teacher-cache-tsdf`; updated Makefile smoke/inspect to use it
  when make is available.
- Documented `mesh_chunk_sidecar.json` in docs/08_API_CONTRACTS.md and recorded
  decision D-0010.
- Added tests/synthetic/test_tsdf_mesh_sidecar.py and updated
  tests/synthetic/test_teacher_cache_replay.py and
  tests/synthetic/test_session_inspect.py for sidecar behavior and Phase 2A
  handoff.
- Rewrote docs/status/active_task.md for Phase 1E and replaced
  docs/status/next_task.md with the Phase 2A prompt.
Commands run:
- python -m unittest tests.synthetic.test_tsdf_mesh_sidecar
- python -m unittest tests.synthetic.test_teacher_cache_replay
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- Get-Command make
- git diff --check
Results:
- Focused TSDF MeshChunk sidecar unittest ran 5 tests and passed.
- Focused teacher-cache replay unittest ran 6 tests and passed.
- Ruff format left 52 files unchanged on the final run.
- Ruff format check passed with 52 files already formatted.
- Ruff lint passed.
- mypy passed with no issues in 38 source files.
- Full unittest discovery ran 67 tests and passed on the final run.
- `Get-Command make` reported: `The term 'make' is not recognized as the name
  of a cmdlet, function, script file, or operable program.` Therefore `make
  test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect` were
  not run.
- `git diff --check` reported no whitespace errors; Git warned that files will
  be converted from LF to CRLF in the working tree.
Known gaps:
- Phase 1E intentionally does not implement GLB/PLY export, trimesh,
  marching-cubes extraction, object-aware meshing, neural inference, external
  model execution, cloud APIs, CUDA, model downloads, or vendored third-party
  code/weights.
- The sidecar mesh is a low-fidelity observed-sample reference artifact for
  pipeline testing only; it is not an accuracy report and does not claim hidden
  or completed geometry as measured.
```

```text
2026-06-01 12:55 local
Task: Phase 2A - dependency-free WorldMap sidecar assembly for CPU TSDF
MeshChunk sidecars.
Changed files:
- Added src/atlas3r/mapping/world_map_sidecar.py with MeshChunk sidecar loading,
  WorldMap construction, deterministic sidecar JSON writing, sidecar loading,
  metadata validation, and deterministic inspect JSON.
- Exported WorldMap sidecar helpers from src/atlas3r/mapping/__init__.py.
- Extended CPU TSDF smoke and teacher-cache TSDF replay writers with
  `write_world_map_sidecar` while preserving existing Phase 0D/1D/1E artifacts.
- Added `--write-world-map-sidecar` to both CPU TSDF smoke commands and
  `atlas3r inspect world-map --input <world_map_sidecar.json>`.
- Updated Makefile smoke/inspect targets to exercise WorldMap sidecar output
  when make is available.
- Documented `world_map_sidecar.json` and inspect behavior in
  docs/08_API_CONTRACTS.md and recorded decision D-0011.
- Added tests/synthetic/test_world_map_sidecar.py and updated the status
  handoff test for Phase 2B.
- Rewrote docs/status/active_task.md and replaced docs/status/next_task.md with
  the Phase 2B prompt.
Commands run:
- python -m unittest tests.synthetic.test_world_map_sidecar
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- Get-Command make
- git diff --check
Results:
- Focused WorldMap sidecar unittest ran 5 tests and passed.
- Ruff format left 54 files unchanged on the final run.
- Ruff format check passed with 54 files already formatted.
- Ruff lint passed.
- mypy passed with no issues in 39 source files.
- Full unittest discovery ran 72 tests and passed.
- `Get-Command make` reported: `The term 'make' is not recognized as the name
  of a cmdlet, function, script file, or operable program.` Therefore `make
  test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect` were
  not run.
- `git diff --check` reported no whitespace errors; Git warned that files will
  be converted from LF to CRLF in the working tree.
Known gaps:
- Phase 2A intentionally does not implement GLB/PLY export, trimesh,
  marching-cubes extraction, object-aware meshing, keyframe map storage, neural
  inference, external model execution, cloud APIs, CUDA, model downloads, or
  vendored third-party code/weights.
- The WorldMap sidecar wraps one observed low-fidelity MeshChunk for pipeline
  testing only; it is not an accuracy report and does not claim hidden or
  completed geometry as measured.
```

```text
2026-06-01 13:13 local
Task: Phase 2B - dependency-free CPU TSDF output folder inspection.
Changed files:
- Added src/atlas3r/mapping/tsdf_output_inspection.py and
  src/atlas3r/mapping/_tsdf_output_inspection_helpers.py with deterministic
  folder inspection for Phase 0D/1D surface artifacts, Phase 1E MeshChunk
  sidecars, and Phase 2A WorldMap sidecars.
- Exported the inspector through src/atlas3r/mapping/__init__.py.
- Added `atlas3r inspect tsdf-output --input <folder> [--mode ...]` and wired
  the Makefile inspect target to exercise the complete folder path.
- Added tests/synthetic/test_tsdf_output_inspection.py for deterministic TSDF
  smoke and teacher-cache replay inspection, metadata mismatch rejection,
  required-sidecar modes, and CLI behavior.
- Updated docs/08_API_CONTRACTS.md, docs/status/active_task.md,
  docs/status/decisions.md, docs/status/next_task.md, and the status handoff
  test for the Phase 2C prompt.
Commands run:
- python -m unittest tests.synthetic.test_tsdf_output_inspection
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- Get-Command make
- git diff --check
Results:
- Focused TSDF output inspection unittest ran 5 tests and passed.
- Ruff format left 57 files unchanged on the final run.
- Ruff format check passed with 57 files already formatted.
- Ruff lint passed.
- mypy passed with no issues in 41 source files after fixing one nullable
  cross-check value.
- Full unittest discovery ran 77 tests and passed.
- `Get-Command make` reported: `The term 'make' is not recognized as the name
  of a cmdlet, function, script file, or operable program.` Therefore `make
  test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect` were
  not run.
- `git diff --check` reported no whitespace errors; Git warned that files will
  be converted from LF to CRLF in the working tree.
Known gaps:
- Phase 2B intentionally does not implement GLB/PLY export, trimesh,
  marching-cubes extraction, object-aware meshing, neural inference, external
  model execution, cloud APIs, CUDA, model downloads, or vendored third-party
  code/weights.
- The folder inspection JSON is a deterministic mapper pipeline diagnostic; it
  is not an accuracy report and does not claim hidden or completed geometry as
  measured.
```

```text
2026-06-01 17:35 local
Task: Phase 2C revised - public DepthObservation mapper input contract and
teacher-cache replay refactor.
Changed files:
- Added src/atlas3r/mapping/observations.py with the validated
  DepthObservation contract and synthetic-frame conversion helper.
- Refactored src/atlas3r/mapping/cpu_tsdf.py so TSDF integration consumes
  DepthObservation while preserving existing synthetic smoke outputs.
- Refactored src/atlas3r/mapping/teacher_cache_replay.py to convert replay
  frames into DepthObservation and removed the replay `type: ignore[arg-type]`
  path.
- Exported DepthObservation and observation conversion helpers from
  src/atlas3r/mapping/__init__.py.
- Added tests/unit/test_mapping_observations.py and updated focused replay and
  status handoff tests.
- Updated docs/08_API_CONTRACTS.md, docs/status/active_task.md,
  docs/status/decisions.md, docs/status/next_task.md, and PLANS.md.
Commands run:
- python -m unittest tests.unit.test_mapping_observations
- python -m unittest tests.synthetic.test_teacher_cache_replay
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- Get-Command make
- git diff --check
Results:
- Focused DepthObservation unittest ran 6 tests and passed.
- Focused teacher-cache replay unittest ran 7 tests and passed.
- Ruff format left 59 files unchanged on the final run.
- Ruff format check passed with 59 files already formatted.
- Ruff lint passed.
- mypy passed with no issues in 42 source files.
- Full unittest discovery ran 84 tests and passed.
- `Get-Command make` reported: `The term 'make' is not recognized as the name
  of a cmdlet, function, script file, or operable program.` Therefore `make
  test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect` were
  not run.
- `git diff --check` reported no whitespace errors; Git warned that files will
  be converted from LF to CRLF in the working tree.
Known gaps:
- Phase 2C revised intentionally does not implement neural inference, external
  model downloads, CUDA, GLB/PLY export, marching cubes, object-aware mesh
  extraction, video decoding, web servers, notebooks, or regression bundles.
- Optional DepthObservation object/rgb/static-mask fields are validated at the
  boundary, but CPU TSDF fusion still uses depth, confidence, uncertainty,
  camera, and pose only.
```

```text
2026-06-01 18:05 local
Task: Phase 2C.1 - mapper boundary cleanup before Phase 2D.
Changed files:
- Added src/atlas3r/data/synthetic_observations.py with the synthetic
  cube-room to DepthObservation converter.
- Added src/atlas3r/mapping/tsdf_grid.py with public TSDF grid shape and voxel
  center helpers.
- Updated src/atlas3r/mapping/observations.py so DepthObservation remains
  generic and synthetic-free.
- Updated src/atlas3r/data/__init__.py, src/atlas3r/mapping/__init__.py,
  src/atlas3r/mapping/cpu_tsdf.py, and
  src/atlas3r/mapping/teacher_cache_replay.py for the new public boundaries.
- Added focused architecture guard assertions in
  tests/unit/test_mapping_observations.py and
  tests/synthetic/test_teacher_cache_replay.py.
- Updated docs/08_API_CONTRACTS.md, docs/status/active_task.md, and
  docs/status/decisions.md; docs/status/next_task.md remains the Phase 2D
  deterministic streaming runtime scheduler prompt.
Commands run:
- python -m ruff format src tests
- python -m ruff format --check src tests
- python -m ruff check src tests
- python -m mypy src
- python -m unittest discover -s tests -p 'test_*.py'
- Get-Command make
- git diff --check
Results:
- Initial mypy caught a tuple typing issue for TSDFVolume.centers_world_m; it
  was fixed and the final mypy run passed with no issues in 44 source files.
- Initial unittest discovery caught a circular import through the eager data
  converter export; it was fixed with a lazy data export and local synthetic
  converter import in CPU TSDF.
- Final Ruff format left 61 files formatted after one file was reformatted.
- Final Ruff format check passed with 61 files already formatted.
- Final Ruff lint passed.
- Final mypy passed with no issues in 44 source files.
- Final unittest discovery ran 85 tests and passed.
- Get-Command make reported: The term 'make' is not recognized as the name of
  a cmdlet, function, script file, or operable program. Therefore make test,
  make lint, make typecheck, make smoke, and make inspect were not run.
- git diff --check reported no whitespace errors; Git warned that files will be
  converted from LF to CRLF in the working tree.
Known gaps:
- Phase 2C.1 intentionally does not implement runtime scheduling, threads,
  CUDA, Metal, neural models, video decoding, marching cubes, GLB/PLY export,
  web servers, notebooks, or new heavy dependencies.
- The synthetic converter is exported lazily from atlas3r.data to avoid import
  cycles while keeping atlas3r.mapping.observations generic.
```
