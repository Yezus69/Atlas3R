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
