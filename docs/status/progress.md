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
