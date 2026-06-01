# Codex Prompt - Atlas3R Phase 0E: Geometry Teacher Adapter Contracts

You are working in the existing public repo `Yezus69/Atlas3R` after Phase 0D.
Read `AGENTS.md`, `README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`, and
the current `docs/status/*` files before coding. Then read only the source and
tests needed for model adapter contracts and CLI discovery.

## Task goal

Create the dependency-safe teacher-adapter contract layer that Phase 1 model
integrations will use. This is still a skeleton/contract task: do not download
model weights, vendor third-party repositories, call cloud APIs, run neural
inference, add CUDA, or add heavyweight visualization/export dependencies.

## Required implementation

1. Add the shared adapter protocol under `src/atlas3r/models/adapters/`.
   - Define `GeometryTeacherAdapter` with:

```python
class GeometryTeacherAdapter(Protocol):
    def predict(self, frames: FrameBatch) -> TeacherPrediction: ...
```

   - Add minimal typed contracts for `FrameBatch`, `TeacherPrediction`, and
     adapter capability/status metadata.
   - Reuse existing `CameraModel`, `PoseEstimate`, `FramePrediction`, and
     uncertainty/confidence conventions instead of creating parallel schemas.

2. Add dependency-safe adapter stubs.
   - Create at least two named stubs for future external teachers, such as
     `VGGTAdapter` and `DepthProAdapter`.
   - Missing optional dependencies must raise clear runtime installation errors
     from adapter construction or `predict`, not import-time crashes.
   - Keep adapter modules small and do not import unavailable third-party
     packages at module import time.

3. Add a small discovery surface.
   - Provide a pure-Python registry/list function for known adapters and their
     availability status.
   - Add a CLI command such as `atlas3r adapters list` that prints the known
     adapters and whether they are available, unavailable, or stub-only.

4. Add tests.
   - The protocol/status contracts validate expected shapes and metadata.
   - Stub adapters import without third-party dependencies installed.
   - Missing dependency errors include the adapter name and installation hint.
   - CLI adapter listing succeeds.
   - Existing Phase 0A-0D tests keep passing.

5. Update docs/status.
   - Rewrite `docs/status/active_task.md` before coding with a concise Phase 0E
     plan and checklist.
   - Update `docs/08_API_CONTRACTS.md` if new public schemas are introduced.
   - Append results to `docs/status/progress.md` after verification.
   - Append to `docs/status/decisions.md` only if an interface or format
     decision changed.
   - Replace this file with the Phase 1A prompt before declaring done.

## Verification commands

Run:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p 'test_*.py'
```

If `make` is available, also run:

```bash
make test
make lint
make typecheck
make smoke
make inspect
```

Record any unavailable command with the exact environment reason in
`docs/status/progress.md`.
