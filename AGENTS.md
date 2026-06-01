# AGENTS.md — Codex Operating Rules for Atlas3R

This file is for Codex. It is intentionally concrete and short enough to remain useful after context compaction.

## Repository mission

Build Atlas3R: a real-time RGB neural 3D mapping system that outputs camera pose, dense metric scene geometry, object instances, and game-engine triangle meshes. The architecture source of truth is in `docs/` and `PLANS.md`.

## Hard constraints

- Do not claim millimeter accuracy unless an evaluation report proves it on a named benchmark or calibration capture.
- Do not hallucinate hidden geometry as measured geometry. Mark completion as predicted/uncertain.
- Do not paste huge code blocks in chat. Write files, run tests, and summarize only the changed paths and results.
- Do not vendor third-party repos or model weights into this repo. Use adapters, installation scripts, and documented external paths.
- Do not create monolithic files. Keep source files under ~500 lines unless there is a strong reason.
- Do not silently change coordinate conventions, tensor shapes, or output schemas. Update `docs/08_API_CONTRACTS.md` and tests first.

## Development workflow

Before coding a large change:

1. Read only the docs relevant to the task.
2. Write a small implementation plan in `docs/status/active_task.md`.
3. Implement the smallest vertical slice that can be tested.
4. Add or update tests before declaring done.
5. Run the relevant `make` command.
6. Update `docs/status/progress.md` with what changed, commands run, and known gaps.

## Done means

A task is not done until:

- types and interfaces are documented,
- unit tests pass,
- synthetic integration tests pass for geometry-facing code,
- the command used to verify is recorded,
- failure modes are handled with explicit errors or confidence outputs.

## Preferred commands

Create these early:

```bash
make format       # ruff/black or equivalent
make lint         # static checks
make typecheck    # mypy or pyright
make test         # unit tests
make smoke        # synthetic cube-room end-to-end smoke test
make profile      # short runtime profile on sample video
```

## Coding style

- Python first for research code; C++/CUDA/Metal only when profiling proves Python is the bottleneck.
- Use typed dataclasses or Pydantic-style schemas at boundaries.
- Keep math functions pure and unit-tested.
- Use explicit coordinate frame names: `T_world_camera`, not `pose`.
- Store transforms as 4×4 matrices plus quaternion/translation convenience accessors.
- Units are meters unless a field explicitly says otherwise.
- Use uncertainty/covariance fields from day one, even if initially populated by heuristics.

## Context control

Codex must keep a compact log instead of relying on conversation history:

```text
docs/status/progress.md      # cumulative short build log
docs/status/active_task.md   # current task plan and checklist
docs/status/decisions.md     # architecture decisions and rationale
```

When context gets compacted, reload `README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`, and the current `docs/status/*` files before continuing.

## Model integration rule

Third-party models must be isolated behind adapters:

```text
src/atlas3r/models/adapters/<model_name>_adapter.py
```

Each adapter must expose the same contract:

```python
class GeometryTeacherAdapter(Protocol):
    def predict(self, frames: FrameBatch) -> TeacherPrediction: ...
```

Adapters must support missing dependencies with a clear installation error, not import-time crashes.

## Accuracy rule

Every geometry output must carry confidence/uncertainty. Mesh chunks must contain metadata:

- source frame IDs,
- observed coverage estimate,
- voxel size,
- coordinate frame,
- metric scale source,
- mean and percentile uncertainty.

