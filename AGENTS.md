# AGENTS.md - Codex Operating Rules for Atlas3R

This file is for Codex. Keep it concrete and compact.

## Repository Mission

Atlas3R is pivoting to a Scale-Aware Monocular Reconstruction Teacher for RGB
videos. The teacher filters videos, reconstructs static geometry when evidence
is strong enough, fuses rays into TSDF/occupancy maps, and accepts metric
pseudo-labels only when scale uncertainty passes a documented gate.

Architecture source of truth: `PLANS.md`, `docs/00_PIVOT_OBJECTIVE.md`,
`docs/01_TEACHER_ARCHITECTURE.md`, `docs/08_API_CONTRACTS.md`, and
`docs/status/*`.

## Hard Constraints

- Do not claim millimeter accuracy unless an evaluation report proves it on a
  named benchmark or calibration capture.
- Do not claim arbitrary unanchored monocular RGB gives measured metric ground
  truth. It does not.
- Do not hallucinate hidden geometry as measured geometry. Mark completion as
  predicted or uncertain.
- Do not vendor third-party repos or model weights into this repo. Use adapters,
  installation scripts, and documented external paths.
- Do not silently change coordinate conventions, tensor shapes, or output
  schemas. Update `docs/08_API_CONTRACTS.md` and tests first.
- Do not create monolithic files. Keep source files under about 500 lines unless
  there is a strong reason.
- Do not implement stale offline-world-builder, VGGT/Depth-Pro witness, or old
  student-runtime code on this pivot branch.

## Development Workflow

Before coding a large change:

1. Read only the docs relevant to the task.
2. Write a small implementation plan in `docs/status/active_task.md`.
3. Implement the smallest vertical slice that can be tested.
4. Add or update tests before declaring done.
5. Run the relevant verification command.
6. Update `docs/status/progress.md` with what changed, commands run, and known
   gaps.

## Done Means

A task is not done until:

- types and interfaces are documented;
- unit tests pass;
- synthetic integration tests pass for geometry-facing code;
- the command used to verify is recorded;
- failure modes are handled with explicit errors or confidence outputs.

## Preferred Commands

Create these when implementation resumes:

```bash
make format && make lint && make typecheck
make test && make smoke
```

This docs-only branch does not currently provide those commands.

## Coding Style

- Python first for research code; C++/CUDA/Metal only when profiling proves
  Python is the bottleneck.
- Use typed dataclasses or Pydantic-style schemas at boundaries.
- Keep math functions pure and unit-tested.
- Use explicit coordinate frame names: `T_world_camera`, not `pose`.
- Store transforms as 4x4 matrices plus quaternion/translation convenience
  accessors.
- Units are meters unless a field explicitly says otherwise.
- Use uncertainty/covariance fields from day one, even if initially heuristic.
- Internal camera geometry should be ray-map first, not pinhole-`K` first.

## Model Integration Rule

Third-party models must be isolated behind adapters:

```text
src/atlas3r/models/adapters/<model_name>_adapter.py
```

Each geometry adapter must expose this conceptual contract:

```python
class VideoGeometryBackbone(Protocol):
    def predict(self, video: VideoInput) -> GeometryBackbonePrediction: ...
```

Adapters must support missing dependencies with clear installation errors, not
import-time crashes.

## Accuracy Rule

Every geometry output must carry confidence or uncertainty. Mesh chunks must
contain metadata:

- source frame IDs;
- observed coverage estimate;
- voxel size;
- coordinate frame;
- metric scale source;
- mean and percentile uncertainty.

Metric output requires a scale posterior and an acceptance decision. If the
posterior is wide, reject the video for metric training or mark it as
non-metric pseudo-label data.

## Context Control

Keep a compact log in `docs/status/progress.md`,
`docs/status/active_task.md`, `docs/status/decisions.md`, and
`docs/status/next_task.md`.

When context gets compacted, reload `README.md`, `PLANS.md`,
`docs/08_API_CONTRACTS.md`, and the current `docs/status/*` files before
continuing.

## Context Budget

Target maximums:

```text
AGENTS.md <= 140 lines
PLANS.md <= 220 lines
docs/08_API_CONTRACTS.md <= 450 lines
docs/status/active_task.md <= 80 lines
docs/status/progress.md <= 180 lines
docs/status/decisions.md <= 180 lines
docs/status/next_task.md <= 150 lines
```
