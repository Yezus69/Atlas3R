# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 3A - minimal public student-model input/output contract and deterministic shape-only stub.
Relevant docs read: AGENTS.md, README.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md, docs/status/decisions.md,
docs/status/next_task.md.
Relevant source/tests read: src/atlas3r/api/contracts.py,
src/atlas3r/api/validation.py, src/atlas3r/mapping/observations.py,
src/atlas3r/models/adapters/contracts.py, tests/unit/test_contracts.py,
tests/unit/test_adapters.py, and tests/unit/test_mapping_observations.py.
Plan:
1. Add `atlas3r.models.student` with `StudentClipInput`,
   `StudentForwardOutput`, and `ShapeOnlyStudentModel`.
2. Validate only NumPy array shapes, intrinsics, transforms, confidence,
   probability, uncertainty, coordinate-frame metadata, and truth-boundary
   flags.
3. Keep the stub deterministic with identity poses, broadcast/copied
   intrinsics, placeholder depth/uncertainty/confidence values, and no global
   state.
4. Add focused unit tests for accepted/rejected contracts, shape-only forward
   outputs, truth-boundary flags, and dependency-safe import.
5. Update API contracts, decisions, progress, and next_task.md after
   verification.
Checklist:
- [x] Read required docs and focused source/tests.
- [x] Rewrite active task for Phase 3A.
- [x] Add student contracts and shape-only stub.
- [x] Add focused unit tests.
- [x] Update API docs, decisions, progress, and next task.
- [x] Run requested Ruff, mypy, unittest, and available make commands.
Known exclusions: Do not add PyTorch, TensorFlow, JAX, Core ML, TensorRT,
CUDA, Metal, OpenCV, video decoding, datasets, training loops, weights,
downloads, external model repos, web servers, notebooks, GLB/PLY export,
marching cubes, runtime scheduler changes, TSDF fusion changes, visualization,
performance claims, or accuracy claims.
```
