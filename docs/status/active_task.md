# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 3B - dependency-free RGB frame-source boundary and deterministic
NPZ/PPM sequence smoke path.
Relevant docs read: AGENTS.md, README.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md, docs/status/decisions.md,
docs/status/next_task.md.
Relevant source/tests read: src/atlas3r/api/contracts.py,
src/atlas3r/models/student/contracts.py, tests/unit/test_contracts.py, and
tests/unit/test_student_model_contracts.py.
Plan:
1. Add `atlas3r.data.frame_source` with an `RGBFrameSource` Protocol and
   stdlib/NumPy NPZ and binary PPM loaders that emit existing `FramePacket`
   records.
2. Validate RGB layouts, deterministic ordering, intrinsics, metadata, and
   placeholder resize/timestamp fields without importing video/image/model
   dependencies.
3. Add focused unit tests for valid NPZ/PPM conversion, rejected layouts and
   intrinsics, frame ordering/IDs, dependency-safe imports, and no student
   package import.
4. Document the public frame-source boundary and smoke fixture format, record a
   decision entry, update progress, and replace next_task.md.
5. Run requested Ruff, mypy, unittest, and available make commands.
Checklist:
- [x] Read required docs and focused source/tests.
- [x] Rewrite active task for Phase 3B.
- [x] Add frame-source contract and dependency-free loaders.
- [x] Add focused unit tests.
- [x] Update API docs, decisions, progress, and next task.
- [x] Run requested Ruff, mypy, unittest, and available make commands.
Known exclusions: Do not add OpenCV, PyAV, imageio, Pillow, ffmpeg, web servers,
notebooks, datasets, training, weights, downloads, external repos, GLB/PLY
export, marching cubes, TSDF/runtime scheduler changes, new inspection bundles,
neural inference, or student model calls.
```
