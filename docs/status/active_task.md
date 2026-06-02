# Active task

Goal: Phase 3D minimal FramePacket/RGBFrameSource to teacher FrameBatch bridge.

Checklist:

- [x] Read only `next_task.md`, API/status docs, and focused source/tests.
- [x] Add a dependency-free helper converting ordered `FramePacket` sequences to `FrameBatch`.
- [x] Add a tiny `RGBFrameSource` wrapper helper only if it keeps tests clearer.
- [x] Reject empty sequences, non-`FramePacket` items, and duplicate `frame_id` values with field-named errors.
- [x] Keep metadata compact and deterministic without changing mapper/runtime/TSDF paths.
- [x] Add focused tests for conversion, preserved order, error cases, import safety, and fixture adapter/runner boundary acceptance.
- [x] Update API/status docs within context budgets.
- [x] Run required Ruff, mypy, unittest, `git diff --check`, and available make commands.

Exclusions: no heavy dependencies, external models, training, video decoding,
runtime scheduler, mapper, TSDF, export, inspection bundle, or CLI changes.
