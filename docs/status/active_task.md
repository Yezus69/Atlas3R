# Active task

Goal: Phase 3C minimal FramePacket to StudentClipInput bridge.

Checklist:

- [x] Read only the requested architecture/status docs and focused source/tests.
- [x] Add dependency-free `atlas3r.data.student_clip` helper.
- [x] Export the helper from `atlas3r.data`.
- [x] Add focused unit tests for valid clips, ordering, errors, import safety, and shape-only forward.
- [x] Update API contracts/status docs within context budgets.
- [x] Run Ruff, mypy, unittest discovery, `git diff --check`, and available make commands.

Exclusions: no heavy dependencies, external models, training, video decoding,
runtime scheduler, mapper, TSDF, export, inspection bundle, or CLI changes.
