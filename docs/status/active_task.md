# Active Task - Phase 6G RGB Teacher Mapping

Goal: build a diagnostic RGB-only teacher bridge from RGB input through
teacher-pseudo depth/pose/intrinsics into the existing sparse TSDF and observed
mesh chunk pipeline.

Branch: `codex/phase6g-rgb-teacher-map`

Checklist:

- [x] Confirm clean worktree and branch from Phase 6F.
- [x] Read required architecture, contract, evaluation, status, runtime,
  mapper, recording, teacher, and CLI context.
- [x] Reuse dependency-safe VGGT teacher path and add RGB input loading for
  image folders, video when optional decoders exist, and recordings in
  RGB-only mode.
- [x] Convert teacher predictions to pseudo `DepthObservation` streams with
  explicit pseudo/truth flags and OpenCV camera-from-world conversion tests.
- [x] Feed pseudo observations through sparse TSDF and `MeshChunkArtifactWriter`
  without weakening measured live replay behavior.
- [x] Add `runtime map-rgb-teacher` CLI, summaries, reports, pseudo recording
  artifacts, and focused unit/CLI/artifact tests.
- [x] Run fake-teacher tests, required verification commands, and a real
  RGB-only evidence run using VGGT or mark Phase 6G blocked with exact failures.
- [x] Update concise docs/status, Phase 6G report, next task, and commit.
