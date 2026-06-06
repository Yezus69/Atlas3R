# Active Task - ViPE Primary Room Reconstruction

## Goal

Use ViPE as the primary offline geometry engine for the real
`room_walk_001/frames` capture, import its camera intrinsics, poses, and dense
depth into Atlas3R, export observed-only map artifacts, and compare against the
current VGGT+DepthPro soft-metric map.

## Starting Evidence

- Branch base: `codex/offline-world-builder-v11-colmap-witness`.
- Working branch: `codex/vipe-primary-room-reconstruction`.
- Room input:
  `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`.
- Existing V10/V11 best map: 1,904,976 points, 2,559 occupied voxels,
  5,600 observed mesh triangles, and 88 trajectory poses.
- Existing best map is unanchored soft-metric teacher-pseudo geometry, not
  measured geometry and not training-quality.
- Known concern from V1.1: compact room-scale bbox around
  `1.72 m x 0.75 m x 0.96 m`.

## Checklist

- [x] Start from V1.1 branch and create
  `codex/vipe-primary-room-reconstruction`.
- [x] Read required status docs and current room map evidence.
- [x] Install or prepare ViPE externally under
  `C:/Users/Asav/source/repos/homebrain/external/vipe`.
- [x] Run ViPE smoke and main room-frame passes, recording commands and logs.
- [x] Add a minimal Atlas3R ViPE importer and one CLI entry point.
- [x] Export `runs/room_walk_001_vipe_import/` map artifacts from real ViPE
  output.
- [x] Compare ViPE import against the current VGGT+DepthPro map.
- [x] Run focused tests and relevant lint/type/test commands.
- [x] Update compact status docs and write the ViPE report.
- [x] Commit source, tests, and docs only.

## Boundaries

- Do not add SAM, object fusion, student training, anchors, or extra report
  plumbing in this task.
- ViPE, LongSplat, downloaded weights, and generated runs stay outside git.
- ViPE output is labeled `teacher_pseudo_vipe_near_metric`,
  `measured_geometry: false`, `physical_accuracy_claim: false`,
  `training_quality: false`, and `observed_only: true`.
