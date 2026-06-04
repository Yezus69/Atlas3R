# Active Task - Phase 5H VGGT Pose/Pointmap Teacher

Goal: run or dependency-safely block a real VGGT pose/depth/pointmap teacher,
convert outputs into Atlas3R teacher-signal caches, evaluate against measured
TUM clip caches, and train only if teacher quality gates pass.

Branch: `codex/phase5h-vggt-pose-pointmap-teacher`

Checklist:

- [x] Start from `codex/phase5g1-multisequence-tum-generalization` and create
  the Phase 5H branch.
- [x] Read the requested contracts, status docs, Phase 5G.1 report, and
  teacher/cache/training/runtime entry points.
- [x] Add a dependency-safe `teachers run-vggt` command with env/CLI VGGT repo
  and checkpoint resolution.
- [x] Convert real or fixture VGGT depth/pose/pointmap outputs into the stable
  teacher-signal cache format without changing the schema.
- [x] Add VGGT-focused inspection metrics for depth, relative pose, aligned ATE,
  pointmaps when present, coverage, confidence, and compact reports.
- [x] Probe the local environment for VGGT, CUDA, checkpoints, and existing TUM
  Phase 5G.1 clip caches; run real VGGT only if configured.
- [x] If VGGT is unavailable, write a precise blocker report with no fake
  predictions or generated data committed.
- [x] Add focused tests for unavailable dependency errors, CLI help, conversion,
  truth-boundary/alignment metadata, and safe relative paths.
- [x] Run required format, lint, typecheck, unittest, diff, and available make
  verification commands.
- [x] Update compact progress, decisions if needed, Phase 5H report, and next
  task.
