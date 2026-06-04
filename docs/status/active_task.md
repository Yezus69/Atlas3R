# Active Task - Phase 5G.1 Multi-Sequence TUM Generalization

Goal: train and evaluate the existing Phase 5G SE(3) temporal student on
multiple measured TUM RGB-D sequences, then report whether it generalizes.

Branch: `codex/phase5g1-multisequence-tum-generalization`

Checklist:

- [x] Start from `codex/phase5g-multisequence-pose-odometry` and create the
  Phase 5G.1 branch.
- [x] Read the requested contracts, status docs, Phase 5G report, and relevant
  training/runtime files.
- [x] Remove the sequence-equality blocker for multi-sequence teacher-signal
  training while preserving single-cache behavior.
- [x] Add focused tests for multi-sequence teacher-cache compatibility and
  summary counts.
- [x] Download or reuse at least two additional supported TUM RGB-D sequences
  if available.
- [x] Prepare block-split manifests, 160x120 clip caches, and measured teacher
  caches for available sequences.
- [x] Run real CUDA/AMP multi-sequence training if CUDA is available; otherwise
  record the blocker and run a CPU smoke.
- [x] Evaluate `student-odometry` per validation sequence and oracle pose on at
  least one sequence.
- [x] Run required format, lint, typecheck, unit, diff, and make verification
  commands.
- [x] Write the Phase 5G.1 generalization report and compact status updates.
