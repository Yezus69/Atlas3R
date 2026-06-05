Core Phase B - Object/Dynamic Teacher Labels And Object-Aware Sparse Fusion

Goal: extend the diagnostic RGB student mapping path with explicit object and
dynamic-scene supervision/fusion boundaries, without claiming final RGB-only
readiness or hidden-geometry completion.

Start from branch `codex/core-smgt-tiny-student-map`. Reload `README.md`,
`PLANS.md`, `docs/01_SYSTEM_ARCHITECTURE.md`, `docs/08_API_CONTRACTS.md`,
`docs/09_EVALUATION.md`, `docs/status/progress.md`,
`docs/status/decisions.md`, `docs/status/active_task.md`, and
`docs/status/core_smgt_tiny_student_map_report.md`.

Context:

- Core Phase A trained a learned diagnostic `SMGTTiny` from the Phase 6H VGGT
  teacher temporal cache and mapped 64 Freiburg RGB frames without VGGT at
  student inference.
- The student output is observed-only sparse TSDF mesh chunks with unverified
  RGB-prior scale, no object-aware fusion, no dynamic filtering, no loop
  closure, and no realtime or benchmark accuracy claim.
- Local evidence paths, when present:
  `runs/core_smgt_tiny_weighted_freiburg1_xyz_val/checkpoint_best.pt` and
  `runs/core_smgt_tiny_student_map_freiburg1_xyz_val`.

Required work:

- Add a compact object/dynamic teacher-label cache contract or adapter boundary
  that can ingest per-frame masks/object IDs/dynamic probabilities without
  vendoring external model weights.
- Extend training data adapters and losses so `SMGTTiny` can consume optional
  object/dynamic labels while keeping missing labels unsupervised rather than
  treating absence as background truth.
- Add mapper-side object/dynamic fields to the RGB student `DepthObservation`
  conversion path, with dynamic pixels excluded or downweighted from the static
  sparse TSDF map.
- Emit object/dynamic diagnostics in summaries, reports, mesh chunk metadata, and
  tests while preserving conservative truth flags.
- Run a synthetic or tiny real diagnostic that proves object IDs or dynamic
  masks affect fusion behavior without poisoning the static map.

Acceptance:

- Contracts and tests cover optional object labels, dynamic masks, missing-label
  behavior, and mapper truth flags.
- The student runtime still works without optional object/dynamic labels.
- No optional external segmentation/tracking dependency imports at module import
  time.
- Verification includes format, lint, typecheck, focused tests, full unittest
  discovery, `git diff --check`, and one object/dynamic fusion evidence command.
