# Active Task - Core SMGT Tiny A2 Generalization Gauntlet

Branch: `codex/core-smgt-tiny-generalization-gauntlet`

## Goal

Harden and falsify SMGT-tiny before object/dynamic fusion: confidence-gated
mapping, strict train/val/heldout teacher-cache discipline, stable training
metrics, heldout RGB-only mapping, long-run diagnostics, and an honest A2
generalization report.

## Checklist

- [x] Confirm clean worktree and switch from `codex/core-smgt-tiny-student-map`.
- [x] Read required architecture, contract, evaluation, status, model, training,
  runtime, mapping, and test files.
- [x] Add confidence/sigma/dynamic valid-mask policy for `map-rgb-student`.
- [x] Add focused confidence-gating tests and preserve conservative truth flags.
- [x] Add deterministic train/val/heldout split manifest utilities and tests.
- [x] Extend SMGT-tiny training CLI/run with heldout split and stable metric
  reporting.
- [x] Train/evaluate SMGT-tiny on strict non-overlapping split.
- [x] Run heldout RGB-only student mapping without VGGT at inference.
- [x] Run long-run memory/latency/map-growth diagnostic.
- [x] Write `docs/status/core_smgt_tiny_a2_generalization_report.md` from
  measured evidence only.
- [x] Update status/architecture/contracts/evaluation docs after evidence.
- [x] Run full verification and record exact commands/results.
- [x] Commit with message `test(models): harden smgt tiny heldout rgb mapping`.

## Stop Conditions

- Unknown user changes appear in the worktree.
- No real heldout RGB input or teacher cache exists and one cannot be generated
  locally.
- CUDA/Torch is unavailable and CPU training is infeasible.
