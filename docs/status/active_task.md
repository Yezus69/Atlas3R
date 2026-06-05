# Active Task - Offline V0.5 Parallel Tracer

Branch: `codex/offline-world-builder-v0-parallel-tracer`

## Goal

Build the first dependency-safe vertical Offline World Builder tracer:
`python -m atlas3r offline build-world` must run ingestion, keyframes,
teacher witness status/proposal cache, camera-scale ledger, consensus world
state, geometry preview, object ledger, render/repair diagnostics, quality
report, and training-cache skeleton in one command.

## Checklist

- [x] Confirm clean worktree on reset branch and create working branch.
- [x] Read required objective, architecture, truth, consensus, API, quality,
  and status docs.
- [x] Expand concise docs for V0.5 module contracts and vertical-slice rule.
- [x] Implement connected offline pipeline modules with explicit artifacts and
  failure points.
- [x] Add focused tests for frame cache, keyframes, proposal cache, consensus,
  geometry preview, object ledger, render/repair, CLI, and full tracer.
- [x] Run dependency-free tiny PPM evidence command with debug flat-depth.
- [x] Search local ignored inputs and run real-input evidence command if
  available.
- [x] Run format, lint, typecheck, unit, CLI, smoke, diff, and make checks as
  available.
- [x] Update progress, current state, next task, and V0.5 report.
- [x] Confirm generated runs/data are not staged and commit.

## Stop Conditions

- Unknown user changes appear in the worktree.
- `import atlas3r` becomes dependent on heavy optional packages.
- PPM/image-folder input cannot be supported dependency-free.
- The package cannot import without rewriting the reset foundation.
