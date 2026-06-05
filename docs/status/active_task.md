# Active Task - Offline World Builder Repo Reset

Branch: `codex/offline-world-builder-repo-reset`

## Goal

Aggressively reset Atlas3R around the Offline World Builder direction:
MP4/RGB input to optimized offline 3D world, inspectable map artifacts, and a
future training cache. Do not keep stale SMGT training/runtime surfaces.

## Checklist

- [x] Confirm clean worktree on `codex/core-smgt-small-v2-measured-pseudo`.
- [x] Create `codex/offline-world-builder-repo-reset`.
- [x] Read current README, PLANS, API contracts, and compact status files.
- [x] Record before-cleanup inventory.
- [x] Delete stale training, runtime, student, phase-report, config, and test code.
- [x] Rewrite docs around Offline World Builder only.
- [x] Keep only dependency-safe contracts, input primitives, teacher boundaries,
  and minimal map artifact inspection utilities.
- [x] Replace tests with focused import, contract, input, teacher, artifact, and
  CLI coverage.
- [x] Remove generated local junk and update `.gitignore`.
- [x] Prune safe merged local stale branches and write prune suggestions.
- [x] Run verification commands and document results.
- [x] Commit with `chore(repo): reset around offline world builder`.

## Stop Conditions

- Unknown user changes appear in the worktree.
- The uploaded Offline World Builder architecture text exists only outside git
  and would be deleted.
- The package cannot import after cleanup without rewriting the repo from
  scratch.
