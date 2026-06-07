# Active Task - Pivot Repository Cleanse

## Goal

Reset Atlas3R to a docs-only pivot baseline for the scale-aware monocular
reconstruction teacher described in `C:/Users/Asav/Downloads/DESIGN_PIVOT.md`.

## Checklist

- [x] Rename current branch to `pivot/scale-aware-reconstruction-teacher`.
- [x] Prune local non-pivot branches and non-pivot remote-tracking refs.
- [x] Remove stale `src/`, `tests/`, old phase docs, Python build config, and
      local tool caches.
- [x] Delete ignored local `data/` and `runs/` directories.
- [x] Replace README, PLANS, AGENTS, contracts, progress, decisions, and next
      task docs with pivot-aligned docs.
- [x] Verify the file inventory and git status.

## Notes

- Hosted remote branches were intentionally left untouched.
- Non-pivot remote-tracking refs were deleted, so `git branch --all` shows only
  the local pivot branch and matching origin pivot tracking ref.
- Ignored local `data/` and `runs/` directories were deleted after the user
  requested a fully clean pivot workspace.
- The old `python -m http.server 8765` viewer process was stopped because it
  held a lock on `runs/`.
- No runtime implementation was added.

## Verification Commands

- Passed: `git status --short --branch`
- Passed: `git branch --all --verbose --no-abbrev`
- Passed: `rg --files`
- Passed: `git diff --cached --check`
- Passed: `Test-Path data` and `Test-Path runs` both returned `False`
