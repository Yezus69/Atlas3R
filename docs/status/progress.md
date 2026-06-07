# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active local branch: `pivot/scale-aware-reconstruction-teacher`.
- Repository has been cleansed to a docs-only pivot baseline.
- Old tracked Python implementation, tests, Python build config, and phase
  reports were removed from the working tree.
- Local non-pivot branches were deleted.
- Non-pivot local remote-tracking refs were deleted; branch refs now show the
  local pivot branch and matching `origin/pivot/scale-aware-reconstruction-teacher`.
- Hosted non-pivot remote branches were not deleted; they can reappear after a
  fetch if they still exist on the host.
- Ignored local `data/` and `runs/` directories were deleted after the user
  requested zero carryover from old captures or generated outputs.
- The old `python -m http.server 8765` viewer process was stopped so `runs/`
  could be removed.
- The pivot source note is
  `C:/Users/Asav/Downloads/DESIGN_PIVOT.md`.

## Pivot Direction

Atlas3R is now oriented around a Scale-Aware Monocular Reconstruction Teacher:

```text
RGB video
  -> reconstructability gate
  -> ViPE/DA3 geometry backbone
  -> SAM2 mask grouping
  -> scale-aware robust optimizer
  -> ray-based TSDF and occupancy fusion
  -> validation and metric gate
```

## Verification

- Passed: `git status --short --branch`; branch is
  `pivot/scale-aware-reconstruction-teacher` with staged follow-up docs for
  deleting `data/` and `runs/`.
- Passed: `git branch --all --verbose --no-abbrev`; only the local pivot branch
  and matching origin pivot tracking ref are listed.
- Passed: `rg --files`; tracked working inventory is README, AGENTS, PLANS, and
  seven pivot/status docs.
- Passed: `git diff --cached --check`.
- Passed: `git ls-files`; tracked files are `.gitignore` plus ten pivot
  documentation files.
- Passed: `Test-Path data` and `Test-Path runs`; both are absent.

## Known Gaps

- No runtime implementation exists on this branch.
- No package scaffold, tests, CLI, or make targets exist yet.
- Hosted remote branches were not pruned from GitHub or any other remote.
- Non-pivot remote-tracking refs can reappear if `git fetch` is run before
  hosted branch deletion.
- Contract docs are draft architecture contracts, not implemented APIs.
