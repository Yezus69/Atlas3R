# Atlas3R

Atlas3R is being cleansed and pivoted into a docs-first baseline for a
Scale-Aware Monocular Reconstruction Teacher.

The target system is:

```text
RGB video
  -> video quality and reconstructability gate
  -> one video geometry backbone
  -> scale-aware global refinement
  -> static/dynamic separation
  -> ray-based TSDF and occupancy fusion
  -> metric-quality gate
```

This branch intentionally contains no runtime implementation. The old Python
package, tests, build config, and phase-history docs were removed so the next
implementation can start from the pivot architecture instead of carrying stale
code forward.

## Current Status

- Active local branch: `pivot/scale-aware-reconstruction-teacher`.
- This is a docs-only pivot baseline.
- No model adapters, CLI commands, package code, tests, or mapper runtime exist
  on this branch yet.
- Branch refs are limited to the local pivot branch and the matching
  `origin/pivot/scale-aware-reconstruction-teacher` tracking ref.
- Local ignored `data/` and `runs/` directories were deleted to prevent old
  captures or generated outputs from contaminating the pivot.

## Pivot Boundary

The teacher must not claim physically measured metric geometry from arbitrary
unanchored monocular RGB. The intended behavior is:

```text
produce metric pseudo-labels only when scale is identifiable enough;
otherwise reject the video or mark outputs as non-metric pseudo-labels
```

Every future geometry output must carry confidence, uncertainty, scale source,
and acceptance status. Unknown space is not free space. Predicted completion is
not measured geometry.

## Target Stack

- Primary geometry backbone: ViPE with the DA3 pipeline.
- Optional hard-video fallback: MegaSaM.
- Mask grouping and propagation: SAM2.
- Optional semantic/anchor prompts: Grounded-SAM2.
- Core project value: the optimizer, validator, and ray-based mapper that turn
  model proposals into accepted or rejected teacher labels.

Third-party repos and model weights must stay outside this repository. Future
code should use adapters and explicit external paths.

## Repository Map

- `AGENTS.md`: operating rules for Codex on this pivot.
- `PLANS.md`: short milestone plan for the pivot.
- `docs/00_PIVOT_OBJECTIVE.md`: pivot objective and truth boundary.
- `docs/01_TEACHER_ARCHITECTURE.md`: target architecture summary.
- `docs/08_API_CONTRACTS.md`: concise draft contract index.
- `docs/status/`: compact current state, decisions, active task, and next task.

## No Implementation Yet

There is no `python -m atlas3r`, no `make test`, and no importable package on
this branch. The next implementation session should create the smallest tested
contract scaffold first, then build vertical slices from those contracts.
