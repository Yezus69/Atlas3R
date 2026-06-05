# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/offline-world-builder-repo-reset`.
- The repository is being reset around Offline World Builder as the immediate
  foundation.
- Old SMGT-tiny, SMGT-small-v2, student training, student runtime mapping,
  measured live replay, TUM training/eval, and phase-report surfaces are removed
  from the active tree.
- The kept source foundation is dependency-safe contracts, input/video
  primitives, teacher witness status registry, an Offline V0 report skeleton,
  and minimal map artifact writers.

## Verification

Verification for the reset is recorded in
`docs/status/repo_reset_report.md`.

## Known Gaps

- No MP4 decoder is implemented yet.
- No teacher model adapter runs yet.
- No consensus optimizer exists yet.
- No mesh, voxel, occupancy, or training cache is produced from real input yet.
- No realtime or accuracy claim exists.
