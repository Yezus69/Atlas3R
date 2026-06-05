Offline V0: implement MP4 ingestion, keyframe extraction, teacher proposal cache schemas, and quality report skeleton.

Start from branch `codex/offline-world-builder-repo-reset`. Reload
`README.md`, `PLANS.md`, `docs/00_PROJECT_OBJECTIVE.md`,
`docs/01_OFFLINE_WORLD_BUILDER_ARCHITECTURE.md`,
`docs/02_DATA_AND_TRUTH_BOUNDARY.md`,
`docs/03_TEACHER_CONSENSUS_AND_OPTIMIZATION.md`,
`docs/04_API_CONTRACTS.md`, `docs/05_EVALUATION_AND_QUALITY.md`, and
`docs/status/*`.

Goal: build Offline V0 only. Add MP4 ingestion behind a dependency-safe adapter,
keyframe extraction metadata, teacher proposal cache schemas, and a quality
report skeleton. Do not train a student and do not restore old runtime mapper
commands.

Acceptance:

- `import atlas3r` remains free of Torch, OpenCV, teacher model, Open3D, and
  external reconstruction imports.
- MP4 decoding has explicit dependency errors and install hints when unavailable.
- Keyframe metadata stores frame IDs, timestamps, camera metadata, source paths,
  and truth boundaries.
- Teacher proposal cache records Depth Pro/VGGT/COLMAP-style proposals without
  making them truth.
- Quality report skeleton includes teacher disagreement, reprojection error,
  render mismatch, scale source, surface confidence, object consistency,
  observed vs predicted geometry, completeness, and failure modes.
- Full verification passes: format, lint, typecheck, unittest discovery, and
  `git diff --check`.
