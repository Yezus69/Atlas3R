# Active Task - Offline V0.6 VGGT Witness

Branch: `codex/offline-world-builder-v06-vggt-witness`

## Checklist

- [x] Confirm clean worktree and create V0.6 branch from V0.5 tracer.
- [x] Read required architecture, truth-boundary, status, source, and test files.
- [x] Add dependency-safe PNG/JPG and MP4 frame decoding through frame cache.
- [x] Add dependency-safe VGGT witness runtime, cache replay, and unavailable status.
- [x] Normalize VGGT camera/depth proposals into proposal-cache artifacts.
- [x] Add minimal overlap Sim3 stitching and rejection metadata.
- [x] Lift teacher depth/K/`T_world_camera` into geometry preview with truth flags.
- [x] Propagate VGGT proposal state through ledgers, world state, diagnostics, quality, and training cache.
- [x] Add focused unit and vertical build-world tests.
- [x] Run required verification and ignored evidence runs.
- [x] Update concise docs/status report and commit.

## Truth Boundary

VGGT output is `teacher_pseudo`, unanchored, observed-only proposal geometry.
It is not measured geometry, not physically accurate, and not training-quality.
