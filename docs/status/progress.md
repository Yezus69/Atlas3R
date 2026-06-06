# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/offline-world-builder-v07-depthpro-disagreement`.
- Offline V0.7 wires Depth Pro through `offline build-world` as a second
  teacher-pseudo geometry witness beside VGGT.
- Depth Pro can run from an external `depth_pro` package/repo or replay
  normalized proposal caches without import-time heavy dependencies.
- Proposal cache writes VGGT and Depth Pro camera/depth streams plus manifest
  counts.
- Camera/scale ledgers, world state, quality report, render diagnostics, and
  training manifest record both witness sources and still block physical
  accuracy/training-quality claims.
- Disagreement artifacts are written under `diagnostics/teacher_disagreement.json`,
  `diagnostics/disagreement_maps.npz`, and `diagnostics/consensus_preview.npz`.
- Geometry preview uses VGGT pose plus diagnostic consensus depth when both
  witnesses exist, falls back to VGGT depth for VGGT-only runs, and stays empty
  for Depth-Pro-only global previews.

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 50 tests.
- Passed: `python -m atlas3r --help`, `python -m atlas3r offline --help`,
  `python -m atlas3r offline build-world --help`,
  `python -m atlas3r teachers list`, and
  `python -m atlas3r smoke contracts`.

## Evidence

- A: `runs/offline_v07_debug_flat_depth` decoded 6 PPM frames, selected 6
  keyframes, wrote 288 debug points and PLY.
- B: `runs/offline_v07_real_decode_no_teachers` decoded 60 TUM PNG frames with
  Pillow, selected 16 keyframes, and correctly wrote 0 geometry points.
- C: `runs/offline_v07_vggt_only` decoded 60 TUM PNG frames, selected 24
  keyframes, wrote 24 VGGT camera/depth proposals, 19,800 points, and PLY.
- D: `runs/offline_v07_depthpro_only` decoded 60 TUM PNG frames, selected 24
  keyframes, wrote 24 Depth Pro camera/depth proposals, and wrote 0 global
  geometry because global pose is missing.
- E: `runs/offline_v07_vggt_depthpro_disagreement` decoded 60 TUM PNG frames,
  selected 24 keyframes, wrote 24 VGGT and 24 Depth Pro depth proposals,
  7,372,800 valid overlap pixels, 18,432 geometry points, and PLY.

## Known Gaps

- VGGT and Depth Pro scales are unanchored teacher proposals.
- The consensus preview is diagnostic only; no optimizer has adjusted depth
  scale/bias, intrinsics, or poses.
- No physical scale anchor, render-repair optimizer, object permanence, final
  mesh reconstruction, or named evaluation report exists.
- Training cache remains a manifest and is not training-quality.
