# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/offline-world-builder-v06-vggt-witness`.
- Offline V0.6 wires PPM/PNG/JPG/MP4 decoding and VGGT teacher proposals
  through `python -m atlas3r offline build-world`.
- Frame cache writes stable normalized PPM copies and records decoder name,
  source URI, timestamps, quality scores, guessed K, and truth boundary.
- VGGT can run from an external runtime or replay normalized proposal caches.
- Proposal cache writes `vggt_cameras.jsonl`, `vggt_depths.npz`,
  `vggt_windows.jsonl`, and manifest counts.
- Camera/scale ledger, world state, geometry preview, diagnostics, quality
  report, and training manifest consume VGGT proposals vertically.
- VGGT output remains `teacher_pseudo`, unanchored, not measured, not
  physically accurate, and not training-quality.

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 39 tests.
- Passed: `python -m atlas3r --help`, `python -m atlas3r offline --help`,
  `python -m atlas3r offline build-world --help`,
  `python -m atlas3r teachers list`, and
  `python -m atlas3r smoke contracts`.
- Passed: `git diff --check`; Git emitted Windows CRLF replacement warnings.
- Unavailable: `make lint`, `make typecheck`, `make test`, and `make smoke`
  because `make` is not installed on this Windows host.

## Evidence

- A: `runs/offline_v06_debug_flat_depth` wrote 288 debug points and PLY.
- B: `runs/offline_v06_real_decode_no_vggt` decoded 60 TUM PNG frames with
  Pillow, selected 16 keyframes, and correctly wrote 0 geometry points.
- C: `runs/offline_v06_vggt_real_geometry` decoded 60 TUM PNG frames, selected
  24 keyframes, ran VGGT, wrote 24 camera/depth proposals, 19,800 geometry
  points, and PLY.

## Known Gaps

- VGGT scale is unanchored and not measured.
- No consensus optimizer, render-repair optimizer, object witness, or final mesh
  reconstruction exists yet.
- Training cache remains a manifest and is not training-quality.
