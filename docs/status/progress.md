# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/offline-world-builder-v0-parallel-tracer`.
- Offline V0.5 adds `python -m atlas3r offline build-world` as a connected
  tracer across frame cache, keyframes, teacher status/proposal cache,
  camera-scale ledger, consensus world state, geometry preview, object ledger,
  render diagnostics, quality report, and training-cache manifest.
- PPM input works dependency-free. MP4 and PNG/JPEG folders produce explicit
  decoder failure points until adapters exist.
- Debug flat-depth can write nonzero NPZ/PLY preview geometry, labeled
  `debug_synthetic`, not measured geometry and not training-quality.

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 27 tests.
- Passed: `git diff --check`; Git emitted Windows CRLF replacement warnings
  only.
- Passed: `python -m atlas3r --help`, `python -m atlas3r offline --help`,
  `python -m atlas3r offline build-world --help`,
  `python -m atlas3r teachers list`, and
  `python -m atlas3r smoke contracts`.
- Unavailable: `make test`, `make lint`, `make typecheck`, and `make smoke`
  because `make` is not installed on this Windows host.
- Evidence A: tiny PPM debug run wrote 288 geometry points to
  `runs/offline_v05_tiny_ppm`.
- Evidence B: local TUM RGB PNG folder wrote the full tree to
  `runs/offline_v05_real_input` with 0 geometry points and explicit decoder,
  geometry, object, render, and teacher failure points.

## Known Gaps

- No MP4, PNG, or JPEG decoder adapter is implemented yet.
- No real teacher model adapter runs yet.
- No consensus optimizer or real render-and-repair loop exists yet.
- Geometry is empty without debug flat-depth or future teacher proposals.
- Training cache is a manifest skeleton only and is not training-quality.
