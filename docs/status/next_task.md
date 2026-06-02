# Codex Prompt - Atlas3R Phase 3D: FrameSource to Teacher FrameBatch Runner Boundary

You are working in `Yezus69/Atlas3R` after Phase 3C.

## Goal

Build the smallest dependency-free bridge from existing `RGBFrameSource` /
`FramePacket` sequences into the existing teacher `FrameBatch` contract so real
external adapters can later consume RGB frame sources.

This is only contract plumbing:

```text
RGBFrameSource / FramePacket sequence
  -> FrameBatch
  -> existing dependency-safe teacher adapter runner boundary
```

## Hard Constraints

- Use only Python stdlib and NumPy.
- Reuse existing `FramePacket`, `RGBFrameSource`, and `FrameBatch`; do not create
  a parallel teacher clip API.
- No real adapter inference, model training, datasets, downloads, external repos,
  weights, video decoding, live camera runtime, runtime scheduler changes, mapper
  changes, TSDF changes, GLB/PLY export, marching cubes, web servers, notebooks,
  or new inspection bundles.
- Do not add OpenCV, PyAV, imageio, Pillow, ffmpeg, PyTorch, TensorFlow, JAX,
  CUDA, Metal, Core ML, TensorRT, or other heavy dependencies.

## Required Slice

- Add a small helper that converts a non-empty ordered `FramePacket` sequence
  into the existing teacher `FrameBatch`.
- Prefer a tiny `RGBFrameSource` wrapper helper only if it keeps tests clearer.
- Preserve input order exactly; do not silently sort.
- Reject non-`FramePacket` items and duplicate `frame_id` values with clear
  field-named errors.
- Keep metadata compact and deterministic.
- Do not change mapper/runtime/TSDF consumption paths.

## Tests

- Valid smoke fixture frames convert to `FrameBatch`.
- Frame ID order is preserved.
- Duplicate frame IDs, empty sequences, and non-packet items are rejected.
- Existing dependency-safe fixture teacher adapter or runner boundary accepts the
  `FrameBatch` without real external model inference.
- Imports do not load heavy ML/video/image dependencies.
- Context-budget tests still pass.

## Verification

Run:

```bash
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p 'test_*.py'
git diff --check
```

If `make` exists, also run:

```bash
make test
make lint
make typecheck
```
