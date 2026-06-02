# Codex Prompt - Atlas3R Phase 3C: FramePacket Clip Builder for Student Boundary

You are working in `Yezus69/Atlas3R` after Phase 3B.1 cleanup.

Read first:

- `AGENTS.md`
- `README.md`
- `PLANS.md`
- `docs/08_API_CONTRACTS.md`
- `docs/status/active_task.md`
- `docs/status/progress.md`
- `docs/status/decisions.md`
- `docs/status/next_task.md`
- `src/atlas3r/api/contracts.py`
- `src/atlas3r/data/frame_source.py`
- `src/atlas3r/models/student/contracts.py`
- focused unit tests for frame sources and student contracts

## Goal

Add only the small stdlib + NumPy bridge that batches existing `FramePacket`
records into the Phase 3A `StudentClipInput` contract. This is contract
plumbing between RGB ingestion and the shape-only student boundary.

Do not add feature scope beyond the bridge.

## Hard Constraints

- No model training, model inference, datasets, weights, downloads, external repos, or new heavy dependencies.
- No OpenCV, PyAV, imageio, Pillow, ffmpeg, PyTorch, TensorFlow, JAX, CUDA, Metal, Core ML, or TensorRT.
- No video decoding, live camera runtime, runtime scheduler changes, TSDF changes, mapper changes, GLB/PLY export, marching cubes, web servers, notebooks, or new inspection bundles.
- Use existing `FramePacket` and `StudentClipInput`; do not create a parallel clip API.
- Preserve deterministic input order. Do not silently sort frames.
- Keep shape-only outputs marked `usable_for_mapping=false`.

## Required Slice

Add a small helper module such as:

```text
src/atlas3r/data/student_clip.py
```

Implement a typed helper:

```python
def student_clip_from_frame_packets(
    frames: Sequence[FramePacket],
    *,
    batch_id: str = "frame-packet-clip",
) -> StudentClipInput: ...
```

Requirements:

- accept a non-empty sequence of `FramePacket` records;
- reject duplicate frame IDs and malformed frame objects with field-named errors;
- stack `rgb_model` arrays into `images_rgb` shaped `1,T,3,H,W`;
- require all `rgb_model` arrays to share the same shape;
- stack validated `K_model` intrinsics into `1,T,3,3`;
- preserve frame IDs, source metadata, batch ID, coordinate frame, and truth-boundary notes;
- use only stdlib and NumPy.

## Tests

Add focused unit tests for:

- valid `FramePacket` sequence to `StudentClipInput`;
- deterministic order preservation;
- duplicate-frame rejection;
- mismatched `rgb_model` shapes rejected;
- malformed frame object rejected;
- optional Phase 3B smoke fixture into the shape-only student stub without mapping readiness;
- dependency-safe imports with no heavy ML/video/image packages.

## Docs And Verification

- Rewrite `docs/status/active_task.md` before coding with a concise Phase 3C checklist.
- Update `docs/08_API_CONTRACTS.md` with the bridge contract.
- Add a compact decision row only if the bridge becomes public API.
- Update `docs/status/progress.md` with commands run and known gaps.
- Replace `docs/status/next_task.md` with the next small Phase 3 prompt.

Run:

```bash
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p 'test_*.py'
```

If `make` exists, also run `make test`, `make lint`, and `make typecheck`.
