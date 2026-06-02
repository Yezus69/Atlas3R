# Codex Prompt - Atlas3R Phase 3C: FramePacket Clip Builder for Student Boundary

You are working in `Yezus69/Atlas3R` after Phase 3B.

Read only these files first unless a test failure requires more context:

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
- focused tests under `tests/unit/` that cover frame sources and student contracts

Do not paste large code blocks in chat. Modify files directly. Keep the task
small and dependency-free. Run tests before declaring done.

## Task goal

Add the minimal dependency-free bridge that batches existing `FramePacket`
records into the Phase 3A `StudentClipInput` contract. This is only contract
plumbing between ingestion and the shape-only student boundary. It must not add
real neural inference, training, mapper/runtime integration, video decoding, or
export.

## Hard constraints

- Do not add OpenCV, PyAV, imageio, Pillow, ffmpeg, PyTorch, TensorFlow, JAX,
  Core ML, TensorRT, CUDA, Metal, web servers, notebooks, datasets, training,
  weights, downloads, external repos, GLB/PLY export, marching cubes, TSDF
  changes, runtime scheduler changes, or new inspection bundles.
- Use only Python stdlib and NumPy.
- Reuse `FramePacket` and `StudentClipInput`; do not create a parallel clip API.
- Keep the bridge explicit about coordinate frame and truth boundary.

## Required implementation

Add a small helper module, for example:

```text
src/atlas3r/data/student_clip.py
```

Implement a typed helper such as:

```python
def student_clip_from_frame_packets(
    frames: Sequence[FramePacket],
    *,
    batch_id: str = "frame-packet-clip",
) -> StudentClipInput: ...
```

Requirements:

- accept a non-empty sequence of `FramePacket` records with unique frame IDs;
- preserve deterministic input order instead of sorting silently;
- stack `rgb_model` arrays into `images_rgb` shaped `1,T,3,H,W`;
- require all frames to share the same `rgb_model` shape;
- stack validated `K_model` into clip-shaped intrinsics `1,T,3,3`;
- preserve frame IDs and minimal source metadata;
- produce explicit path-named or field-named errors for malformed inputs.

Optionally add one tiny smoke helper if it is the smallest way to test the
bridge end to end from the Phase 3B smoke fixture into the shape-only student
stub. The smoke helper must keep `usable_for_mapping=false` and must not feed
mapper/runtime paths.

## Tests

Add focused unit tests covering:

- valid `FramePacket` sequence to `StudentClipInput`;
- deterministic order and duplicate-frame rejection;
- mismatched `rgb_model` shapes rejected;
- invalid/missing frame packet rejected;
- optional smoke path from Phase 3B fixture into the shape-only student stub
  without claiming mapping readiness;
- imports do not require heavy ML/video/image dependencies.

## Docs/status

- Rewrite `docs/status/active_task.md` before coding with a concise Phase 3C
  checklist.
- Update `docs/08_API_CONTRACTS.md` with the bridge contract and limitations.
- Append a decision entry only if the bridge becomes public API.
- Append verification results to `docs/status/progress.md`.
- Replace `docs/status/next_task.md` with the next small Phase 3 prompt.

## Verification commands

Run:

```bash
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p 'test_*.py'
```

If `make` exists, also run:

```bash
make test
make lint
make typecheck
```

Record unavailable commands and exact reasons in `docs/status/progress.md`.
