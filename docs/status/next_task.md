# Codex Prompt - Atlas3R Phase 3B: Dependency-Free RGB Frame Source Contract and NPZ/PPM Sequence Smoke

You are working in `Yezus69/Atlas3R` after Phase 3A.

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
- `src/atlas3r/models/student/contracts.py`
- focused tests under `tests/unit/` that cover API and student contracts

Do not paste large code blocks in chat. Modify files directly. Keep the task
small and dependency-free. Run tests before declaring done.

## Task goal

Define the minimal RGB frame-source boundary for future video/live ingestion and
add a deterministic dependency-free smoke path that can produce existing
`FramePacket` records from local NumPy `.npz` clips or simple binary PPM image
sequences.

This task is only ingestion contract plumbing. It must not run neural inference
or call the student model.

## Hard constraints

- Do not add OpenCV, PyAV, imageio, Pillow, ffmpeg, web servers, notebooks,
  datasets, training, weights, downloads, external repos, GLB/PLY export,
  marching cubes, TSDF/runtime scheduler changes, or new inspection bundles.
- Use only Python stdlib and NumPy.
- Reuse existing `FramePacket` and camera intrinsics contracts instead of
  creating a parallel frame API.
- Keep source files small and tests focused.

## Required implementation

Add a small frame-source package, for example:

```text
src/atlas3r/data/frame_source.py
```

Implement a minimal typed boundary such as:

```python
class RGBFrameSource(Protocol):
    def frames(self) -> Iterator[FramePacket]: ...
```

Add dependency-free helpers for:

- loading an `.npz` clip with required `rgb_u8` shaped `T,H,W,3` and `K`
  shaped `3,3` or `T,3,3`;
- loading a directory of simple binary `P6` PPM files sorted by name, with a
  supplied or sidecar intrinsics matrix;
- converting each frame into the existing `FramePacket` contract with
  deterministic `frame_id`, placeholder `timestamp_ns`, `rgb_model` shaped
  `3,H,W`, `K_model`, `K_original`, identity `resize_transform`, and metadata
  that states the source format.

Add one smoke helper or CLI subcommand only if it is the smallest way to test
the source end to end. It should write a tiny deterministic `.npz` or PPM
sequence fixture into a temp/output folder and validate the resulting
`FramePacket` objects. Do not add an inspection command or runtime scheduler
integration.

## Tests

Add focused unit tests covering:

- valid `.npz` clip to `FramePacket` conversion;
- valid PPM sequence to `FramePacket` conversion;
- invalid RGB layout/channel count rejected;
- invalid intrinsics rejected through existing helpers;
- deterministic frame ordering and frame IDs;
- imports do not require OpenCV/PyAV/Pillow/imageio or any heavy dependency;
- the frame-source path does not import or run `atlas3r.models.student`.

## Docs/status

- Rewrite `docs/status/active_task.md` before coding with a concise Phase 3B
  checklist.
- Update `docs/08_API_CONTRACTS.md` with the frame-source boundary and smoke
  fixture format.
- Append a decision entry only if the frame-source boundary becomes public API.
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
