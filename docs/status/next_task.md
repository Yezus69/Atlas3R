# Codex Prompt - Atlas3R Phase 3A: Student Model Contract and Shape-Only Forward Skeleton

You are working in `Yezus69/Atlas3R` after Phase 2E.

Read only these files first unless a test failure requires more context:

- `AGENTS.md`
- `PLANS.md`
- `docs/08_API_CONTRACTS.md`
- `docs/status/active_task.md`
- `docs/status/progress.md`
- `docs/status/decisions.md`
- focused API/model adapter tests under `tests/unit/`
- focused synthetic runtime/teacher-cache tests under `tests/synthetic/` only if
  they are needed to preserve existing contracts.

Do not paste large code blocks in chat. Modify files directly. Keep changes
small. Preserve all existing CLI commands and artifacts. Run tests before
declaring done.

## Task goal

Start Phase 3 by defining the public student model boundary and a dependency-safe
shape-only forward skeleton for the Streaming Metric Geometry Transformer MVP.
This is a contract and tensor-shape slice only.

Do not add training loops, datasets, downloads, model weights, third-party model
repos, web servers, notebooks, video decoding, GLB/PLY export, CUDA-specific
kernels, real-time performance claims, or accuracy claims.

## Required implementation

Add a small student-model contract under `src/atlas3r/models/student/` that
captures:

- input clip/frame IDs, image tensor shape, camera intrinsics availability,
  optional previous pose/context metadata, and coordinate frame;
- output per-frame depth, depth uncertainty, confidence, pose, intrinsics,
  normals, pointmap, dynamic/static mask, optional object embedding, and dense
  match summary fields;
- explicit batch, time, channel, height, and width shape validation;
- confidence and uncertainty fields from day one;
- dependency-safe behavior when optional tensor/model dependencies are missing.

Add a shape-only forward skeleton that validates inputs and returns deterministic
placeholder outputs with the right shapes and truth-boundary metadata. The
skeleton must not claim neural inference, learned geometry, speed, or accuracy.

If a tensor library is used, keep imports dependency-safe and raise a clear
installation/configuration error when unavailable. Prefer the smallest slice
that can be tested without downloading weights.

## Documentation and status

- Rewrite `docs/status/active_task.md` before coding with a concise Phase 3A
  checklist.
- Update `docs/08_API_CONTRACTS.md` with the student model input/output
  contracts and shape-only skeleton truth boundary.
- Append to `docs/status/decisions.md` if the student contract becomes a new
  public format/API.
- Append results to `docs/status/progress.md` after verification.
- Replace `docs/status/next_task.md` with the next Phase 3 prompt before
  declaring done.

## Verification commands

Run:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p 'test_*.py'
```

If `make` is available, also run:

```bash
make test
make lint
make typecheck
make smoke
make inspect
```

Record unavailable commands and exact reasons in `docs/status/progress.md`.
