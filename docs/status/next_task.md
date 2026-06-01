# Next task

```text
# Codex Prompt - Atlas3R Phase 0B: Deterministic Synthetic Cube-Room Generator

You are working in the existing Atlas3R repo after Phase 0A. Reload `README.md`,
`PLANS.md`, `docs/08_API_CONTRACTS.md`, and `docs/status/*` before coding.
Read any additional docs only if they are directly relevant.

Task goal:
Implement Phase 0B: a deterministic synthetic cube-room generator and
`atlas3r smoke synthetic-cube-room`. Do not implement neural models, teacher
adapters, TSDF fusion, learned mesh extraction, or runtime video capture yet.

Required implementation:
- Create a synthetic cube-room scene with known camera intrinsics, camera poses,
  analytic depth maps, object masks, and a ground-truth triangle mesh.
- Use the Phase 0A coordinate convention: meters; camera frame x right, y down,
  z forward; `T_A_B` maps homogeneous points from frame B into frame A.
- Use the Phase 0A contracts and validation helpers for cameras, poses,
  mesh chunks, and world/session metadata.
- Add projection/unprojection integration tests using
  `atlas3r.camera.pinhole` and `atlas3r.pose.transforms`.
- Add a smoke command:
  `atlas3r smoke synthetic-cube-room --output <folder>`
  which writes a tiny `.atlas3r` session folder containing metadata, poses,
  cameras, optional depth `.npz` files, object records, and mesh chunk metadata.
- Keep outputs deterministic with fixed scene parameters and no global random
  state. If randomness is useful, require an explicit seed.

Required tests:
- Synthetic intrinsics and poses validate with Phase 0A contracts.
- Analytic depth agrees with projection/unprojection for selected pixels.
- Object masks align with the generated object geometry.
- Ground-truth mesh validates as `MeshChunk`/`WorldMap` data.
- Smoke command creates the expected session folder files.

Verification commands:
- `python -m ruff format --check src tests`
- `python -m ruff check src tests`
- `python -m mypy src`
- `python -m unittest discover -s tests -p test_*.py`
- If available: `make test`, `make lint`, `make typecheck`, and `make smoke`.

After coding:
- Update `docs/status/active_task.md`.
- Append commands/results/known gaps to `docs/status/progress.md`.
- Update `docs/status/decisions.md` only for interface or coordinate decisions.
```
