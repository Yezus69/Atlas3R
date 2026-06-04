# Phase 5H VGGT Pose/Pointmap Teacher Report

Branch: `codex/phase5h-vggt-pose-pointmap-teacher`

Base branch: `codex/phase5g1-multisequence-tum-generalization`

## Result

Phase 5H added a dependency-safe real VGGT runner boundary and focused VGGT
teacher evaluation helpers, but real VGGT execution is blocked in this
environment. No VGGT predictions, teacher caches, `.npz` payloads, PLY files,
model weights, checkpoints, or generated run folders were committed.

Conclusion: blocked, not useful yet. The blocker is external dependency
availability, not a measured teacher-quality failure.

## Environment Availability

Local probes:

- `importlib.util.find_spec("vggt")`: `None`
- `ATLAS3R_VGGT_REPO`: unset
- `ATLAS3R_VGGT_CHECKPOINT`: unset
- PyTorch: importable
- CUDA: available
- CUDA device count: 3
- First CUDA device: `NVIDIA GeForce RTX 4090`

Available Phase 5G.1 TUM inputs:

- `data/tum_rgbd/freiburg1_xyz_phase5g1_clip_cache_val`
- `data/tum_rgbd/freiburg1_xyz_phase5g1_measured_teacher_val`
- matching Phase 5G.1 train/val caches for `freiburg1_desk` and
  `freiburg2_xyz`

## Code Added

- `atlas3r teachers run-vggt` public command.
- `VGGTRunConfig` and `VGGTExternalTeacherRunner`.
- Optional VGGT loader that accepts an importable `vggt` package or
  `ATLAS3R_VGGT_REPO`, and an optional CLI/env checkpoint.
- Conversion from real or fixture VGGT depth/confidence/pose/intrinsic/pointmap
  outputs into the existing teacher-signal cache format.
- Diagnostic `diagnostic_sim3` and `diagnostic_se3` source-pose alignment, with
  explicit metadata that aligned pseudo-labels are not measured geometry.
- VGGT-specific JSON/JSONL/Markdown evaluation for depth, ATE-like camera-center
  error, RPE-like relative pose error, optional pointmaps, coverage, and
  confidence.

The implementation uses the existing teacher-signal cache schema. No new cache
schema was added.

## Real Run Attempt

Command:

```bash
python -m atlas3r teachers run-vggt \
  --clip-cache data/tum_rgbd/freiburg1_xyz_phase5g1_clip_cache_val \
  --output runs/phase5h_vggt_freiburg1_xyz_val \
  --device cuda \
  --max-clips 1 \
  --align-to-source-pose diagnostic_sim3
```

Result:

- Exit code: `2`
- Error: `VGGT is unavailable: missing optional dependency: vggt; set ATLAS3R_VGGT_REPO or install VGGT. Install VGGT in the active environment or set ATLAS3R_VGGT_REPO to a local VGGT checkout; keep VGGT code and weights outside Atlas3R.`
- `runs/phase5h_vggt_freiburg1_xyz_val`: not created

## Teacher Metrics

No teacher cache was generated because VGGT was unavailable. Therefore:

- Teacher cache count: 0
- Teacher-vs-measured depth metrics: not run
- Teacher-vs-measured pose metrics: not run
- Pointmap metrics: not run
- Quality gates: not evaluated

## Student Training

Not run. The task requires training only if a real VGGT teacher passes at least
one quality gate against measured TUM depth/pose/pointmap evidence. There is no
real VGGT teacher evidence in this environment.

## Verification

Commands run:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m unittest tests.unit.test_external_teachers -v
python -m mypy src
python -m unittest discover -s tests -p "test_*.py"
git diff --check
where.exe make
```

Results:

- `python -m unittest tests.unit.test_external_teachers -v`: passed 10 tests.
- `python -m ruff format src tests`: passed; 152 files left unchanged.
- `python -m ruff format --check src tests`: passed; 152 files already
  formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 115 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 201 tests; the
  pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `where.exe make`: no `make` found, so `make test`, `make lint`, and
  `make typecheck` were unavailable in this Windows shell.

## Next Task

Install or point Atlas3R at a real VGGT checkout and checkpoint outside this
repo, then rerun `atlas3r teachers run-vggt` on the Phase 5G.1 TUM validation
caches. Only train the temporal student if the resulting VGGT cache improves a
depth, relative-pose, or pointmap quality gate against measured TUM data.
