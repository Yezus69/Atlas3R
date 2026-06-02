# Phase 4C Real-Data Run Report

Branch: `codex/overnight-realdata-tum-rgbd`

Code commit: `0c7b31856e41c87a8da2a25fac4bd2c3c97d16e2`

## Commands Run

- `python -m pip install -e ".[dev,train]"`: passed.
- `python -m ruff format src tests`: passed.
- `python -m ruff format --check src tests`: passed.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed.
- `python -m unittest discover -s tests -p "test_*.py"`: passed, 146 tests.
- `python -m atlas3r train tum-rgbd-depth-pose --help`: passed.
- CUDA probe: Torch `2.1.0+cu121`, CUDA available, 3 devices, first device
  `NVIDIA GeForce RTX 4090`.
- `python -m atlas3r datasets tum-rgbd download --sequence freiburg1_xyz
  --output data/tum_rgbd`: failed before extraction with
  `[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate`.

## Real-Data Result

- Download succeeded: no.
- Manifest frame counts: not available; manifest preparation did not run.
- Long CUDA training: not run because official TUM download failed.
- Final/best train and validation metrics: not available.
- Checkpoint and preview paths: not produced.
- Generated checkpoints/run folders committed: no; ignored by `.gitignore`.

## Blocker

Python stdlib HTTPS certificate verification cannot validate
`https://cvg.cit.tum.de` in this environment. The next unlock is fixing local
certificate trust for Python stdlib HTTPS or placing the official
`rgbd_dataset_freiburg1_xyz` files under ignored `data/tum_rgbd/`, then running
the prepare and CUDA training commands from the Phase 4C prompt.

## Next Best Task

Once official data is available, run manifest preparation and the CUDA overnight
training command with `--max-runtime-minutes 480`, then run the Phase 4D real
checkpoint inference and mapping diagnostic prompt only after a real checkpoint
exists.
