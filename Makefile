.PHONY: format lint typecheck test smoke inspect profile

PYTHON ?= python
export PYTHONPATH := src

format:
	$(PYTHON) -m ruff format src tests

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

smoke:
	$(PYTHON) -m atlas3r smoke tsdf-cube-room --output build/smoke/tsdf_cube_room --write-mesh-sidecar
	$(PYTHON) -m atlas3r smoke synthetic-cube-room --output build/smoke/synthetic_cube_room.atlas3r
	$(PYTHON) -m atlas3r adapters run --adapter fixture-cube-room --input build/smoke/synthetic_cube_room.atlas3r --output build/smoke/teacher_cache --store-arrays
	$(PYTHON) -m atlas3r smoke teacher-cache-tsdf --input build/smoke/teacher_cache --output build/smoke/teacher_cache_tsdf --write-mesh-sidecar

inspect:
	$(PYTHON) -m atlas3r smoke synthetic-cube-room --output build/smoke/synthetic_cube_room.atlas3r
	$(PYTHON) -m atlas3r inspect session --input build/smoke/synthetic_cube_room.atlas3r --output build/smoke/preview
	$(PYTHON) -m atlas3r adapters run --adapter fixture-cube-room --input build/smoke/synthetic_cube_room.atlas3r --output build/smoke/teacher_cache --store-arrays
	$(PYTHON) -m atlas3r inspect teacher-cache --input build/smoke/teacher_cache
	$(PYTHON) -m atlas3r smoke teacher-cache-tsdf --input build/smoke/teacher_cache --output build/smoke/teacher_cache_tsdf --write-mesh-sidecar

profile:
	$(PYTHON) -m atlas3r profile --help
