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
	$(PYTHON) -m atlas3r smoke synthetic-cube-room --output build/smoke/synthetic_cube_room.atlas3r

inspect:
	$(PYTHON) -m atlas3r smoke synthetic-cube-room --output build/smoke/synthetic_cube_room.atlas3r
	$(PYTHON) -m atlas3r inspect session --input build/smoke/synthetic_cube_room.atlas3r --output build/smoke/preview

profile:
	$(PYTHON) -m atlas3r profile --help
