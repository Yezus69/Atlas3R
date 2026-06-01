.PHONY: format lint typecheck test smoke profile

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
	$(PYTHON) -m atlas3r smoke --help

profile:
	$(PYTHON) -m atlas3r profile --help
