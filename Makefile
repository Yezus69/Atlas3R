.PHONY: format lint typecheck test smoke profile

PYTHON ?= python
export PYTHONPATH := src

format:
	$(PYTHON) -m ruff format src tests

lint:
	$(PYTHON) -m ruff format --check src tests
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

smoke:
	$(PYTHON) -m atlas3r smoke contracts
	$(PYTHON) -m atlas3r teachers list

profile:
	$(PYTHON) -m atlas3r offline --help
