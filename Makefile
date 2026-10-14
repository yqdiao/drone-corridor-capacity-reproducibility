PYTHON ?= python3

.PHONY: reproduce figures checks verify

reproduce: figures checks verify

figures:
	$(PYTHON) checks/generate_numerical_figures.py

checks:
	$(PYTHON) checks/run_checks.py --json checks/results.json

verify:
	$(PYTHON) checks/verify_outputs.py
