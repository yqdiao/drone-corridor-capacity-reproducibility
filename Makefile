PYTHON ?= python3

.PHONY: reproduce figures checks audit verify

reproduce: figures checks audit
	$(MAKE) verify

figures:
	$(PYTHON) checks/generate_numerical_figures.py

checks:
	$(PYTHON) checks/run_checks.py --json checks/results.json

audit:
	$(PYTHON) checks/check_additions.py

verify:
	$(PYTHON) checks/verify_outputs.py
