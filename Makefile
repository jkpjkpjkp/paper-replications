PYTHON ?= .venv/bin/python
export OPENBLAS_NUM_THREADS=2

.PHONY: test reproduce report

test:
	$(PYTHON) scripts/check_numerics.py

reproduce: test
	$(PYTHON) scripts/run_suite.py
	$(PYTHON) scripts/run_diagnostics.py
	$(PYTHON) -m scripts.run_extensions sensitivities
	$(PYTHON) -m scripts.run_extensions convergence
	$(PYTHON) -m scripts.run_surfaces
	$(PYTHON) scripts/report.py
	$(PYTHON) scripts/verify.py

report:
	$(PYTHON) scripts/report.py
	$(PYTHON) scripts/verify.py
