PY ?= .venv/bin/python

.PHONY: help install lock test bench figures clean exp-establishment exp-immune exp-immune-collapse exp-resistance

help:
	@echo "install            create .venv with uv and install requirements.txt"
	@echo "lock               write requirements-lock.txt from the current .venv"
	@echo "test               run the unit tests (about 1 min)"
	@echo "bench              time the compiled SSA in each regime"
	@echo "figures            rebuild every figure from the committed CSVs in results/"
	@echo "exp-establishment  rerun the establishment experiment (writes results/establishment.csv)"
	@echo "exp-immune         rerun the immune-activation experiment (writes results/immune_activation.*)"
	@echo "exp-immune-collapse  follow collapsed responses to day 200 (writes results/immune_collapse.csv)"
	@echo "exp-resistance     rerun the therapy grid (writes results/resistance*.csv and the Figure 6 ensemble)"
	@echo "clean              remove generated figures and caches"

exp-establishment:
	$(PY) scripts/exp_establishment.py

exp-immune:
	$(PY) scripts/exp_immune_activation.py

exp-immune-collapse:
	$(PY) scripts/exp_immune_collapse.py

exp-resistance:
	$(PY) scripts/exp_resistance.py

install:
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -r requirements.txt -e .

lock:
	uv pip freeze --python $(PY) | grep -v '^-e\|viraldyn' > requirements-lock.txt

test:
	$(PY) -m pytest -q

bench:
	$(PY) scripts/bench_ssa.py

figures:
	@for f in scripts/fig*.py; do [ -e "$$f" ] && $(PY) "$$f" || true; done

clean:
	rm -rf .pytest_cache viraldyn/__pycache__ tests/__pycache__ scripts/__pycache__
	find figures -type f ! -name .gitkeep -delete
