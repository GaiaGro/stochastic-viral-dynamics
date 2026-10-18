PY ?= .venv/bin/python

.PHONY: help install lock test bench figures clean

help:
	@echo "install  create .venv with uv and install requirements.txt"
	@echo "lock     write requirements-lock.txt from the current .venv"
	@echo "test     run the unit tests (about 1 min)"
	@echo "bench    time the compiled SSA in each regime"
	@echo "figures  rebuild every figure from the committed CSVs in results/"
	@echo "clean    remove generated figures and caches"

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
