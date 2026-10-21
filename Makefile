# Convenience targets for Linux and macOS. On Windows use: python run_all.py
PY ?= python

.PHONY: all quick data test benchmark optimisation analysis figures tables verify paper reference clean

all:
	$(PY) run_all.py
quick:
	$(PY) run_all.py --quick
reference:
	$(PY) run_all.py --from-reference
data:
	$(PY) scripts/download_data.py
test:
	$(PY) -m pytest -q
benchmark: data
	$(PY) experiments/run_benchmark.py
optimisation: data
	$(PY) experiments/run_optimisation.py
analysis:
	$(PY) experiments/analyse.py
figures:
	$(PY) experiments/make_figures.py
tables:
	$(PY) experiments/make_tables.py
verify:
	$(PY) experiments/verify.py
paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
clean:
	rm -rf results/quick .pytest_cache
	cd paper && latexmk -C
