# One command regenerates every table and figure from raw results.
# `make experiment` re-runs the experiments from scratch.

PY      := .venv/bin/python
WORKERS ?= 10
E1      := experiments/exp01_bbob_d10/config.yaml
E2OUT   := experiments/exp02_structural_bias/results
E3      := experiments/exp03_cec2017_d10/config.yaml

.PHONY: all experiment report test clean setup exp01 exp02 exp03 verify

## regenerate all tables and figures from existing raw results
all: report

## full re-run: every experiment from scratch (hours)
experiment: exp01 exp02 exp03 report

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[ioh,cec,dev]" "setuptools<82"

exp01:
	$(PY) -m mhbench.cli run $(E1) --workers $(WORKERS)

exp02:
	$(PY) -m mhbench.cli bias --out $(E2OUT) --runs 200 --dim 10 \
		--budget 10000 --workers $(WORKERS)

exp03:
	$(PY) -m mhbench.cli run $(E3) --workers $(WORKERS)

report:
	@$(PY) -m mhbench.cli report $(E1)
	@test -f $(E3:config.yaml=results)/runs.csv && $(PY) -m mhbench.cli report $(E3) || \
		echo "[report] exp03 not run yet, skipping"

test:
	$(PY) -m pytest tests/ -q

## assert every claim in the README is backed by a file on disk
verify:
	$(PY) scripts/verify_claims.py

clean:
	rm -rf experiments/*/results/figures
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

freeze:
	$(PY) -m pip freeze > requirements.lock
