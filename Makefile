# The unified test / bench / CI entry point.
#
# Every person runs the same targets; CI runs the same targets. If a check
# exists only in CI, it does not exist. See docs/team/testing-ci.md.

PY      ?= python3
TESTRUN ?= $(PY) run_tests.py
STRICT  ?= 0

.DEFAULT_GOAL := help
.PHONY: help bootstrap fixtures golden golden-check lint fmt test test-unit test-contract \
        test-integration test-e2e test-map bench ci clean

help: ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

bootstrap: ## install dev deps (ruff) — needs no Triton, no torch, no GPU
	$(PY) -m pip install -e '.[dev]'

fixtures: ## regenerate fixtures/raw with Triton (manual, reviewed; see VERSIONS.txt)
	$(PY) src/triton_tritonflow/harness/extract_fixtures.py --out fixtures/raw
	$(MAKE) golden

golden: ## re-derive the observations block of fixtures/GOLDEN.json
	$(PY) tools/snapshot_fixtures.py --write

golden-check: ## CI gate: fixtures and GOLDEN.json agree
	$(PY) tools/snapshot_fixtures.py --check

lint: ## ruff
	$(PY) -m ruff check .

fmt: ## ruff, fixing what it can
	$(PY) -m ruff check . --fix && $(PY) -m ruff format .

test-map: ## module <-> test and contract <-> contract-test mapping (R5)
	$(PY) tools/check_test_map.py

test-unit: ## T-L1
	$(TESTRUN) --suite unit

test-contract: ## T-L2
	$(TESTRUN) --suite contract

test-integration: ## T-L3
	$(TESTRUN) --suite integration

test-e2e: ## T-L4
	$(TESTRUN) --suite e2e

test: ## run all tests
	$(TESTRUN)

bench: ## T-L5: publish bench/results.json. Never gates a merge.
	$(PY) bench/run.py --out bench/results.json

ci: lint golden-check test-map test ## exactly what .github/workflows/ci.yml runs
	@echo "ci: local reproduction passed. PRs are gated on STRICT_REAL=$(STRICT)."

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
