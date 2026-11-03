# ChurnAI developer commands.
#
# Every recipe is a plain `python -m ...` call so it behaves the same under
# cmd.exe (GNU Make on Windows) and bash (Linux/macOS/CI). Activate your
# virtualenv first, or override the interpreter:  make train PYTHON=path/to/python
#
# Typical flow from a clean clone:  make setup -> make data -> make train -> make serve

PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8000
MLFLOW_PORT ?= 5000
SRC_DIRS = src tests scripts

.DEFAULT_GOAL := help
.PHONY: help setup lint format test test-fast coverage profile data train pipeline serve mlflow-ui check clean clean-artifacts

help:
	@echo "ChurnAI - available targets:"
	@echo "  setup            install pinned dependencies and the churnai package"
	@echo "  lint             ruff static checks"
	@echo "  format           auto-fix lint issues where possible"
	@echo "  test             full test suite"
	@echo "  test-fast        unit tests only, skips model training"
	@echo "  coverage         full suite with a line-coverage report"
	@echo "  profile          EDA report and leakage scan - reports/data_profile.md"
	@echo "  data             validate raw data and write train/test splits"
	@echo "  train            train candidates, log to MLflow, save the champion"
	@echo "  pipeline         data + train"
	@echo "  serve            run the API on http://$(HOST):$(PORT) - docs at /docs"
	@echo "  mlflow-ui        browse experiments on http://127.0.0.1:$(MLFLOW_PORT)"
	@echo "  check            lint + test, the same gate CI runs"
	@echo "  clean            remove tool caches"
	@echo "  clean-artifacts  remove generated data, models, reports and MLflow store"

# ------------------------------------------------------------------ setup ---
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements/dev.txt
	$(PYTHON) -m pip install -e . --no-deps

# ---------------------------------------------------------------- quality ---
lint:
	$(PYTHON) -m ruff check $(SRC_DIRS)

format:
	$(PYTHON) -m ruff check $(SRC_DIRS) --fix

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow"

coverage:
	$(PYTHON) -m pytest --cov=churnai --cov-report=term-missing

check: lint test

# --------------------------------------------------------------- pipeline ---
profile:
	$(PYTHON) -m churnai.pipelines.profile_data

data:
	$(PYTHON) -m churnai.pipelines.prepare_data

train:
	$(PYTHON) -m churnai.pipelines.train

pipeline: data train

# ---------------------------------------------------------------- serving ---
serve:
	$(PYTHON) -m uvicorn churnai.api.app:create_app --factory --host $(HOST) --port $(PORT)

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

# ---------------------------------------------------------------- cleanup ---
clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache', 'htmlcov')]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').glob('.coverage*')]"

clean-artifacts:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('data/processed', 'data/rejected', 'models', 'reports', 'mlartifacts')]"
	$(PYTHON) -c "import pathlib; pathlib.Path('mlflow.db').unlink(missing_ok=True)"
