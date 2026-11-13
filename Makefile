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
IMAGE ?= churnai
TAG ?= local

.DEFAULT_GOAL := help
.PHONY: help setup lint format test test-fast coverage profile data tune train shap-backfill compare ab-fetch ab-online simulate-batches pipeline serve mlflow-ui docker-build docker-run check clean clean-artifacts

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
	@echo "  tune             Optuna studies, best params written to configs/models.yaml"
	@echo "  shap-backfill    add SHAP plots to earlier MLflow candidate runs"
	@echo "  compare          offline A/B: champion vs every candidate, statistical tests"
	@echo "  ab-fetch         download the champion and challenger bundles from MLflow"
	@echo "  ab-online        simulated online A/B: power check, replay, verdict"
	@echo "  simulate-batches drifted future-year batches in data/batches for the drift study"
	@echo "  pipeline         data + train"
	@echo "  serve            run the API on http://$(HOST):$(PORT) - docs at /docs"
	@echo "  mlflow-ui        browse experiments on http://127.0.0.1:$(MLFLOW_PORT)"
	@echo "  docker-build     build the API image $(IMAGE):$(TAG) - needs a trained model"
	@echo "  docker-run       run that image on http://localhost:$(PORT)"
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

tune:
	$(PYTHON) -m churnai.pipelines.tune

shap-backfill:
	$(PYTHON) -m churnai.pipelines.backfill_shap

compare:
	$(PYTHON) -m churnai.pipelines.compare_models

ab-fetch:
	$(PYTHON) -m churnai.pipelines.fetch_ab_bundles

ab-online:
	$(PYTHON) -m churnai.pipelines.ab_online

simulate-batches:
	$(PYTHON) -m churnai.pipelines.simulate_batches

pipeline: data train

# ---------------------------------------------------------------- serving ---
serve:
	$(PYTHON) -m uvicorn churnai.api.app:create_app --factory --host $(HOST) --port $(PORT)

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

# ----------------------------------------------------------------- docker ---
docker-build:
	docker build --build-arg BUILD=$(TAG) -t $(IMAGE):$(TAG) .

docker-run:
	docker run --rm -p $(PORT):8000 $(IMAGE):$(TAG)

# ---------------------------------------------------------------- cleanup ---
clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache', 'htmlcov')]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').glob('.coverage*')]"

clean-artifacts:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('data/processed', 'data/rejected', 'data/batches', 'models', 'reports', 'mlartifacts', 'logs')]"
	$(PYTHON) -c "import pathlib; pathlib.Path('mlflow.db').unlink(missing_ok=True)"
