# ChurnAI

[![CI](https://github.com/gehna88/MLOps_ChurnAI-/actions/workflows/ci.yml/badge.svg)](https://github.com/gehna88/MLOps_ChurnAI-/actions/workflows/ci.yml)

**An end-to-end MLOps pipeline for bank customer churn prediction.**

ChurnAI takes a churn classifier from raw CSV to a tested, explainable, monitored
prediction API. It predicts whether a bank customer will close their account, explains
*why* with SHAP, and watches live traffic for data drift. The focus is on the full loop
(reproducibility, testing, tracking, serving and monitoring) rather than squeezing out
the last point of accuracy.

> MLOps course project · BS Computer Science, Fall 2026 · Gehna Bhatia (29054), M. Zayed Nauman (29047)

---

## Contents

- [Milestone 1 submission map](#milestone-1-submission-map)
- [Results](#results)
- [Best MLflow run](#best-mlflow-run)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Commands](#commands)
- [API](#api)
- [Project structure](#project-structure)
- [Key design decisions](#key-design-decisions)
- [Configuration](#configuration)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributors](CONTRIBUTION.md)

---

## Milestone 1 submission map

Where to find each Milestone 1 requirement (tag `v1.0-milestone1`):

| Requirement | Where |
|---|---|
| Pitch, architecture diagram, quick start, dataset and task | This README: [top](#churnai), [Architecture](#architecture), [Quickstart](#quickstart), [Dataset](#dataset-and-task) |
| Data preparation, separate from training, with validation | [`src/churnai/pipelines/prepare_data.py`](src/churnai/pipelines/prepare_data.py), [`src/churnai/data/validation.py`](src/churnai/data/validation.py) |
| Training script: held-out evaluation, versioned model artefact | [`src/churnai/pipelines/train.py`](src/churnai/pipelines/train.py) writes `models/<model>-v<version>-<run>.joblib` and `models/model.joblib` |
| MLflow: 3+ runs, params and metrics, best run justified | 4 runs per training; see [Best MLflow run](#best-mlflow-run) |
| FastAPI service with `/predict`, Pydantic validation, `/docs` | [`src/churnai/api/app.py`](src/churnai/api/app.py) (app factory), [`routes.py`](src/churnai/api/routes.py), [`schemas.py`](src/churnai/api/schemas.py) |
| pytest: data prep, model loading, prediction endpoint | [`tests/test_prepare_data.py`](tests/test_prepare_data.py), [`tests/test_training.py`](tests/test_training.py), [`tests/test_api.py`](tests/test_api.py); how to run in [Testing](#testing) |
| Names, ERP IDs and member-to-task table | [`CONTRIBUTION.md`](CONTRIBUTION.md) |
| AI tools used, for what, and our review | [`AI_USAGE.md`](AI_USAGE.md) |
| CI pipeline | [`.github/workflows/ci.yml`](.github/workflows/ci.yml): runs on Python 3.11 on every pull request and push to `main` (see [Continuous integration](#continuous-integration)) |

> The FastAPI app lives in the `churnai` package (`src/churnai/api/app.py`) rather than a
> loose `src/app.py`, so it is importable, installable and testable like the rest of the code.

---

## Dataset and task

- **Data:** [Bank Customer Churn](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
  by Radheshyam Kollipara (Kaggle). It covers 10,000 customers of a European bank in France,
  Germany and Spain, with 18 columns: demographics, balance, products, activity, card
  type, satisfaction and loyalty points. It is stored at `data/raw/Customer-Churn-Records.csv`.
- **Task:** binary classification. Predict `Exited` (1 means the customer closed their
  account). About 20% of customers churned.
- **Excluded columns:**
  - `RowNumber`, `CustomerId` and `Surname` are identifiers, dropped so the model cannot
    memorise customers.
  - `Complain` is target leakage (see [Results](#results)).
- **Primary metric:** ROC-AUC, because the classes are imbalanced and retention teams rank
  customers by risk. Recall, precision and F1 on the churn class are logged alongside it.

---

## Results

Four candidates are compared with 5-fold stratified cross-validation on the training split.
The champion is chosen by **CV ROC-AUC only**; the held-out test set (20%, never used for
selection or threshold tuning) gives the unbiased estimate.

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test recall | Test precision |
|---|---|---|---|---|---|
| **XGBoost (shallow), champion** | **0.862** | **0.872** | 0.718 | 0.667 | 0.588 |
| XGBoost (deep) | 0.861 | 0.871 | 0.720 | 0.699 | 0.589 |
| LightGBM | 0.858 | 0.868 | 0.713 | 0.694 | 0.564 |
| Logistic Regression (baseline) | 0.766 | 0.777 | 0.474 | 0.593 | 0.459 |

- The tree models beat the interpretable linear baseline by about **9 AUC points**,
  because churn depends on non-linear effects such as age and number of products.
- The decision threshold (0.61) is tuned on out-of-fold predictions to maximise F1. At
  that threshold the model catches two thirds of churners.
- CV and test scores agree closely, so there is no sign of overfitting. Any test AUC above
  0.95 would abort training as suspected leakage.

**Data findings** (from `make profile`, see `reports/data_profile.md`):

- `Complain` predicts churn almost perfectly on its own (single-feature AUC **0.998**). It
  is a consequence of churn, not a cause, so it is excluded. A test guarantees it can never
  reach training.
- About 36% of customers hold a zero balance. That is a real behavioural segment, so it is
  kept and flagged as an engineered feature.
- Classes are imbalanced (about 20% churn). This is handled with a stratified split and
  class weights (`scale_pos_weight` / `class_weight`).

---

## Best MLflow run

Every `make train` logs **one MLflow run per candidate** (4 runs) to the `churnai`
experiment. Each run records:
- the model family and hyperparameters
- CV and test metrics: ROC-AUC, PR-AUC, F1, recall, precision, accuracy and Brier score
- the tuned threshold, the confusion matrix, and the fitted model

The champion is registered as `churnai-classifier` with the alias `champion`.

**Best run: `xgb_shallow`**

| | |
|---|---|
| MLflow run ID | `65c7276acf3e4faf90c5f0e29fbd9108` |
| Registry version | `churnai-classifier` **v2** (`@champion`) |
| Served as | `xgb_shallow-v2-65c7276a` (returned in every `/predict` response) |
| Hyperparameters | `n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, scale_pos_weight=3.908` |
| CV ROC-AUC (selection metric) | **0.862** |
| Test ROC-AUC (held-out) | **0.872** |
| Decision threshold | 0.61 (tuned on out-of-fold predictions for F1) |

**Why this run wins:**

1. **It has the highest cross-validated ROC-AUC.** ROC-AUC is the primary metric because
   classes are imbalanced and retention teams rank customers by risk. Selection uses CV on
   the training split only, so the test set stays an unbiased check.
2. **It is the simplest of the near-tied models.** `xgb_deep` (0.861) and LightGBM (0.858)
   are within 0.004 AUC, which is within fold-to-fold noise. The shallow model (depth 3)
   wins and is the least likely to overfit, so there is no reason to prefer a more complex one.
3. **The held-out test confirms it.** The test AUC (0.872) is consistent with CV (0.862),
   so there is no sign of overfitting, and it is far below the 0.95 leakage alarm.
4. **It clearly beats the interpretable baseline.** Logistic Regression scores 0.766, so
   the gain over a linear model is about 0.09 AUC.

**Reproduce and inspect:**

```bash
make data && make train   # logs 4 new runs and registers a new champion version
make mlflow-ui            # http://127.0.0.1:5000 → experiment "churnai"
```

![MLflow runs table](docs/images/mlflow-runs.png)
![MLflow run comparison](docs/images/mlflow-compare.png)
![MLflow model registry](docs/images/mlflow-registry.png)

> **Note on the MLflow requirement for LLM projects:** ChurnAI is a classical tabular ML
> project, so experiments compare *training runs* (model families and hyperparameters)
> rather than prompts, LLMs or temperature settings.

---

## Architecture

Two pipelines connected by a single hand-off artefact, the **model bundle**. Every
deployed model maps back to exactly one MLflow run.

```mermaid
flowchart LR
    subgraph Training pipeline
        A[Raw CSV] --> B[prepare_data<br/>validate · drop leakage · stratified split]
        B --> C[train<br/>4 candidates · 5-fold CV · threshold tuning]
        C --> D[(MLflow<br/>runs + model registry)]
        C --> E[models/model.joblib<br/>pipeline · threshold · drift reference · SHAP background]
    end
    subgraph Serving
        E --> F[FastAPI service]
        F --> G["/predict · /predict/batch"]
        F --> H["/explain (SHAP)"]
        F --> I["/health · /metrics (latency + PSI drift)"]
    end
```

**Inside the model pipeline:** feature engineering, then preprocessing (scaling and
one-hot encoding), then the estimator, all in one scikit-learn `Pipeline`. Preprocessing
is saved *inside* the model, so the API cannot drift from training.

---

## Quickstart

Requires **Python 3.11+** and Git. Run everything from the repository root.

**In short:**

```bash
git clone https://github.com/gehna88/MLOps_ChurnAI-.git && cd MLOps_ChurnAI-
make setup && make data && make train && make serve
```

Run these inside an activated virtualenv (step 1 below).

### 0. Clone

```bash
git clone https://github.com/gehna88/MLOps_ChurnAI-.git
cd MLOps_ChurnAI-
```

### 1. Create and activate a virtualenv

**Windows (PowerShell):**

```powershell
python -m venv $env:USERPROFILE\.venvs\churnai
& $env:USERPROFILE\.venvs\churnai\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> On Windows, keep the venv outside OneDrive or other synced folders.

### 2. Install, prepare, train, serve

| Step | With `make` | Without `make` |
|---|---|---|
| Install | `make setup` | `pip install -r requirements/dev.txt` then `pip install -e . --no-deps` |
| Prepare data | `make data` | `python -m churnai.pipelines.prepare_data` |
| Train | `make train` | `python -m churnai.pipelines.train` |
| Serve | `make serve` | `uvicorn churnai.api.app:create_app --factory --port 8000` |

Then open **http://127.0.0.1:8000/docs** for interactive API docs.

> **Windows and `make`:** GNU Make is not installed by default. Install it with
> `winget install ezwinports.make` from a non-admin PowerShell (then reopen the terminal),
> or use the *Without `make`* column. Both are equivalent.

---

## Commands

| `make` target | Equivalent command | Purpose |
|---|---|---|
| `setup` | `pip install -r requirements/dev.txt && pip install -e . --no-deps` | Install pinned dependencies and the package |
| `lint` | `ruff check src tests scripts` | Static checks |
| `test` | `pytest` | Full suite (74 tests) |
| `test-fast` | `pytest -m "not slow"` | Unit tests only (seconds) |
| `coverage` | `pytest --cov=churnai --cov-report=term-missing` | Suite with coverage report |
| `check` | lint + test | The gate CI runs |
| `profile` | `python -m churnai.pipelines.profile_data` | EDA report and leakage scan |
| `data` | `python -m churnai.pipelines.prepare_data` | Validate and split data |
| `train` | `python -m churnai.pipelines.train` | Train, track and register the champion |
| `pipeline` | data + train | End-to-end retrain |
| `serve` | `uvicorn churnai.api.app:create_app --factory --port 8000` | Run the API |
| `mlflow-ui` | `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` | Browse experiments |
| `clean` | | Remove tool caches |
| `clean-artifacts` | | Remove generated data, models, reports and the MLflow store |

### What each pipeline produces

| Command | Outputs |
|---|---|
| `profile` | `reports/data_profile.md` |
| `data` | `data/processed/{train,test}.csv`, `summary.json`; rejected rows (with reasons) in `data/rejected/` |
| `train` | `models/model.joblib` (served), `models/<model>-v<version>-<run>.joblib` (versioned copy), `models/metadata.json`, `reports/leaderboard.csv`, one MLflow run per candidate, champion registered as `churnai-classifier@champion` |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Churn probability, label and model version for one customer |
| `POST` | `/predict/batch` | Same for up to 1,000 customers |
| `POST` | `/explain` | Prediction plus the top 5 features driving it (SHAP, log-odds) |
| `GET` | `/health` | Liveness and whether a model is loaded |
| `GET` | `/metrics` | Request counts, error counts, latency (avg/p50/p95/p99), drift report |
| `GET` | `/model` | Full metadata and lineage of the served model |
| `GET` | `/docs` | Interactive Swagger UI |

Invalid input (out-of-range values, unknown categories, missing or extra fields) is
rejected with **422** before it reaches the model. Every response carries an
`X-Request-ID` header, and every request is logged as one JSON line.

### Example: predict

**bash / curl:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"credit_score": 619, "geography": "France", "gender": "Female", "age": 42,
       "tenure": 2, "balance": 0, "num_of_products": 1, "has_cr_card": 1,
       "is_active_member": 1, "estimated_salary": 101348.88,
       "satisfaction_score": 2, "card_type": "DIAMOND", "point_earned": 464}'
```

**PowerShell:**

```powershell
$c = @{credit_score=619; geography="France"; gender="Female"; age=42; tenure=2; balance=0;
       num_of_products=1; has_cr_card=1; is_active_member=1; estimated_salary=101348.88;
       satisfaction_score=2; card_type="DIAMOND"; point_earned=464} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -Body $c -ContentType "application/json"
```

**Response:**

```json
{
  "churn_probability": 0.6481,
  "churn_prediction": true,
  "threshold": 0.61,
  "model_version": "xgb_shallow-v2-65c7276a"
}
```

`model_version` is `<model>-v<registry version>-<MLflow run id>`, so any prediction can be
traced back to the exact training run.

### Example: explain

`POST /explain` with the same body returns:

```json
{
  "churn_probability": 0.6481,
  "base_value": 0.0256,
  "units": "log_odds",
  "top_features": [
    {"feature": "NumOfProducts",  "value": 1,        "shap_value":  0.5096, "direction": "increases_churn_risk"},
    {"feature": "IsActiveMember", "value": 1,        "shap_value": -0.3841, "direction": "decreases_churn_risk"},
    {"feature": "Gender",         "value": "Female", "shap_value":  0.2637, "direction": "increases_churn_risk"},
    {"feature": "Geography",      "value": "France", "shap_value": -0.1503, "direction": "decreases_churn_risk"},
    {"feature": "Balance",        "value": 0.0,      "shap_value":  0.1498, "direction": "increases_churn_risk"}
  ]
}
```

One-hot columns are aggregated back to their original feature (for example, every
`Geography=*` column becomes `Geography`). Contributions are additive:
`base_value + Σ shap_value` equals the model's log-odds exactly. A test enforces this for
every model family.

### Input fields

| Field | Type | Allowed values |
|---|---|---|
| `credit_score` | int | 300 – 900 |
| `geography` | string | `France`, `Germany`, `Spain` |
| `gender` | string | `Female`, `Male` |
| `age` | int | 18 – 100 |
| `tenure` | int | 0 – 60 (years with the bank) |
| `balance` | float | 0 – 10,000,000 |
| `num_of_products` | int | 1 – 4 |
| `has_cr_card` | int | 0 or 1 |
| `is_active_member` | int | 0 or 1 |
| `estimated_salary` | float | 0 – 10,000,000 |
| `satisfaction_score` | int | 1 – 5 |
| `card_type` | string | `SILVER`, `GOLD`, `PLATINUM`, `DIAMOND` |
| `point_earned` | int | 0 – 10,000 |

These rules are generated from the same data contract (`src/churnai/data/schema.py`) used
for offline validation, so training-time and serving-time checks are identical.

---

## Project structure

```
ChurnAI/
├── configs/
│   ├── config.yaml            # paths, split, CV, MLflow, explainability, monitoring settings
│   └── models.yaml            # candidate models and hyperparameters
├── data/raw/                  # source dataset (processed outputs are generated, not committed)
├── requirements/              # pinned lock files: serve.txt ⊂ train.txt ⊂ dev.txt
├── scripts/healthcheck.py     # container health check (stdlib only)
├── src/churnai/
│   ├── config.py              # typed, validated settings (pydantic) with env overrides
│   ├── logging_utils.py       # one-line JSON logging
│   ├── exceptions.py          # project exception hierarchy
│   ├── data/                  # schema (data contract), validation, preparation, profiling, io
│   ├── features/              # feature engineering and the shared sklearn Pipeline
│   ├── models/                # model factory, training/selection, evaluation, bundle, MLflow tracking
│   ├── explain/               # SHAP explainer
│   ├── monitoring/            # PSI drift detection, request metrics
│   ├── api/                   # FastAPI app factory, routes, schemas, inference service
│   └── pipelines/             # CLI entry points: profile_data, prepare_data, train
├── tests/                     # 74 tests, about 95% line coverage
├── Makefile
└── pyproject.toml             # package metadata, ruff and pytest configuration
```

---

## Key design decisions

| Decision | Why |
|---|---|
| **One data contract** (`schema.py`) drives offline validation *and* the API request schema | Training and serving can't disagree about what valid input is |
| **Allow-list of features**, not a deny-list | A new upstream column, even a leaky one, can never silently reach training |
| **Preprocessing inside the model `Pipeline`** | The API reuses exactly the fitted transformations; no serving-side reimplementation |
| **Select on CV, report on test** | Model choice and threshold never see the test set, so the test score is unbiased |
| **Leakage guard** (test AUC > 0.95 aborts training) | Implausibly good results are treated as a bug, not a win |
| **Model bundle** (pipeline + threshold + lineage + drift reference + SHAP background) | One artefact to ship; the API needs nothing else and no MLflow at runtime |
| **`ExperimentTracker` interface** (MLflow or no-op) | Training logic is unit-testable without an MLflow server |
| **Model factory registry** | Adding a model family means registering one builder plus a YAML entry |
| **App factory, no global state** | Each test builds an isolated app with its own model |
| **Graceful degradation** | If the model is missing, the API still starts and `/health` reports `unavailable` instead of crash-looping |
| **Pinned, layered requirements** | `serve.txt` keeps the runtime image lean; training and dev tools stay out of it |

### Drift monitoring

At training time a reference distribution of every feature, and of the predicted
probability, is stored in the bundle. The API keeps a rolling window of the last 500
requests and computes the **Population Stability Index** per feature. PSI > 0.2 flags
drift in `/metrics`. When drift is flagged: retrain (`make pipeline`), compare in MLflow,
then redeploy.

---

## Configuration

All settings live in `configs/config.yaml` and `configs/models.yaml`. Paths are relative
to the project root, so the project runs from any clean clone.

Environment variables override config without editing files:

| Variable | Purpose |
|---|---|
| `CHURNAI_HOME` | Project root (default: current directory) |
| `CHURNAI_CONFIG` | Path to an alternative config file |
| `CHURNAI_MODEL_DIR` | Where the model bundle is read and written |
| `CHURNAI_MLFLOW_TRACKING_URI` | For example, a remote MLflow server |
| `CHURNAI_MLFLOW_ARTIFACT_DIR` | MLflow artifact location |
| `CHURNAI_LOG_LEVEL` | `DEBUG`, `INFO` (default), ... |
| `CHURNAI_LOG_FORMAT` | `json` (default for the API) or `text` |

**To add a model**, append a candidate to `configs/models.yaml`:

```yaml
- name: xgb_wide
  estimator: xgboost          # logistic_regression | xgboost | lightgbm
  params: {n_estimators: 600, max_depth: 4, learning_rate: 0.02, scale_pos_weight: auto}
```

---

## Testing

```bash
make test        # full suite (74 tests)        or: pytest
make test-fast   # unit tests only, seconds     or: pytest -m "not slow"
make coverage    # full suite with coverage     or: pytest --cov=churnai --cov-report=term-missing
```

The three areas the milestone asks for are covered by:
- **Data preparation:** `tests/test_prepare_data.py` and `tests/test_validation.py`
- **Model loading:** `tests/test_training.py`. It covers bundle load and save, the versioned
  copy, and a clear error when no model exists.
- **Prediction endpoint:** `tests/test_api.py`. It covers `/predict`, `/predict/batch`,
  `/explain` and 422 validation.

Tests run against an isolated temporary project (a stratified sample of the real data
plus fast model settings). They never touch your `models/` or `mlflow.db`, and they don't
require `make train` to have been run first.

| Area | What is verified |
|---|---|
| Data | Bad rows rejected with reasons; leakage and identifier columns can never reach training; deterministic stratified split |
| Models | Factory, `scale_pos_weight=auto`, threshold tuning, CV-only selection, leakage guard |
| Explainability | Top-k ordering; additivity for XGBoost, LightGBM and Logistic Regression |
| Monitoring | PSI maths; no false alarm on training-like traffic; detects shifted populations |
| API | All endpoints, 422 validation, batch limits, request-ID propagation, behaviour without a model |
| MLflow | Runs logged, champion registered, bundle lineage matches the registry |

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every pull request to
`main` and every push to `main`, on a fresh Ubuntu machine with **Python 3.11**. Each run proves the
project works from a clean clone:

1. `make setup`: install the pinned dependencies
2. `make lint`: ruff
3. `make test`: the full pytest suite
4. `make profile`, `make data`, `make train`: the full pipeline, including MLflow tracking and registration
5. Start the API, check `/health`, and call `/predict` with a real customer

The training reports and model metadata are uploaded as a downloadable artifact on each
run (**Actions → a run → Artifacts**).

---

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| **1. Model and API** | Data pipeline, MLflow-tracked training, FastAPI `/predict` and `/explain`, tests, Makefile, README | ✅ |
| **2. Deployment** | Multi-stage Docker image, extend CI to build and push to GHCR, Render deployment, RUNBOOK, rollback drill | 🔜 |
| **3. Final presentation** | Live demo of pipeline, MLflow, CI and API | ⏳ |

`/health`, `/metrics`, JSON logging and PSI drift monitoring (Milestone 2 scope) are
already implemented.

---

**Team:** Gehna Bhatia (29054) and M. Zayed Nauman (29047). See [CONTRIBUTION.md](CONTRIBUTION.md)
for who did what.
