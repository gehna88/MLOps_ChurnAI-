# Contributions

**ChurnAI: an end-to-end MLOps pipeline for bank customer churn prediction**
MLOps course project, BS Computer Science, Fall 2026

| Name | ERP ID | GitHub |
|---|---|---|
| Gehna Bhatia | 29054 | [@gehna88](https://github.com/gehna88) |
| M. Zayed Nauman | 29047 | [@zayed-nauman](https://github.com/zayed-nauman) |

Each of us opened our own commits on the `milestone-1` branch, and the other reviewed the
pull request into `main`. The commit history on GitHub
(`git log --format="%an | %s"`) records exactly who committed what.

---

## Milestone 1: Model and API (`v1.0-milestone1`)

| Member | Area | Tasks | Main files |
|---|---|---|---|
| **Gehna Bhatia** | Project setup | Repository, typed YAML configuration with env-var overrides, JSON logging, exception hierarchy, pinned requirements | `pyproject.toml`, `requirements/`, `configs/config.yaml`, `src/churnai/config.py`, `logging_utils.py`, `exceptions.py` |
| **Gehna Bhatia** | Data preparation | Data contract (schema), row-level validation (nulls, ranges, categories, duplicates) with rejected-row audit, dropping leakage and identifier columns, stratified train/test split | `src/churnai/data/schema.py`, `validation.py`, `preparation.py`, `io.py`, `pipelines/prepare_data.py` |
| **Gehna Bhatia** | Training and MLflow | 5-fold CV training, OOF threshold tuning, champion selection, leakage guard, MLflow run logging, model registry with `champion` alias, versioned model bundle | `src/churnai/models/training.py`, `bundle.py`, `tracking.py`, `pipelines/train.py` |
| **Gehna Bhatia** | Monitoring | PSI drift detection with a reference profile stored in the model bundle | `src/churnai/monitoring/drift.py` |
| **Gehna Bhatia** | Explainability | SHAP explanations aggregated to business features, with an additivity guarantee | `src/churnai/explain/explainer.py` |
| **Gehna Bhatia** | Documentation | Best MLflow run justification, submission map, this file | `README.md` (Best MLflow run section), `CONTRIBUTION.md` |
| **M. Zayed Nauman** | Features and models | Feature engineering, shared scikit-learn pipeline, model factory (Logistic Regression, XGBoost, LightGBM), evaluation metrics | `src/churnai/features/`, `src/churnai/models/factory.py`, `evaluation.py`, `configs/models.yaml` |
| **M. Zayed Nauman** | FastAPI service | `/predict`, `/predict/batch`, `/explain`, `/health`, `/metrics`, `/model`; Pydantic validation from the data contract; request logging and metrics middleware; health-check script | `src/churnai/api/`, `src/churnai/monitoring/metrics.py`, `scripts/healthcheck.py` |
| **M. Zayed Nauman** | Data profiling and CLI | Automated EDA report, single-feature leakage scan, shared pipeline CLI | `src/churnai/data/profiling.py`, `pipelines/profile_data.py`, `pipelines/_cli.py` |
| **M. Zayed Nauman** | Testing | pytest suite for data preparation, validation, training and model loading, MLflow integration, drift, CLI and API (74 tests, about 95% coverage) | `tests/` |
| **M. Zayed Nauman** | Tooling and docs | Makefile (`setup`, `data`, `train`, `serve`, `test`, `lint`), README | `Makefile`, `README.md` |
| **Gehna Bhatia** | CI | GitHub Actions workflow: Python 3.11 lint, tests, full pipeline and an API smoke test on every pull request and push to `main` | `.github/workflows/ci.yml` |
| **Both** | Review | Cross-reviewed the Milestone 1 pull request; ran the full pipeline and test suite on our own machines | |

### Reflections

**Gehna Bhatia:** 
I Developed a data preparation pipeline, training and MLflow tracking flow, drift monitoring and the SHAP explanations. It was with a single feature called Complain that I learned that an AUC of 0.998 was the most dangerous bug, appearing too good to be true. I also learned the importance of picking the model on cross-validation and not the test set to maintain fairness of the final score.

**M. Zayed Nauman:** 
I built the feature engineering and model factory, the FastAPI service, the test suite and the Makefile. I learned how to prevent training and serving from becoming separate worlds by keeping the preprocessing steps inside the model pipeline, and having the API's validation generated at the same time. I was able to write tests that can be executed on an isolated temporary project, also giving me a glimpse of how to make tests fast and independent of the previous training run.

---

## Milestone 2: Containerisation, CI/CD and Deployment


| Member | Tasks |
|---|---|
| M. Zayed Nauman | Multi-stage Dockerfile,monitoring and drift section of the RUNBOOK, RUNBOOK.md |
| Gehna Bhatia | Render deployment,extend CI to build and push images to GHCR rollback drill |
