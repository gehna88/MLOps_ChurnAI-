# ChurnAI inference service.
#
# Two stages keep the runtime image small: the builder installs the pinned serving
# dependencies and the churnai package into a virtualenv; the runtime stage copies only
# that virtualenv, the config and the trained model. Optuna and pytest never enter the
# image. MLflow, matplotlib and the dataset do, for the drift study: the API serves the
# study's results and graphs (GET /drift/study...), can run a new study
# (POST /drift/study/run) and logs it to the data-drift MLflow experiment.
#
# The model and the study are not built here: run `make data && make train &&
# make drift-study` first, so models/model.joblib and reports/drift_study/ exist.
# CI does exactly that before `docker build`.
#
#   docker build -t churnai .
#   docker run -p 8000:8000 churnai          # then open http://localhost:8000/docs

ARG PYTHON_VERSION=3.11

# ------------------------------------------------------------------ builder ---
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
# Dependencies first, so this layer is cached until the lock file changes.
COPY requirements/serve.txt requirements/drift.txt requirements/
RUN pip install -r requirements/serve.txt -r requirements/drift.txt

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps .

# ------------------------------------------------------------------ runtime ---
FROM python:${PYTHON_VERSION}-slim AS runtime

# The image tag (CI passes e.g. sha-1a2b3c4); reported by /health as "build".
ARG BUILD=local

# LightGBM needs the OpenMP runtime, which the slim image does not ship.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURNAI_HOME=/app \
    CHURNAI_LOG_FORMAT=json \
    CHURNAI_BUILD=${BUILD} \
    CHURNAI_MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db \
    CHURNAI_MLFLOW_ARTIFACT_DIR=/app/mlflow/artifacts \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs/config.yaml configs/config.yaml
COPY scripts/healthcheck.py scripts/healthcheck.py
COPY models/model.joblib models/model.joblib
# Drift study: the model catalogue (to retrain the champion's config), the dataset, the
# pinned requirements MLflow records, and the study CI ran before this build.
COPY configs/models.yaml configs/models.yaml
COPY requirements/serve.txt requirements/serve.txt
COPY data/raw/Customer-Churn-Records.csv data/raw/Customer-Churn-Records.csv
COPY reports/drift_study reports/drift_study

# Run as an unprivileged user. It writes logs/, the MLflow store in mlflow/ and new drift
# studies in reports/ (point CHURNAI_MLFLOW_TRACKING_URI at /home/... on Azure to keep runs
# across restarts). MLflow's SQL store also creates ./mlruns as its default artifact root on
# start-up, even though our experiments store artifacts under mlflow/artifacts.
RUN useradd --create-home --uid 10001 churnai \
    && mkdir -p logs mlflow mlruns \
    && chown -R churnai:churnai logs mlflow mlruns reports
USER churnai

EXPOSE 8000

# Healthy only when /health reports a loaded model (see scripts/healthcheck.py).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]

# Hosting platforms such as Render inject PORT; locally it defaults to 8000.
CMD ["sh", "-c", "exec uvicorn churnai.api.app:create_app --factory --host 0.0.0.0 --port ${PORT}"]
