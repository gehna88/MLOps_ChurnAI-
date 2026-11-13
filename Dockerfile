# ChurnAI inference service.
#
# Two stages keep the runtime image small: the builder installs the pinned serving
# dependencies and the churnai package into a virtualenv; the runtime stage copies only
# that virtualenv, the config and the trained model. Training tools (MLflow, Optuna,
# pytest) never enter the image.
#
# The model is not built here: run `make data && make train` first so that
# models/model.joblib exists. CI does exactly that before `docker build`.
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
COPY requirements/serve.txt requirements/serve.txt
RUN pip install -r requirements/serve.txt

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
    PORT=8000

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs/config.yaml configs/config.yaml
COPY scripts/healthcheck.py scripts/healthcheck.py
COPY models/model.joblib models/model.joblib

# Run as an unprivileged user; logs/ is the only path the service writes to.
RUN useradd --create-home --uid 10001 churnai \
    && mkdir -p logs && chown churnai:churnai logs
USER churnai

EXPOSE 8000

# Healthy only when /health reports a loaded model (see scripts/healthcheck.py).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]

# Hosting platforms such as Render inject PORT; locally it defaults to 8000.
CMD ["sh", "-c", "exec uvicorn churnai.api.app:create_app --factory --host 0.0.0.0 --port ${PORT}"]
