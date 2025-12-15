# Reproducible container build for OptiConn.
# This image intentionally does NOT bundle third-party executables like DSI Studio.
# All Python dependencies are downloaded during `docker build`.

ARG PYTHON_IMAGE=python:3.10.12-slim

FROM ${PYTHON_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/opticonn

# System packages: keep minimal; wheels cover most scientific deps, but build tools help when a wheel isn't available.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      git \
 && rm -rf /var/lib/apt/lists/*

# Install the Python package (non-editable) for a clean, reviewable environment.
# Copy only what is needed for installation first to maximize Docker cache reuse.
COPY pyproject.toml README.md LICENSE opticonn.py ./
COPY scripts ./scripts
COPY constraints.txt ./constraints.txt

ARG INSTALL_DEV=0
RUN python -m pip install --upgrade pip setuptools wheel \
 && if [ "$INSTALL_DEV" = "1" ]; then \
      python -m pip install --constraint /opt/opticonn/constraints.txt ".[dev]"; \
    else \
      python -m pip install --constraint /opt/opticonn/constraints.txt .; \
    fi

FROM base AS runtime

# Default to the CLI entrypoint.
ENTRYPOINT ["opticonn"]
CMD ["--help"]

FROM base AS docs

# Docs are a separate target so the runtime image stays minimal.
COPY mkdocs.yml ./mkdocs.yml
COPY docs ./docs
RUN python -m pip install -r docs/requirements.txt

EXPOSE 8000
CMD ["python", "-m", "mkdocs", "serve", "-a", "0.0.0.0:8000"]
