# Reproducible container build for OptiConn.
# This image intentionally does NOT bundle third-party executables like DSI Studio.
# All Python dependencies are downloaded during `docker build`.

ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim AS base

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

ARG INSTALL_DEV=0
RUN python -m pip install --upgrade pip setuptools wheel \
 && if [ "$INSTALL_DEV" = "1" ]; then \
      python -m pip install ".[dev]"; \
    else \
      python -m pip install .; \
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
