#!/usr/bin/env bash
set -euo pipefail

# Serve the MkDocs-based documentation locally.
# Usage: ./scripts/docs_serve.sh [--port PORT]

PORT=8000
if [[ ${1:-} == "--port" ]]; then
  PORT=${2:-8000}
fi

# If a project venv exists, print a hint but do not auto-source (activation can be environment-specific)
if [[ -f "braingraph_pipeline/bin/activate" ]]; then
  echo "Hint: activate the project venv before running this script:" >&2
  echo "  source braingraph_pipeline/bin/activate" >&2
fi

# Ensure mkdocs is installed
if ! command -v mkdocs >/dev/null 2>&1; then
  echo "mkdocs not found. Install with: python -m pip install mkdocs mkdocs-material" >&2
  exit 2
fi

echo "Serving docs at http://127.0.0.1:${PORT} (CTRL+C to stop)"
mkdocs serve -a 127.0.0.1:${PORT}
