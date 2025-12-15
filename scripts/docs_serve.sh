#!/usr/bin/env bash
set -euo pipefail

# Serve the MkDocs-based documentation locally.
# Usage: ./scripts/docs_serve.sh [--port PORT]

PORT=8000
if [[ ${1:-} == "--port" ]]; then
  PORT=${2:-8000}
fi

is_port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi

  # Fallback: try binding to the port; if it fails, it's in use.
  python - <<PY >/dev/null 2>&1
import socket
s = socket.socket()
try:
    s.bind(('127.0.0.1', int('${port}')))
except OSError:
    raise SystemExit(0)
else:
    raise SystemExit(1)
finally:
    try:
        s.close()
    except Exception:
        pass
PY
  # Python exits 0 when in-use, 1 when free.
  [[ $? -eq 0 ]]
}

# Avoid a noisy MkDocs traceback when the port is already taken.
if is_port_in_use "${PORT}"; then
  original_port="${PORT}"
  for try_port in $(seq "$((PORT + 1))" "$((PORT + 20))"); do
    if ! is_port_in_use "${try_port}"; then
      PORT="${try_port}"
      echo "Port ${original_port} is already in use; switching to ${PORT}." >&2
      break
    fi
  done

  if [[ "${PORT}" == "${original_port}" ]]; then
    echo "Port ${PORT} is already in use." >&2
    echo "Try: ./scripts/docs_serve.sh --port 8001" >&2
    exit 3
  fi
fi

# If a project venv exists, print a hint but do not auto-source (activation can be environment-specific)
if [[ -f "braingraph_pipeline/bin/activate" ]]; then
  echo "Hint: activate the project venv before running this script:" >&2
  echo "  source braingraph_pipeline/bin/activate" >&2
fi

# Ensure mkdocs is installed in the *current Python environment*.
# (Avoid accidentally picking up a different venv's `mkdocs` from $PATH.)
if ! python -c "import mkdocs" >/dev/null 2>&1; then
  echo "MkDocs is not installed in the active Python environment ($(command -v python))." >&2
  echo "Install docs dependencies with:" >&2
  echo "  python -m pip install -r docs/requirements.txt" >&2
  exit 2
fi

echo "Serving docs at http://127.0.0.1:${PORT} (CTRL+C to stop)"
python -m mkdocs serve -a 127.0.0.1:${PORT}
