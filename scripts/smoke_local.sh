#!/usr/bin/env bash
set -euo pipefail

# Local smoke-test helper for OptiConn (Docker-based)
# Usage: ./scripts/smoke_local.sh [--platform <platform>] [--image-name <name>]

PLATFORM=""
IMAGE_NAME="opticonn:runtime"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="--platform=$2"; shift 2 ;;
    --image-name) IMAGE_NAME="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--platform <platform>] [--image-name <name>]"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

echo "Building Docker image: ${IMAGE_NAME} ${PLATFORM}"
docker build ${PLATFORM} --target runtime -t "${IMAGE_NAME}" .

echo "Running CLI help check"
docker run --rm ${PLATFORM} "${IMAGE_NAME}" --help

echo "Running Python import smoke test"
docker run --rm ${PLATFORM} --entrypoint python "${IMAGE_NAME}" -c "import opticonn; import scripts.opticonn_hub; print('imports-ok')"

echo "Running config validator (no DSI Studio required)"
docker run --rm ${PLATFORM} --entrypoint python "${IMAGE_NAME}" scripts/json_validator.py configs/braingraph_default_config.json --suggest-fixes || true

echo "Local smoke test completed."
