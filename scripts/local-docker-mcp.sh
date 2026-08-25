#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="agent-spreadsheet-full:test"

# Local test working directory (ignored by git).
TEST_WORKDIR="${TEST_WORKDIR:-$PROJECT_ROOT/test_workdir}"

# Backwards-compat: WORKSPACE_ROOT previously controlled the /data mount.
HOST_DATA_DIR="${HOST_DATA_DIR:-${WORKSPACE_ROOT:-$TEST_WORKDIR/data}}"
HOST_SCREENSHOT_DIR="${HOST_SCREENSHOT_DIR:-$TEST_WORKDIR/screenshots}"

mkdir -p "$HOST_DATA_DIR" "$HOST_SCREENSHOT_DIR"

# Canonicalize for stable path mapping strings.
HOST_DATA_DIR="$(realpath "$HOST_DATA_DIR")"
HOST_SCREENSHOT_DIR="$(realpath "$HOST_SCREENSHOT_DIR")"

docker build -q -f "$PROJECT_ROOT/Dockerfile.full" -t "$IMAGE_NAME" "$PROJECT_ROOT" >/dev/null

exec docker run --rm -i \
    -v "$HOST_DATA_DIR:/data" \
    -v "$HOST_SCREENSHOT_DIR:/screenshots" \
    -e SPREADSHEET_MCP_SCREENSHOT_DIR=/screenshots \
    -e "SPREADSHEET_MCP_PATH_MAP=/data=$HOST_DATA_DIR,/screenshots=$HOST_SCREENSHOT_DIR" \
    "$IMAGE_NAME" \
    --workspace-root /data \
    --transport stdio \
    --recalc-enabled \
    --vba-enabled
