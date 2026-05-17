#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PY="$ROOT_DIR/.venv/bin/python"

usage() {
  cat <<EOF
Usage: ./scripts/start_local.sh [all|api|review]

  all     Start the local API and Streamlit review UI. This is the default.
  api     Start only the local API used by the Chrome extension.
  review  Start only the Streamlit review UI.
EOF
}

case "$MODE" in
  all|api|review) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [ ! -x "$VENV_PY" ]; then
  echo "Creating .venv..."
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
fi

echo "Installing/updating Python dependencies..."
"$VENV_PY" -m pip install -q -r requirements.txt

echo "Initializing local database..."
"$VENV_PY" -m app.cli init-db >/dev/null

start_api() {
  echo "Starting API at http://127.0.0.1:8765"
  "$VENV_PY" -m uvicorn app.api:api --host 127.0.0.1 --port 8765
}

start_review() {
  echo "Starting review UI at http://127.0.0.1:8501"
  "$VENV_PY" -m streamlit run app/review_app.py --server.address 127.0.0.1 --server.port 8501
}

if [ "$MODE" = "api" ]; then
  start_api
elif [ "$MODE" = "review" ]; then
  start_review
else
  api_pid=""

  cleanup() {
    if [ -n "$api_pid" ] && kill -0 "$api_pid" 2>/dev/null; then
      kill "$api_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  start_api &
  api_pid="$!"

  echo "API is running in the background for the Chrome extension."
  start_review
fi
