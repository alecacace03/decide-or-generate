#!/usr/bin/env bash
# One-command launcher: creates a virtualenv, installs dependencies, downloads the two
# models on first run (~2 GB) and opens the dashboard at http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This prototype runs on MLX, which needs a Mac with Apple silicon (M1 or newer)." >&2
  exit 1
fi

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ and run ./run.sh again." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

# Plain HTTPS downloads are more reliable than the Hugging Face xet transport on some networks.
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m dual_engine.server "$@"
