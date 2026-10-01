#!/usr/bin/env bash
# AIlobotomy launcher for Linux and macOS.
# The first run creates .venv and installs PyTorch; later runs start instantly.
set -e
cd "$(dirname "$0")"

export HF_HOME="$PWD/hf_cache"
PY="${PYTHON:-python3}"

if [ ! -x ".venv/bin/python" ]; then
  echo
  echo "  First run: setting up the environment, this takes a few minutes"
  echo
  if ! command -v "$PY" >/dev/null 2>&1; then
    echo "  Python 3.10+ not found. Install it and run ./start.sh again."
    exit 1
  fi
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  if [ "$(uname)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  No NVIDIA GPU found, installing the CPU build of PyTorch"
    .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  else
    .venv/bin/python -m pip install torch
  fi
  if ! .venv/bin/python -m pip install -r requirements.txt; then
    echo "  Installation failed. Check your connection and run ./start.sh again."
    rm -rf .venv
    exit 1
  fi
fi

exec .venv/bin/python server.py
