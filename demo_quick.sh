#!/bin/bash
# Quick Demo for TritonFlow - Automated 5-Stage Live Compiler Demo
# Usage: bash demo_quick.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${DIR}/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

exec "$PYTHON" "${DIR}/demo.py" --quick "$@"
