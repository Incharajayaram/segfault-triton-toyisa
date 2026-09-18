#!/usr/bin/env bash
# Setup script for TritonFlow development environment
set -e

echo "=== 1. Creating clean virtual environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "=== 2. Upgrading pip & installing dependencies ==="
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .

echo "=== 3. Building C++ Emulator Extension ==="
mkdir -p build && cd build
cmake .. -DPython3_EXECUTABLE="$(which python3)" -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"
cd ..
cp build/src/tritonflow/emu/cpp/_emu_cpp*.so src/tritonflow/emu/ 2>/dev/null || true

echo "=== 4. Running verification suite ==="
pytest -q

echo "🎉 TritonFlow environment successfully configured and verified!"
