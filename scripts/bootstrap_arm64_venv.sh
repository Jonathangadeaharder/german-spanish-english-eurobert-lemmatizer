#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="/opt/homebrew/bin/uv"
UV_PYTHON="$HOME/.local/share/uv/python/cpython-3.13-macos-aarch64-none/bin/python3.13"

cd "$PROJECT_ROOT"

"$UV_BIN" sync --python "$UV_PYTHON"

"$UV_BIN" run --python "$UV_PYTHON" python - <<'PY'
import platform
import torch

print("python_machine:", platform.machine())
print("torch:", torch.__version__)
print("mps_available:", torch.backends.mps.is_available())
print("mps_built:", torch.backends.mps.is_built())
PY
echo "Created uv-managed project environment at $PROJECT_ROOT/.venv"
