#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
UV_PYTHON_INSTALL_MIRROR=https://github.com/astral-sh/python-build-standalone/releases/download uv python install 3.8.20
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.8.20 .venv
fi
uv pip sync --python .venv/bin/python --find-links https://download.pytorch.org/whl/torch_stable.html environment.lock.txt
cd sources
../.venv/bin/python setup.py build_ext --inplace
