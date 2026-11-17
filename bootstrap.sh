#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
task_commit=3575f1f94a14620e089422f52cf9e1a6766ec7b1
if [ ! -d upstream/.git ]; then
    git clone --filter=blob:none --depth 1 --no-checkout https://github.com/AdaMLLab/up-for-ocp.git upstream
    git -C upstream sparse-checkout set core tests/configs tests/datasets/raw tests/datasets/processed/AXP tests/datasets/processed/AAPL tests/datasets/processed/AMZN tests/datasets/processed/GOOGL tests/datasets/processed/elec2
    git -C upstream fetch --depth 1 origin "$task_commit"
    git -C upstream checkout --detach "$task_commit"
fi
test "$(git -C upstream rev-parse HEAD)" = "$task_commit"
if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.12.13 .venv
fi
uv pip sync --python .venv/bin/python requirements.lock.txt
mkdir -p reference/pid
task_pid_base=https://raw.githubusercontent.com/aangelopoulos/conformal-time-series/b729c3f5ff633bfc43f0f7ca08199b549c2573ac
if [ ! -f reference/pid/methods.py ]; then
    curl -fL --retry 2 "$task_pid_base/core/methods.py" -o reference/pid/methods.py
fi
if [ ! -f reference/pid/LICENSE ]; then
    curl -fL --retry 2 "$task_pid_base/LICENSE" -o reference/pid/LICENSE
fi
sha256sum --check reference/pid/SHA256SUMS
