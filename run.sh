#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/runtime${PYTHONPATH:+:$PYTHONPATH}"
exec "${PDU_PYTHON:-python3}" "$@"
