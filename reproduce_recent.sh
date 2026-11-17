#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Requires the user's existing local CRSP daily data and historical security names.
.venv/bin/python analysis/recent/prepare.py
.venv/bin/python analysis/legacy_execution_audit.py
.venv/bin/python analysis/recent/run.py --device "${HERMES_DEVICE:-cuda}"
.venv/bin/python analysis/recent/rescore.py
.venv/bin/python analysis/recent/report.py
.venv/bin/python analysis/recent/verify.py
