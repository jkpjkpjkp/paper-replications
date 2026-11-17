#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv venv --allow-existing --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements.lock.txt
if [ ! -f upstream/estimate/src/data/US/sp500/baseline_data_sp500.npy ]; then
    git clone https://github.com/thanhtrunghuynh93/estimate.git upstream/estimate
    git -C upstream/estimate checkout b3e4e100cdb478bdd8803005445102f37926113d
fi
.venv/bin/python scripts/prepare_data.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_suite.py --device "${HERMES_DEVICE:-cuda}"
if [ ! -f upstream/Temporal_Relational_Stock_Ranking/data/relation.tar.gz ]; then
    git clone --filter=blob:none --sparse https://github.com/fulifeng/Temporal_Relational_Stock_Ranking.git upstream/Temporal_Relational_Stock_Ranking
    git -C upstream/Temporal_Relational_Stock_Ranking checkout cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d
    git -C upstream/Temporal_Relational_Stock_Ranking sparse-checkout set data/2013-01-01 preprocess
fi
.venv/bin/python scripts/prepare_rsr.py
.venv/bin/python scripts/run_companions.py --device "${HERMES_DEVICE:-cuda}"
.venv/bin/python scripts/reevaluate.py --device "${HERMES_DEVICE:-cuda}"
.venv/bin/python analysis/no_skip_significance.py
.venv/bin/python scripts/summarize.py
.venv/bin/python scripts/verify_results.py
