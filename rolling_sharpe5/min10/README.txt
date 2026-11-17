MINIMUM TEN FACTOR POOLS

Open results/report.html.
Keep prior strict passers, top up to10 using worst signed annual t over prior5years.
Primary original universe: complete1990-2022;2023 option data unavailable.
No-options-category sensitivity: complete1990-2025. Each pool has10 factors.
EARC fees5bp, incumbent factors gross. Equal weight and fixed prior-data mean-variance allocation.

Reproduce from project root:
.venv/bin/python rolling_sharpe5/min10/select_pools.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python rolling_sharpe5/min10/analyze.py
.venv/bin/python rolling_sharpe5/min10/report.py
