# Economic-value diagnostics for the LiMT US transfer

Re-evaluate the completed LiMT US predictions using the fixed portfolio and
utility settings in `protocol.json`. This is an analysis-only extension; it
neither retrains the model nor reproduces the entire referenced RFS experiment.
The primary portfolios are before-cost, dollar-neutral and 100% gross. Risk-scaled
utility and inference are separate diagnostics.

Prepare the parent LiMT US workspace with `--reuse-results`, or first complete
the base experiment. The required inputs are the original data, three model
prediction sets and LightGBM baseline. Set the two interpreter variables as in
[the parent guide](../../README.md), then run from the **limt-us workspace root**:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$LIMT_MODEL_PYTHON" -m pytest -q
"$LIMT_ANALYSIS_PYTHON" experiments/rfs_frontier/run.py
"$LIMT_ANALYSIS_PYTHON" experiments/rfs_frontier/checks/source_audit.py
"$LIMT_ANALYSIS_PYTHON" experiments/rfs_frontier/checks/finalize.py
```

Outputs are generated under this experiment's working directory. Inspect the
report, source payoff audit and `final_checks.json`. `sources/upstream/` retains
reference R source and configuration; it is not the entry point for this Python
re-evaluation. The original source mapping is preserved in the local archive as
`sources/protocol-mapping.txt` and restored by workspace preparation.
