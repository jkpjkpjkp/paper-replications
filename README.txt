Sparse Tangent — original-market published-result replication attempt

COMPARISON REPORTING
Reports now lead with the requested annualized-Sharpe interpretation:
SR_annual = sqrt(252) * SR_daily. Both the mean and the sample SD over seeds
receive this factor. Paper table means and SDs are copied unchanged. Section5.3
explicitly calls its metric daily; interpreting its displayed values as annualized
is an assumption, not a verified author convention. Daily comparisons remain
visible, and maximum drawdown is never annualized. Numerical matches require both
mean and SD agreement and complete seed sets; they do not prove exact replication.
The shared conversion is in reproduction/compare_published.py. Model training,
return traces and the frozen protocol.json retain their original definitions.

PFL BASELINE (current work)
  .venv/bin/python reproduction/replay_pfl.py --check-training
This reuses verified MSE predictors and completed daily traces, fills missing
runs through unmodified upstream/src/main.py, and independently retrains seed16
in all four markets to compare every model parameter and all six hard-OSCAR /
mSSRM return traces. Results: results/pfl_replay/report.html; direct_checks.json
records measured source equivalence. Run without --check-training to rebuild
the report and fill missing runs without repeating independent retraining.

PFL uses prediction MSE only, the published512/256 network and batch64, and
released Adam/tanh/dropout.1/weight-decay1e-4/300 sampled updates. One model per
market/seed feeds every cardinality and optimizer. oscar_hard is the conventional
OSCAR baseline; soft oscar is a distinct, potentially dense selector. This is a
release-faithful replay on documented current inputs, not an exact reproduction
of the unavailable author experiment. SD-relaxation's published cardinality
matching is absent from the release and is not claimed as reproduced.

The earlier paper_method.py trial below is a separate literal-equation
reconstruction with explicit assumptions; it is not the primary PFL replay.

DATA AND CLEANUP
reproduction/download_specification.py is the maintained downloader; it writes
to a separate output directory and preserves the frozen experimental panels:
  .venv/bin/python reproduction/download_specification.py --output results/new_download
Superseded acquire*.py helpers were removed after checking callers. Deletion
hashes and any redundant-checkpoint pruning are recorded in
sources/cleanup_2026-11-11.json. Historical acquisition receipts remain intact.

2026-11-11 fidelity audit: results/fidelity_audit/report.html and audit.txt.
Fresh public-source downloads and unmodified-source reruns are isolated there.
The paper comparison now defaults to hard OSCAR for Historic/PFL (Section 5);
the former soft-baseline comparison remains in fidelity_audit/soft_baseline.
No exact published-number reproduction has been established.

PUBLISHED-METHOD IMPLEMENTATION (separate experiment)
reproduction/paper_method.py implements Algorithm 1 / Eq. 5 as printed:
t>=0, squared-vector loss, full mini-batch epochs, consistent raw-return units,
soft sigmoid gates, rolling covariance and a 512/256 predictor. An equivalent
unit-ball conic coordinate system improves conditioning without changing the
constraints. Conventional OSCAR uses a hard selected support and Eq. 1 risk.
reproduction/paper_method_protocol.json separates published requirements from
unreported choices, including the five-full-epoch trial budget and SGD update.
The stable-membership cohort is still unverified; this is not an exact-number
claim. results/paper_method/report.html reports run completion, failures,
actual holdings, and invalid-budget cases rather than hiding those cases.

Rebuild that report:
  .venv/bin/python reproduction/report_paper_method.py
Run a new isolated trial (choose a new --output directory):
  .venv/bin/python reproduction/paper_method.py \
    --data data/daily/euro50_160101_251231.csv --method dfl \
    --seed 16 --rho 10 --epochs 5 --output results/new_paper_method_trial
The five-epoch budget is a declared reconstruction choice; the paper does not
publish the number of epochs. No trial's test result selects that budget.

Target publication:
Jeon, Choi, Bae, Lee & Kim (2026), Decision-focused Sparse Tangent Portfolio
Optimization, ICML/PMLR 306, pages 51108–51125.
https://proceedings.mlr.press/v306/jeon26a.html

This is a new original-market release replay, separate from the earlier corrected
CRSP methodological transfer. The old /home/jkp/data/sparse-tangent-reproduction
directory was already absent when this run began. We reused the official source
recovered by paper-claim-validation, its exact original commit, the original
dependency pins, published target tables and prior source-equivalence diagnostics.

Official code: https://github.com/feuerwerksh/Diffble-card-SR
Pinned commit: 8bc1e802b834bf82fe144e98cdd9e2227607c3e7
upstream/src remains unchanged. Check with git -C upstream diff -- src.

RUN TARGET / HONEST LIMIT
The objective is to rerun the original four-market setting and compare published
cells, not to substitute a corrected portfolio task. Author financial CSVs are
not distributed. Current downloads do not establish the original cohort or price
vintage. Euro has47 assets as in the paper; current FTSE has92 vs93 and Nikkei206
vs208. KOSPI acquisition is separately recorded. The current Korean fallback
constituent list comes from an archived Wikipedia revision after the KRX public
endpoint returned LOGOUT; it is not the author's missing KRX export.
The Korean return generator completed with159 assets versus paper162. Column
order and provider receipts are retained. The FTSE generator can also retain
valid bare-code US collisions (verified AAL is American Airlines, while AAL.L is
Anglo American). Raw inputs are retained and the issue limits any literal claim
to reconstructing the original market panel.

Paper explicit values are used: 2016–2025, chronological80/20, lookback100,
MLP512/256, batch64, lr.001, alpha.5, covariance shrinkage.10 and top-k32 bisections.
The five seeds are the author's aggregate.py DEFAULT_SEEDS:
16,37,47,214,1947. The paper does not print its seed IDs. Other unspecified choices
use release defaults:300 sampled-batch updates, tanh, dropout.1, weight decay1e-4.
The release performs one sampled batch per so-called epoch, not full passes.
These are documented reconstruction choices, not newly discovered paper details.
The public OpenReview forum and original submitted PDF were also inspected.
That draft uses256/128 and fixed covariance and reports different learned Table1
values (Euro DFL10%1.818 versus final0.955). This run targets final512/256 and
rolling covariance. The original anonymous repository is now expired. Evidence
is archived in sources/setting_reconciliation.json and the associated documents.

Release preprocessing and optimization are deliberately preserved: full-panel
StandardScaler, normalized prediction/covariance units, norm(L*y) risk constraint,
t>=1, soft gates without repairing actual support, no leverage/position/turnover
limits, per-asset-mean MSE plus negative raw realized return, and raw daily Sharpe.
No risk-free subtraction, transaction costs or market hedge. Source artifacts
remain daily; the comparison reports additionally apply sqrt(252) as described above.

Unresolved paper/release differences remain explicit: full-training Historic mean
caption vs rolling source mean; squared-vector loss vs source asset-mean MSE;
paper t>=0 vs source t>=1; unreleased SD realized-cardinality matching procedure.
Primary Historic/PFL OSCAR comparator now uses selector=oscar_hard, following
Section 5's explicit support restriction. This choice follows the specification,
not numerical agreement. The old soft comparison is retained as a sensitivity.
DFL always uses the differentiable oscar selector.

EXECUTION
systemctl --user status sparse-original-study.service
runtime/status.json       current stage and job
runtime/jobs.json         all88 jobs and completion/failure/data-block states
runtime/training_status.json  current update progress
runtime/sdr_status.json       per-date SDR progress
runtime/logs/             separate job logs
artifacts_released/       unmodified author artifact schema and saved checkpoints
results/published_comparison/report.html  all84 Table1 cells, including unfinished

The queue runs: Historic; interleaved PFL and DFL for each market/seed;
Euro alpha sensitivity; full-stride Historic/PFL SD-relaxation. It never interpolates
SDR dates. Slow SDPs can dominate total wall time; the queue runs independently of
the chat. Check actual status rather than interpreting a service launch as success.
Data-blocked markets remain visible and become eligible automatically if acquired.
Numerically failed jobs remain failures rather than changing solvers or dropping
dates. A report only becomes complete after all required runs actually finish.

The initial run used a shared4GiB cgroup cap. On2026-11-11 the user explicitly
removed the RAM restraint: MemoryHigh, MemoryMax and MemorySwapMax are now infinity.
CPUQuota400% and one active experiment worker remain. The frozen protocol retains
the original resource declaration so training checkpoint identities remain valid;
runtime/resource_override.json records the authorized execution-only change.
runtime/resources.json and resource_history.json distinguish actual cgroup records.

The memory adapter preserves the same64 sampled observations, model forward and
dropout order, source objective, solvers and Adam update, while releasing conic
derivative contexts after reverse-order per-sample backpropagation. Current Euro
reference comparison: two full64 Adam updates, gradient and parameter error0,
same Torch RNG, in runtime/equivalence.json. Prior21 checks are archived in sources/.
Equivalence is certified on finite-loss paths; nonfinite paths stop explicitly.
Adam and RNG state checkpoint every25 updates; SDR checkpoints every10 dates.

Recreate environment:
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r upstream/requirements.txt
The resolved environment is saved in sources/environment.lock.txt.

Manual job example (production resource policy is in the systemd unit):
  .venv/bin/python reproduction/run_native.py --mode dfl --dataset euro50 \
    --seed 16 --alpha 0.5 --hidden_dims 512 256 --batchsize 64 --epochs 300 \
    --cuda -1 --splits test --save_dir artifacts_released --selector oscar

The released source and paper are authoritative evidence for what was implemented
and specified; numeric matching on a contemporary snapshot is not author-exact
replication. The comparison reporter always preserves that distinction.
