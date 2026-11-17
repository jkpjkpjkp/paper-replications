# From vision to value: replication of Table 1

Reproduce the size-by-signal portfolio experiment in Table 1 of **Byun, Na and
Song, *From vision to value: Stock chart image-driven factors and their pricing
power*** ([SSRN 5407028](https://ssrn.com/abstract=5407028)). The comparison uses
the January 25, 2026 preprint, page 11.

This package starts with the authors' released monthly portfolio returns and
recomputes all **48 numeric cells**: mean returns and Newey-West t-statistics for
ViT and CNN, three size groups, and Low/Neutral/High/High-minus-Low portfolios.
All 48 match the paper at its printed precision of three decimals. It runs on a
CPU without credentials, a GPU, raw CRSP data, or neural-network training.

## Quick start

Use Python **3.11 or 3.12** and [uv](https://docs.astral.sh/uv/). From this
directory (or the extracted source release):

```bash
uv sync --frozen --python 3.11
uv run --frozen v2v-replicate
```

The first run downloads two aggregate CSVs (about 402 KB total) from the pinned
upstream commit and verifies both SHA-256 hashes before calculation. Dependencies
and their distribution hashes are pinned in `uv.lock`. The default input cache
is `.cache/vision-to-value/`; results go to `outputs/table1/`.

Expected output:

```text
Table 1: 48/48 numeric cells match at 3 decimals (288 months, HAC lag 6).
Outputs: outputs/table1
```

To reuse downloaded inputs without network access:

```bash
uv run --frozen v2v-replicate --offline
```

Here `--offline` applies to data acquisition. For a completely disconnected
machine, install the environment beforehand and use
`.venv/bin/python -m vision_to_value --offline` (Windows:
`.venv\Scripts\python -m vision_to_value --offline`).

An existing checkout of the authors' repository can be used directly:

```bash
uv run --frozen v2v-replicate --upstream-dir /path/to/vit-sdf-lasso --offline
```

The runner reads only `sdf_analysis/sorted_portfolio/{vit,cnn}_3by3.csv` and
requires the pinned file hashes regardless of the checkout's current HEAD. It
does not modify that checkout or run its notebooks. `--cache-dir` and
`--output-dir` accept explicit paths; defaults are relative to the current working
directory. Re-running replaces the four named result files in the output
directory. Use different output directories for simultaneous runs.

## Replication method

1. **Pin the source.** Inputs come from
   [finxlab/vit-sdf-lasso at bde5d07442fdea2c230641ebfbdd0ea00b5021fc](https://github.com/finxlab/vit-sdf-lasso/tree/bde5d07442fdea2c230641ebfbdd0ea00b5021fc).
   URLs, byte counts, and hashes are recorded in
   [`inputs.json`](src/vision_to_value/inputs.json). A bad hash is an error;
   there is no fallback to the latest upstream version.
2. **Validate the panel.** Each model must have exactly 2,592 rows: nine
   size/signal cells for every month from January 2001 to December 2024, inclusive.
   Reject duplicate keys, missing months or cells, invalid group codes, and
   nonfinite returns. Sort and join by calendar month, never by row position.
3. **Use the paper's return series.** Select `Cap_VW_return`, in decimal units.
   Size codes 1/2/3 map to Micro/Small/Big; signal codes 1/2/3 map to
   Low/Neutral/High. For each size group construct H-L month by month as
   `High - Low`. This gives 24 series, each with 288 observations.
4. **Estimate monthly performance.** Compute `mean_pct = 100 * mean(return)`.
   Fit a constant-only OLS regression for each series. Use Newey-West HAC with
   **six lags, Bartlett weights, and no finite-sample correction**, matching the
   released notebook. The t-statistic is the intercept divided by its HAC
   standard error. Returns are not annualized; there is no risk-free subtraction
   or additional fee adjustment at this stage.
5. **Compare to the paper.** Independently transcribed targets are in
   [`table1_expected.csv`](src/vision_to_value/table1_expected.csv). Compare each
   recomputed mean and t-statistic after rounding to three decimals. Save the
   full-precision computed value and difference as well as the pass/fail flag.
   Matching printed numbers does not imply bitwise identity.

For an independent numerical check, the tests compute the constant-only HAC
variance directly. With demeaned returns `e[t]`, `T = 288`, and lag `k`:

```text
Var(mean) = (sum(e[t]^2)
             + 2 * sum_{k=1..6} (1 - k/7) * sum_{t=k..T-1} e[t] * e[t-k]) / T^2
t = mean(return) / sqrt(Var(mean))
```

The paper describes June NYSE size breakpoints at the 20th and 50th percentiles,
signal thresholds at the 30th and 70th percentiles of NYSE non-Micro stocks, and
monthly capped value weights. The released sorting code caps capitalization at
the NYSE 80th percentile and applies a 10-basis-point entry and exit fee via
`(1 - 0.001) * (1 + raw_return) * (1 - 0.001) - 1` before portfolio aggregation.
These operations are already embedded in the supplied `Cap_VW_return`. This
replication does not rebuild or independently validate those upstream operations.

## Outputs and interpretation

| File | Contents |
| --- | --- |
| `table1.csv` | 24 series: mean return in percent, HAC t-statistic, month count |
| `comparison.csv` | 48 published/computed values, differences, and match flags |
| `summary.json` | Overall result, sample dates, inference settings and scope |
| `manifest.json` | Input URLs/hashes, upstream commit, Python/package versions, code and output hashes |

High-minus-Low monthly mean returns, with HAC t-statistics in parentheses:

| Size | ViT | CNN |
| --- | ---: | ---: |
| Micro | 0.799% (3.055) | 0.404% (2.483) |
| Small | 0.262% (2.060) | 0.234% (1.682) |
| Big | 0.179% (1.718) | -0.002% (-0.012) |

Exit status is **0** for 48/48 matches, **1** for numerical mismatches (diagnostics
are still saved), and **2** for invalid inputs or execution errors. If a download
fails, retry or provide the pinned checkout. A corrupt cached CSV must be removed
explicitly before downloading it again.

This is an **artifact-based numerical replication of one experiment**. It does
not retrain CNN/ViT models, regenerate images, verify stock-level predictions,
reconstruct CRSP portfolios, run the SDF/LASSO experiments, or establish that the
published strategy is economically implementable. No claim is made about equality
to a different version of the paper. The local CRSP/CZ evaluation uses a different
protocol and is outside this public package.

## Tests and release

```bash
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python tools/build_release.py
```

Tests use synthetic data and run offline. They check HAC against the direct
formula, H-L date alignment, complete comparison coverage, bad inputs, checksum
failures, cached/offline acquisition, and CLI exit behavior. CI additionally
downloads the pinned inputs and checks 48/48 matches on Python 3.11 and 3.12.
Local validation evidence is recorded in [`docs/VALIDATION.md`](docs/VALIDATION.md).

`tools/build_release.py` creates
`dist/vision-to-value-replication-0.1.0.tar.gz`, its SHA-256 sidecar, and an internal
file-hash manifest. An explicit file allowlist includes only source, tests,
documentation, the dependency lock, CI, and notices. **Use this archive as the
source for a public repository**, rather than uploading the mixed research
workspace. It contains no downloaded inputs, PDFs, weights, local datasets,
environments, or legacy results. The archive is prepared locally; the script
does not publish it.

The previous root `README.txt` was preserved unchanged as `README-evaluation.txt`
in the working directory. That legacy guide, `scripts/`, `requirements.txt`,
`data/`, `results/`, `sources/`, `upstream/`, and `vendor-py311/` remain local and
are excluded from both the public archive and normal git staging. The old
`requirements.txt` belongs to that evaluation; this package uses `pyproject.toml`
and `uv.lock`. These legacy files are absent from an extracted public release.

## Attribution

Replication code is MIT licensed; see [`LICENSE`](LICENSE). The authors' upstream
license is retained in [`docs/UPSTREAM_LICENSE`](docs/UPSTREAM_LICENSE).
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records attribution, paper
version, input provenance, and the distinction between code and data rights.
[`CITATION.cff`](CITATION.cff) contains the paper citation. Cite the paper and state
the replication package version and pinned upstream commit when using results.
