# Paper replication sources

For replication quality, read [REPLICATION_STATUS.txt](REPLICATION_STATUS.txt)
and each project's `STATUS.txt`. The 0–3 scale and update guidance are in
[AGENTS.md](AGENTS.md); runnable code alone does not establish a paper match.

Each paper folder contains its implementation, required configurations/dependencies,
and a README with the input contract and commands for rerunning that work. Generated
outputs and historical progress notes are kept in `.replication-local/`. Shared raw
inputs remain in their existing directories; they are not copied into paper folders
or included in the source release.

## Searchable paper references

[PAPER_TEXT_INDEX.txt](PAPER_TEXT_INDEX.txt) lists the paper references in 48
project folders. Top-level projects keep their `PAPER_*` files in `.agent/`;
nested projects retain their existing locations. The index lists each reference
directory. Each has `PAPER_TEXT.txt` and `PAPER_INDEX.json`, recording the
original source, its SHA-256 hash, coverage, page/line locations and search chunks.
PDF-based references also have `PAPER_LAYOUT.txt` for checking table columns.
The extracts preserve the available paper version; equations and figures still
need the original. Abstracts, linked guides and independent study notes are
explicitly distinguished from paper extracts.
Extract source paths and index `path_relative_to_project` values remain relative
to the project root; indexed text/layout filenames are relative to the index.

```bash
rg -n 'search phrase' --glob PAPER_TEXT.txt --glob '!.replication-local/**' --hidden --no-ignore .
python _replication/prepare_paper_texts.py --project cex-dex-hmm
```

The source mapping is `_replication/paper_sources.json`. The generator reads
archived inputs without modifying them and refuses to overwrite edited extracts.
Crypto-only experiment projects are grouped in `CRYPTO/`; see
[CRYPTO/INDEX.txt](CRYPTO/INDEX.txt) for scope. Projects with stock/bond experiments
remain outside that group, including the equity adaptation of ArchetypeTrader.

## Rerun a paper

Start at the collection root and list the available implementations:

```bash
python _replication/run.py list
python _replication/run.py prepare --project alphaqcm
cd .replication-local/workspace/alphaqcm
```

Follow the environment and experiment commands in that paper's README. The launcher
copies only source code into the working directory and links existing local inputs.
It restores local source-reference artifacts and environments when present. On a new
machine, install the documented environment and provide the declared inputs first.
The default preparation does not restore previous training results.

To change a model, configuration or data vintage, create a separate run workspace:

```bash
python _replication/run.py prepare --project alphaqcm --workspace /path/to/new-run
```

Choose `--fresh-inputs` with a new workspace to rebuild derived inputs from the
shared raw datasets. Cached input directories are linked for reuse; do not run a
preparation step that overwrites those links. Choose `--reuse-results` for an explicit
historical replay: generated outputs are restored into independent writable copies
(using filesystem reflinks where supported). The historical archive remains intact.
Existing staged code is not overwritten; use a new workspace after changing source.

The launcher preserves documented legacy path conventions inside the runtime and
records applied path substitutions in `.prepare-PROJECT.json`. It does not change
model equations, invent missing data, or certify a match to published results.

## Paper directories

Papers using language models in their research method, data pipeline, or evaluated
prediction task are grouped under `LLM/`. Writing/code assistance and an LLM
comparison baseline alone do not qualify; see
`LLM/INDEX.txt` for the reason each project is included. Paper and study folders
have `ABSTRACT.txt` files containing a title and concise summary without authors.

| Paper or study | Rerun guide | Monthly firm table |
| --- | --- | --- |
| alphaprobe | [README](LLM/alphaprobe/README.md) | cz-monthly |
| alphaqcm | [README](alphaqcm/README.md) | csi300-price-inputs |
| alphasage | [README](alphasage/README.md) | cz-monthly |
| arboreal | [README](arboreal/README.md) | finance-monthly |
| archetypetrader | [README](archetypetrader/README.md) | monthly-equity |
| cex-dex-hmm | [README](CRYPTO/cex-dex-hmm/README.md) | Not exported; see the paper’s data contract |
| climate-disclosure-source-shift | [README](LLM/climate-disclosure-source-shift/README.md) | Not exported; see the paper’s data contract |
| corporate-graph-anomalies | [README](corporate-graph-anomalies/README.md) | Not exported; see the paper’s data contract |
| diffusion-dfl | [README](diffusion-dfl/README.md) | crsp-substitute |
| doubleadapt | [README](doubleadapt/README.md) | cz-monthly |
| DRO | [README](DRO/README.md) | Not exported; see the paper’s data contract |
| financial-bias-triggers | [README](LLM/financial-bias-triggers/README.md) | Not exported; see the paper’s data contract |
| financial-emoji-transfer | [README](LLM/financial-emoji-transfer/README.md) | Not exported; see the paper’s data contract |
| fraudgnam | [README](fraudgnam/README.md) | Not exported; see the paper’s data contract |
| graph-risk-placement | [README](graph-risk-placement/README.md) | Not exported; see the paper’s data contract |
| healthcare-insurance-fraud | [README](healthcare-insurance-fraud/README.md) | Not exported; see the paper’s data contract |
| hermes | [README](hermes/README.md) | sp500; nasdaq; nyse |
| hga-mt | [README](hga-mt/README.md) | price-only |
| Kronos | [README](Kronos/README.md) | Not exported; see the paper’s data contract |
| limt | [README](limt/README.md) | csi300; csi500 |
| liu26y | [README](liu26y/README.md) | Not exported; see the paper’s data contract |
| Lookahead | [README](Lookahead/README.md) | Not exported; see the paper’s data contract |
| MDP Pricing | [README](MDP%20Pricing/README.md) | Not exported; see the paper’s data contract |
| monthly-coforecast | [README](monthly-coforecast/README.md) | monthly-universe |
| multihorizon-forecast | [README](multihorizon-forecast/README.md) | daily-input-monthly |
| PDE | [README](PDE/README.md) | Not exported; see the paper’s data contract |
| pdu | [README](pdu/README.md) | nasdaq; nyse |
| Portfolio Pilot | [README](LLM/Portfolio%20Pilot/README.md) | Not exported; see the paper’s data contract |
| prism-vq | [README](prism-vq/README.md) | csi300; sp500 |
| prophets-profit | [README](LLM/prophets-profit/README.md) | equity-monthly |
| proximal-portfolio | [README](proximal-portfolio/README.md) | Not exported; see the paper’s data contract |
| RADAR | [README](LLM/RADAR/README.md) | Reuses prism-vq/sp500 |
| rise | [README](rise/README.md) | Not exported; see the paper’s data contract |
| risk-aware-utility-reranking | [README](risk-aware-utility-reranking/README.md) | Not exported; see the paper’s data contract |
| Set Sequence | [README](Set%20Sequence/README.md) | Not exported; see the paper’s data contract |
| song26l | [README](song26l/README.md) | Not exported; see the paper’s data contract |
| stable-alpha | [README](stable-alpha/README.md) | Not exported; see the paper’s data contract |
| temg-tta | [README](CRYPTO/temg-tta/README.md) | Not exported; see the paper’s data contract |

LiMT's US implementation lives in `limt/extensions/us/` and is staged as
`limt-us`. The Lookahead adapter lives in `Lookahead/reproduction/` and is staged
as `lfm-quant`. These extensions share a paper folder while retaining their own
runtime and experiment definitions. `DRO` and `Kronos` remain at the collection
root; `Portfolio Pilot` is grouped under `LLM/`.

The simultaneously maintained `tra`, `SIT`, `stable`, `vision-to-value-evaluation`
and `blocked` projects were left outside this cleanup. Manuscript workspaces
`paper` and `ijcai-paper`, and existing shared data/download tools, remain in place.
Source-only audits and old queues are retained in the local archive rather than
being presented as completed paper implementations.

## Shared inputs

| Existing root | Input role |
| --- | --- |
| `crsp-daily/`, `crsp-monthly/` | Daily/monthly security returns and identifiers |
| `Chen_Zimmermann_Factor_Zoo_2026_10/` | Firm characteristics and factor documentation |
| `compustat/`, `ibes/` | Fundamentals, links and analyst information |
| `csi/`, `global-market-data/` | Existing regional price/universe snapshots |
| `factors-daily/` | Factor-return series |
| `song26l-us/` | Existing US minute bars and security mappings |
| `ravenpack/`, `sp500-earnings-transcripts/` | Existing news/text inputs where required |

`--data-root /path/to/shared-input-root` selects another location with the same
relative layout. Each paper README names any additional original-release inputs,
model checkpoints, or source-reference files required. A public checkout contains
code; it does not grant or supply access to these datasets.

## Generate month–firm–return tables

Install `_replication/requirements-tables.txt` in a suitable environment, then:

```bash
python _replication/month_firm_returns.py --list
python _replication/month_firm_returns.py --project alphaprobe --data-root .
python _replication/month_firm_returns.py --project all --data-root .
```

Outputs are under `.replication-local/tables/PROJECT/TABLE/`, with Parquet data and
JSON provenance containing source/output hashes and timing definitions. Use
`--output-root /path/to/tables` to change the destination. Existing inputs are read
in place; the output is a derived table, not another raw-data archive.

| Field | Definition |
| --- | --- |
| `month` | Calendar month in which the return is realized (`YYYY-MM`) |
| `firm` | Original security identifier; PERMNO is a security, not a company PERMCO |
| `return` | Simple decimal return: 0.01 means 1% |
| `return_observed` | Missing returns remain missing, not zero |
| `prediction_score` | Present only for a model with a correctly aligned monthly score |

The exporter distinguishes price returns from total returns. Daily series are
compounded only on the supplied source calendar; missing required observations
produce missing monthly returns. Price endpoints are not forward-filled. Daily
boundary months are conservatively omitted. Five- and twenty-trading-day model
labels are not renamed as calendar-month returns, and daily scores are not
presented as monthly forecasts. Exact conventions are in
`_replication/table_specs.json` and each table's provenance record.

## Verified local table coverage

| Project | Table | Rows | Securities | Return months |
| --- | --- | ---: | ---: | --- |
| alphaprobe | cz-monthly | 97,641 | 2,968 | 2022-01–2025-12 |
| alphasage | cz-monthly | 97,641 | 2,968 | 2022-01–2025-12 |
| doubleadapt | cz-monthly | 97,641 | 2,968 | 2022-01–2025-12 |
| monthly-coforecast | monthly-universe | 420,000 | 4,377 | 1991-01–2025-12 |
| arboreal | finance-monthly | 417,706 | 4,601 | 2005-01–2025-12 |
| hermes | sp500 | 35,550 | 474 | 2016-02–2022-04 |
| hermes | nasdaq | 59,508 | 1,026 | 2013-02–2017-11 |
| hermes | nyse | 100,746 | 1,737 | 2013-02–2017-11 |
| hga-mt | price-only | 3,698 | 86 | 2017-11–2021-05 |
| diffusion-dfl | crsp-substitute | 44,388 | 324 | 2005-07–2016-11 |
| pdu | nasdaq | 58,482 | 1,026 | 2013-02–2017-10 |
| pdu | nyse | 99,009 | 1,737 | 2013-02–2017-10 |
| limt | csi300 | 83,286 | 661 | 2010-02–2020-07 |
| limt | csi500 | 161,028 | 1,278 | 2010-02–2020-07 |
| limt-us | us-total-return | 313,010 | 1,385 | 2007-02–2025-11 |
| prism-vq | csi300 | 139,620 | 780 | 2010-02–2024-12 |
| prism-vq | sp500 | 169,911 | 837 | 2008-02–2024-12 |
| alphaqcm | csi300-price-inputs | 117,335 | 757 | 2010-02–2022-12 |
| archetypetrader | monthly-equity | 214,058 | 3,696 | 2017-01–2025-12 |
| prophets-profit | equity-monthly | 287,834 | 3,022 | 2000-01–2025-12 |
| multihorizon-forecast | daily-input-monthly | 1,007,648 | 11,490 | 1990-02–2025-11 |

## Build the source release

```bash
python _replication/package.py
```

This uses the explicit source catalogue and produces
`.replication-local/release/replication-source.tar.gz` plus its file/hash manifest.
Shared data, generated tables/results, environments, credentials and Git histories
are excluded. The root `.gitignore` also permits only the selected source folders.
Existing author licenses and notices stay with their code; this cleanup does not
assign a blanket license to third-party material. Nothing is uploaded or published.

RADAR writes a provenance reference to the existing PRISM-VQ SP500 table; it does
not duplicate that table. The current export contains 21 direct tables with
4,025,740 rows, plus this shared-table reference.

## Local archive and verification

`.replication-local/artifacts/` preserves original inputs, results, environments,
source evidence and old READMEs. Move journals and validation reports are directly
under `.replication-local/`; the original coordinator and nested audit directory
layouts are preserved inside the archive. Do not delete that directory if you
intend to reuse the existing cached experiments.

The cleanup was checked with numerical/unit tests, source/command checks, staged
workspace tests, and independent table schema/key/provenance validation. Full GPU
training suites and paid API generation were not rerun. Individual guides preserve
known missing-input, partial-implementation and scientific-fidelity limitations.

Run the shared launcher checks with `python -m unittest discover -s _replication -p test_run.py` and table checks with `python _replication/month_firm_returns.py --self-test` in the table environment. Historical hash-bound checkpoints may reject relocated source; retain the unchanged archive for original-source comparisons.
