SONG ET AL. (2026): U.S. S&P 500 DATA FOR TESTING THE LABEL HORIZON PARADOX

Purpose and completion

Reproduction work continues in reproduction/. Its README.txt describes the
explicit feature set, five-seed LSTM comparison, engineering checks and automatic
coordinator. That coordinator builds available tensors during downloading and
starts primary modeling only after the full data audits pass. See
reproduction/run_status.json and reproduction/reproduction.log for its progress.

This folder assembles U.S. inputs for Scenario 1 / Appendix E.3 of "The Label
Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting."
The paper uses a market-close decision, next-trading-day close return as the
evaluation target, and returns to intermediate minutes of that next day as
candidate supervision labels. U.S. inputs cover 390 regular-session minutes
and form 26 nonoverlapping 15-minute feature patches on full-length days.

Minute extraction is resumable and may still be running. The detached full run
was originally launched on 2026-11-03 with four dedicated workers. Following
intermittent WRDS connection failures on 2026-11-05, the coordinator starts
with three workers and retries with two, then one, if an attempt fails.
Completed receipts are retained; sale-condition filters and aggregation are
unchanged. run_status.json records
the supervisor state, download.pid identifies its process, and download.log
records per-session progress. The supervisor retries an interrupted download
up to three attempts, then automatically runs the complete independent audit.
This README does not
certify that all bars have arrived. Read status.json for downloader progress
and root-level validation.json for the latest independent audit. Full-period
completion requires validation.json complete=true, 1,655 verified sessions,
zero pending sessions and zero integrity errors. A benchmark_complete status
or a successful --partial validation command does not establish completion.
support/validation.json certifies the supporting-data build only.

The planned period contains 1,655 XNYS sessions from 2019-01-02 through
2025-08-01. August 1 is an extra label session for the final July 31 test input.
Observed CRSP market dates exactly matched the exchange_calendars XNYS
schedule when support was built. Thirteen sessions have official early closes.

Scope and limits of reproduction

The paper specifies seven minute variables: open, high, low, close, amount,
volume and transaction count. The supplied data supports constructing these
inputs and all next-day minute price labels. It does not contain trained
models, predictions or a completed reproduction of the reported results.

The original feature list and original experimental code are not public.
Appendix A.3 describes feature categories but omits the complete indicators
and formulas. The exact historical-universe, vendor-cleaning and missing-data
choices are also not fully specified. Our dated membership, trade filters,
session boundaries and adjustment policy are explicit below so a U.S. test
can be repeated without claiming an exact reconstruction of those choices.

author-demo/ is the authors' separate public demonstration from
https://github.com/Chenhui-Song/label-horizon-paradox . It uses daily-frequency
A-share Alpha158 factors and 1-10 day VWAP labels. Its README explicitly says
it is not the paper's experimental codebase. It is retained as a methodological
reference; it is not a substitute for this U.S. minute dataset, and its internal
DolphinDB data connection is not accessible here.

Folder layout

  metadata/       WRDS historical S&P 500 membership/name intervals, official
                  CRSP-TAQ links, TAQ daily master files and trade partitions.
                  Each downloaded file has a receipt with SQL, row count,
                  byte size and SHA-256.
  support/        Dated membership, trading calendar, raw daily CRSP auxiliary
                  data and resolved daily TAQ symbol mappings; build audits.
  bars/YYYY/      One YYYY-MM-DD.parquet and matching .receipt.json per session
                  once extracted and transferred. Error files document failed
                  attempts; their presence alone is not a completion signal.
  reference/      Downloaded TAQ / CTS vendor specification PDFs.
  author-demo/    Separate public daily A-share example described above.
  .venv/          Project Python environment; requirements.txt pins packages.
  status.json     Latest downloader state, which can include a benchmark run.
  download-plan.json  Full download dates, filters and worker implementation.
  validation.json Independent audit of received files and remaining work.
  validation_gaps.parquet  Security/day coverage gaps for audited bar files.

Universe and security identification

metadata/sp500_membership.parquet contains raw CRSP dsp500list_v2 membership
intervals. Endpoints are treated as inclusive.
support/sp500_daily_membership.parquet gives the exact daily universe and
two flags: membership_active and
membership_previous_session. The union includes stocks that were members on
the prior trading session so their next-day labels can still be observed
after index removal. Select membership_active for the decision-day universe.

support/crsp_daily_auxiliary.parquet preserves all 50 source daily CRSP columns
and adds membership/calendar fields. The source files are external to this
project at ../crsp-daily/data/crsp_a_stock/dsf_v2/2019.parquet through
2025.parquet. The support builder rechecks their transfer receipt hashes,
sizes and row counts. Raw missing prices, returns, CUSIPs and factors remain
missing. The build found 34 active membership days without a CRSP daily row
and 47 auxiliary rows with missing cumulative adjustment factors.

support/taq_daily_mappings.parquet resolves daily symbols by preferring the
best available official WRDS CRSP-TAQ match level (lowest numeric level).
Where that link is absent, it requires an exact same-day match of dated CRSP
cusip[8] to the first eight characters of TAQ master cusip[9]. This fallback
is needed in every year, including 2025 when the official link has no rows.
Static header CUSIPs and guessed ticker histories are not used. TAQ symbol_15
is parsed as ROOT plus optional SUFFIX; blank suffix is normalized to NULL.

There are 834,212 resolved security/day mappings out of 834,313 active/prior
universe rows. The 101 unresolved days comprise 59 without a dated CRSP row,
41 without a valid dated CUSIP and one ambiguous assignment. Conflicting
symbol/date-to-PERMNO candidates are excluded, not guessed. Inspect
support/taq_mapping_unresolved.parquet, taq_mapping_rejected.parquet and
mapping_validation.json for exact records, provenance and source hashes.

Minute data conventions

Source trades are WRDS NYSE Daily TAQ consolidated trades,
taqm_YYYY.ctm_YYYYMMDD. Aggregation occurs within the authorized WRDS account;
raw trade files are not downloaded. Each received daily Parquet has a remote
receipt and is checked locally against its SHA-256, size, bar count and
aggregated trade/share/dollar totals.

Session time is America/New_York, with daylight-saving changes reflected in
the calendar's UTC open/close fields. minute_start is a timezone-naive NEW
YORK LOCAL timestamp, not UTC. Minutes are left-closed/right-open:
minute_index=0 means [09:30,09:31), and index=389 means [15:59,16:00).
Regular extraction uses 09:30 <= trade time < the official session close.
An exact 16:00 print and after-hours trades are outside that interval.

Early-close sessions retain their actual scheduled minutes, normally 210,
rather than being padded to 390. Their expected_minutes and early_close
flags come from the XNYS schedule, not inferred trading volume. A fixed
26-patch model must explicitly exclude or mask shortened sessions and report
that choice, including its effect on the following-day label window.

The seven principal variables are:
  open, high, low, close   Eligible trade prices, in unadjusted dollars/share.
  amount                  Sum of price * shares, in dollars; not close*volume.
  volume                  Sum of qualifying trade sizes, in shares.
  trade_count             Number of qualifying TAQ trade prints.

Prices and dollar amounts are float64; share volumes and counts are int64.
Keys are (date, permno, minute_index). Symbol components and taq_symbol_count
are included for provenance. raw_open/raw_high/raw_low/raw_close retain OHLC
from the broader volume-eligible print set. eligible_amount/eligible_volume/
eligible_trade_count use the same narrower eligibility as principal OHLC.
SIP time bounds and other worker diagnostics permit additional audits.

The driver currently applies these explicit SQL predicates:
  Base: price > 0 AND size > 0 AND tr_corr IN ('00','01')
  OHLC: coalesce(t.tr_scond,'') ~ '^[ @EFKX56O]*$'
  Volume/amount/count: coalesce(t.tr_scond,'') !~ '[MQ9]'

Thus principal OHLC and principal volume totals intentionally use different
sale-condition eligibility. The OHLC regular expression permits only the
listed characters, including blank/@. Broad totals exclude any condition
string containing M, Q or 9. Both sets use the base price/size/correction
filter; only correction-code values 00 and 01 are retained. The predicates,
worker source hash and dated mapping hash are recorded in each receipt.
These are disclosed data-construction choices, not a claim that the authors
used the same vendor filters.

Missingness is preserved. A minute without qualifying trades has no row.
A minute with broad trades but no OHLC-eligible print can have positive
volume and null principal OHLC. No zero-return, zero-volume, forward-filled
or synthetic-price rows are inserted. validation_gaps.parquet distinguishes
unmapped securities, mapped securities with no trades, absent trade minutes
and missing eligible-price minutes. Missing session files remain pending
download work; they are different from source gaps in a received session.

Returns, adjustments and date splits

Minute prices are unadjusted. Join on (permno, date) to the daily auxiliary
file and divide each price by that day's positive dlycumfacpr to construct
split-adjusted price levels. For example, a next-day intermediate price
return is (P_next,minute / F_next) / (P_today,last / F_today) - 1, with F the
daily cumulative price adjustment factor and the last eligible continuous
session price as the endpoint. Null/nonpositive factors or missing endpoint
prices require an explicit mask/exclusion; no replacement is supplied.

Price-ratio labels do not add ordinary cash dividends. CRSP dlyret is a total
return measure and dlyretx is a price return measure; they must not be silently
interchanged with one another or with minute price-ratio labels. Dividend
amounts and both return fields are retained for a separately defined test.

The paper gives overlapping month labels at split boundaries. This package
chooses the following nonoverlapping decision-date intervals:
  train:           2019-01-01 <= date < 2023-07-01  (1,132 sessions)
  validation:      2023-07-01 <= date < 2024-07-01  (250 sessions)
  test:            2024-07-01 <= date <= 2025-07-31 (272 sessions)
  label_extension: 2025-08-01                       (1 session)

Calendar previous_session/next_session fields identify actual trading days.
Purge training/validation examples whose future label windows cross their
split boundary, including the June 30, 2023 and June 28, 2024 input days for
next-session labels. August 1 is allowed only to finish July 31 test labels;
it is not another train/validation/test decision day. Fit any feature scaling
or imputation on the training data only. Cross-sectional label/output
standardization and the exact candidate horizons remain model-build steps.

Rebuild, resume and validate

Run from /home/jkp/data/song26l-us. Existing metadata files and completed day
files are checked and reused when their receipts/configuration match. Do not
start a second downloader while this folder's .download.lock is held.

  cd /home/jkp/data/song26l-us
  .venv/bin/python download_metadata.py
  .venv/bin/python build_support.py
  .venv/bin/python build_mappings.py
  .venv/bin/python run_full_download.py

For an individual download without the supervisor, use
  .venv/bin/python download_minutes.py --workers 4
and then run validate_download.py. Receipts record the exact worker hash;
changing the worker implementation invalidates reuse of its earlier receipts.
Use --dates YYYY-MM-DD only for an explicitly limited benchmark. For a
progress audit during a download, run:

  .venv/bin/python validate_download.py --partial

The final validator checks all 1,655 planned sessions, source/mapping receipts,
hashes, schemas, keys, times, OHLC bounds and aggregate totals. Read its complete
flag and errors/pending lists even when --partial returns exit status zero.

Access uses the existing external WRDS JupyterHub token at
/home/jkp/data/wrds-crsp-download/wrds-jupyter.token and bridge helpers in
/home/jkp/data/ibes. The private token is not copied into this project.
The token authenticates the WRDS account; source-table permissions still
depend on that account's subscriptions. Rebuilding on another machine needs
the external verified CRSP daily files, required helper modules and a valid
private token, with configured paths updated accordingly.
