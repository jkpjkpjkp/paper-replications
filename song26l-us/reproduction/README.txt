U.S. LSTM TEST OF THE LABEL HORIZON PARADOX

This is an independent test using the paper's disclosed data structure and
bi-level objective. The original proprietary features are unavailable, so
this experiment cannot establish an exact replication of the published numbers.
The first primary comparison covers LSTM, five random seeds, and all 390
next-day minute candidate labels. It does not cover all ten architectures.

CURRENT STATE

run_status.json describes the automatic reproduction coordinator. While raw
data is downloading, it incrementally builds tensors. It starts primary fitting
only when both ../validation.json and tensor_status.json report complete=true.
Model checkpoints and horizon selections have configuration and checksum receipts
and can be reused after an interrupted fitting run. Final test evaluation happens
after all five seeds' baseline, selection and retraining fits have finished.

smoke/ and smoke.log are engineering checks on dates before July 2024. They
exercise the actual LSTM optimizer and higher-order derivative path, but produce
no financial test metrics. Their shortened epoch/date settings are not the primary
experiment. full_batch_memory_check.json records a real-data 10-support-day,
10-query-day GPU backward check. verify_engineering.py also checks numerical
hypergradients, lost-connection recovery, boundary exclusion, and Nvidia's real
June 2024 split. That split's adjusted label is approximately +0.6495%; the
unadjusted price ratio would incorrectly appear close to -89.94%.

DATA AND SPLITS

Each decision-day S&P 500 member supplies 26 consecutive 15-minute patches.
Input and following label session must both have the normal 390-minute length.
An input stock needs a positive same-day price adjustment factor and eligible
15:59-minute close, plus a valid next-session factor and final-minute close.
The label is (next_minute_close / next_factor) / (today_close / today_factor)-1.
All 390 horizons are retained; individual unavailable minute labels are masked.
At least 100 valid stocks are required for a day. No missing raw price is filled.

Primary decision-date splits are fixed in protocol.json:
  train:      2019-01-01 through 2023-06-30
  validation: 2023-07-01 through 2024-06-30
  test:       2024-07-01 through 2025-07-31
Inputs whose next-session labels enter a different split are removed. August 1,
2025 supplies labels only for July 31. All short input/label sessions are excluded;
their source bars are retained in the parent dataset. See tensor_status.json for
exact excluded dates, sample counts, and missing values.

THE EXPLICIT 72 FEATURES PER PATCH

The raw channels are open, high, low, close, dollar amount, shares and trade count.
Nominal price channels are divided by that stock-day's final eligible close.
Amount, shares and counts are each divided by their own stock-day mean.
All of those normalizers are known at the decision close.

For each of the seven normalized channels, six statistics are calculated over
the observed values in the patch: arithmetic mean; last/first-1; (max-min)/abs(mean);
population standard deviation; third central moment/std^3; and fourth central
moment/variance^2-3. Skewness requires three observed values, kurtosis four.
Constant nonempty channels receive zero skewness/kurtosis. This gives 42 features.

Twenty-one features are pairwise Pearson correlations for all unordered pairs
of channels, using jointly observed minutes. Three observations are required;
a constant channel gives zero correlation. Three additional features are:
  (patch dollar amount / patch shares) / mean eligible patch close - 1
  log(1 + patch shares / patch trade count)
  log(1 + patch dollars / patch trade count)
Four features repeat the stock-day log close and log(1 + daily amount/shares/count).
Two features are the missing eligible-price and missing broad-trade fractions.
Undefined descriptors remain NaN in cache. Feature names and exact code hashes
are included in tensor_status.json. The total is 42+21+3+4+2=72.

Training-only per-feature means and population standard deviations are fitted
across training stocks and patches. Missing features receive the training mean,
and standardized values are clipped to [-10,10]. Validation and test statistics
are never used to fit this preprocessing. There is no target winsorization.

MODEL AND HORIZON LEARNING

protocol.json freezes the configuration before primary test evaluation. It uses
a one-layer 64-hidden-unit LSTM and linear score head. Every loss is a mean of
daily cross-sectional losses, with predictions and labels standardized on the
same observed stocks. The implementation uses 2-2*IC, equal to standardized MSE
for nonconstant predictions; an uninformative constant score receives loss 2.
Baseline and final selected-horizon fits use AdamW at
1e-3, weight decay 1e-4, a 20-date batch, at most 50 epochs and validation patience
5. Both are initialized from the same seed and use the same batch ordering.

BLO warms up for three epochs on the mean of all 390 standardized candidate
labels. Each 20-date training batch is randomly split into disjoint support and
query dates. A differentiable one-step SGD look-ahead uses the weighted sum of
support losses at learning rate 1e-6. Query loss always uses final-minute labels.
Adam updates the 390 softmax horizon logits at 1e-3, with entropy coefficient
1e-3. Look-ahead parameters are written back to the student. Query dates are
training dates; the chronological validation set is reserved for early stopping.
The selected horizon is argmax of the best validation checkpoint's weights;
exact ties choose the latest minute, so uniform weights select the fixed target.
The selected label is then used to retrain from scratch.

The paper's displayed Equation 13 uses gradient-descent look-ahead, while
Appendix A.5 says models use AdamW. This implementation follows Equation 13
for the meta step and AdamW for warmup/retraining; the choice is explicit in
protocol.json. Hidden size, initial learning rate, weight decay, feature details,
missingness policy and exact July cutoffs are additional disclosed choices that
were not fully specified in the paper. No financial test result is used to make
those choices.

RESULTS WHEN COMPLETE

results/status.json must state complete. results/metrics.json will contain daily
Pearson IC, Spearman RankIC, unannualized ICIRs, mean top-decile price return and
sqrt(252)-annualized Sharpe for each seed and the five-seed mean. Standard
deviations use ddof=1. The top decile uses ceil(0.1*N) names with highest score.
These are gross signal metrics, not net execution returns. The paired IC
difference is averaged across seeds by date, with a 20-session circular block
bootstrap confidence interval (10,000 resamples). Predictions and daily metrics
are retained as CSV. us_lstm_comparison.png will show the comparison and learned
horizon weights. The source paper's US LSTM ICs are 0.0314 (standard) and 0.0459
(proposed), but different features and cleaning prevent a like-for-like numerical
comparison. Paper source: /home/jkp/papers/predict/song26l.pdf, Appendix E.3,
Table 6; raw input and preprocessing setup is in Appendix A.

COMMANDS

From /home/jkp/data/song26l-us:
  .venv/bin/python reproduction/build_tensors.py
  .venv/bin/python reproduction/verify_engineering.py --features --download
  /home/jkp/data/alphapareto-reproduction/.venv/bin/python reproduction/verify_engineering.py --model
  /home/jkp/data/alphapareto-reproduction/.venv/bin/python reproduction/train_lstm.py --smoke
  .venv/bin/python reproduction/run_reproduction.py

The coordinator is already running in a detached process; reproduction.pid
records its PID and reproduction.log its output. Do not start a duplicate.
It uses the existing verified Torch 2.0.1 CUDA environment in the other local
reproduction folder without modifying that environment. The data builder uses
this project's .venv. The model manifest records package versions and runtime.
To use a different compatible model environment, pass --model-python /path/to/python
to run_reproduction.py. At most 40% of GPU memory is reserved by Torch; the tested
full 10/10-day meta batch used about 1.7 GiB of allocated GPU memory.
