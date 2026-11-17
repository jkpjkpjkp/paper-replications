# Reproduction protocol and evidence

The experiment was specified before scoring the full test set. The paper and author code were retrieved on 2026-11-06. Search and retrieval details are in `sources/search_log.json`; the complete paper read is saved in `sources/paper-browser.txt`.

Data availability update (2026-11-17): all six author-share pickle files are
present in `../.replication-local/artifacts/limt/AAAI 2026-HiMT-Dataset/` relative
to this source folder. Size, hash and sample-index inspection confirms 2008-start
training data for both universes; see `.agent/AUTHOR_DATA_INVENTORY.json`.
References below to unretrieved author data describe the original community-data
run. Its completed results still use the 2010-start snapshot; using the newly
verified inputs requires a separate preprocessing audit and training run.

## Data and time alignment

- Local provider: `/home/jkp/data/csi/qlib/cn_data`, pinned Hugging Face revision `ee0fa6b2299cb5e927d2df491888c84f6c9c027a`; archive SHA256 `50d6fbb35634eb858f0e665e4811cbcbdcaccfba1d1e0084e8b7227deae17757`. Its vendor prices and membership history are not independently certified.
- Stock universe: historical CSI membership intervals. Stocks need an observed current close and positive current volume at inference. Future label availability never decides the input universe or backtest candidates.
- A current member's lookback uses its observable historical prices even before its entry into the index; membership filters decision dates. The unretrieved author data may pad those earlier sequence rows differently.
- Features: Qlib Alpha158. This provider has raw prices and a separate cumulative factor, so every Alpha158 price reference is replaced by raw price times factor. Volume remains in the source's units. No external factors or financial statements enter the model.
- Features are normalized with training-member medians and scaled MAD, clipped to[-3,3]. Training fit ends on 2014-12-29, after purging labels across the split. Missing values are filled forward within each observed 21-day window, then by the window's feature mean, then0 for an entirely missing feature. This is a concrete causal interpretation of the README's incompletely implemented `ffill + mean` instruction.
- For a signal at close t, the labels are `close[t+2]/close[t+1]-1`, `log(volume[t+2])-mean(log(volume[t-3:t+2]))`, and `(high[t+2]-low[t+2])/vwap[t+2]`. The volume slice is half-open and contains t-3 through t+1. Return closes use the adjustment factor; the range/VWAP ratio cancels price scaling.
- Training labels are standardized within each date and task, with sample standard deviation, matching the released `torch.std` convention. Missing targets are masked task by task.
- Train: available 2010 data through 2014; validation2015-2016; test2017 through 2020-08-01. Windows begin after 20 observed trading dates. A sample is retained only when its t+2 realization stays inside the split. Thus the last test signal is 2020-07-29 and its return is realized on 2020-07-31.

## Model and optimization

The primary experiment follows `LiMT_v2_MMoE_RVol2Vola2`, the released cross-stock-then-temporal model with the three-task fusion head. Classes are selected directly from the pinned source, avoiding unrelated missing imports. Equivalent batched SDPA attention and reused deterministic expert outputs reduce computation. With dropout disabled, outputs, input gradients, and parameter gradients are tested against the original code in float64. The fused dropout implementation has the same distribution, but does not promise identical random masks to the authors' older PyTorch runtime.

Fixed configuration: hidden width 256; cross-stock heads 4; temporal heads 2; experts 8; dropout 0.5; Adam at 1e-5; weights return/volume/range0.65/0.30/0.05; gradient values clipped to[-3,3]. The latter two architectural defaults come from the public CLI; the task weights are one declared selection in the paper's grid. There is no validation or test tuning after seeing test results. Train one complete cross-section per optimization step, shuffle dates by a deterministic seed/epoch generator, use at most 100 epochs, and stop after 21 non-improving validation epochs, as in the released `patience > 20` logic. Select by validation return IC. Seeds0,1,2 run on both universes. Only the selected checkpoint is tested. FP32 is used; TF32 is disabled for the primary suite.

| Component | Paper equations | Released code used here |
| --- | --- | --- |
| Temporal attention logits | Divide by square root of head width | No division |
| Cross-task scalar gates | Each auxiliary representation supplies its own gate | Shared backbone representation supplies both gates |
| Attention blocks | Attention, output projection and residual LayerNorm | Pre-normalized Q/K/V, concatenated heads without output projection, residual FFN, sinusoidal position encoding |
| Head counts | Cross4 / temporal2 | CLI defaults reverse these; this run uses the paper's counts |
| Expert count/dropout | Not fully specified in the paper | CLI defaults8 /0.5 |

These disagreements make a claim of simultaneous equation-exact and code-exact reproduction impossible. The primary run explicitly chooses the released model; `upstream/` remains untouched.

## Comparator

LightGBM uses the same sample universe, split, normalization and daily return targets. It receives the current Alpha158 row, whose158 factors already summarize historical data. Configuration:31 leaves, learning rate0.05, minimum 100 rows per leaf, feature and row fractions0.9, maximum 1000 trees, early stopping 50 rounds on validation MSE, two threads, seeds 0, 1, 2. This is a transparent independent comparator; the authors' fourteen baseline configurations and full search results were not released.

## Independent-task comparison (2026-11-09)

The added comparison replaces joint training with three independent single-task fits: return, volume shock (liquidity), and range/VWAP (volatility). Every fit has its own original 256-dimensional feature/position embedding, cross-stock attention, temporal attention, eight-expert bank, task gate and task projection. Cross-task fusion and unused task branches are absent. Corresponding active modules receive the same seed-matched initialization as the joint model. Expert banks and attention parameters are not shared across the three fits. This is a model-separation ablation with a larger total parameter and compute budget, not a capacity-matched ablation.

All saved joint configuration values are inherited for the paired universe/seed: data, eligibility, label definitions, normalization, splits, date order, learning rate, Adam settings, dropout, expert count, FP32/TF32 setting, gradient clipping, epoch limit and patience. The single-task MSE retains its original scalar loss weight (0.65/0.30/0.05), including its interaction with Adam's epsilon. Each independent model selects by its own target's validation IC using the same 21-epoch patience; the joint reference continues to select by validation return IC. This change in checkpoint-selection target is explicit because an independent auxiliary model cannot be selected by a return prediction it does not produce. Seeds 0, 1, 2 are run on both universes, for eighteen independent fits. Existing completed reference runs are reused; the interrupted CSI500 seed-2 joint fit resumes from its saved epoch before comparison. The RNG restoration fix copies loaded RNG-state tensors to CPU, as required by PyTorch; it does not alter the training configuration or epoch shuffle.

The three selected forecasts are joined on exactly matching signal date, realization date and symbol keys, retaining missing labels. No test-target values influence checkpoint selection. Comparisons report per-task IC/ICIR/RankIC, all existing portfolio sizing rules, and the same dollar-neutral cost scenarios. Portfolio capital, caps, fees, target construction and execution timing remain fixed. The paired daily IC difference is separate minus joint after averaging seeds, with the existing deterministic 20-day circular block bootstrap and 2000 draws. The report includes per-seed records, selected epochs, total parameters and summed epoch time; runtime comparisons are descriptive because GPU contention may differ.

Before reporting, paired saved configurations and data summaries must match, manifest/checkpoint hashes must agree, validation-selected epochs must be verified, full valid/test prediction keys and labels must match the panel and joint reference, and each independent checkpoint must reconstruct one held-out cross-section within declared FP32 tolerance. Smoke runs use a separate output directory and are rejected by full comparison audits. Outputs are under `results/separate/`, with the final report at `results/separate/comparison.md`.

## Portfolio convention

The repository does not release APO execution code. The independent engine implements equations22-26 and the appendix's three sizing rules, with the following declared conventions:

- Signal at t, execute at t+1 close, value the next close at t+2. This matches the return target and imposes a full trading-day delay.
- Form top/bottom return deciles using only information observable at t. De-normalize liquidity forecasts with cross-sectional volume-shock and range moments already observed at t, and the five-day log-volume mean through t. Never use moments of future labels.
- Tushare volumes are lots of 100 shares: multiply raw volume times raw VWAP by 100 for CNY traded value. This constant cancels the volume-shock label but changes cash capacity. Units are supported by the [Tushare daily schema](https://tushare.pro/document/2?doc_id=27).
- Allocate initial CNY 1 billion to each side. Apply the equation26 target-position cap using t turnover, and an independent2% cap on each stock's executed trade using t+1 turnover. Partial fills are retained as holdings; residual capital remains cash. All four long-short sizing rules receive the same caps and 10 bp cost on traded value. No renormalization occurs after clipping.
- Previous holdings drift with prices. Costs reduce equity once. Return denominators use total equity, initially CNY 2 billion. Cash earns0. Funding costs and short borrow fees are not provided. Short availability is assumed, with no locate, board-lot or price-limit restrictions. These are material execution limitations.
- Turnover and deployed gross notional are divided by the initial allocation: CNY 2 billion for long-short portfolios and CNY 1 billion for the long-only reference. The CSV field `gross_exposure` stores this fixed-allocation ratio. Daily returns use current equity.
- Suspended assets cannot trade. Missing closes are carried for valuation; absent delisting proceeds cannot be recovered from this provider. Do not interpret this engine as a fully verified executable A-share simulation.
- OSR is the count fraction of stock-level nonzero orders filled in full; partial fills fail. Filled-value ratio is also reported because the paper does not specify OSR's denominator precisely.
- Annualized return is252 times daily mean; Sharpe uses sample volatility and zero risk-free rate. Drawdown and cumulative return use compounded NAV. The paper's exact annualization and capital-denominator conventions are not released, so its AR/Sharpe cannot be matched by definition alone.
- `top50` is a separate long-only50-stock/20-replacement reference with the same trade cap and 10 bp cost. It is an approximation of the paper's Qlib top-k strategy, not a rerun of their absent exchange settings. Its excess return is measured against archived CSI300/CSI500 price-index daily returns from Eastmoney.
- APO r=0.1 is fixed from the paper. Sensitivity at0.1,...,0.9 is descriptive on the test set and never selects a model or a reported optimal parameter.

## Interpretation

An exact headline reproduction needs the authors' processed 2008-start data, their selected configurations and checkpoints, and their execution settings. The work here establishes a runnable, tested adaptation on available real data and reports what it measures. Seed dispersion is initialization variability, not a confidence interval for future returns. Paired daily IC differences, when reported, use a circular 20-trading-day block bootstrap; this exploratory interval is conditional on this sample and protocol.
