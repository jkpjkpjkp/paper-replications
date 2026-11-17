# Published protocol: implementation check

Checked 2026-11-11T06:47:03.634725+00:00. Protocol `us-paper-equations-v2`.

**Published numerical results remain unreplicated.** These checks establish implementation behavior, not equivalence to the authors’ private inputs or configuration.

| Item | Published source | Specification | Observed implementation / limit |
|---|---|---|---|
| Raw inputs | A.2, p.12 | Seven one-minute fields | Names/schema aligned; vendor trade eligibility is an assumption |
| Feature aggregation | A.3–A.4, pp.12–13 | 26 non-overlapping 15-minute patches (US) | Patch locality tested; exact 66 features and D are not established |
| Ordinary training | A.5.1, p.13 | AdamW; cross-sectionally standardized MSE | MSE–IC identity and streamed optimizer update tested |
| Batch size | A.5.1, p.13 | 20 trading days | Full 20-date windows; last 17 training dates omitted by fixed batching |
| Warm-up | 4.2; A.5.2, pp.6,13 | Three epochs; mean standardized candidate levels | Same AdamW settings as ordinary training; label calculation tested |
| Support/query split | A.5.2, pp.13–14 | Random 10/10 partition within each batch | Disjoint date indices in train.select_horizon; epoch RNG recovery tested |
| Inner learning rate | A.5.2, p.14 | 1e-6 | Differentiable single SGD step, retained student writeback |
| Outer learning rate | A.5.2, p.14 | 1e-3 | SGD on softmax logits: explicit equation interpretation, optimizer still ambiguous |
| Entropy coefficient | A.5.2, p.14 | 1e-3 | Subtract entropy from minimized query objective |
| Early stopping | A.5.1, p.13 | Patience 5; restore best validation checkpoint | Checkpoint/recovery tests pass; no primary epoch cap |
| Final model | 4.2, p.6 | Argmax horizon then retrain | Fresh same-seed model; all 15 fits required before backbone test metrics |

The baseline/warm-up learning rate (1e-3), weight decay (1e-4), architecture sizes, seed IDs 0–4, preprocessing, missing-bar handling, corporate-action policy, point-in-time membership convention, and boundary purging are implementation choices. The paper does not fully specify them. Appendix A.3 gives representative feature categories and leaves D unspecified. The 66 implemented statistics must not be represented as the original tensor.

The [authors’ public repository](https://github.com/EFundAI/label-horizon-paradox), checked live on November 11, identifies itself as a separate daily Alpha158 demo and says the original experimental features are not public. No new original feature/configuration release was found there.

## Independent numerical evidence

All **38 regression tests passed** (training/data/backbone checks and progress-integrity checks). The primary model source hashes and completed checkpoint receipts still match the frozen run.

A consistent snapshot of LSTM seed 0 at BLO epoch 137 was checked on 20 real TRAINING dates, with a seeded random 10/10 split. A separate implementation of standardized MSE and first-order inner updates recomputes the query objective at perturbed logits. Two independent directions and two step sizes agree with the analytic hypergradient to maximum absolute error 1.28e-12. No validation or test scores were read for this diagnostic.

At that snapshot, maximum/uniform weight = 1.000000031242. The provisional argmax is 53 minutes; its weight exceeds the runner-up (52 minutes) by only 1.95e-12. This is a nearly uniform distribution, not evidence of a stable selected optimum.

The task hypergradient norm is 2.11e-07. At the same LR of 1e-3, a fresh SGD step changes the logits by norm 2.11e-10; a fresh AdamW step (epsilon 1e-8, zero weight decay) changes them by 0.00889. This is a one-step numerical sensitivity check, not a trained alternative or evidence that AdamW improves financial performance. It confirms that choosing an optimizer can materially change the trajectory. The running primary protocol remains unchanged.

## Work still outstanding

The active coordinator is executing the five-seed LSTM comparison, followed by GRU and TCN. Each requires 15 completed fits before test evaluation. See [live status](../STATUS.md) for actual completed counts; the monitor refreshes it every 30 seconds while the coordinator runs. Early stopping has no reliable finish-time estimate while validation continues improving.

Original CSI minute inputs, exact feature definitions/configurations, seven other architectures, and the remaining original-paper experiments are still unavailable or unfinished. The older daily CSI results are not minute-paper replications.

[Published paper](https://proceedings.mlr.press/v306/song26l.html) · [Machine-readable check](protocol_check.json) · [Regression log](protocol-check-tests.log) · [Frozen numerical diagnostic](diagnostics/hypergradient-20261111/result.json)

Re-run: `.venv/bin/python check_hypergradient.py` (reuses its frozen snapshot), then `.venv/bin/python check_protocol.py`. Neither changes the fitted primary models.
