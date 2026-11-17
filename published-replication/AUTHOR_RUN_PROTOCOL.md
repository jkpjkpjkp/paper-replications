# Released-data LiMT run, 2026-11-17

This run uses the six author-share samplers whose sizes, hashes and actual
date indexes are recorded in `.agent/AUTHOR_DATA_INVENTORY.json`. It reproduces
the main LiMT predictive experiment on CSI300 and CSI500. It does not claim to
reproduce the private APO execution engine or every baseline and ablation.

Paper settings: Alpha158, 21-day windows, the serialized 2008–2014 training,
2015–2016 validation and 2017–2020 test splits, hidden width 256, learning rate
1e-5, four cross-stock heads, two temporal heads, up to 100 epochs, and three
random initializations. Seeds 0, 1 and 2 are local choices because the paper
does not identify its actual seed integers.

All 36 combinations of return weights {0.63, 0.65, 0.68, 0.70}, volume weights
{0.30, 0.33, 0.35}, and volatility weights {0.03, 0.05, 0.08} are evaluated.
Each combination has three seeds on each universe: 216 fits. The paper does
not impose a sum-to-one constraint, so combinations are not silently removed
or renormalized. Hidden width and learning rate are fixed at the selected
values explicitly reported in the paper. Candidate checkpoint selection uses
validation return IC. Per-universe weight selection uses the mean of the
three seeds' best validation ICs; this aggregation convention is explicit
because the paper does not disclose how tuning was combined across seeds.
Only the selected three checkpoints per universe are evaluated on the test set.
Ties follow the frozen candidate order. No test score enters selection.

The model, sampler, training step and prediction methods execute the released
upstream classes unchanged. The unused, unavailable `Embed` import is omitted
when loading model definitions. The wrapper resolves the pickle's
`__main__.TSDataSampler` name to the released sampler and adds progress records,
atomic epoch checkpoints and RNG restoration. It does not use the earlier
community-data adapter, regenerate features, purge boundary observations,
change sample membership, or replace the original attention implementation.

Released CLI/trainer defaults supply dropout 0.5, eight experts, Adam, gradient
value clipping at 3, and early stopping after 21 non-improving validation epochs.
The released day sampler repeats the same seed-permuted date order each epoch;
that behavior is preserved. These defaults are not independently verified as
the private published-run configuration. The public model also differs from
the paper's temporal-attention and fusion equations; see `PROTOCOL.md`.

The files' serialized fill setting is `ffill+zero`. Inspected training labels
are on raw scales, despite the README's description of pre-normalized original
training labels. Consequently, the released training loop's daily z-score
normalization is retained for all three labels. No additional extreme-label
filter is applied. The release's missing-value padding and sample boundaries
are preserved. Input hashes/date checks establish the identity of these local
files, not an independent reconstruction of the authors' preprocessing.

The separate CPU and CUDA runtimes pin PyTorch 1.11.0 and pandas 1.5.3.
The sampler and required Qlib-derived utilities are bundled in the released
`utils.py`; they are executed directly. Exact hardware identity is not possible
on this workstation. A serial scheduler uses CUDA for a new trial when at least
2.5 GB is free and the CUDA environment passes a live tensor-operation check;
it uses CPU otherwise. It preserves a trial's chosen device on resume.
CPU execution uses two threads. No existing process is interrupted.

Modern glibc rejects the legacy wheel's executable-stack request. The local
runtime clears the ELF GNU_STACK execute flag with `patchelf --clear-execstack`;
the numerical code is unchanged. Before/after library hashes are recorded in
`cpu-runtime-compatibility.json` and, after CUDA setup finishes,
`cuda-runtime-status.json`. Five tests passed in the pinned CPU environment,
including equality with the released training loop and interrupted-run recovery.
Both universes also passed separate three-date real-data smoke checks.

Run artifacts are isolated under
`/mnt/data/home/jkp/replications/limt-author-2008-20261117/`.
The executable source snapshot, frozen plan/configurations, package lock,
checks, logs, progress records and resumable checkpoints live there.

Entry points:

```bash
python scripts/run_author_suite.py --output /path/to/new/run --prepare-only
python scripts/run_author_suite.py --output /path/to/new/run --device auto
```

Use the separately pinned author environment, not the community experiment's
environment. Re-running the same frozen suite resumes completed epoch
checkpoints; conflicting configuration/source/runtime fingerprints are rejected.
Smoke checks use a separate `smoke/` directory and cannot enter full selection.
