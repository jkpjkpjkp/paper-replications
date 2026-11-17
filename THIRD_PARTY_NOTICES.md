# Third-party sources

`ocp.py` contains independent Algorithm 1 code and optimized adaptations of the following MIT-licensed implementations. The original files and copyright notices are retained:

- AdaMLLab/up-for-ocp, commit `3575f1f94a14620e089422f52cf9e1a6766ec7b1`: [upstream/LICENSE](upstream/LICENSE). Adapted update rules: UP experimental version, KT, SF-OGD, DtACI, and inverse quantile CDF semantics.
- aangelopoulos/conformal-time-series, commit `b729c3f5ff633bfc43f0f7ca08199b549c2573ac`: [reference/pid/LICENSE](reference/pid/LICENSE). Adapted P/PI update rules with one-step forecasting and scorecasting disabled. The original implementation is retained at `reference/pid/methods.py`.

The paper remains copyrighted by its authors under its published terms. Raw datasets and forecast caches were downloaded from the authors' repository for reproduction; the code licenses do not establish separate redistribution rights for the source datasets. Sources and hashes are in `PROVENANCE.json` and `data/manifest.json`.
