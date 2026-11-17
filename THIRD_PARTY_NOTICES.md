# Attribution and data provenance

This independent replication implements the Table 1 calculations documented in
`sdf_analysis/portfolio_performance.ipynb` in
[finxlab/vit-sdf-lasso](https://github.com/finxlab/vit-sdf-lasso/tree/bde5d07442fdea2c230641ebfbdd0ea00b5021fc),
commit `bde5d07442fdea2c230641ebfbdd0ea00b5021fc`.
The upstream software is MIT licensed, copyright (c) 2025 Financial Innovation &
Decision Analytics Lab. Its complete notice is retained in `docs/UPSTREAM_LICENSE`.

The 48 expected numeric cells are transcribed from Table 1, page 11, of the
January 25, 2026 SSRN preprint by Jun Young Byun, Yosep Na, and Jae Wook Song,
*From vision to value: Stock chart image-driven factors and their pricing power*,
[SSRN 5407028](https://ssrn.com/abstract=5407028). They were visually checked against
the retained PDF. The PDF SHA-256 is
`a357cdcca59e52b1ee276a614409cb34c86c166791f0fead52b8e1508934f31f`.
The PDF itself is not included in this distribution. This is a preprint-version
comparison; equality to another paper version is not claimed.

The two aggregate input CSVs are fetched directly from the authors' pinned public
release and verified against `src/vision_to_value/inputs.json`. Neither input CSVs,
security-level observations, CRSP/WRDS data, Chen-Zimmermann data, model weights,
nor local evaluation outputs are bundled in the source release. The code license
does not grant rights to underlying third-party datasets. Data access remains
subject to the relevant providers' terms.

NumPy, pandas, SciPy and statsmodels are installed as dependencies; their own
licenses apply. They are not vendored in the release. This project is not an
official release by the paper's authors.
