# TRA Paper Reproduction

Reproduction tooling for the **ALSTM+TRA** results in *Learning Multiple Stock
Trading Patterns with Temporal Routing Adaptor and Optimal Transport* (KDD 2021).
This workspace combines the released Qlib implementation with a pinned runtime,
six-seed execution scripts, and independent metric verification.

**Exact Qlib subfolder:**
[`examples/benchmarks/TRA/` at the pinned upstream commit](https://github.com/microsoft/qlib/tree/3fc2f8c93c91baac5f0dd1af23dd8ef215abe2d4/examples/benchmarks/TRA).
The local copy is [`sources/examples/benchmarks/TRA/`](sources/examples/benchmarks/TRA/).

## Source and scope

The vendored source is Qlib **v0.7.1**, commit
`3fc2f8c93c91baac5f0dd1af23dd8ef215abe2d4`, recorded in
[`source-commit.txt`](source-commit.txt). The runner uses the original paper
implementation in that subfolder, including its model, dataset, and ALSTM YAML
configurations. It verifies the files against
[`sources/source-verification.json`](sources/source-verification.json) before training.

| Reference | Location |
| --- | --- |
| Paper | [arXiv:2106.12950](https://arxiv.org/abs/2106.12950) |
| Upstream TRA documentation | [README.md](sources/examples/benchmarks/TRA/README.md) |
| Original model and dataset | [src/](sources/examples/benchmarks/TRA/src/) |
| Original experiment configurations | [configs/](sources/examples/benchmarks/TRA/configs/) |
| Original evaluation notebook | [Reports.ipynb](sources/examples/benchmarks/TRA/Reports.ipynb) |
| Reproduction protocol and data checksums | [protocol.json](protocol.json) |

## Requirements

- Linux x86_64, the platform used for this reproduction.
- `uv`, Bash, and a C++ compiler for building Qlib's extensions.
- Python 3.8.20, installed by the environment restoration script.
- The released `feature.pkl`, `label.pkl`, and `ret.pkl` data files.
- For GPU execution, an NVIDIA GPU and driver compatible with the pinned
  PyTorch 1.7.0+cu110 build. CPU execution is also supported.

The full runtime is pinned in [`environment.lock.txt`](environment.lock.txt),
including NumPy 1.17.4, pandas 1.1.2, and scikit-learn 0.23.2.
[`paper-requirements.txt`](paper-requirements.txt) preserves the original direct
requirements; the lock file records the environment used by these wrappers.

## Setup

Run all commands from this workspace's root directory.

### 1. Restore the environment

```bash
bash scripts/restore_environment.sh
mkdir -p logs results
```

The script creates `.venv` if needed, synchronizes the locked dependencies, and
builds the vendored Qlib extensions in place. The runner imports Qlib from `sources/`.

### 2. Prepare the data

Obtain the three released pickle files using the link in the
[upstream data instructions](sources/examples/benchmarks/TRA/data/README.md).
Both the runner and evaluator expect this exact layout relative to the workspace:

```text
<parent>/
├── tra/                         # This workspace
└── ordered-reproductions/
    └── data/TRA/kdd21/
        ├── feature.pkl
        ├── label.pkl
        └── ret.pkl
```

The data directory is fixed at `../ordered-reproductions/data/TRA/kdd21/` in
[`scripts/run_stage.py`](scripts/run_stage.py) and
[`scripts/evaluate.py`](scripts/evaluate.py). Data is stored separately from this
workspace. Each stage checks its SHA-256 hashes against `protocol.json`.

## Usage

### Preflight

Choose a device and check both stages before starting a full run:

```bash
export TRA_DEVICE=cpu
.venv/bin/python scripts/run_stage.py --seed 1000 --stage initialization --preflight
.venv/bin/python scripts/run_stage.py --seed 1000 --stage tra --preflight
```

Use `export TRA_DEVICE=cuda` for GPU execution; `cuda` is the default when the
variable is unset. Each preflight performs one forward/backward optimizer step
and writes to a separate preflight directory. It does not produce a trained
backbone or a full reproduction result. Completed preflights refuse to overwrite
their existing output.

### Run the experiments

With `TRA_DEVICE` set in the same shell:

```bash
.venv/bin/python -u scripts/run_all.py
```

The runner executes initialization followed by the TRA stage for each seed, in
the order `1000, 1, 2000, 3000, 4000, 5000`. Every seed uses its own pretrained
backbone. Full CPU execution can take several days, depending on early stopping.
This command runs in the foreground; it does not install a background service.

Completed stages are hash-verified before reuse. Stages with an existing
`model.bin` refuse to overwrite it; there is no mid-epoch resume support.

### Inspect and summarize results

```bash
.venv/bin/python scripts/status.py
.venv/bin/python scripts/summarize.py
```

Keep `TRA_DEVICE` set to the device whose results you want to summarize. The
status script reads saved status files and counts completed main stages; it does
not check whether a recorded process is still alive.

| Output | Location |
| --- | --- |
| Overall run status | `results/status.json` |
| CPU stage artifacts and status | `results/paper-runtime-cpu/<seed>/<stage>/` |
| CUDA stage artifacts and status | `results/paper-runtime/<seed>/<stage>/` |
| Preflight outputs | `results/preflight-cpu/` or `results/preflight/` |
| Training logs | `logs/<device>-<seed>-<stage>.log` |
| Summary | `results/report.txt`, `results/aggregate.json` |
| Metrics after a main stage completes | `results/per-seed.csv`, `results/comparison.csv` |
| All five-seed subsets after six seeds complete | `results/all-five-seed-subsets.csv` |

CPU and CUDA stage artifacts use separate directories. Summary files are shared
and regenerated for the selected device.

## Reproduction protocol

The protocol is recorded in [`protocol.json`](protocol.json). Validation IC and
the released early-stopping rule select checkpoints. Final test predictions
come from the native `fit()` call; the evaluator checks all eight reported metrics
against the original notebook without drawing new routing samples. There is no
test-based tuning or seed selection.

Known limits of the released experiment are retained explicitly:

- Table 1 describes five seeds, while `run.sh` names six. All six are reported;
  once complete, every five-seed subset is reported descriptively.
- The YAML and notebook use ALSTM width 256, while Appendix B lists 32, 64, and 128.
- The released sequence boundary behavior and stochastic evaluation are preserved.
- The original Python patch version, hardware, and full transitive dependency
  versions are unspecified. CPU execution is a recorded hardware deviation.

These scripts do not establish exact numerical reproduction merely by completing
training or matching rounded table values.

## Contributing

For changes to the wrappers, include the affected command, runtime versions, and
relevant logs. Changes to model behavior, data, or evaluation should be documented
as a separate experiment so the pinned reproduction remains traceable.

## Citation

Please cite the original TRA paper when using this work:

```bibtex
@inproceedings{HengxuKDD2021,
  author = {Hengxu Lin and Dong Zhou and Weiqing Liu and Jiang Bian},
  title = {Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport},
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
  year = {2021},
  publisher = {ACM}
}
```

The [upstream README](sources/examples/benchmarks/TRA/README.md) also provides the
Qlib citation.

## License

The vendored Qlib code is licensed under the [MIT License](sources/LICENSE),
copyright Microsoft Corporation. No separate license is currently declared for
the reproduction wrapper scripts.
