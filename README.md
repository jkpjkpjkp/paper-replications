# Paper replications

35 paper and study projects, kept in **one GitHub repository**. Each paper has an independent `paper/<name>` branch; `main` contains the catalogue and shared tools.

Only source code, configuration, references, and small written assessments are included. Shared datasets, generated results, checkpoints, environments, `LLM/`, and `CRYPTO/` stay local. Data access and environments must be supplied separately.

## Browse the papers

| Local folder | Paper branch |
| --- | --- |
| `DRO` | [paper/dro](https://github.com/jkpjkpjkp/paper-replications/tree/paper/dro) |
| `Kronos` | [paper/kronos](https://github.com/jkpjkpjkp/paper-replications/tree/paper/kronos) |
| `Lookahead` | [paper/lookahead](https://github.com/jkpjkpjkp/paper-replications/tree/paper/lookahead) |
| `MDP Pricing` | [paper/mdp-pricing](https://github.com/jkpjkpjkp/paper-replications/tree/paper/mdp-pricing) |
| `NIPS26/stable-alpha` | [paper/stable-alpha](https://github.com/jkpjkpjkp/paper-replications/tree/paper/stable-alpha) |
| `NODATA/hga-mt` | [paper/hga-mt](https://github.com/jkpjkpjkp/paper-replications/tree/paper/hga-mt) |
| `NODATA/sparse-tangent` | [paper/sparse-tangent](https://github.com/jkpjkpjkp/paper-replications/tree/paper/sparse-tangent) |
| `PDE` | [paper/pde](https://github.com/jkpjkpjkp/paper-replications/tree/paper/pde) |
| `SIT` | [paper/sit](https://github.com/jkpjkpjkp/paper-replications/tree/paper/sit) |
| `Set Sequence` | [paper/set-sequence](https://github.com/jkpjkpjkp/paper-replications/tree/paper/set-sequence) |
| `alphaqcm` | [paper/alphaqcm](https://github.com/jkpjkpjkp/paper-replications/tree/paper/alphaqcm) |
| `alphasage` | [paper/alphasage](https://github.com/jkpjkpjkp/paper-replications/tree/paper/alphasage) |
| `arboreal` | [paper/arboreal](https://github.com/jkpjkpjkp/paper-replications/tree/paper/arboreal) |
| `corporate-graph-anomalies` | [paper/corporate-graph-anomalies](https://github.com/jkpjkpjkp/paper-replications/tree/paper/corporate-graph-anomalies) |
| `diffusion-dfl` | [paper/diffusion-dfl](https://github.com/jkpjkpjkp/paper-replications/tree/paper/diffusion-dfl) |
| `doubleadapt` | [paper/doubleadapt](https://github.com/jkpjkpjkp/paper-replications/tree/paper/doubleadapt) |
| `earnings-timing` | [paper/earnings-timing](https://github.com/jkpjkpjkp/paper-replications/tree/paper/earnings-timing) |
| `fraudgnam` | [paper/fraudgnam](https://github.com/jkpjkpjkp/paper-replications/tree/paper/fraudgnam) |
| `graph-risk-placement` | [paper/graph-risk-placement](https://github.com/jkpjkpjkp/paper-replications/tree/paper/graph-risk-placement) |
| `healthcare-insurance-fraud` | [paper/healthcare-insurance-fraud](https://github.com/jkpjkpjkp/paper-replications/tree/paper/healthcare-insurance-fraud) |
| `hermes` | [paper/hermes](https://github.com/jkpjkpjkp/paper-replications/tree/paper/hermes) |
| `ijcai-paper` | [paper/ijcai-paper](https://github.com/jkpjkpjkp/paper-replications/tree/paper/ijcai-paper) |
| `limt` | [paper/limt](https://github.com/jkpjkpjkp/paper-replications/tree/paper/limt) |
| `liu26y` | [paper/liu26y](https://github.com/jkpjkpjkp/paper-replications/tree/paper/liu26y) |
| `monthly-coforecast` | [paper/monthly-coforecast](https://github.com/jkpjkpjkp/paper-replications/tree/paper/monthly-coforecast) |
| `pdu` | [paper/pdu](https://github.com/jkpjkpjkp/paper-replications/tree/paper/pdu) |
| `prism-vq` | [paper/prism-vq](https://github.com/jkpjkpjkp/paper-replications/tree/paper/prism-vq) |
| `proximal-portfolio` | [paper/proximal-portfolio](https://github.com/jkpjkpjkp/paper-replications/tree/paper/proximal-portfolio) |
| `rise` | [paper/rise](https://github.com/jkpjkpjkp/paper-replications/tree/paper/rise) |
| `risk-aware-utility-reranking` | [paper/risk-aware-utility-reranking](https://github.com/jkpjkpjkp/paper-replications/tree/paper/risk-aware-utility-reranking) |
| `NODATA/song26l` | [paper/song26l](https://github.com/jkpjkpjkp/paper-replications/tree/paper/song26l) |
| `stable` | [paper/stable](https://github.com/jkpjkpjkp/paper-replications/tree/paper/stable) |
| `tra` | [paper/tra](https://github.com/jkpjkpjkp/paper-replications/tree/paper/tra) |
| `transalpha` | [paper/transalpha](https://github.com/jkpjkpjkp/paper-replications/tree/paper/transalpha) |
| `vision-to-value-evaluation` | [paper/vision-to-value-evaluation](https://github.com/jkpjkpjkp/paper-replications/tree/paper/vision-to-value-evaluation) |

Lookahead/reproduction and limt/extensions/us remain with their parent papers. `NIPS26` and `NODATA` are organizational groups; their individual papers have separate branches. The scratch catalogue `paper/` and `multihorizon-forecast` are excluded.

## Clone

Clone one paper:

```bash
git clone --single-branch --branch paper/alphaqcm git@github.com:jkpjkpjkp/paper-replications.git alphaqcm
```

For the original shared-tools layout, clone `main` first and place the selected paper inside it:

```bash
git clone --single-branch --branch main git@github.com:jkpjkpjkp/paper-replications.git paper-replications
cd paper-replications
git clone --single-branch --branch paper/alphaqcm git@github.com:jkpjkpjkp/paper-replications.git alphaqcm
```

Consult each branch’s README and `STATUS.txt` for commands, required inputs and replication scope. A grade of 2 may cover a baseline or released-artifact analysis; it does not establish a whole-paper reproduction.

[REPLICATION_STATUS.txt](REPLICATION_STATUS.txt) retains the complete local assessment, including excluded projects. Some evidence paths refer to local-only files. [WORKSPACE.md](WORKSPACE.md) retains the broader workspace guide; its links to excluded folders require the local workspace.

[UPLOAD.md](UPLOAD.md) explains how to refresh and push these branches without importing old datasets or Git histories. Third-party notices and licenses stay with their respective source trees; no blanket license is assigned to the collection.
