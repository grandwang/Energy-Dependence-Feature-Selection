# EDFS — minimal reproducible release

Official pre-release code for our manuscript "Energy–Dependence Feature Selection".

A small, executable release of EDFS for feature selection with partially labeled
training data. The frozen implementation is exposed as `EDFS`.

This release covers **three datasets, one seed (74076), one selector label ratio
(0.10), three feature budgets (5, 10, 20), and 3-nearest-neighbor evaluation**.

| Dataset | Samples | Features | Classes |
|---|---:|---:|---:|
| COIL20 | 1,440 | 1,024 | 20 |
| madelon | 2,600 | 500 | 2 |
| waveform (`mlj2018_waveform`) | 5,000 | 40 | 3 |

## Quick start

Use Python 3.9–3.12 in a fresh environment. Validation was performed on Windows
x64 with Python 3.9.13 and the exact versions in `requirements.txt`.

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --scope smoke
python download_data.py
python run.py --scope benchmark
```

The smoke check uses three small synthetic reference cases and needs no downloaded
data. Dataset files are obtained from upstream hosts, validated, and stored in
ignored `data/`. See [DATASETS.md](DATASETS.md). To use an existing exact copy:

```sh
python run.py --scope benchmark --data-dir /path/to/data
```

Every run creates `outputs/<unique-run>/results.json`. Exit codes: `0` means the
requested checks passed; `1` means historical selection/metric differences;
`2` means execution or validation error. A benchmark pass covers nine selection
prefixes and 36 metric values. Historical prediction vectors and process arrays
are unavailable; no full-chain or full-paper reproduction claim is made.
Feature indices must match exactly; metric absolute tolerance is `1e-14`.
Synthetic process tolerance is `1e-12`. Numeric libraries can affect comparisons.

## Protocol

The three frozen partitions are inherited from the original experiment, with
70% outer training / 30% testing. Standardization is fitted on outer training
only (`ddof=0`; standard deviations below `1e-12` become 1).
Only 10% of the training labels are available to feature selection.
**The downstream 3-NN classifier uses all outer-training labels.**

The selector combines completed Energy relevance with a Ledoit–Wolf correlation
structure and forward greedy selection. Numerical guards and tie behavior are
preserved. `protocol.json` records the exact released experiment, and
`reference/expected.json` contains only its historical 3-NN reference fields.

## Using EDFS

```python
from edfs import EDFS

# Apply training-fitted preprocessing consistently to X_l, X_u, and X_test.
selector = EDFS().fit(X_l, y_l, X_u)
indices = selector.select(20)   # ordered, zero-based indices
X_test_selected = selector.transform(X_test)
```

## License and citation

First-party code and accompanying first-party release materials are distributed
under **GPL-3.0-only**; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
External datasets and installed dependencies retain their own terms.
See [CITATION.md](CITATION.md) for the software identifier.

## 中文说明

这是包含完整可运行选择器的最小复现发布：三个数据集、一个固定种子、一个标签率、
三个特征预算和一个分类器。可先运行 smoke，再下载数据运行 benchmark。
不包含完整论文实验网格、其他方法、论文文件、开发历史或 Python 运行时。
GPL 允许按许可证使用、修改和再分发；缩小发布范围不意味着能阻止方法被研究或复用。
