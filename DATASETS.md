# Dataset provenance

No third-party raw or processed dataset matrices are redistributed in this
repository. The downloader contacts the original public hosts; access and use
remain subject to the dataset owners' terms. GPL does not apply to those data.

| Released ID | Source | Upstream MAT keys |
|---|---|---|
| COIL20 | [scikit-feature collection](https://jundongl.github.io/scikit-feature/datasets.html) | `X`, `Y` |
| madelon | [scikit-feature collection](https://jundongl.github.io/scikit-feature/datasets.html) | `X`, `Y` |
| mlj2018_waveform | [Simple Strategies for Semi-Supervised Feature Selection](https://github.com/sechidis/2018-MLJ-Semi-supervised-feature-selection/tree/master/Datasets) | `X_data`, `Y_labels` |

Exact URLs, dimensions, accepted file hashes, and normalized array hashes are
recorded in `datasets.json`. Public upstream files were compared with the original
processed experiment files: all X and y values, sample order, and feature order
match. Storage bytes differ (MAT headers, dtype, or variable names), so both
verified file identities are accepted. Normalized hashes use contiguous
little-endian float64 X and flattened y, in C order. No row sorting, resampling,
feature replacement, or regenerated splits are used.

If an upstream URL becomes unavailable, obtain the same file from its source and
place it under the filename in `datasets.json`, or supply `--data-dir`.
The loader rejects unknown bytes and changed arrays. Do not bypass checks with
a different version of a similarly named dataset.

If Python cannot reach the hosts (for example, because it does not inherit a
system proxy), download the `download_url` entries from `datasets.json` with your
browser, rename each file to its `file` value, and place it in `data/`.
The benchmark performs the same verification on manually downloaded files.

The three bundled split NPZ files contain indices only, with no raw features
or labels. Their hashes appear in `protocol.json`.
