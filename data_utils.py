# SPDX-License-Identifier: GPL-3.0-only
"""Read only the three documented datasets; verify bytes and numerical identity."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()

def load_dataset(path, item):
    if sha(path) not in item['accepted_file_sha256']:
        raise ValueError('Unrecognized dataset bytes: ' + item['dataset'])
    mat = loadmat(path)
    xkey, ykey = ('X', 'Y') if 'X' in mat and 'Y' in mat else item['upstream_keys']
    X = np.asarray(mat[xkey], dtype=np.float64)
    y = np.asarray(mat[ykey]).reshape(-1)
    if X.shape != (item['samples'], item['features']) or y.size != len(X):
        raise ValueError('Unexpected dataset shape')
    if not np.isfinite(X).all() or not np.isfinite(y).all() or np.unique(y).size != item['classes']:
        raise ValueError('Invalid values or class count')
    if array_sha(X) != item['X_sha256'] or array_sha(y) != item['y_sha256']:
        raise ValueError('Dataset values or row/feature order changed')
    return X, y
