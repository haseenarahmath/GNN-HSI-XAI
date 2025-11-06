"""
HSI loader:
- If .mat paths provided, reads (H,W,B) cube and (H,W) labels.
- Else, generates a synthetic cube.
- Builds a kNN graph in feature-space; returns (x:[N,B], adj:[N,N], y:[N]).
"""
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.neighbors import kneighbors_graph

def _normalize_features(x):
    # row-wise l2 norm
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
    return x / denom

def _to_torch_dense_adj(knn_graph):
    A = knn_graph.todense().astype(np.float32)
    A = (A + A.T) / 2.0  # symmetrize
    row_sum = np.asarray(A.sum(1)).reshape(-1) + 1e-6
    A = A / row_sum[:, None]  # row normalize
    return torch.from_numpy(np.array(A))

def _load_mat_pair(mat_x, mat_y, key_x, key_y):
    X = loadmat(mat_x)[key_x]  # H x W x B
    Y = loadmat(mat_y)[key_y]  # H x W
    assert X.ndim == 3 and Y.ndim == 2, "Expected cube HxWxB and labels HxW"
    H, W, B = X.shape
    Xf = X.reshape(-1, B).astype(np.float32)   # [N,B]
    Yf = Y.reshape(-1).astype(np.int64)        # [N]
    # shift labels to 0..C-1 if they start at 1; -1 remains unlabeled if present
    if Yf.min() == 1:
        Yf = Yf - 1
    mask = Yf >= 0
    return Xf[mask], Yf[mask], [f"band_{i}" for i in range(B)]

def _synthetic_cube(H=32, W=32, B=64, C=9, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(H, W, B)).astype(np.float32)
    Y = rng.integers(0, C, size=(H, W))
    # make class-specific discriminative bands
    for c in range(C):
        band = (c * (B // C)) % max(B, 1)
        X[Y == c, band] += 3.0
    return X, Y

def load_hsi_as_graph(dataset="synthetic", data_dir="data",
                      mat_x=None, mat_y=None, mat_x_key="X", mat_y_key="y",
                      knn=8, device="cpu"):
    if mat_x and mat_y and Path(mat_x).exists() and Path(mat_y).exists():
        Xf, Yf, band_names = _load_mat_pair(mat_x, mat_y, mat_x_key, mat_y_key)
    else:
        # Minimal synthetic fallback by dataset name
        if dataset.lower() == "indian_pines":
            H, W, B, C = 32, 32, 200, 16
        elif dataset.lower() == "pavia":
            H, W, B, C = 32, 32, 103, 9
        elif dataset.lower() == "salinas":
            H, W, B, C = 32, 32, 224, 16
        else:
            H, W, B, C = 32, 32, 64, 9
        X, Y = _synthetic_cube(H, W, B, C)
        Xf = X.reshape(-1, B).astype(np.float32)
        Yf = Y.reshape(-1).astype(np.int64)
        band_names = [f"band_{i}" for i in range(B)]

    Xf = _normalize_features(Xf)
    A = kneighbors_graph(Xf, knn, mode="connectivity", include_self=True)
    adj = _to_torch_dense_adj(A)
    x = torch.from_numpy(Xf)
    y = torch.from_numpy(Yf)
    return x.to(device), adj.to(device), y.to(device), band_names
