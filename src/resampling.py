"""Ways to rebalance the TRAINING data only (never the test fold).

bootstrap : duplicate random minority samples (with replacement)  - the paper's method
smote     : create NEW synthetic minority samples by interpolating between a minority sample and one of
            its k nearest minority neighbours (Chawla et al., 2002). Implemented from scratch so no extra
            dependency is needed. Binary (one-hot) columns are copied from one of the two parents instead
            of being interpolated, so they stay 0/1.

For K classes, every minority class is raised to  ratio * (size of the majority class).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_samples(X, n_new, k, cat_idx, rng):
    n = len(X)
    if n < 2:  # nothing to interpolate with -> fall back to duplication
        return X[rng.integers(0, n, n_new)]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    neigh = nn.kneighbors(X, return_distance=False)[:, 1:]
    base = rng.integers(0, n, n_new)
    other = neigh[base, rng.integers(0, k, n_new)]
    lam = rng.random((n_new, 1))
    synth = X[base] + lam * (X[other] - X[base])
    for c in cat_idx:
        synth[:, c] = np.where(lam[:, 0] < 0.5, X[base, c], X[other, c])
    return synth


def rebalance(X, y, strategy, ratio=1.0, cat_idx=(), rng=None, k=5):
    """Return (X_new, y_new). strategy in {'none', 'bootstrap', 'smote'}."""
    if strategy == "none":
        return X, y
    rng = rng if rng is not None else np.random.default_rng(0)
    classes, counts = np.unique(y, return_counts=True)
    target = int(round(ratio * counts.max()))
    Xs, ys = [X], [y]
    for c, n in zip(classes, counts):
        if n >= target:
            continue
        Xc = X[y == c]
        n_new = target - n
        if strategy == "bootstrap":
            new = Xc[rng.integers(0, n, n_new)]
        elif strategy == "smote":
            new = smote_samples(Xc, n_new, k, cat_idx, rng)
        else:
            raise ValueError(strategy)
        Xs.append(new)
        ys.append(np.full(n_new, c, dtype=y.dtype))
    return np.vstack(Xs), np.concatenate(ys)
