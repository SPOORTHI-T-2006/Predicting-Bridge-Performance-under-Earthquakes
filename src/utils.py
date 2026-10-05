"""Helper functions: data loading and class-wise accuracy."""
import numpy as np


def load_csv(path):
    """Load a CSV whose header has feature columns starting with 'x' and a label column starting with 'y'.

    Returns X of shape (m, n_features) and y of shape (m,).
    """
    with open(path, "r", newline="") as f:
        headers = f.readline().strip().split(",")
    x_cols = [i for i, h in enumerate(headers) if h.startswith("x")]
    y_cols = [i for i, h in enumerate(headers) if h.startswith("y")]
    X = np.loadtxt(path, delimiter=",", skiprows=1, usecols=x_cols)
    y = np.loadtxt(path, delimiter=",", skiprows=1, usecols=y_cols)
    if X.ndim == 1:
        X = X[:, None]
    return X, y


def class_accuracy(y_true, y_pred, label):
    """Accuracy restricted to samples whose true label equals `label` (recall of that class)."""
    mask = y_true == label
    return float(np.mean(y_pred[mask] == y_true[mask])) if mask.any() else float("nan")
