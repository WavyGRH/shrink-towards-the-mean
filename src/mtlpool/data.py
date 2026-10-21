"""Loading, splitting and standardising the School data."""
import numpy as np
import scipy.io as sio

from .config import CONT_COLS


def load_school(path):
    """Return lists (Xs, ys): one design matrix (n_i x 28, last column = bias) and target per school."""
    d = sio.loadmat(path)
    Xs = [np.asarray(x, float) for x in d["X"].ravel()]
    ys = [np.asarray(y, float).ravel() for y in d["Y"].ravel()]
    return Xs, ys


def split(Xs, ys, ratio, rng):
    """Per-school random split: max(1, round(ratio * n_i)) training students per school."""
    tr, te = [], []
    for X, y in zip(Xs, ys):
        p = rng.permutation(len(y))
        k = max(1, int(round(ratio * len(y))))
        tr.append((X[p[:k]], y[p[:k]]))
        te.append((X[p[k:]], y[p[k:]]))
    return tr, te


def standardise(tr, te, cols=CONT_COLS):
    """Z-score the numeric columns with training-set statistics; binary columns are untouched."""
    cols = list(cols)
    A = np.vstack([X for X, _ in tr])
    mu, sd = A[:, cols].mean(0), A[:, cols].std(0) + 1e-12

    def f(X):
        X = X.copy()
        X[:, cols] = (X[:, cols] - mu) / sd
        return X

    return [(f(X), y) for X, y in tr], [(f(X), y) for X, y in te]


def unzip(pairs):
    return [X for X, _ in pairs], [y for _, y in pairs]


def make_split(Xs, ys, ratio, s, seed_base):
    """The split used for (ratio, split index s) throughout the paper."""
    rng = np.random.default_rng(seed_base + s)
    return standardise(*split(Xs, ys, ratio, rng))
