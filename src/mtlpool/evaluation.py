"""Cross-validation, metrics and paired tests."""
import numpy as np
from scipy import stats

from .config import K_FOLDS
from .data import unzip


def sse(W, Xs, ys):
    return sum(float(np.sum((X @ W[:, i] - y) ** 2)) for i, (X, y) in enumerate(zip(Xs, ys)))


def cv_select(fit, grid, tr, rng, k_folds=K_FOLDS):
    """K-fold CV inside the training set, folds formed within each school.

    Every school keeps at least one training student in every fold (schools with a single
    training student are never held out). Returns (best value, its index in the grid).
    """
    folds = []
    for X, y in tr:
        lab = np.full(len(y), -1)
        if len(y) >= 2:
            lab = rng.permutation(len(y)) % k_folds
        folds.append(lab)
    err = np.zeros(len(grid))
    for k in range(k_folds):
        a = [(X[f != k], y[f != k]) for (X, y), f in zip(tr, folds)]
        b = [(X[f == k], y[f == k]) for (X, y), f in zip(tr, folds)]
        Xa, ya = unzip(a)
        Xb, yb = unzip(b)
        for j, h in enumerate(grid):
            err[j] += sse(fit(Xa, ya, h), Xb, yb)
    j = int(np.argmin(err))
    return grid[j], j


def metrics(W, te, y_train_all):
    """Test metrics. ev_within follows Argyriou et al. (2008); ev_pooled uses all test scores."""
    Xt, yt = unzip(te)
    errs = [X @ W[:, i] - y for i, (X, y) in enumerate(zip(Xt, yt))]
    allerr = np.concatenate(errs)
    ally = np.concatenate(yt)
    mse = float(np.mean(allerr ** 2))
    mse_task = np.mean([np.mean(e ** 2) for e, y in zip(errs, yt) if len(y) > 1])
    var_task = np.mean([np.var(y) for y in yt if len(y) > 1])
    return {"ev_pooled": 100 * (1 - mse / np.var(ally)),
            "ev_within": 100 * (1 - mse_task / var_task),
            "rmse": np.sqrt(mse),
            "nmse": mse / np.var(ally),
            "rmse_z": np.sqrt(mse) / np.std(y_train_all)}


def paired_test(loss_a, loss_b, ratio):
    """Compare two methods on the same splits (losses, lower is better).

    Positive mean_diff means b is better. p_nb is the corrected resampled t-test of
    Nadeau and Bengio (2003), with variance factor 1/J + n_test/n_train.
    """
    a, b = np.asarray(loss_a), np.asarray(loss_b)
    d = a - b
    J = len(d)
    se = np.sqrt((1 / J + (1 - ratio) / ratio) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else np.inf
    return {"mean_diff": float(d.mean()), "rel_diff_pct": float(100 * d.mean() / a.mean()),
            "b_wins": int((d > 0).sum()), "p_nb": float(2 * stats.t.sf(abs(t), J - 1)),
            "p_plain": float(stats.ttest_rel(a, b).pvalue)}
