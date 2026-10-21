"""Offline unit tests on synthetic data (no download needed): python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtlpool import (cv_select, independent_ridge, l21_mtl, mean_lowrank, mean_regularized, metrics,  # noqa: E402
                     paired_test, pooled_ridge, prox_l21, prox_trace, split, standardise, trace_mtl)
from mtlpool.solvers import gram_stats  # noqa: E402


def make_tasks(t=8, d=5, n=(15, 40), seed=0, shared=True):
    rng = np.random.default_rng(seed)
    w0 = rng.normal(size=d)
    Xs, ys = [], []
    for _ in range(t):
        k = rng.integers(*n)
        X = np.c_[rng.normal(size=(k, d - 1)), np.ones(k)]
        w = (w0 if shared else 0) + 0.3 * rng.normal(size=d)
        Xs.append(X)
        ys.append(X @ w + 0.5 * rng.normal(size=k))
    return Xs, ys


# ---------- proximal operators ----------
def test_prox_l21_worked_examples():
    V = np.array([[3.0, 4.0], [2.0, 0.5], [0.6, 0.6]])
    out = prox_l21(V, 1.0)
    np.testing.assert_allclose(out[0], [2.4, 3.2])
    np.testing.assert_allclose(out[1], np.array([2.0, 0.5]) * (1 - 1 / np.hypot(2, 0.5)))
    np.testing.assert_allclose(out[2], [0.0, 0.0])


def test_prox_trace_worked_example():
    np.testing.assert_allclose(prox_trace(np.array([[2.0, 1.0], [1.0, 2.0]]), 1.0), np.ones((2, 2)), atol=1e-12)


@pytest.mark.parametrize("prox, reg", [(prox_l21, lambda W: np.linalg.norm(W, axis=1).sum()),
                                       (prox_trace, lambda W: np.linalg.svd(W, compute_uv=False).sum())])
def test_prox_is_minimiser(prox, reg):
    """prox(V) minimises 0.5||W - V||^2 + tau R(W): random perturbations never do better."""
    rng = np.random.default_rng(1)
    V, tau = rng.normal(size=(6, 4)), 0.7
    obj = lambda W: 0.5 * np.sum((W - V) ** 2) + tau * reg(W)
    P = prox(V, tau)
    assert all(obj(P) <= obj(P + 1e-3 * rng.normal(size=V.shape)) + 1e-12 for _ in range(200))


# ---------- estimators ----------
def test_pooled_ridge_matches_closed_form():
    Xs, ys = make_tasks()
    lam = 0.3
    W = pooled_ridge(Xs, ys, lam)
    X, y = np.vstack(Xs), np.concatenate(ys)
    assert np.allclose(W, W[:, :1])                       # one shared model
    grad = X.T @ (X @ W[:, 0] - y) / len(y) + lam * W[:, 0]   # stationarity of (1/2n)||Xw - y||^2 + (lam/2)||w||^2
    assert np.linalg.norm(grad) < 1e-10


def test_mean_regularized_kkt():
    """Stationarity of sum_i (1/2n_i)||X_i(b + v_i) - y_i||^2 + (lam/2) sum ||v_i||^2 + (mu_b/2)||b||^2."""
    Xs, ys = make_tasks()
    lam = 0.5
    W = mean_regularized(Xs, ys, lam)
    G, B = gram_stats(Xs, ys)
    grads = np.stack([G[i] @ W[:, i] - B[:, i] for i in range(len(Xs))], 1)
    # v_i = w_i - b; stationarity in v_i gives grad_i + lam v_i = 0, so b = w_i + grad_i / lam for every i
    b_est = W + grads / lam
    assert np.allclose(b_est, b_est[:, :1], atol=1e-7)


def test_mean_regularized_limits():
    Xs, ys = make_tasks()
    ind = independent_ridge(Xs, ys, 1e-6)
    np.testing.assert_allclose(mean_regularized(Xs, ys, 1e-9), ind, atol=1e-3)
    W = mean_regularized(Xs, ys, 1e7)
    assert np.allclose(W, W[:, :1], atol=1e-4)            # all schools share the mean model


def test_proposition_thresholds():
    """Trace norm: W = 0 iff lam >= ||B||_2. l2,1: W = 0 iff lam >= max_j ||B^j||_2."""
    Xs, ys = make_tasks()
    _, B = gram_stats(Xs, ys)
    for fit, thr in ((trace_mtl, np.linalg.norm(B, 2)), (l21_mtl, np.linalg.norm(B, axis=1).max())):
        assert np.abs(fit(Xs, ys, 1.02 * thr)).max() < 1e-10
        assert np.abs(fit(Xs, ys, 0.98 * thr)).max() > 1e-4


def test_mean_lowrank_collapses_to_mean_above_threshold():
    """Mean + low-rank: V = 0 iff lam >= ||Gamma_0||_2, Gamma_0 = [G_i b_inf - B_i]."""
    Xs, ys = make_tasks()
    G, B = gram_stats(Xs, ys)
    b_inf = np.linalg.solve(G.sum(0) + 1e-6 * np.eye(G.shape[1]), B.sum(1))
    thr = np.linalg.norm(np.stack([G[i] @ b_inf - B[:, i] for i in range(len(Xs))], 1), 2)
    _, V = mean_lowrank(Xs, ys, 1.02 * thr, return_parts=True)
    assert np.abs(V).max() < 1e-8
    _, V = mean_lowrank(Xs, ys, 0.98 * thr, return_parts=True)
    assert np.abs(V).max() > 1e-6


def test_mean_lowrank_parts_add_up():
    Xs, ys = make_tasks()
    W, (b, V) = mean_lowrank(Xs, ys, 0.05), mean_lowrank(Xs, ys, 0.05, return_parts=True)
    np.testing.assert_allclose(W, b[:, None] + V, atol=1e-10)


def test_partial_pooling_helps_related_tasks():
    """On tasks that share a mean model, mean-reg MTL beats independent ridge with few samples."""
    Xs, ys = make_tasks(t=20, d=6, n=(8, 12), seed=3)
    tr, te = split(Xs, ys, 0.5, np.random.default_rng(0))
    Xa, ya = [x for x, _ in tr], [y for _, y in tr]
    ind = metrics(independent_ridge(Xa, ya, 1e-2), te, np.concatenate(ya))["nmse"]
    mr = metrics(mean_regularized(Xa, ya, 1.0), te, np.concatenate(ya))["nmse"]
    assert mr < ind


# ---------- data handling and evaluation ----------
def test_split_keeps_one_training_student():
    Xs, ys = make_tasks(n=(2, 5))
    tr, te = split(Xs, ys, 0.05, np.random.default_rng(0))
    assert all(len(y) >= 1 for _, y in tr)
    assert all(len(a[1]) + len(b[1]) == len(y) for a, b, y in zip(tr, te, ys))


def test_standardise_uses_training_statistics():
    Xs, ys = make_tasks(d=6)
    tr, te = split(Xs, ys, 0.5, np.random.default_rng(0))
    trs, tes = standardise(tr, te, cols=(0, 1))
    A = np.vstack([X for X, _ in trs])
    np.testing.assert_allclose(A[:, :2].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(A[:, :2].std(0), 1, atol=1e-9)
    np.testing.assert_array_equal(np.vstack([X for X, _ in tes])[:, 2:], np.vstack([X for X, _ in te])[:, 2:])


def test_cv_select_is_deterministic_and_on_grid():
    Xs, ys = make_tasks()
    tr = list(zip(Xs, ys))
    grid = np.logspace(-3, 2, 6)
    h1 = cv_select(mean_regularized, grid, tr, np.random.default_rng(5))
    h2 = cv_select(mean_regularized, grid, tr, np.random.default_rng(5))
    assert h1 == h2 and h1[0] == grid[h1[1]]


def test_metrics_perfect_and_zero_predictor():
    Xs, ys = make_tasks()
    te = list(zip(Xs, ys))
    W = np.stack([np.linalg.lstsq(X, X @ np.arange(X.shape[1]), rcond=None)[0] for X in Xs], 1)
    te_exact = [(X, X @ np.arange(X.shape[1])) for X in Xs]
    assert metrics(W, te_exact, np.concatenate(ys))["nmse"] < 1e-20
    m = metrics(np.zeros((Xs[0].shape[1], len(Xs))), te, np.concatenate(ys))
    y = np.concatenate(ys)
    assert np.isclose(m["nmse"], np.mean(y ** 2) / np.var(y))


def test_paired_test_direction():
    a = np.array([1.0, 1.1, 0.9, 1.05, 1.0, 0.95, 1.02, 1.01, 0.99, 1.0])
    b = a - 0.05 + 0.002 * np.random.default_rng(0).normal(size=a.size)
    out = paired_test(a, b, ratio=0.75)
    assert out["b_wins"] == 10 and out["mean_diff"] > 0 and out["p_nb"] < 0.01
