"""The six estimators compared in the paper. Each takes (Xs, ys, hyperparameter) and returns W (d x t)."""
import numpy as np

from .config import FISTA_ITERS, MU_B
from .solvers import fista, prox_l21, prox_trace


def pooled_ridge(Xs, ys, mu):
    """One ridge model for all students (each student weighted equally), copied to every school."""
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    w = np.linalg.solve(X.T @ X / len(y) + mu * np.eye(X.shape[1]), X.T @ y / len(y))
    return np.tile(w[:, None], (1, len(Xs)))


def independent_ridge(Xs, ys, mu):
    """A separate ridge model per school."""
    return np.column_stack([
        np.linalg.solve(X.T @ X / len(y) + mu * np.eye(X.shape[1]), X.T @ y / len(y))
        for X, y in zip(Xs, ys)])


def l21_mtl(Xs, ys, lam):
    """L(W) + lam * ||W||_{2,1}: shared feature selection."""
    return fista(Xs, ys, lam, prox_l21)


def trace_mtl(Xs, ys, lam):
    """L(W) + lam * ||W||_*: shared low-dimensional subspace."""
    return fista(Xs, ys, lam, prox_trace)


def _task_stats(Xs, ys):
    G = np.stack([X.T @ X / len(y) for X, y in zip(Xs, ys)])   # t x d x d
    B = np.stack([X.T @ y / len(y) for X, y in zip(Xs, ys)])   # t x d
    return G, B


def mean_regularized(Xs, ys, lam):
    """W = b 1^T + V, minimising L + lam/2 ||V||_F^2 + mu_b/2 ||b||^2, solved exactly.

    With R_i = (G_i + lam I)^-1:  (lam sum_i G_i R_i + mu_b I) b = lam sum_i R_i B_i,
    and v_i = R_i (B_i - G_i b).
    """
    G, B = _task_stats(Xs, ys)
    I = np.eye(G.shape[1])
    R = np.stack([np.linalg.inv(Gi + lam * I) for Gi in G])
    A = lam * np.einsum("tij,tjk->ik", G, R) + MU_B * I
    rhs = lam * np.einsum("tij,tj->i", R, B)
    b = np.linalg.solve(A, rhs)
    V = np.einsum("tij,tj->it", R, B - np.einsum("tij,j->ti", G, b))
    return b[:, None] + V


def mean_lowrank(Xs, ys, lam, iters=FISTA_ITERS, return_parts=False):
    """W = b 1^T + V, minimising L + lam ||V||_* + mu_b/2 ||b||^2.

    FISTA on V for the reduced objective min_b L(b 1^T + V): the shared mean b is minimised
    exactly at every gradient evaluation (Algorithm 1 in the paper).
    """
    G, B = _task_stats(Xs, ys)
    d, t = G.shape[1], G.shape[0]
    Ginv = np.linalg.inv(G.sum(0) + MU_B * np.eye(d))

    def bstar(V):
        return Ginv @ (B.sum(0) - np.einsum("tij,jt->i", G, V))

    def grad(V):
        b = bstar(V)
        return np.einsum("tij,jt->it", G, V + b[:, None]) - B.T

    L = max(np.linalg.norm(Gi, 2) for Gi in G)
    V = Z = np.zeros((d, t))
    s = 1.0
    for _ in range(iters):
        V_new = prox_trace(Z - grad(Z) / L, lam / L)
        s_new = (1 + np.sqrt(1 + 4 * s * s)) / 2
        Z = V_new + ((s - 1) / s_new) * (V_new - V)
        V, s = V_new, s_new
    b = bstar(V)
    return (b, V) if return_parts else b[:, None] + V


ESTIMATORS = {
    "pooled ridge": pooled_ridge,
    "independent ridge": independent_ridge,
    "l2,1 MTL": l21_mtl,
    "trace-norm MTL": trace_mtl,
    "mean-reg MTL": mean_regularized,
    "mean+low-rank MTL": mean_lowrank,
}
