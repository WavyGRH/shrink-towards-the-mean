"""Proximal operators and FISTA for regularised multi-task least squares.

Loss: L(W) = sum_i (1 / 2 n_i) ||X_i w_i - y_i||^2, with W = [w_1, ..., w_t] (d x t).
"""
import numpy as np

from .config import FISTA_ITERS


def prox_l21(V, tau):
    """Prox of tau * sum_j ||row_j||_2: shrink each row, zeroing whole features."""
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return np.maximum(0, 1 - tau / np.maximum(norms, 1e-12)) * V


def prox_trace(V, tau):
    """Prox of tau * ||V||_*: singular value thresholding."""
    U, s, Vt = np.linalg.svd(V, full_matrices=False)
    return (U * np.maximum(s - tau, 0)) @ Vt


def gram_stats(Xs, ys):
    """Per-task G_i = X_i^T X_i / n_i (t x d x d) and B = [X_i^T y_i / n_i] (d x t)."""
    G = np.stack([X.T @ X / len(y) for X, y in zip(Xs, ys)])
    B = np.stack([X.T @ y / len(y) for X, y in zip(Xs, ys)], 1)
    return G, B


def loss_gradient(G, B):
    """Gradient of L: column i equals G_i w_i - B_i."""
    return lambda W: np.einsum("tij,jt->it", G, W) - B


def fista(Xs, ys, lam, prox, iters=FISTA_ITERS):
    """Minimise L(W) + lam * R(W) by FISTA with step 1/L, L = max_i ||X_i||_2^2 / n_i."""
    G, B = gram_stats(Xs, ys)
    grad = loss_gradient(G, B)
    L = max(np.linalg.norm(X, 2) ** 2 / len(X) for X in Xs)
    W = Z = np.zeros((Xs[0].shape[1], len(Xs)))
    s = 1.0
    for _ in range(iters):
        W_new = prox(Z - grad(Z) / L, lam / L)
        s_new = (1 + np.sqrt(1 + 4 * s * s)) / 2
        Z = W_new + ((s - 1) / s_new) * (W_new - W)
        W, s = W_new, s_new
    return W
