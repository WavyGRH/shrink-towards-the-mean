"""Optimisation experiment (Figure 5): trace-norm MTL, raw vs standardised features, ISTA vs FISTA.

Uses the first split of the 75% protocol and lambda = 3. F* is the best objective in 100,000 FISTA steps.
"""
import argparse
import json
import os

import numpy as np

from _common import DATA, RAW
from mtlpool import config
from mtlpool.data import load_school, split, standardise, unzip
from mtlpool.solvers import gram_stats, loss_gradient, prox_trace


def run(Xs, ys, lam, iters, momentum):
    G, B = gram_stats(Xs, ys)
    grad = loss_gradient(G, B)
    yy = sum(float(y @ y) / (2 * len(y)) for y in ys)

    def F(W):
        quad = 0.5 * np.einsum("it,tij,jt->", W, G, W) - np.einsum("it,it->", W, B) + yy
        return quad + lam * np.linalg.svd(W, compute_uv=False).sum()

    L = max(np.linalg.norm(Gi, 2) for Gi in G)
    W = Z = np.zeros(B.shape)
    s = 1.0
    hist = []
    for _ in range(iters):
        W_new = prox_trace(Z - grad(Z) / L, lam / L)
        if momentum:
            s_new = (1 + np.sqrt(1 + 4 * s * s)) / 2
            Z = W_new + ((s - 1) / s_new) * (W_new - W)
            s = s_new
        else:
            Z = W_new
        W = W_new
        hist.append(F(W))
    return np.array(hist), L


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--out", default=str(RAW / "optimisation.json"))
    p.add_argument("--ref-iters", type=int, default=100_000)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lam", type=float, default=3.0)
    a = p.parse_args()

    Xs, ys = load_school(a.data)
    tr, te = split(Xs, ys, 0.75, np.random.default_rng(config.SPLIT_SEED_BASE))
    trs, _ = standardise(tr, te)
    out = {}
    for tag, data in (("raw", tr), ("standardised", trs)):
        Xa, ya = unzip(data)
        ref, L = run(Xa, ya, a.lam, a.ref_iters, True)
        fstar = float(ref.min())
        for mom in (False, True):
            h, _ = run(Xa, ya, a.lam, a.iters, mom)
            out[f"{tag}-{'fista' if mom else 'ista'}"] = {"L": float(L), "fstar": fstar,
                                                          "rel_gap": ((h - fstar) / fstar).tolist()}
        print(f"{tag}: L = {L:.2f}, gap at 500: ISTA {out[f'{tag}-ista']['rel_gap'][499]:.2e}, "
              f"FISTA {out[f'{tag}-fista']['rel_gap'][499]:.2e}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
