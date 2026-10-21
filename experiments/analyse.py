"""Statistics behind every number in the paper: summaries, paired tests, per-school gains,
learned structure, shrinkage of the shared model. Writes results/analysis.json."""
import argparse
import json

import numpy as np

from _common import DATA, RAW, RESULTS
from mtlpool import config
from mtlpool.data import load_school, make_split, unzip
from mtlpool.estimators import ESTIMATORS, mean_lowrank
from mtlpool.evaluation import paired_test

KEYS = ["nmse", "ev_within", "ev_pooled", "rmse", "rmse_z"]
COMPARISONS = [("independent ridge", "trace-norm MTL"), ("independent ridge", "l2,1 MTL"),
               ("l2,1 MTL", "trace-norm MTL"), ("pooled ridge", "trace-norm MTL"),
               ("pooled ridge", "mean-reg MTL"), ("pooled ridge", "mean+low-rank MTL"),
               ("trace-norm MTL", "mean+low-rank MTL"), ("mean-reg MTL", "mean+low-rank MTL"),
               ("trace-norm MTL", "mean-reg MTL")]


def load_results(path):
    data = json.load(open(path))
    return data["results"] if isinstance(data, dict) else data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--results", default=str(RAW / "benchmark.json"))
    p.add_argument("--out", default=str(RESULTS / "analysis.json"))
    a = p.parse_args()

    res = {}
    for r, s, o in load_results(a.results):
        res.setdefault((round(r, 2), s), {}).update(o)
    ratios = sorted({r for r, _ in res})
    J = max(s for _, s in res) + 1
    M = list(config.METHODS)
    Xs, ys = load_school(a.data)

    def vals(r, mth, key):
        return np.array([res[(r, s)][mth][key] for s in range(J)])

    def refit(r, s, mth):
        tr, te = make_split(Xs, ys, r, s, config.SPLIT_SEED_BASE)
        return ESTIMATORS[mth](*unzip(tr), res[(r, s)][mth]["hyper"]), tr, te

    summary = {str(r): {mth: {k: [float(vals(r, mth, k).mean()), float(vals(r, mth, k).std())] for k in KEYS}
                        | {"hyper_median": float(np.median(vals(r, mth, "hyper"))),
                           "edges": int(sum(res[(r, s)][mth]["edge"] for s in range(J)))}
                        for mth in M} for r in ratios}
    tests = {str(r): {f"{x} vs {y}": paired_test(vals(r, x, "nmse"), vals(r, y, "nmse"), r)
                      for x, y in COMPARISONS} for r in ratios}
    best = {str(r): min(M, key=lambda mth: summary[str(r)][mth]["nmse"][0]) for r in ratios}

    # per-school errors at 20% training (Figure 3)
    sizes = np.array([len(y) for y in ys])
    per_school = {mth: np.zeros(len(ys)) for mth in
                  ["pooled ridge", "independent ridge", "trace-norm MTL", "mean-reg MTL", "mean+low-rank MTL"]}
    for s in range(J):
        for mth in per_school:
            W, _, te = refit(0.2, s, mth)
            for i, (X, y) in enumerate(te):
                per_school[mth][i] += np.mean((X @ W[:, i] - y) ** 2) / J

    # structure selected by CV (Figure 4b)
    def srank(W, tol=1e-6):
        sv = np.linalg.svd(W, compute_uv=False)
        return int((sv > tol * max(sv[0], 1e-12)).sum())

    structure = {}
    for r in ratios:
        ranks, rows, vranks = [], [], []
        for s in range(J):
            Wt, _, _ = refit(r, s, "trace-norm MTL")
            Wl, tr, _ = refit(r, s, "l2,1 MTL")
            _, V = mean_lowrank(*unzip(tr), res[(r, s)]["mean+low-rank MTL"]["hyper"], return_parts=True)
            ranks.append(srank(Wt)); rows.append(int((np.linalg.norm(Wl, axis=1) > 1e-8).sum())); vranks.append(srank(V))
        structure[str(r)] = {"trace_rank": [float(np.mean(ranks)), float(np.std(ranks))],
                             "l21_rows": [float(np.mean(rows)), float(np.std(rows))],
                             "mlr_dev_rank": [float(np.mean(vranks)), float(np.std(vranks))]}

    # spectra at 75%, first split (Figure 4a)
    spectra = {}
    for mth in ["independent ridge", "trace-norm MTL"]:
        W, _, _ = refit(0.75, 0, mth)
        sv = np.linalg.svd(W, compute_uv=False); spectra[mth] = (sv / sv[0]).tolist()
    _, tr, _ = refit(0.75, 0, "pooled ridge")
    _, V = mean_lowrank(*unzip(tr), res[(0.75, 0)]["mean+low-rank MTL"]["hyper"], return_parts=True)
    sv = np.linalg.svd(V, compute_uv=False); spectra["mean+low-rank MTL (deviations V)"] = (sv / sv[0]).tolist()

    # shrinkage of the shared model: norm of the average school model relative to pooled ridge
    shrinkage = {}
    for r in (0.05, 0.2, 0.75):
        rows = []
        for s in range(J):
            Wp, _, _ = refit(r, s, "pooled ridge")
            ref = np.linalg.norm(Wp[:, 0])
            rows.append([np.linalg.norm(refit(r, s, mth)[0].mean(1)) / ref
                         for mth in ("trace-norm MTL", "mean-reg MTL")])
        rows = np.array(rows)
        shrinkage[str(r)] = {"trace": [float(rows[:, 0].mean()), float(rows[:, 0].std())],
                             "meanreg": [float(rows[:, 1].mean()), float(rows[:, 1].std())]}

    # model matrices for Figure 1 (20%, first split), schools sorted by size
    order = np.argsort(sizes)
    W_fig = {mth: refit(0.2, 0, mth)[0][:, order].tolist()
             for mth in ["pooled ridge", "mean+low-rank MTL", "trace-norm MTL", "independent ridge"]}

    facts = {"tasks": len(ys), "students": int(sizes.sum()), "d": int(Xs[0].shape[1]),
             "n_min": int(sizes.min()), "n_median": float(np.median(sizes)), "n_max": int(sizes.max()),
             "y_mean": float(np.concatenate(ys).mean()), "y_std": float(np.concatenate(ys).std()),
             "const_cols": [j for j in range(Xs[0].shape[1]) if all(np.ptp(X[:, j]) == 0 for X in Xs)],
             "rank_min": int(min(np.linalg.matrix_rank(X) for X in Xs)),
             "rank_median": float(np.median([np.linalg.matrix_rank(X) for X in Xs])),
             "rank_max": int(max(np.linalg.matrix_rank(X) for X in Xs)),
             "rank_pooled": int(np.linalg.matrix_rank(np.vstack(Xs))),
             "between_share": float(np.var(np.concatenate([np.full(len(v), v.mean()) for v in ys]))
                                    / np.var(np.concatenate(ys)))}

    out = {"summary": summary, "tests": tests, "best": best, "structure": structure, "spectra": spectra,
           "per_school": {k: v.tolist() for k, v in per_school.items()}, "sizes": sizes.tolist(),
           "W_fig": W_fig, "facts": facts, "shrinkage": shrinkage}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"wrote {a.out}")
    for r in ratios:
        line = "  ".join(f"{m.split()[0]} {summary[str(r)][m]['nmse'][0]:.4f}" for m in M)
        print(f"  {int(round(r * 100)):2d}%  best: {best[str(r)]:18s} | {line}")


if __name__ == "__main__":
    main()
