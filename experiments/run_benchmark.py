"""Main experiment: six estimators x training fractions x random splits, with 5-fold CV.

    python experiments/run_benchmark.py                      # full study (about 20 to 25 min on 2 cores)
    python experiments/run_benchmark.py --splits 2 --ratios 0.1 0.75 --out results/quick/benchmark.json
"""
import argparse
import json
import os
import platform
import time
from multiprocessing import Pool

import numpy as np

from _common import DATA, RAW
import mtlpool
from mtlpool import config
from mtlpool.data import load_school, make_split, unzip
from mtlpool.estimators import ESTIMATORS
from mtlpool.evaluation import cv_select, metrics


def job(args):
    path, ratio, s, methods = args
    Xs, ys = load_school(path)
    tr, te = make_split(Xs, ys, ratio, s, config.SPLIT_SEED_BASE)
    Xtr, ytr = unzip(tr)
    out = {}
    for name in methods:
        fit, grid = ESTIMATORS[name], config.GRIDS[name]
        # the same CV folds for every method on a given split
        h, idx = cv_select(fit, grid, tr, np.random.default_rng(config.CV_SEED_BASE + s))
        W = fit(Xtr, ytr, h)
        res = {k: float(v) for k, v in metrics(W, te, np.concatenate(ytr)).items()}
        res["hyper"] = float(h)
        res["edge"] = bool(idx in (0, len(grid) - 1))
        out[name] = res
    return ratio, s, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--out", default=str(RAW / "benchmark.json"))
    p.add_argument("--splits", type=int, default=config.N_SPLITS)
    p.add_argument("--ratios", type=float, nargs="+", default=list(config.RATIOS))
    p.add_argument("--methods", nargs="+", default=list(config.METHODS))
    p.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    a = p.parse_args()

    jobs = [(a.data, r, s, a.methods) for r in a.ratios for s in range(a.splits)]
    print(f"{len(jobs)} jobs ({len(a.ratios)} ratios x {a.splits} splits), {len(a.methods)} methods, "
          f"{a.workers} workers")
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap(job, jobs), 1):
            res.append(r)
            print(f"  [{i}/{len(jobs)}] ratio {r[0]:.2f} split {r[1]} done", flush=True)
    meta = {"mtlpool": mtlpool.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "splits": a.splits, "ratios": a.ratios, "methods": a.methods,
            "minutes": round((time.time() - t0) / 60, 2)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "results": [[r, s, o] for r, s, o in res]}, f, indent=1)
    print(f"wrote {a.out} in {meta['minutes']} min")


if __name__ == "__main__":
    main()
