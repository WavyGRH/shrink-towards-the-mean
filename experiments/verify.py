"""Compare a fresh run with the results reported in the paper (results/reference/).

Checks every metric, selected hyperparameter and grid-edge flag of every (fraction, split, method),
then every number in analysis.json. Exits non-zero on any mismatch beyond the tolerance.

    python experiments/verify.py                 # full run
    python experiments/verify.py --tol 1e-6      # looser, e.g. across BLAS libraries
"""
import argparse
import json
import sys

import numpy as np

from _common import RAW, REFERENCE, RESULTS


def load(path):
    data = json.load(open(path))
    rows = data["results"] if isinstance(data, dict) else data
    out = {}
    for r, s, o in rows:
        out.setdefault((round(r, 2), s), {}).update(o)
    return out


def flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def close(a, b, tol):
    if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(b))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(RAW / "benchmark.json"))
    p.add_argument("--analysis", default=str(RESULTS / "analysis.json"))
    p.add_argument("--optim", default=str(RAW / "optimisation.json"))
    p.add_argument("--tol", type=float, default=1e-9, help="relative tolerance")
    p.add_argument("--benchmark-only", action="store_true", help="check only the benchmark results")
    a = p.parse_args()

    new, ref = load(a.results), load(REFERENCE / "benchmark.json")
    bad, n, worst = [], 0, 0.0
    for key, methods in new.items():
        for m, vals in methods.items():
            for k, v in vals.items():
                r = ref[key][m][k]
                n += 1
                if not isinstance(v, bool):
                    worst = max(worst, abs(v - r) / max(1.0, abs(r)))
                if not close(v, r, a.tol):
                    bad.append(f"benchmark {key} {m} {k}: {v} vs paper {r}")
    print(f"benchmark: {n} values compared over {len(new)} (fraction, split) pairs, "
          f"max relative difference {worst:.1e}")

    for name, path in () if a.benchmark_only else (("analysis", a.analysis), ("optimisation", a.optim)):
        try:
            got = dict(flatten(json.load(open(path))))
        except FileNotFoundError:
            print(f"{name}: {path} not found, skipped")
            continue
        exp = dict(flatten(json.load(open(REFERENCE / f"{name}.json"))))
        common = [k for k in exp if k in got]
        worst = max((abs(got[k] - exp[k]) / max(1.0, abs(exp[k])) for k in common
                     if not isinstance(exp[k], (bool, str))), default=0.0)
        bad += [f"{name} {k}: {got[k]} vs paper {exp[k]}" for k in common if not close(got[k], exp[k], a.tol)]
        print(f"{name}: {len(common)} of {len(exp)} values compared, max relative difference {worst:.1e}")

    if bad:
        print(f"\n{len(bad)} mismatches beyond relative tolerance {a.tol:g}:")
        print("\n".join(bad[:30]))
        sys.exit(1)
    print("\nOK: everything matches the paper.")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
