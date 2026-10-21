"""Reproduce the whole paper with one command (works the same on Windows, macOS and Linux).

    python run_all.py            # data, benchmark, optimisation, analysis, figures, tables, verification
    python run_all.py --quick    # 2 splits x 3 fractions, a smoke test of the pipeline in a few minutes
    python run_all.py --from-reference   # skip the experiments, rebuild figures and tables from the paper's results
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def step(title, *args):
    print(f"\n== {title} ==", flush=True)
    t = time.time()
    subprocess.run([PY, *map(str, args)], cwd=ROOT, check=True)
    print(f"   ({time.time() - t:.0f} s)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--from-reference", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()
    w = ["--workers", str(a.workers)] if a.workers else []

    if a.from_reference:
        ref = ROOT / "results" / "reference"
        step("figures", "experiments/make_figures.py", "--analysis", ref / "analysis.json",
             "--optim", ref / "optimisation.json")
        step("tables", "experiments/make_tables.py", "--analysis", ref / "analysis.json")
        return

    step("data", "scripts/download_data.py")
    if a.quick:
        out = ROOT / "results" / "quick"
        step("benchmark (quick)", "experiments/run_benchmark.py", "--splits", "2", "--ratios", "0.05", "0.2",
             "0.75", "--out", out / "benchmark.json", *w)
        step("optimisation (quick)", "experiments/run_optimisation.py", "--ref-iters", "5000", "--iters", "600",
             "--out", out / "optimisation.json")
        step("verify the quick splits against the paper", "experiments/verify.py", "--results", out / "benchmark.json", "--benchmark-only")
        return

    step("benchmark (about 20 to 25 min on 2 cores)", "experiments/run_benchmark.py", *w)
    step("optimisation", "experiments/run_optimisation.py")
    step("analysis", "experiments/analyse.py")
    step("figures", "experiments/make_figures.py")
    step("tables", "experiments/make_tables.py")
    step("verify against the paper", "experiments/verify.py")


if __name__ == "__main__":
    main()
