"""Vector figures for the paper. Every figure is saved as SVG and as PDF (same vector drawing).

    python experiments/make_figures.py [--analysis ...] [--optim ...] [--out figures]
"""
import argparse
import json
import logging
import os
import numpy as np
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

from _common import FIGURES, RAW, RESULTS, fresh_or_reference

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--analysis", default=str(fresh_or_reference(RESULTS / "analysis.json", "analysis.json")))
_p.add_argument("--optim", default=str(fresh_or_reference(RAW / "optimisation.json", "optimisation.json")))
_p.add_argument("--out", default=str(FIGURES))
ARGS = _p.parse_args()
logging.getLogger("fontTools").setLevel(logging.ERROR)   # harmless font timestamp warnings
A = json.load(open(ARGS.analysis))
O = json.load(open(ARGS.optim))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "lines.linewidth": 1.25, "lines.markersize": 3.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "svg.fonttype": "path", "pdf.fonttype": 42, "svg.hashsalt": "mtlpool",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

M = ["pooled ridge", "independent ridge", "l2,1 MTL", "trace-norm MTL", "mean-reg MTL", "mean+low-rank MTL"]
LABEL = {"pooled ridge": "Pooled ridge", "independent ridge": "Independent ridge",
         "l2,1 MTL": r"$\ell_{2,1}$ MTL", "trace-norm MTL": "Trace-norm MTL",
         "mean-reg MTL": "Mean-reg. MTL", "mean+low-rank MTL": "Mean + low-rank MTL"}
COLOR = {"pooled ridge": "#4d4d4d", "independent ridge": "#a6a6a6", "l2,1 MTL": "#E69F00",
         "trace-norm MTL": "#0072B2", "mean-reg MTL": "#009E73", "mean+low-rank MTL": "#D55E00"}
MARK = {"pooled ridge": "s", "independent ridge": "o", "l2,1 MTL": "^", "trace-norm MTL": "D",
        "mean-reg MTL": "P", "mean+low-rank MTL": "v"}
RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.60, 0.75]
TEXTW = 6.5


def save(fig, name):
    # fixed metadata so that rebuilding a figure gives a byte-identical file
    fig.savefig(os.path.join(ARGS.out, f"{name}.svg"), metadata={"Date": None})
    fig.savefig(os.path.join(ARGS.out, f"{name}.pdf"), format="pdf", metadata={"CreationDate": None}); plt.close(fig)


def pct_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([r * 100 for r in RATIOS]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"Training fraction per school $\rho$ (%)")


# ---------------- Figure 1: learned model matrices ----------------
def fig_overview():
    names = ["pooled ridge", "mean+low-rank MTL", "trace-norm MTL", "independent ridge"]
    mats = [np.array(A["W_fig"][n])[:27] for n in names]          # drop the intercept row
    vmax = np.percentile(np.abs(np.concatenate([x.ravel() for x in mats])), 98)
    fig, axes = plt.subplots(1, 4, figsize=(TEXTW, 1.75), sharey=True)
    for ax, n, W in zip(axes, names, mats):
        im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        sv = np.linalg.svd(W, compute_uv=False); r = int((sv > 1e-6 * sv[0]).sum())
        ax.set_title(f"{LABEL[n]}\n(rank {r})", pad=3)
        ax.set_xticks([]); ax.set_yticks([0, 13, 26]); ax.set_yticklabels(["1", "14", "27"])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.5)
    axes[0].set_ylabel("Feature")
    fig.supxlabel("Columns: the 139 schools, ordered from smallest to largest", fontsize=8, y=0.02)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6.5, width=0.4)
    cb.set_label("Weight", fontsize=7.5)
    save(fig, "fig1_model_matrices")


# ---------------- Figure 2: learning curves ----------------
def fig_curves():
    S = A["summary"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.45))
    x = np.array(RATIOS) * 100
    for mth in M:
        mu = np.array([S[str(r)][mth]["nmse"][0] for r in RATIOS])
        sd = np.array([S[str(r)][mth]["nmse"][1] for r in RATIOS])
        lw = 1.8 if mth == "mean-reg MTL" else 1.15
        a.errorbar(x, mu, yerr=sd, color=COLOR[mth], marker=MARK[mth], lw=lw, capsize=1.5, elinewidth=0.6,
                   label=LABEL[mth], zorder=3 if mth == "mean-reg MTL" else 2)
        if mth in ("trace-norm MTL", "mean-reg MTL", "mean+low-rank MTL"):
            base = np.array([S[str(r)]["pooled ridge"]["nmse"][0] for r in RATIOS])
            b.plot(x, 100 * (mu - base) / base, color=COLOR[mth], marker=MARK[mth], lw=lw,
                   zorder=3 if mth == "mean-reg MTL" else 2)
    for ax in (a, b):
        pct_axis(ax)
    a.set_ylabel("Test nMSE (lower is better)")
    a.set_title(r"(a) Learning curves, mean $\pm$ s.d. over 10 splits", loc="left")
    b.axhline(0, color=COLOR["pooled ridge"], lw=0.9, ls=(0, (4, 2)))
    b.text(76, 0.8, "pooled ridge", ha="right", va="bottom", fontsize=7, color=COLOR["pooled ridge"])
    b.set_ylabel("nMSE relative to pooled ridge (%)")
    b.set_title("(b) Change versus one pooled model", loc="left")
    b.set_ylim(-9, 30)
    a.legend(loc="upper right", ncol=1, handlelength=2.2)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_learning_curves")


# ---------------- Figure 3: who benefits ----------------
def fig_schools():
    P = {k: np.array(v) for k, v in A["per_school"].items()}; n = np.array(A["sizes"])
    pairs = [("independent ridge", "trace-norm MTL", "(a) Trace-norm vs. independent"),
             ("trace-norm MTL", "mean-reg MTL", "(b) Mean-reg. vs. trace-norm"),
             ("pooled ridge", "mean-reg MTL", "(c) Mean-reg. vs. pooled")]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.3), sharex=True)
    bins = np.quantile(n, np.linspace(0, 1, 7))
    for ax, (base, meth, title) in zip(axes, pairs):
        g = 100 * (P[base] - P[meth]) / P[base]
        ax.scatter(n, g, s=7, color=COLOR[meth], alpha=0.55, lw=0, zorder=2)
        idx = np.clip(np.digitize(n, bins[1:-1]), 0, 5)
        cx = [np.median(n[idx == k]) for k in range(6)]; cy = [np.median(g[idx == k]) for k in range(6)]
        ax.plot(cx, cy, color="black", lw=1.2, marker="o", ms=3, zorder=3, label="median per size sextile")
        ax.axhline(0, color="#888888", lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator([25, 50, 100, 200]))
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Students in school (log scale)")
        ax.set_title(title, loc="left")
        ax.text(0.97, 0.03, f"better on {int((g > 0).sum())} of {len(g)} schools", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.8,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=4)
    axes[0].set_ylabel("Per-school MSE reduction (%)")
    axes[0].legend(loc="lower left", fontsize=6.6, bbox_to_anchor=(0.0, 0.08))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_per_school")


# ---------------- Figure 4: learned structure ----------------
def fig_structure():
    sp = A["spectra"]; st = A["structure"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.35))
    styles = {"independent ridge": (COLOR["independent ridge"], "o", "Independent ridge, $W$"),
              "trace-norm MTL": (COLOR["trace-norm MTL"], "D", "Trace-norm MTL, $W$"),
              "mean+low-rank MTL (deviations V)": (COLOR["mean+low-rank MTL"], "v", "Mean + low-rank, deviations $V$")}
    for k, (c, mk, lab) in styles.items():
        v = np.array(sp[k]); keep = v > 1e-6
        a.semilogy(np.arange(1, len(v) + 1)[keep], v[keep], color=c, marker=mk, ms=2.8, lw=1.0, label=lab)
    a.set_ylim(5e-4, 1.6); a.set_xlabel(r"Singular value index $k$"); a.set_ylabel(r"$\sigma_k/\sigma_1$")
    a.set_title(r"(a) Spectra at $\rho = 75\%$ (split 1)", loc="left"); a.legend(loc="lower right")
    x = np.array(RATIOS) * 100
    for key, c, mk, lab in [("trace_rank", COLOR["trace-norm MTL"], "D", "Trace-norm MTL: rank of $W$"),
                            ("mlr_dev_rank", COLOR["mean+low-rank MTL"], "v", "Mean + low-rank: rank of $V$"),
                            ("l21_rows", COLOR["l2,1 MTL"], "^", r"$\ell_{2,1}$ MTL: active features")]:
        mu = np.array([st[str(r)][key][0] for r in RATIOS]); sd = np.array([st[str(r)][key][1] for r in RATIOS])
        b.errorbar(x, mu, yerr=sd, color=c, marker=mk, capsize=1.5, elinewidth=0.6, label=lab)
    pct_axis(b); b.set_ylabel("Count (of 28)"); b.set_ylim(0, 29)
    b.set_title("(b) Structure chosen by cross-validation", loc="left"); b.legend(loc="upper left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_structure")


# ---------------- Figure 5: optimisation ----------------
def fig_optim():
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    k = np.arange(1, 3001)
    for key, c, ls, lab in [("raw-ista", "#a6a6a6", "-", "Raw features, ISTA"),
                            ("raw-fista", "#4d4d4d", "-", "Raw features, FISTA"),
                            ("standardised-ista", "#56B4E9", "-", "Standardised, ISTA"),
                            ("standardised-fista", "#0072B2", "-", "Standardised, FISTA")]:
        g = np.maximum(np.array(O[key]["rel_gap"]), 1e-12)
        ax.loglog(k, g, color=c, ls=ls, lw=1.2, label=lab)
    ax.axvline(500, color="#D55E00", lw=0.8, ls=(0, (3, 2)))
    ax.text(440, 3e-11, "500-iteration\nbudget", fontsize=6.8, color="#D55E00", va="bottom", ha="right")
    ax.set_xlabel(r"Iteration $k$"); ax.set_ylabel(r"$(F(W_k)-F^\star)/F^\star$")
    ax.set_ylim(1e-11, 3); ax.legend(loc="lower left", fontsize=6.6)
    save(fig, "fig5_optimisation")


if __name__ == "__main__":
    os.makedirs(ARGS.out, exist_ok=True)
    fig_overview(); fig_curves(); fig_schools(); fig_structure(); fig_optim()
    print(f"figures written to {ARGS.out}")
