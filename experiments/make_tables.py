"""Generate the LaTeX tables of the paper from results/analysis.json, so no number is transcribed by hand.

    python experiments/make_tables.py [--analysis results/analysis.json] [--out paper/tables]
"""
import argparse
import json
import os

import numpy as np

from _common import RESULTS, TABLES, fresh_or_reference

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--analysis", default=str(fresh_or_reference(RESULTS / "analysis.json", "analysis.json")))
_p.add_argument("--out", default=str(TABLES))
ARGS = _p.parse_args()
os.makedirs(ARGS.out, exist_ok=True)
A = json.load(open(ARGS.analysis))
S, T = A["summary"], A["tests"]
M = ["pooled ridge", "independent ridge", "l2,1 MTL", "trace-norm MTL", "mean-reg MTL", "mean+low-rank MTL"]
TEX = {"pooled ridge": "Pooled ridge", "independent ridge": "Independent ridge",
       "l2,1 MTL": r"$\ell_{2,1}$ MTL", "trace-norm MTL": "Trace-norm MTL",
       "mean-reg MTL": "Mean-reg.\\ MTL", "mean+low-rank MTL": "Mean + low-rank MTL"}
RATIOS = ["0.05", "0.1", "0.15", "0.2", "0.3", "0.6", "0.75"]


def pm(x, d=1):
    return f"${x[0]:.{d}f} \\pm {x[1]:.{d}f}$"


# ---------- Table 1: reproduction ----------
pub_ev = {"pooled ridge": ("Aggregate", "$22.7 \\pm 1.3$", ""),
          "independent ridge": ("Independent / STL", "$23.8 \\pm 2.1$", "$23.5 \\pm 1.9$"),
          "l2,1 MTL": ("MTL-FEAT (var.\\ sel.)", "$24.8 \\pm 2.0$", ""),
          "trace-norm MTL": ("MTL-FEAT / MTFL", "$26.7 \\pm 2.0$", "$26.7 \\pm 2.0$"),
          "mean-reg MTL": ("", "", ""), "mean+low-rank MTL": ("", "", "")}
pub_rmse = {"independent ridge": "$10.67 \\pm 0.20$"}
rows = []
for mth in M:
    name, a08, zy10 = pub_ev[mth]
    rows.append(f"{TEX[mth]} & {name} & {pm(S['0.75'][mth]['ev_within'])} & {a08} & {zy10} & "
                f"{pm(S['0.6'][mth]['rmse'], 2)} & {pub_rmse.get(mth, '')} \\\\")
extra = [r"Multi-task GP & MTGP & & & $29.2 \pm 1.6$ & & \\",
         r"MTRL & MTRL & & & $29.9 \pm 1.8$ & & \\",
         r"No-group MTL & No-group MTL & & & & & $10.18 \pm 0.15$ \\",
         r"GO-MTL & GO-MTL & & & & & $10.04 \pm 0.24$ \\"]
t1 = r"""\begin{table}[t]
\centering
\caption{Reproduction of published School results. Explained variance (EV, within schools, \%; higher is better) under the protocol of \citet{argyriou2008convex} ($\rho = 75\%$) and RMSE in exam points (lower is better) under the protocol of \citet{kumar2012learning} ($\rho = 60\%$). Ours: mean $\pm$ s.d.\ over 10 random splits. Published values are copied from the cited papers; blank cells were not reported or not run. The bottom four rows are published methods we did not re-implement.}
\label{tab:repro}
\small
\setlength{\tabcolsep}{2.7pt}
\begin{tabular}{@{}llccccc@{}}
\toprule
& & \multicolumn{3}{c}{EV within schools (\%), $\rho = 75\%$} & \multicolumn{2}{c}{RMSE, $\rho = 60\%$} \\
\cmidrule(lr){3-5}\cmidrule(l){6-7}
Method & Published counterpart & Ours & A08 & ZY10 & Ours & KD12 \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + "\n".join(extra) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
{\footnotesize A08: \citet{argyriou2008convex}; ZY10: \citet{zhang2010convex}; KD12: \citet{kumar2012learning}.}
\end{table}
"""
open(os.path.join(ARGS.out, "tab_repro.tex"), "w").write(t1)

# ---------- Table 2: main results (nMSE) ----------
def cell(r, mth):
    mu, sd = S[r][mth]["nmse"]
    vals = sorted(S[r][x]["nmse"][0] for x in M)
    s = f"{mu:.3f}".lstrip("0")
    s = f"\\textbf{{{s}}}" if mu == vals[0] else (f"\\underline{{{s}}}" if mu == vals[1] else s)
    return f"{s}\\,{{\\scriptsize$\\pm${sd:.3f}}}".replace("$\\pm$0.", "$\\pm$.")

lines = [f"{TEX[mth]} & " + " & ".join(cell(r, mth) for r in RATIOS) + r" \\" for mth in M]
t2 = r"""\begin{table}[t]
\centering
\caption{Test nMSE (MSE divided by the variance of test scores; lower is better) as a function of the training fraction per school $\rho$. Mean $\pm$ s.d.\ over 10 random splits; best in bold, second best underlined.}
\label{tab:main}
\small
\setlength{\tabcolsep}{3.6pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
& \multicolumn{7}{c}{Training fraction per school $\rho$} \\
\cmidrule(l){2-8}
Method & 5\% & 10\% & 15\% & 20\% & 30\% & 60\% & 75\% \\
\midrule
""" + "\n".join(lines[:2]) + "\n\\midrule\n" + "\n".join(lines[2:4]) + "\n\\midrule\n" + "\n".join(lines[4:]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(ARGS.out, "tab_main.tex"), "w").write(t2)

# ---------- Table 3: paired comparisons ----------
COMP = [("independent ridge", "trace-norm MTL"), ("pooled ridge", "trace-norm MTL"),
        ("pooled ridge", "mean-reg MTL"), ("trace-norm MTL", "mean-reg MTL"),
        ("pooled ridge", "mean+low-rank MTL"), ("trace-norm MTL", "mean+low-rank MTL"),
        ("mean-reg MTL", "mean+low-rank MTL")]
SHORT = {"pooled ridge": "Pooled", "independent ridge": "Indep.", "trace-norm MTL": "Trace",
         "mean-reg MTL": "MR", "mean+low-rank MTL": "MLR", "l2,1 MTL": "$\\ell_{2,1}$"}

def pc(r, a, b):
    t = T[r][f"{a} vs {b}"]
    mark = "$^{\\dagger}$" if t["p_nb"] < 0.05 else ""
    return f"${-t['rel_diff_pct']:+.1f}${mark}\\,{{\\scriptsize({t['b_wins']})}}"

lines = [f"{SHORT[b]} vs.\\ {SHORT[a]} & " + " & ".join(pc(r, a, b) for r in RATIOS) + r" \\" for a, b in COMP]
t3 = r"""\begin{table}[t]
\centering
\caption{Paired comparisons on the same splits. Each cell gives the relative nMSE change of the first method against the second (\%, negative means the first method is better) and, in parentheses, the number of the 10 splits on which the first method wins. $^{\dagger}$: two-sided $p < 0.05$ under the corrected resampled $t$-test of \citet{nadeau2003inference}, which is conservative when the test set is much larger than the training set (small $\rho$). Indep.: independent ridge; Pooled: pooled ridge; Trace: trace-norm MTL; MR: mean-regularized MTL; MLR: mean + low-rank MTL.}
\label{tab:paired}
\footnotesize
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
& \multicolumn{7}{c}{Training fraction per school $\rho$} \\
\cmidrule(l){2-8}
Comparison & 5\% & 10\% & 15\% & 20\% & 30\% & 60\% & 75\% \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(ARGS.out, "tab_paired.tex"), "w").write(t3)

# ---------- Table 4 (appendix): metric definitions ----------
lines = [f"{TEX[mth]} & {pm(S['0.75'][mth]['ev_within'])} & {pm(S['0.75'][mth]['ev_pooled'])} & "
         f"{pm(S['0.75'][mth]['nmse'], 4)} \\\\" for mth in M]
t4 = r"""\begin{table}[h]
\centering
\caption{The same predictions ($\rho = 75\%$) scored under two definitions of explained variance and under nMSE. Pooled EV uses the variance of all test scores; within-school EV uses the average within-school variance, as in \citet{argyriou2008convex}.}
\label{tab:metrics}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
Method & EV within schools (\%) & EV pooled (\%) & nMSE \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(ARGS.out, "tab_metrics.tex"), "w").write(t4)

# ---------- Table 5 (appendix): hyperparameters ----------
def fmt(v):
    if v >= 100:
        return f"{v:.0f}"
    if v >= 0.01:
        return f"{v:.2g}"
    e = int(np.floor(np.log10(v))); return f"${v / 10 ** e:.1f}\\cdot 10^{{{e}}}$"

def hyp(mth):
    return " & ".join(fmt(S[r][mth]['hyper_median']) for r in RATIOS)
grids = {"pooled ridge": r"$\mu$: $10^{-6}$ to $10^{2}$ (17)",
         "independent ridge": r"$\mu$: $10^{-6}$ to $10^{2}$ (17)",
         "l2,1 MTL": r"$\lambda$: $10^{-1.5}$ to $10^{2.5}$ (12)",
         "trace-norm MTL": r"$\lambda$: $10^{-1.5}$ to $10^{2.5}$ (12)",
         "mean-reg MTL": r"$\lambda$: $10^{-4}$ to $10^{3}$ (15)",
         "mean+low-rank MTL": r"$\lambda$: $10^{-2.5}$ to $10^{3.5}$ (15)"}
lines = [f"{TEX[mth]} & {grids[mth]} & {hyp(mth)} \\\\" for mth in M]
edges = sum(S[r][mth]["edges"] for r in RATIOS for mth in M)
t5 = r"""\begin{table}[h]
\centering
\caption{Hyperparameter grids (log-spaced; number of values in parentheses) and the median value selected by 5-fold cross-validation at each training fraction. """ + f"Across all {7 * 10 * 6} selections, {edges} fell on the boundary of a grid." + r"""}
\label{tab:hyper}
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}llccccccc@{}}
\toprule
Method & Grid (values) & 5\% & 10\% & 15\% & 20\% & 30\% & 60\% & 75\% \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(ARGS.out, "tab_hyper.tex"), "w").write(t5)
print("tables written; grid-edge selections:", edges)

