"""Extra tables for the expanded experiments section (aggregates only).
Writes paper/tables/{templates,maintenance,contrast_all,sensitivity}.tex and appends macros to paper/numbers.tex.
Run after make_numbers.py.
"""
import json, os
import numpy as np
from scipy.stats import trim_mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, ".."); RES = os.path.join(ROOT, "results"); PAPER = os.path.join(ROOT, "paper")
DS = [("RMA", "rma"), ("Amazon-MI", "amazon"), ("Amazon-IS", "amazon2")]
R = {lab: {e: json.load(open(os.path.join(RES, f"final_{k}_eps{e}.json"))) for e in ("05", "10")} for lab, k in DS}
MS = {lab: json.load(open(os.path.join(RES, f"final_{k}_minsup200_eps05.json"))) for lab, k in DS}
mean = lambda xs: float(np.mean(xs))


def pooled_t8(runs):
    agg = {}
    for x in runs:
        for var, d in x["semview"]["quality"].get("T8", {}).items():
            a = agg.setdefault(var, {"n": 0, "c": [0, 0], "c2": [0, 0], "rc": [0, 0], "rc2": [0, 0], "vr": []})
            a["n"] += d["n"]
            for k, kk in (("c", "cov"), ("c2", "cov2"), ("rc", "rcov"), ("rc2", "rcov2")): a[k][0] += d[kk][0]; a[k][1] += d[kk][1]
            if d["vratio_mean"] is not None: a["vr"].append(d["vratio_mean"])
    # empirical Var(coordinated)/Var(independent): per query across PRN salts of its workload seed (measured, not Thm 2)
    emp = {}
    for ws in sorted({x["wseed"] for x in runs}):
        Rr = [x["semview"]["quality"]["t8err"] for x in runs if x["wseed"] == ws and "t8err" in x["semview"]["quality"]]
        if not Rr: continue
        for i in range(len(Rr[0])):
            a1 = np.array([r[i][1] for r in Rr]); a2 = np.array([r[i][2] for r in Rr])
            if a2.var() > 1e-12: emp.setdefault(Rr[0][i][0], []).append(a1.var() / a2.var())
    for var, rr in emp.items():
        # 5%-trimmed mean over queries: same aggregate as the prediction (mean over queries), robust to the noise of 40-salt variances
        if var in agg: agg[var]["emp"] = float(trim_mean(rr, 0.05))
    # pooled form (v7 records carry each query's design variances): predicted sum Var_coord / sum Var_indep over all queries,
    # measured sum of across-salt variances of the coordinated and independent estimates; a ratio of sums has no F-ratio bias
    if runs and "t8err" in runs[0]["semview"]["quality"] and len(runs[0]["semview"]["quality"]["t8err"][0]) == 5:
        pn, pd_, mn, md = {}, {}, {}, {}
        for x in runs:
            for r in x["semview"]["quality"]["t8err"]:
                pn[r[0]] = pn.get(r[0], 0) + r[3]; pd_[r[0]] = pd_.get(r[0], 0) + r[4]
        for ws in sorted({x["wseed"] for x in runs}):
            Rr = [x["semview"]["quality"]["t8err"] for x in runs if x["wseed"] == ws]
            for i in range(len(Rr[0])):
                v = Rr[0][i][0]
                mn[v] = mn.get(v, 0) + np.var([r[i][1] for r in Rr]); md[v] = md.get(v, 0) + np.var([r[i][2] for r in Rr])
        for v in agg:
            if v in pn and pd_[v] > 0 and md.get(v, 0) > 0:
                agg[v]["pred_pooled"] = pn[v] / pd_[v]; agg[v]["meas_pooled"] = mn[v] / md[v]
    return agg


# per-template quality (floored coverage needs per-template data: use HT per template + T3 separation)
L = ["\\begin{table}[t]\\centering\\caption{Per-template quality of \\sys{} at $\\epsilon=0.05$: HT coverage of T1--T3 targets, and for top-reason queries (T3) the share that separate the top two values and the share of separated answers that are correct.}\\label{tab:templates}",
     "\\setlength{\\tabcolsep}{3pt}\\small\\begin{tabular}{lrrrrr}\\toprule Data & T1 & T2 & T3 & T3 sep. & T3 correct\\\\ \\midrule"]
for lab, d in R.items():
    q = [x["semview"]["quality"] for x in d["05"]["runs"]]
    L.append(f"{lab} & {mean([z['coverage_T1'] for z in q]):.3f} & {mean([z['coverage_T2'] for z in q]):.3f} & {mean([z['coverage_T3'] for z in q]):.3f} & "
             f"{mean([z['T3_separated_share'] for z in q]):.3f} & {mean([z['T3_correct_when_separated'] for z in q]):.3f}\\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(os.path.join(PAPER, "tables", "templates.tex"), "w").write("\n".join(L) + "\n")

# maintenance and cost decomposition (oracle-label counts, price-independent)
L = ["\\begin{table}[t]\\centering\\caption{Where \\sys{}'s oracle labels go ($\\epsilon=0.05$, mean per run): bought by the plan, by lazy top-ups and by the pilot, and the number of lazy raises.}\\label{tab:maint}",
     "\\setlength{\\tabcolsep}{2.5pt}\\small\\begin{tabular}{lrrrrr}\\toprule Data & tuples & plan & lazy & pilot & raises\\\\ \\midrule"]
for lab, d in R.items():
    runs = d["05"]["runs"]; c = lambda k: mean([x["semview"]["counts"][k] for x in runs])
    edit = mean([x["semview"]["cost"]["human_oracle"]["total_with_edits"] / x["semview"]["cost"]["human_oracle"]["total"] - 1 for x in runs])
    L.append(f"{lab} & {d['05']['n_tuples']:,} & {c('plan_only'):,.0f} & {c('oracle_main') - c('plan_only'):,.0f} & {c('pilot'):,.0f} & "
             f"{mean([x['semview']['lazy'] for x in runs]):.0f}\\\\".replace(",", "{,}"))
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(os.path.join(PAPER, "tables", "maintenance.tex"), "w").write("\n".join(L) + "\n")

# contrasts on all datasets
names = {"mom": "month", "yoy": "year", "pp30": "30d/30d", "roll7": "rolling"}
short = {"RMA": "RMA", "Amazon-MI": "MI", "Amazon-IS": "IS"}
L = ["\\begin{table}[t]\\centering\\caption{Comparison queries (T4), $\\epsilon=0.05$: coverage of floored difference and ratio intervals; variance of the coordinated synopsis relative to independent per-window samples, predicted by Theorem~\\ref{thm:contrast}(b) and measured over 40 PRN salts, pooled over queries. Variants are defined in Section~\\ref{sec:exp}.}\\label{tab:contrast_all}",
     "\\setlength{\\tabcolsep}{3pt}\\small\\begin{tabular}{llrrrr}\\toprule Data & variant & queries & diff. & ratio & Var ratio (pred./meas.)\\\\ \\midrule"]
for lab, d in R.items():
    f7 = os.path.join(RES, "v7", f"matched_{dict(DS)[lab]}_eps05.json")
    T = pooled_t8(json.load(open(f7))["runs"] if os.path.exists(f7) else d["05"]["runs"])
    for var in ("mom", "yoy", "pp30", "roll7"):
        if var not in T: continue
        a = T[var]
        L.append(f"{short[lab]} & {names[var]} & {a['n']:,} & {a['c2'][1] / a['c2'][0]:.3f} & {a['rc2'][1] / max(a['rc2'][0], 1):.3f} & {a.get('pred_pooled', mean(a['vr'])):.3f} / {a.get('meas_pooled', a.get('emp', float('nan'))):.3f}\\\\".replace(",", "{,}"))
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(os.path.join(PAPER, "tables", "contrast_all.tex"), "w").write("\n".join(L) + "\n")


def b5ratio(runs, price="human_oracle"):
    return mean([x["semview"]["cost"][price]["total"] / x[f"b5_{price}"]["total"] for x in runs])


# sensitivity: minimum slice support 50 vs 200
L = ["\\begin{table}[t]\\centering\\caption{Effect of the minimum slice support ($\\epsilon=0.05$): \\sys{} cost relative to B5, HT coverage over all targets / worst group.}\\label{tab:sens}",
     "\\setlength{\\tabcolsep}{3pt}\\small\\begin{tabular}{lrrrr}\\toprule & \\multicolumn{2}{c}{min.\\ support 50} & \\multicolumn{2}{c}{min.\\ support 200}\\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5} Data & vs.\\ B5 & HT cov. & vs.\\ B5 & HT cov.\\\\ \\midrule"]


def worst(runs, key="groups"):
    g = {}
    for x in runs:
        for k, (n, c) in x["semview"]["quality"][key].items():
            a = g.setdefault(k, [0, 0]); a[0] += n; a[1] += c
    big = [c / n for k, (n, c) in g.items() if n >= 1000] or [c / n for k, (n, c) in g.items()]
    return min(big)


for lab in R:
    a, b = R[lab]["05"]["runs"], MS[lab]["runs"]
    L.append(f"{lab} & {b5ratio(a):.2f} & {mean([x['semview']['quality']['coverage'] for x in a]):.3f} / {worst(a):.3f} & "
             f"{b5ratio(b):.2f} & {mean([x['semview']['quality']['coverage'] for x in b]):.3f} / {worst(b):.3f}\\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(os.path.join(PAPER, "tables", "sensitivity.tex"), "w").write("\n".join(L) + "\n")

# macros
vr = [mean(pooled_t8(R[lab]["05"]["runs"])["roll7"]["vr"]) for lab in R]
plan_share = [mean([x["semview"]["counts"]["plan_only"] / x["semview"]["counts"]["oracle_main"] for x in R[lab]["05"]["runs"]]) for lab in R]
t3sep = [mean([x["semview"]["quality"]["T3_separated_share"] for x in R[lab]["05"]["runs"]]) for lab in R]
with open(os.path.join(PAPER, "numbers_extra.tex"), "w") as fh:  # own file, overwritten: no duplicate macros
    fh.write(f"\\newcommand{{\\NumRollRange}}{{{min(vr):.2f}--{max(vr):.2f}}}\n")
    fh.write(f"\\newcommand{{\\NumPlanShareRange}}{{{min(plan_share):.0%}--{max(plan_share):.0%}}}\n".replace("%", "\\%"))
    fh.write(f"\\newcommand{{\\NumTsepRange}}{{{min(t3sep):.2f}--{max(t3sep):.2f}}}\n")
print("roll7 vr", [round(v, 3) for v in vr], "plan share", [round(v, 3) for v in plan_share], "T3 sep", [round(v, 3) for v in t3sep])
