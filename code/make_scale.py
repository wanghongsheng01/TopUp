"""Scale-up table from results/scale_g/*_k{1,2,4,8}_eps05.json (run_scale_g.sh): every tuple replicated k times.
B5-shared is compared at matched validity, as in Table I: the smallest sample inflation gamma in {2.5, 3} whose share of
certified targets reaches TopUp's and whose realised errors are within eps for >= 95% of the queries (else gamma = 3).
Writes paper/tables/scale.tex and paper/numbers_scale.tex."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "scale_g"); PAPER = os.path.join(HERE, "..", "paper")
m = lambda xs: float(np.mean(xs))
HUMAN, EPS = 0.05, 0.05

L = ["\\begin{table}[t]\\centering\\caption{Scale-up ($\\epsilon=0.05$, human-oracle price, mean of %d paired runs): every tuple replicated $k$ times with fresh random numbers. \\sys's cost relative to matched B5-shared (Table~\\ref{tab:cost}), cheaper in every paired run, and to O-full; share of targets certified (\\sys{} / B5-shared).}\\label{tab:scale}",
     "\\setlength{\\tabcolsep}{3pt}\\small\\begin{tabular}{llrrrr}\\toprule Data & $k$ & tuples & vs.\\ B5-sh. & vs.\\ O-full & met\\\\ \\midrule"]
out = {}
for lab, key in (("RMA", "rma"), ("Amazon-IS", "amazon2")):
    for k in (1, 2, 4, 8):
        f = os.path.join(RES, f"{key}_k{k}_eps05.json")
        if not os.path.exists(f): continue
        r = json.load(open(f)); runs = r["runs"]; n = r["n_tuples"]
        sv = np.array([x["semview"]["cost"]["human_oracle"]["total"] for x in runs])
        met_s = m([x["semview"]["quality"]["target_met"] for x in runs])
        best = None
        for g in ("2.5", "3.0"):
            q = [x[f"b5shared_g{g}_human_oracle"]["quality"] for x in runs]
            met_b = m([z["target_met"] for z in q]); acc_b = m([np.mean(np.abs(z["qerr"]) <= EPS + 1e-12) for z in q])
            best = g
            if met_b >= met_s - 1e-9 and acc_b >= 0.95: break
        b = np.array([x[f"b5shared_g{best}_human_oracle"]["total"] for x in runs])
        met_b = m([x[f"b5shared_g{best}_human_oracle"]["quality"]["target_met"] for x in runs])
        nor = m([x["semview"]["counts"]["oracle_main"] + x["semview"]["counts"]["pilot"] for x in runs])
        rec = dict(n=n, gamma=best, vs_b=m(sv / b), cheaper=float((sv < b).mean()), vs_of=m(sv) / (n * HUMAN), phi=nor / n,
                   met_s=met_s, met_b=met_b, cov=m([x["semview"]["quality"]["coverage_cons"] for x in runs]))
        out[f"{lab}|{k}"] = rec
        L.append(f"{lab if k == 1 else ''} & {k} & {n:,} & {rec['vs_b']:.2f} & {rec['vs_of']:.2f} & {met_s:.2f} / {met_b:.2f}\\\\".replace(",", "{,}"))
L[0] = L[0] % len(runs)
assert all(v["cheaper"] == 1.0 for v in out.values())
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(os.path.join(PAPER, "tables", "scale.tex"), "w").write("\n".join(L) + "\n")
json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
o = {lab: [out[f"{lab}|{k}"] for k in (1, 2, 4, 8)] for lab in ("RMA", "Amazon-IS")}
with open(os.path.join(PAPER, "numbers_scale.tex"), "w") as fh:
    fh.write(f"\\newcommand{{\\NumScaleOfullOne}}{{{min(v[0]['vs_of'] for v in o.values()):.2f}--{max(v[0]['vs_of'] for v in o.values()):.2f}$\\times$}}\n")
    fh.write(f"\\newcommand{{\\NumScaleOfullEight}}{{{min(v[-1]['vs_of'] for v in o.values()):.2f}$\\times$}}\n" if round(o['RMA'][-1]['vs_of'], 2) == round(o['Amazon-IS'][-1]['vs_of'], 2)
             else f"\\newcommand{{\\NumScaleOfullEight}}{{{min(v[-1]['vs_of'] for v in o.values()):.2f}--{max(v[-1]['vs_of'] for v in o.values()):.2f}$\\times$}}\n")
    fh.write(f"\\newcommand{{\\NumScaleBOne}}{{{min(v[0]['vs_b'] for v in o.values()):.2f}--{max(v[0]['vs_b'] for v in o.values()):.2f}$\\times$}}\n")
    fh.write(f"\\newcommand{{\\NumScaleBEight}}{{{min(v[-1]['vs_b'] for v in o.values()):.2f}--{max(v[-1]['vs_b'] for v in o.values()):.2f}$\\times$}}\n")
    fh.write(f"\\newcommand{{\\NumScaleMaxTuples}}{{{max(v[-1]['n'] for v in o.values()):,}}}\n".replace(",", "{,}"))
