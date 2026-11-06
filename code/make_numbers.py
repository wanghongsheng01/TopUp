"""Generate paper/numbers.tex and paper/tables/*.tex from the final replay JSON files (aggregates only).

Usage: python make_numbers.py results/final_rma_eps05.json results/final_rma_eps10.json results/final_rma_minsup200_eps05.json
Public datasets are picked up from results/final_{amazon,amazon2}_eps{05,10}.json when present.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PAPER = os.path.join(ROOT, "paper")
os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
f05, f10, fms = sys.argv[1:4]
R05, R10, RMS = (json.load(open(f)) for f in (f05, f10, fms))


def tot(runs, m, price="llm_oracle"):
    if m in ("semview", "b7", "a_lazy", "p_only"): return np.array([x[m]["cost"][price]["total"] for x in runs])
    return np.array([x[f"{m}_{price}"]["total"] for x in runs])


def ratio(runs, m, price="llm_oracle"): return float(np.mean(tot(runs, "semview", price) / tot(runs, m, price)))


def pooled(runs, m, key):
    g = {}
    for x in runs:
        for k, (n, c) in x[m]["quality"][key].items():
            a = g.setdefault(k, [0, 0]); a[0] += n; a[1] += c
    big = [(c / n, k, n) for k, (n, c) in g.items() if n >= 1000] or [(c / n, k, n) for k, (n, c) in g.items()]
    return min(big)


def t8(runs, m="semview"):
    agg = {}
    for x in runs:
        for var, d in x[m]["quality"].get("T8", {}).items():
            a = agg.setdefault(var, {"n": 0, "c": [0, 0], "c2": [0, 0], "rc": [0, 0], "vr": [], "hw": []})
            a["n"] += d["n"]; a["hw"].append(d["hw_mean"])
            for k, kk in (("c", "cov"), ("c2", "cov2"), ("rc", "rcov")):
                a[k][0] += d[kk][0]; a[k][1] += d[kk][1]
            if d["vratio_mean"] is not None: a["vr"].append(d["vratio_mean"])
    return agg


r05, r10, rms = R05["runs"], R10["runs"], RMS["runs"]
q05 = [x["semview"]["quality"] for x in r05]
n_tuples = R05["n_tuples"]; n_hist = r05[0]["semview"]["counts"].get("n_hist", 0); n_cur = n_tuples - n_hist
spend = sum(json.loads(l)["usd"] for l in open(os.path.join(ROOT, "cache", "usage.log")))
# the first RMA pilot (2026-11-03 morning, 9,936 notes) predates the usage ledger; its spend ($9.42) is from 08_试点结果.md
spend += 9.42
ofull = lambda R, price: R["n_tuples"] * (R["prices"]["llm_oracle"] if price == "llm_oracle" else 0.05)  # every tuple any query reads, history included
sv05 = tot(r05, "semview"); sv05h = tot(r05, "semview", "human_oracle")
cnt = lambda runs, k: np.mean([x["semview"]["counts"][k] for x in runs])
sens = sorted(glob.glob(os.path.join(ROOT, "results", "t8sens", "eps0.05_w*.json")))
sens_r = [float(np.mean(tot(json.load(open(f))["runs"], "semview") / tot(json.load(open(f))["runs"], "b5"))) for f in sens]
T = t8(r05)
roll = T.get("roll7", {"vr": [np.nan]})
N = {
    "NumPilotTuples": f"{n_cur:,}".replace(",", "{,}"), "NumHistTuples": f"{n_hist:,}".replace(",", "{,}"),
    "NumQueries": f"{r05[0]['n_queries']:,}".replace(",", "{,}"), "NumRuns": str(len(r05)),
    "NumCostVsLadder": f"{ratio(r05, 'b4'):.3f}$\\times$", "NumCostVsLadderX": f"{1 / ratio(r05, 'b4'):.0f}$\\times$",
    "NumCostVsCache": f"{ratio(r05, 'b5'):.2f}$\\times$", "NumCostVsCacheTen": f"{ratio(r10, 'b5'):.2f}$\\times$",
    "NumCostVsCacheX": f"{1 / ratio(r05, 'b5'):.1f}--{1 / ratio(r10, 'b5'):.1f}$\\times$",
    "NumCostVsIdealX": f"{1 / ratio(r05, 'b4ideal'):.0f}$\\times$",
    "NumCovHT": f"{np.mean([z['coverage'] for z in q05]):.3f}", "NumCovCons": f"{np.mean([z['coverage_cons'] for z in q05]):.3f}",
    "NumWorstHT": f"{pooled(r05, 'semview', 'groups')[0]:.3f}", "NumWorstCons": f"{pooled(r05, 'semview', 'groups2')[0]:.3f}",
    "NumMinsupCov": f"{np.mean([x['semview']['quality']['coverage'] for x in rms]):.3f}",
    "NumMinsupWorst": f"{pooled(rms, 'semview', 'groups')[0]:.3f}",
    "NumBias": f"{np.mean([abs(np.mean([x['semview']['quality']['qerr'][i] for x in r05 if x['wseed'] == ws])) for ws in sorted({x['wseed'] for x in r05}) for i in range(len(r05[0]['semview']['quality']['qerr']))]):.4f}",
    "NumContrastRatio": f"{np.mean(roll['vr']):.2f}", "NumContrastGainX": f"{1 / np.mean(roll['vr']):.1f}$\\times$",
    "NumOracleShare": f"{(cnt(r05, 'oracle_main') - cnt(r05, 'oracle_hist') + cnt(r05, 'pilot')) / n_cur:.0%}".replace("%", "\\%"),
    "NumVsOfull": f"{np.mean(sv05) / ofull(R05, 'llm_oracle'):.2f}$\\times$",
    "NumVsOfullHuman": f"{np.mean(sv05h) / ofull(R05, 'human'):.2f}$\\times$",
    "NumVsOfullMinsup": f"{np.mean(tot(rms, 'semview', 'human_oracle')) / ofull(RMS, 'human'):.2f}$\\times$",  # same (human) price as NumVsOfullHuman
    "NumOfullGap": f"{ofull(R05, 'llm_oracle') / np.mean(sv05):.1f}$\\times$",
    "NumSensRange": f"{min(sens_r):.3f}--{max(sens_r):.3f}" if sens_r else "--",
    "NumBSevenX": f"{1 / ratio(r05, 'b7'):.2f}$\\times$", "NumALazyX": f"{1 / ratio(r05, 'a_lazy'):.2f}$\\times$",
    "NumALazyCov": f"{np.mean([x['a_lazy']['quality']['coverage'] for x in r05]):.3f}",
    "NumPlanShare": f"{cnt(r05, 'plan_only') / cnt(r05, 'oracle_main'):.0%}".replace("%", "\\%"),
    "NumEditCost": f"+{np.mean([x['semview']['cost']['human_oracle']['total_with_edits'] / x['semview']['cost']['human_oracle']['total'] - 1 for x in r05]):.0%}".replace("%", "\\%"),
    "NumCzero": f"\\${R05['prices']['llm_oracle'] * 1000:.2f}", "NumCone": f"\\${R05['c1_bundle'] * 1000:.3f}",
    "NumApiSpend": f"\\${spend:.2f}",
    "NumWTone": "", "NumWTtwo": "", "NumWTthree": "", "NumWTeight": "",
    "NumReviewsProcessed": "23{,}805", "NumReviewRedundancy": "2.9", "NumChatSemShare": "35\\%", "NumEditShare": "41\\%",
}
w8 = 0.15; N.update(NumWTone=f"{0.35 / 0.85 * (1 - w8):.0%}", NumWTtwo=f"{0.25 / 0.85 * (1 - w8):.0%}",
                    NumWTthree=f"{0.25 / 0.85 * (1 - w8):.0%}", NumWTeight=f"{w8:.0%}")
for k in ("NumWTone", "NumWTtwo", "NumWTthree", "NumWTeight"): N[k] = N[k].replace("%", "\\%")
with open(os.path.join(PAPER, "numbers.tex"), "w") as fh:
    fh.write("% generated by code/make_numbers.py from " + ", ".join(os.path.basename(f) for f in (f05, f10, fms)) + "\n")
    for k, v in N.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# Table: cost
rows = [("semview", "\\sys"), ("b7", "B7 uniform"), ("a_lazy", "A-lazy"), ("b6", "B6 SemViews-adapted"), ("b5", "B5 Ladder+cache"), ("b5ideal", "B5-ideal"),
        ("b4", "B4 Ladder per query"), ("b4ideal", "B4-ideal")]
L = ["\\begin{table}[t]\\centering\\caption{Total labelling cost over the 62-day horizon (mean of %d runs) and \\sys's cost relative to each method.}\\label{tab:cost}" % len(r05),
     "\\setlength{\\tabcolsep}{3pt}\\small\\begin{tabular}{lrrrr}\\toprule",
     " & \\multicolumn{2}{c}{$\\epsilon=0.05$} & \\multicolumn{2}{c}{$\\epsilon=0.10$}\\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
     "Method & LLM oracle & human oracle & LLM oracle & human oracle\\\\ \\midrule"]
for m, name in rows:
    cells = []
    for R in (r05, r10):
        for price in ("llm_oracle", "human_oracle"):
            c = np.mean(tot(R, m, price)); rr = ratio(R, m, price)
            cells.append(f"\\${c:,.2f}" + ("" if m == "semview" else f" ({rr:.2f})"))
    L.append(f"{name} & " + " & ".join(cells) + "\\\\")
L.append(f"O-full & \\${ofull(R05, 'llm_oracle'):.2f} & \\${ofull(R05, 'human'):,.0f} & \\${ofull(R10, 'llm_oracle'):.2f} & \\${ofull(R10, 'human'):,.0f}\\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(os.path.join(PAPER, "tables", "main_cost.tex"), "w").write("\n".join(L).replace(",", "{,}") + "\n")

# Table: validity
L = ["\\begin{table}[t]\\centering\\caption{Validity of \\sys{} intervals (nominal 0.95). Worst group: slice type $\\times$ size bucket, pooled over runs.}\\label{tab:valid}",
     "\\small\\begin{tabular}{lrrrr}\\toprule & \\multicolumn{2}{c}{$\\epsilon=0.05$} & \\multicolumn{2}{c}{$\\epsilon=0.10$}\\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
     "Metric & HT & floored & HT & floored\\\\ \\midrule"]
for lab, kh, kc in (("Coverage, all targets", "coverage", "coverage_cons"),):
    v = [np.mean([x["semview"]["quality"][k] for x in R]) for R in (r05, r10) for k in (kh, kc)]
    L.append(f"{lab} & " + " & ".join(f"{z:.3f}" for z in v) + "\\\\")
v = [pooled(R, "semview", g)[0] for R in (r05, r10) for g in ("groups", "groups2")]
L.append("Worst group & " + " & ".join(f"{z:.3f}" for z in v) + "\\\\")
v = [np.mean([x["semview"]["quality"][k] for x in R]) for R in (r05, r10) for k in ("mean_halfwidth", "mean_halfwidth_cons")]
L.append("Mean half-width & " + " & ".join(f"{z:.4f}" for z in v) + "\\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(os.path.join(PAPER, "tables", "validity.tex"), "w").write("\n".join(L) + "\n")

# Table: contrasts
names = {"mom": "Prior month", "yoy": "Previous year", "pp30": "30d vs.\\ preceding 30d", "roll7": "Rolling 30d, 7d apart"}
L = ["\\begin{table}[t]\\centering\\caption{Comparison queries (T8) on RMA at $\\epsilon=0.05$: coverage of the difference (HT / floored) and of the ratio, and the variance of the coordinated synopsis relative to independent per-window samples.}\\label{tab:contrast}",
     "\\setlength{\\tabcolsep}{3pt}\\small\\begin{tabular}{lrrrrr}\\toprule Variant & overlap & queries & diff.\\ cov. & ratio cov. & Var ratio\\\\ \\midrule"]
for var in ("mom", "yoy", "pp30", "roll7"):
    if var not in T: continue
    a = T[var]; f = lambda z: z[1] / z[0] if z[0] else float("nan")
    L.append(f"{names[var]} & {'0.77' if var == 'roll7' else '0'} & {a['n']:,} & {f(a['c']):.3f} / {f(a['c2']):.3f} & {f(a['rc']):.3f} & {np.mean(a['vr']):.3f}\\\\".replace(",", "{,}"))
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(os.path.join(PAPER, "tables", "contrast.tex"), "w").write("\n".join(L) + "\n")
print(json.dumps(N, indent=1))


# ------------------------------------------------------------------ multi-dataset tables (RMA + public, eps = 0.05 and 0.10)
DS = [("RMA", "rma"), ("Amazon-MI", "amazon"), ("Amazon-IS", "amazon2")]
RES = os.path.join(ROOT, "results")
have = {lab: {e: json.load(open(os.path.join(RES, f"final_{k}_eps{e}.json")))["runs"]
              for e in ("05", "10") if os.path.exists(os.path.join(RES, f"final_{k}_eps{e}.json"))} for lab, k in DS}
have = {k: v for k, v in have.items() if len(v) == 2}
if len(have) > 1:
    rows = [("semview", "\\sys"), ("p_only", "\\sys{} plan only"), ("b7", "B7 uniform"), ("a_lazy", "A-lazy"),
            ("b6", "B6 SemViews-ad."), ("b5", "B5 Ladder+cache"), ("b5shared", "B5-shared"), ("b5ideal", "B5-ideal"),
            ("b4", "B4 Ladder"), ("b4ideal", "B4-ideal")]
    L = ["\\begin{table*}[t]\\centering\\caption{Total labelling cost over the 62-day horizon (human-oracle price, \\$0.05 per label; mean of %d runs) and, in parentheses, \\sys's cost relative to each method.}\\label{tab:cost}" % len(r05),
         "\\small\\begin{tabular}{l" + "rr" * len(have) + "}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for lab in have) + "\\\\",
         " & " + " & ".join("$\\epsilon=0.05$ & $\\epsilon=0.10$" for _ in have) + "\\\\ \\midrule"]
    for m, name in rows:
        cells = []
        for lab, d in have.items():
            for e in ("05", "10"):
                c = np.mean(tot(d[e], m, "human_oracle")); rr = ratio(d[e], m, "human_oracle")
                cells.append(f"{c:,.0f}" + ("" if m == "semview" else f" ({rr:.2f})"))
        L.append(f"{name} & " + " & ".join(cells) + "\\\\")
    cells = []
    for lab, d in have.items():
        n = d["05"][0]["semview"]["counts"]
        cells += [f"{n['cheap_scan'] * 0.05:,.0f}"] * 2
    L.append("O-full & " + " & ".join(cells) + "\\\\")
    cells = []
    for lab, d in have.items():
        f = os.path.join(RES, f"olazy_{dict(DS)[lab]}.json")
        if not os.path.exists(f): cells += ["--", "--"]; continue
        ol = json.load(open(f))["touched_mean"] * 0.05
        for e in ("05", "10"):
            cells.append(f"{ol:,.0f} ({np.mean(tot(d[e], 'semview', 'human_oracle')) / ol:.2f})")
    L.append("O-lazy & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table*}"]
    open(os.path.join(PAPER, "tables", "main_cost.tex"), "w").write("\n".join(L).replace(",", "{,}") + "\n")
    L = ["\\begin{table}[t]\\centering\\caption{Coverage of 95\\% intervals (all targets / worst slice-type $\\times$ size group), HT vs.\\ floored, and the share of targets whose realised error is within $\\epsilon$ (acc.).}\\label{tab:valid}",
         "\\setlength{\\tabcolsep}{3pt}\\small\\begin{tabular}{llrrr}\\toprule Data & $\\epsilon$ & HT & floored & acc.\\\\ \\midrule"]
    for lab, d in have.items():
        for e in ("05", "10"):
            q = [x["semview"]["quality"] for x in d[e]]
            L.append(f"{lab} & 0.{e} & {np.mean([z['coverage'] for z in q]):.3f} / {pooled(d[e], 'semview', 'groups')[0]:.3f} & "
                     f"{np.mean([z['coverage_cons'] for z in q]):.3f} / {pooled(d[e], 'semview', 'groups2')[0]:.3f} & "
                     f"{np.mean([np.mean(np.abs(z['qerr']) <= int(e) / 100) for z in q]):.3f}\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(PAPER, "tables", "validity.tex"), "w").write("\n".join(L) + "\n")
    with open(os.path.join(PAPER, "numbers.tex"), "a") as fh:
        pub = [k for k in have if k != "RMA"]
        rb5 = [ratio(have[k][e], "b5", "human_oracle") for k in have for e in ("05", "10")]
        rb4 = [ratio(have[k][e], "b4", "human_oracle") for k in have for e in ("05", "10")]
        cov = [np.mean([x["semview"]["quality"]["coverage_cons"] for x in have[k][e]]) for k in have for e in ("05", "10")]
        fh.write(f"\\newcommand{{\\NumBFiveRangeAll}}{{{min(rb5):.2f}--{max(rb5):.2f}$\\times$}}\n")
        fh.write(f"\\newcommand{{\\NumBFourRangeAll}}{{{min(rb4):.3f}--{max(rb4):.3f}$\\times$}}\n")
        fh.write(f"\\newcommand{{\\NumCovConsAll}}{{{min(cov):.3f}--{max(cov):.3f}}}\n")
        fh.write(f"\\newcommand{{\\NumDatasets}}{{{len(have)}}}\n")
        rol = []
        for lab, d in have.items():
            f = os.path.join(RES, f"olazy_{dict(DS)[lab]}.json")
            if os.path.exists(f):
                ol = json.load(open(f))["touched_mean"] * 0.05
                rol += [np.mean(tot(d[e], "semview", "human_oracle")) / ol for e in ("05", "10")]
        rb7 = [ratio(have[k][e], "b7", "human_oracle") for k in have for e in ("05", "10")]
        acc = [np.mean([np.mean(np.abs(x["semview"]["quality"]["qerr"]) <= int(e) / 100) for x in have[k][e]]) for k in have for e in ("05", "10")]
        if rol: fh.write(f"\\newcommand{{\\NumOlazyRange}}{{{min(rol):.2f}--{max(rol):.2f}$\\times$}}\n")
        rbs = [ratio(have[k][e], "b5shared", "human_oracle") for k in have for e in ("05", "10")]
        rpo = [ratio(have[k][e], "p_only", "human_oracle") for k in have for e in ("05", "10")]
        tm = [np.mean([x["semview"]["quality"]["target_met"] for x in have[k][e]]) for k in have for e in ("05", "10")]
        tmp = [np.mean([x["p_only"]["quality"]["target_met"] for x in have[k][e]]) for k in have for e in ("05", "10")]
        fh.write(f"\\newcommand{{\\NumBFiveSharedRange}}{{{min(rbs):.2f}--{max(rbs):.2f}$\\times$}}\n")
        fh.write(f"\\newcommand{{\\NumPonlyRange}}{{{min(rpo):.2f}--{max(rpo):.2f}$\\times$}}\n")
        fh.write(f"\\newcommand{{\\NumTargetMet}}{{{min(tm):.3f}--{max(tm):.3f}}}\n")
        fh.write(f"\\newcommand{{\\NumTargetMetPonly}}{{{min(tmp):.3f}--{max(tmp):.3f}}}\n")
        fh.write(f"\\newcommand{{\\NumBSevenRange}}{{{min(rb7):.2f}--{max(rb7):.2f}$\\times$}}\n")
        fh.write(f"\\newcommand{{\\NumAccRange}}{{{min(acc):.3f}--{max(acc):.3f}}}\n")
    # Table: validity of every method at equal targets (v5 replays record intervals for the per-query baselines too)
    qual = lambda runs, m: [x[m]["quality"] if m in ("semview", "b7", "a_lazy", "p_only") else x[f"{m}_human_oracle"]["quality"] for x in runs]
    acc_of = lambda q, e: np.mean(np.abs(q["qerr"]) <= int(e) / 100 + 1e-12)
    vrows = [("semview", "\\sys"), ("p_only", "\\sys{} plan only"), ("b7", "B7 uniform"), ("a_lazy", "A-lazy"),
             ("b5shared", "B5-shared"), ("b5", "B5 Ladder+cache")]
    if all("quality" in have[k]["05"][0]["b5shared_human_oracle"] for k in have):
        V = {}
        for k in have:
            for e in ("05", "10"):
                for m, _ in vrows:
                    q = qual(have[k][e], m)
                    V[(k, e, m)] = (np.mean([z["coverage_cons"] for z in q]), np.mean([acc_of(z, e) for z in q]),
                                    np.mean([z["target_met"] for z in q]), np.mean([z["hw_ratio_median"] for z in q]),
                                    np.mean([z["target_met_110"] for z in q]))
        json.dump({"|".join(k): v for k, v in V.items()}, open(os.path.join(RES, "validity_all_methods.json"), "w"), indent=0)
        L = ["\\begin{table*}[t]\\centering\\caption{Validity at equal targets ($\\epsilon=0.05$, mean of %d runs): coverage of the floored 95\\%% interval (cov.), share of queries whose realised error is within $\\epsilon$ (acc.), and share of T1/T2/T8 targets whose certified half-width is within $\\epsilon$ (met). Per-query planners use the sample-variance interval of~\\cite{ladder}, floored by the pilot moments.}\\label{tab:basevalid}" % len(r05),
             "\\setlength{\\tabcolsep}{4pt}\\small\\begin{tabular}{l" + "rrr" * len(have) + "}\\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for lab in have) + "\\\\",
             " " + "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(have))),
             "Method & " + " & ".join("cov. & acc. & met" for _ in have) + "\\\\ \\midrule"]
        for m, name in vrows:
            L.append(f"{name} & " + " & ".join(f"{V[(k, '05', m)][0]:.3f} & {V[(k, '05', m)][1]:.3f} & {V[(k, '05', m)][2]:.2f}" for k in have) + "\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{table*}"]
        open(os.path.join(PAPER, "tables", "basevalid.tex"), "w").write("\n".join(L) + "\n")
        with open(os.path.join(PAPER, "numbers.tex"), "a") as fh:
            rng_ = lambda m, j, d=3: f"{min(V[(k, e, m)][j] for k in have for e in ('05', '10')):.{d}f}--{max(V[(k, e, m)][j] for k in have for e in ('05', '10')):.{d}f}"
            for m, tag in (("b5shared", "BFiveShared"), ("b5", "BFive"), ("b7", "BSeven"), ("a_lazy", "ALazy"), ("semview", "Sys")):
                fh.write(f"\\newcommand{{\\NumCov{tag}}}{{{rng_(m, 0)}}}\n\\newcommand{{\\NumAcc{tag}}}{{{rng_(m, 1)}}}\n"
                         f"\\newcommand{{\\NumMet{tag}}}{{{rng_(m, 2, 2)}}}\n\\newcommand{{\\NumHwr{tag}}}{{{rng_(m, 3, 2)}}}\n")
            fh.write(f"\\newcommand{{\\NumMetTolBFiveShared}}{{{rng_('b5shared', 4, 2)}}}\n")
            cv = [v[0] for v in V.values()]
            fh.write(f"\\newcommand{{\\NumCovAllMethods}}{{{min(cv):.3f}--{max(cv):.3f}}}\n")
    print("multi-dataset tables for", list(have))
