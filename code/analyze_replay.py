"""Summarise replay JSON files into the tables of 08_试点结果.md (prints markdown; aggregates only)."""
import json, sys
import numpy as np
from scipy import stats

SHARED = ["semview", "b7", "a_lazy", "p_only"]


def ci(x):
    x = np.asarray(x, float)
    return f"{x.mean():.4g} [{np.percentile(x, 2.5):.4g}, {np.percentile(x, 97.5):.4g}]"


def main(path):
    r = json.load(open(path)); runs = r["runs"]
    print(f"## {path}\n\neps={r['eps']}  tuples={r['n_tuples']}  runs={len(runs)}  queries/run={runs[0]['n_queries']}  "
          f"c1(bundle)=${r['c1_bundle'] * 1000:.4f}/1k  c0(LLM oracle)=${r['prices']['llm_oracle'] * 1000:.4f}/1k  c0(human)=$0.05/label\n")
    for price in r["prices"]:
        print(f"### cost, oracle price = {price}\n\n| method | total $ (mean [2.5%, 97.5%]) | oracle labels | ratio SemView/method |\n|---|---|---|---|")
        sv = np.array([x["semview"]["cost"][price]["total"] for x in runs])
        rows = {m: (np.array([x[m]["cost"][price]["total"] for x in runs]),
                    np.array([x[m]["counts"]["oracle_main"] + x[m]["counts"]["pilot"] for x in runs])) for m in SHARED}
        for m in [k for k in ["b4", "b5", "b4ideal", "b5ideal", "b6", "b5shared"] if f"{k}_{price}" in runs[0]]:
            rows[m] = (np.array([x[f"{m}_{price}"]["total"] for x in runs]), np.array([x[f"{m}_{price}"]["n_oracle"] for x in runs]))
        for m, (c, n) in rows.items():
            print(f"| {m} | {ci(c)} | {n.mean():.0f} | {ci(sv / c)} |")
        e = np.array([x["semview"]["cost"][price]["total_with_edits"] for x in runs])
        pil = np.array([x["semview"]["cost"][price]["pilot"] for x in runs]); sc = np.array([x["semview"]["cost"][price]["cheap_scan"] for x in runs])
        print(f"\nSemView breakdown: cheap scan {sc.mean():.4g}, pilot {pil.mean():.4g}, oracle main {(sv - pil - sc).mean():.4g}; "
              f"with edit re-labels {e.mean():.4g} (+{(e / sv - 1).mean() * 100:.1f}%)\n")
    print("### quality (shared-synopsis methods)\n\n| method | coverage | T1 | T2 | T3 | worst pooled group (n>=1000) | share of queries w/ instance coverage<0.85 | T3 separated | T3 correct when separated | mean half-width | oracle labels/query in window | lazy raises/run |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in SHARED:
        q = [x[m]["quality"] for x in runs]
        g = {}
        for qq in q:
            for k, (n, c) in qq["groups"].items():
                a = g.setdefault(k, [0, 0]); a[0] += n; a[1] += c
        big = [(k, c / n, n) for k, (n, c) in g.items() if n >= 1000] or [(k, c / n, n) for k, (n, c) in g.items()]
        worst = min(big, key=lambda z: z[1])
        inst = []
        for ws in sorted({x["wseed"] for x in runs}):
            Q = np.array([x[m]["quality"]["qcov"] for x in runs if x["wseed"] == ws])
            inst.append((Q.mean(0) < 0.85).mean())
        print(f"| {m} | {np.mean([z['coverage'] for z in q]):.3f} | {np.mean([z['coverage_T1'] for z in q]):.3f} | "
              f"{np.mean([z['coverage_T2'] for z in q]):.3f} | {np.mean([z['coverage_T3'] for z in q]):.3f} | "
              f"{worst[0]} {worst[1]:.3f} (n={worst[2]}) | {np.mean(inst):.3f} | {np.mean([z['T3_separated_share'] for z in q]):.3f} | "
              f"{np.mean([z['T3_correct_when_separated'] for z in q]):.3f} | {np.mean([z['mean_halfwidth'] for z in q]):.4f} | "
              f"{np.mean([z['mean_n0_per_query'] for z in q]):.1f} | {np.mean([x[m]['lazy'] for x in runs]):.0f} |")
    print("\n### conservative CI (max(HT, pilot design variance))\n\n| method | coverage | worst pooled group | share of queries w/ instance coverage<0.85 | mean half-width |\n|---|---|---|---|---|")
    for m in SHARED:
        q = [x[m]["quality"] for x in runs]; g = {}
        for qq in q:
            for k, (n, c) in qq["groups2"].items():
                a_ = g.setdefault(k, [0, 0]); a_[0] += n; a_[1] += c
        big = [(k, c / n, n) for k, (n, c) in g.items() if n >= 1000] or [(k, c / n, n) for k, (n, c) in g.items()]
        w = min(big, key=lambda z: z[1]); inst = []
        for ws in sorted({x["wseed"] for x in runs}):
            Q = np.array([x[m]["quality"]["qcov2"] for x in runs if x["wseed"] == ws]); inst.append((Q.mean(0) < 0.85).mean())
        print(f"| {m} | {np.mean([z['coverage_cons'] for z in q]):.3f} | {w[0]} {w[1]:.3f} (n={w[2]}) | {np.mean(inst):.3f} | {np.mean([z['mean_halfwidth_cons'] for z in q]):.4f} |")
    print("\n### unbiasedness check (per query, signed error averaged over PRN salts)\n")
    for m in SHARED:
        fr, mb = [], []
        for ws in sorted({x["wseed"] for x in runs}):
            E = np.array([x[m]["quality"]["qerr"] for x in runs if x["wseed"] == ws])
            if len(E) < 3: continue
            t = E.mean(0) / (E.std(0, ddof=1) / np.sqrt(len(E)) + 1e-12)
            fr.append((np.abs(t) > 3).mean()); mb.append(np.abs(E.mean(0)).mean()); dfree = len(E) - 1
        if fr: print(f"- {m}: share of queries with |t|>3 = {np.mean(fr):.4f} (expected {2 * stats.t.sf(3, dfree):.4f} if unbiased); mean |bias| = {np.mean(mb):.5f}")
    if "T8" in runs[0]["semview"]["quality"]:
        print("\n### T8 period-over-period comparisons (pooled over runs)\n\n| method | variant | queries | coverage (HT) | coverage (cons.) | ratio coverage (delta) | mean half-width | Var(coordinated)/Var(independent) |\n|---|---|---|---|---|---|---|---|")
        for m in SHARED:
            agg = {}
            for x in runs:
                for var, d in x[m]["quality"].get("T8", {}).items():
                    a_ = agg.setdefault(var, {"n": 0, "c": [0, 0], "c2": [0, 0], "rc": [0, 0], "vr": [], "hw": []})
                    a_["n"] += d["n"]; a_["hw"].append(d["hw_mean"])
                    for k_, kk in (("c", "cov"), ("c2", "cov2"), ("rc", "rcov")):
                        a_[k_][0] += d[kk][0]; a_[k_][1] += d[kk][1]
                    if d["vratio_mean"] is not None: a_["vr"].append(d["vratio_mean"])
            for var, a_ in sorted(agg.items()):
                f_ = lambda z: f"{z[1] / z[0]:.3f}" if z[0] else "-"
                print(f"| {m} | {var} | {a_['n']} | {f_(a_['c'])} | {f_(a_['c2'])} | {f_(a_['rc'])} | {np.mean(a_['hw']):.4f} | {np.mean(a_['vr']):.3f} |")
    if "t8err" in runs[0]["semview"]["quality"]:
        print("\n### empirical Var(coordinated)/Var(independent) across salts (T8, semview)\n")
        ratios = {}
        for ws in sorted({x["wseed"] for x in runs}):
            R_ = [x["semview"]["quality"]["t8err"] for x in runs if x["wseed"] == ws]
            for i in range(len(R_[0])):
                var = R_[0][i][0]; a1 = np.array([r[i][1] for r in R_]); a2 = np.array([r[i][2] for r in R_])
                if a2.var() > 1e-12: ratios.setdefault(var, []).append(a1.var() / a2.var())
        for var, rr in sorted(ratios.items()): print(f"- {var}: median {np.median(rr):.3f}, mean {np.mean(rr):.3f} over {len(rr)} queries")
    print("\npooled group coverage (semview):")
    g = {}
    for x in runs:
        for k, (n, c) in x["semview"]["quality"]["groups"].items():
            a = g.setdefault(k, [0, 0]); a[0] += n; a[1] += c
    for k in sorted(g): print(f"  {k}: {g[k][1] / g[k][0]:.3f} (n={g[k][0]})")


if __name__ == "__main__":
    for p in sys.argv[1:]: main(p)
