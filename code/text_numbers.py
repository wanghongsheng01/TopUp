"""Recompute every number that the paper's prose states literally (not via numbers.tex), for re-checking after reruns."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
DS = {"RMA": "rma", "Amazon-MI": "amazon", "Amazon-IS": "amazon2"}
R = {k: {e: json.load(open(os.path.join(RES, f"final_{v}_eps{e}.json"))) for e in ("05", "10")} for k, v in DS.items()}
MS = {k: json.load(open(os.path.join(RES, f"final_{v}_minsup200_eps05.json"))) for k, v in DS.items()}
OL = {k: json.load(open(os.path.join(RES, f"olazy_{v}.json"))) for k, v in DS.items()}
m = lambda xs: float(np.mean(xs))
rng = lambda xs, f="{:.3f}": f"{f.format(min(xs))}--{f.format(max(xs))}"
H = "human_oracle"
sv = lambda runs, p=H: np.array([x["semview"]["cost"][p]["total"] for x in runs])
bl = lambda runs, b, p=H: np.array([x[f"{b}_{p}"]["total"] for x in runs])
nor = lambda runs: m([x["semview"]["counts"]["oracle_main"] + x["semview"]["counts"]["pilot"] for x in runs])
out = {}
out["B7 saving %"] = rng([100 * (1 - m(sv(R[k][e]["runs"]) / np.array([x["b7"]["cost"][H]["total"] for x in R[k][e]["runs"]]))) for k in R for e in R[k]], "{:.0f}")
out["B5 saving %"] = rng([100 * (1 - m(sv(R[k][e]["runs"]) / bl(R[k][e]["runs"], "b5"))) for k in R for e in R[k]], "{:.0f}")
out["TopUp/B5 %"] = rng([100 * m(sv(R[k][e]["runs"]) / bl(R[k][e]["runs"], "b5")) for k in R for e in R[k]], "{:.0f}")
out["TopUp/O-lazy %"] = rng([100 * m(sv(R[k][e]["runs"])) / (OL[k]["touched_mean"] * 0.05) for k in R for e in R[k]], "{:.0f}")
out["B5 labels / TopUp labels"] = rng([m([x["b5_human_oracle"]["n_oracle"] for x in R[k][e]["runs"]]) / nor(R[k][e]["runs"]) for k in R for e in R[k]], "{:.2f}")
out["B6 vs B5 cost diff %"] = rng([100 * abs(m(bl(R[k][e]["runs"], "b6")) / m(bl(R[k][e]["runs"], "b5")) - 1) for k in R for e in R[k]], "{:.1f}")
out["O-lazy touched share %"] = rng([100 * OL[k]["touched_mean"] / OL[k]["n_tuples"] for k in R], "{:.0f}")
out["B5ideal ratio (min is parity)"] = {k: [round(m(sv(R[k][e]["runs"]) / bl(R[k][e]["runs"], "b5ideal")), 2) for e in R[k]] for k in R}
q = lambda k, e: [x["semview"]["quality"] for x in R[k][e]["runs"]]
out["half-width HT eps05"] = rng([m([z["mean_halfwidth"] for z in q(k, "05")]) for k in R])
out["half-width floored eps05"] = rng([m([z["mean_halfwidth_cons"] for z in q(k, "05")]) for k in R])
out["half-width eps10 (HT..floored)"] = rng([m([z[h] for z in q(k, "10")]) for k in R for h in ("mean_halfwidth", "mean_halfwidth_cons")])
out["T3 correct when separated min"] = round(min(m([z["T3_correct_when_separated"] for z in q(k, "05")]) for k in R), 3)


def worst(runs, key):
    g = {}
    for x in runs:
        for kk, (n, c) in x["semview"]["quality"][key].items():
            a = g.setdefault(kk, [0, 0]); a[0] += n; a[1] += c
    big = [c / n for kk, (n, c) in g.items() if n >= 1000] or [c / n for kk, (n, c) in g.items()]
    return min(big)


out["worst floored eps05 per dataset"] = {k: round(worst(R[k]["05"]["runs"], "groups2"), 3) for k in R}
out["worst HT eps05 per dataset"] = {k: round(worst(R[k]["05"]["runs"], "groups"), 3) for k in R}
t8 = {}
for k in R:
    agg = {}
    for x in R[k]["05"]["runs"]:
        for var, d in x["semview"]["quality"].get("T8", {}).items():
            a = agg.setdefault(var, {"c2": [0, 0], "rc": [0, 0], "rc2": [0, 0], "vr": []})
            for kk, s in (("c2", "cov2"), ("rc", "rcov"), ("rc2", "rcov2")): a[kk][0] += d[s][0]; a[kk][1] += d[s][1]
            if d["vratio_mean"] is not None: a["vr"].append(d["vratio_mean"])
    t8[k] = agg
out["T8 diff floored cov"] = rng([a["c2"][1] / a["c2"][0] for k in t8 for a in t8[k].values()])
out["T8 ratio cov"] = rng([a["rc"][1] / max(1, a["rc"][0]) for k in t8 for a in t8[k].values()])
out["T8 floored ratio cov"] = rng([a["rc2"][1] / max(1, a["rc2"][0]) for k in t8 for a in t8[k].values()])
out["T8 roll var ratio"] = {k: round(m(t8[k]["roll7"]["vr"]), 3) for k in t8}
out["T8 yoy ratio cov"] = {k: round(t8[k]["yoy"]["rc"][1] / t8[k]["yoy"]["rc"][0], 3) for k in t8}
out["lazy raises/run"] = rng([m([x["semview"]["lazy"] for x in R[k]["05"]["runs"]]) for k in R], "{:.0f}")
out["pilot share of oracle labels %"] = rng([100 * m([x["semview"]["counts"]["pilot"] for x in R[k]["05"]["runs"]]) / nor(R[k]["05"]["runs"]) for k in R], "{:.0f}")
out["A-lazy HT cov"] = rng([m([x["a_lazy"]["quality"]["coverage"] for x in R[k][e]["runs"]]) for k in R for e in R[k]])
out["f (oracle share of all tuples) eps05"] = {k: round(nor(R[k]["05"]["runs"]) / R[k]["05"]["n_tuples"], 3) for k in R}
out["f eps10"] = {k: round(nor(R[k]["10"]["runs"]) / R[k]["10"]["n_tuples"], 3) for k in R}
out["kappa* eps05"] = {k: round(1 / (1 - nor(R[k]["05"]["runs"]) / R[k]["05"]["n_tuples"]), 2) for k in R}
out["minsup: B5 ratio drop (points)"] = rng([100 * (m(sv(R[k]["05"]["runs"]) / bl(R[k]["05"]["runs"], "b5")) - m(sv(MS[k]["runs"]) / bl(MS[k]["runs"], "b5"))) for k in R], "{:.0f}")
out["minsup: worst HT gain (points)"] = rng([100 * (worst(MS[k]["runs"], "groups") - worst(R[k]["05"]["runs"], "groups")) for k in R], "{:.0f}")
out["RMA cheap bundle $/1k, oracle $/1k"] = (round(R["RMA"]["05"]["c1_bundle"] * 1000, 4), round(R["RMA"]["05"]["prices"]["llm_oracle"] * 1000, 3))
out["RMA kappa LLM, human"] = (round(R["RMA"]["05"]["prices"]["llm_oracle"] / R["RMA"]["05"]["c1_bundle"], 1), round(0.05 / R["RMA"]["05"]["c1_bundle"], 0))
out["targets per run"] = {k: R[k]["05"]["runs"][0]["semview"]["quality"]["targets"] for k in R}
out["RMA tuples"] = R["RMA"]["05"]["n_tuples"]
for k, v in out.items(): print(f"{k}: {v}")

# blocking by lazy raises (A-lazy vs TopUp): raises per run, share of queries blocked, lazy oracle labels (blocking time at 200 labels/min)
lz = lambda runs, mm: (m([x[mm]["lazy"] for x in runs]), m([x[mm]["counts"]["oracle_main"] - x[mm]["counts"]["plan_only"] for x in runs]), m([x["n_queries"] for x in runs]))
B = {(k, e, mm): lz(R[k][e]["runs"], mm) for k in R for e in R[k] for mm in ("semview", "a_lazy")}
for mm in ("semview", "a_lazy"):
    out[f"{mm} raises/run (both eps)"] = rng([B[(k, e, mm)][0] for k in R for e in R[k]], "{:.0f}")
    out[f"{mm} blocked query share % (both eps)"] = rng([100 * B[(k, e, mm)][0] / B[(k, e, mm)][2] for k in R for e in R[k]], "{:.1f}")
    out[f"{mm} blocking minutes at 200/min (both eps)"] = rng([B[(k, e, mm)][1] / 200 for k in R for e in R[k]], "{:.0f}")
out["A-lazy/TopUp raises ratio"] = rng([B[(k, e, "a_lazy")][0] / B[(k, e, "semview")][0] for k in R for e in R[k]], "{:.1f}")
out["A-lazy/TopUp blocking time ratio"] = rng([B[(k, e, "a_lazy")][1] / B[(k, e, "semview")][1] for k in R for e in R[k]], "{:.1f}")
print(json.dumps({k: out[k] for k in list(out)[-8:]}, indent=1))
