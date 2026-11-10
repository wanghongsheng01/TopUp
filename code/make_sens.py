"""Sensitivity of the hand-set constants (results/sens, run_sens.sh): pilot size 200/800 (default 400) and moment
shrinkage omega 2/50 (default 10), eps=0.05, 40 paired runs. The default is the same 40 (workload seed, salt) runs of
results/scale_g/rma_k1_eps05.json (RMA) and results/v7/matched_amazon_eps05.json (Amazon-MI).
Writes paper/numbers_sens.tex and results/sens/summary.json."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results"); PAPER = os.path.join(HERE, "..", "paper")
EPS = 0.05
DEF = {"rma": os.path.join(RES, "scale_g", "rma_k1_eps05.json"), "amazon": os.path.join(RES, "v7", "matched_amazon_eps05.json")}


def stats(runs):
    c = {(x["wseed"], x["salt"]): x["semview"]["cost"]["human_oracle"]["total"] for x in runs}
    q = [x["semview"]["quality"] for x in runs]
    return c, float(np.mean([z["target_met"] for z in q])), float(np.mean([np.mean(np.abs(z["qerr"]) <= EPS + 1e-12) for z in q])), float(np.mean([z["coverage_cons"] for z in q]))


out = {}
for d, lab in (("rma", "RMA"), ("amazon", "Amazon-MI")):
    c0, *_ = stats(json.load(open(DEF[d]))["runs"])
    for v in ("SV_PILOT=200", "SV_PILOT=800", "SV_KAPPA=2", "SV_KAPPA=50"):
        c, met, acc, cov = stats(json.load(open(os.path.join(RES, "sens", f"{d}_{v}.json")))["runs"])
        keys = sorted(c)
        assert all(k in c0 for k in keys)
        out[f"{lab}|{v}"] = dict(rel_cost=float(np.mean([c[k] for k in keys]) / np.mean([c0[k] for k in keys])), met=met, acc=acc, cov=cov, n=len(keys))
print(json.dumps(out, indent=1))
rc = [v["rel_cost"] for v in out.values()]; mets = [v["met"] for v in out.values()]; accs = [v["acc"] for v in out.values()]
with open(os.path.join(PAPER, "numbers_sens.tex"), "w") as fh:
    fh.write(f"\\newcommand{{\\NumSensDev}}{{{int(np.ceil(100 * max(abs(r - 1) for r in rc)))}\\%}}\n")
    fh.write(f"\\newcommand{{\\NumSensMet}}{{{min(mets):.2f}--{max(mets):.2f}}}\n")
    fh.write(f"\\newcommand{{\\NumSensAcc}}{{{min(accs):.3f}--{max(accs):.3f}}}\n")
json.dump(out, open(os.path.join(RES, "sens", "summary.json"), "w"), indent=1)
