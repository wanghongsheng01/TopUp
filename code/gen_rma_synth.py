"""Synthetic stand-in for the production RMA table, generated from its released aggregates (results/rma_synth_stats.json,
written by make_rma_stats.py). Per stratum and day it draws the published number of arrivals, the oracle class from the
stratum's class counts, the cheap labellers' and the legacy code's outputs from the pooled confusion given the oracle class,
and an update delay from the published histogram. Writes public/rma_synth/ in the layout of the public datasets, so the
replay runs on it unchanged:  SV_DATA=rma_synth SV_C=4 SV_YOY=1 SV_LABS=ds-flash-b10,kimi-k2.6-b10 python code/replay.py ...

Usage: python gen_rma_synth.py [stats.json] [seed]
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "rma_synth_stats.json")
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
S = json.load(open(src)); out = os.path.join(HERE, "..", "public", "rma_synth"); os.makedirs(os.path.join(out, "labels"), exist_ok=True)
conf = S["confusion_given_oracle"]; PRED = np.array([-1, 0, 1, 2, 3]); CODE = np.array([-1, 1, 2, 3, 4])
delays = np.array([int(k) for k in S["edit_delay_hist"]]); dw = np.array(list(S["edit_delay_hist"].values()), float); dw /= dw.sum()
rows = []
for s in S["strata"]:
    pc = np.array(s["oracle_class_counts"], float) + 0.5; pc /= pc.sum()  # small add-one style smoothing for empty classes
    days = [(d, n) for d, n in enumerate(s["arrivals_cur"]) if n] + [(int(d), n) for d, n in s["arrivals_hist"].items()]
    for d, n in days:
        y = rng.choice(4, size=n, p=pc)
        for k in range(n):
            rows.append((s["area"], s["market"], s["category"], d, int(y[k])))
p = pd.DataFrame(rows, columns=["area_name", "market_name", "category_lv1_name", "day", "gold"])
p["key"] = [f"r{i:06d}" for i in range(len(p))]
edited = rng.random(len(p)) < S["edit_share"]
p["edit_day"] = np.where(edited, p.day + rng.choice(delays, size=len(p), p=dw), -1)
cl = np.array(conf["legacy_code_lv1"], float)
p["code_lv1"] = [CODE[rng.choice(5, p=cl[y] / cl[y].sum())] for y in p.gold]
cols = ["key", "day", "edit_day", "area_name", "market_name", "category_lv1_name", "code_lv1", "gold"]
for suf, part in (("", p[p.day >= 0]), ("_hist", p[p.day < 0])):
    part[cols].to_parquet(os.path.join(out, f"prepared{suf}.parquet"), index=False)
    for l in S["labellers"]:
        cm = np.array(conf[l], float)
        pred = [PRED[rng.choice(5, p=cm[y] / cm[y].sum())] for y in part.gold]
        pd.DataFrame({"key": part.key, "pred": pred, "cost_usd": S["mean_cost_usd"][l]}).to_csv(os.path.join(out, "labels", f"{l}{suf}.csv"), index=False)
print(len(p), "rows ->", out)
