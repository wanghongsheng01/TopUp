"""Aggregate, de-identified statistics of the production RMA table, the only form in which it may leave the company.

Reads the de-identified pilot tables and label files in private/ (never the note text) and writes
results/rma_synth_stats.json with, per anonymised stratum (channel-country x category, renamed S001...):
  - arrival counts per day for the current quarter (days 0..91) and the history quarter (negative days),
  - counts of the oracle's classes,
and, pooled over the table:
  - each cheap labeller's and the legacy code's confusion counts given the oracle class,
  - mean per-note cost of every labeller,
  - the share of rows with an update and the histogram of update delays (days).
Market, area and category names are replaced by ids; no row-level value is written.
gen_rma_synth.py turns the file into a synthetic table that the replay reads with SV_DATA=rma_synth.
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); PRIV = os.path.join(HERE, "..", "private")
LABS, ORACLE = ["ds-flash-b10", "kimi-k2.6-b10"], "ds-pro-think"
COLS = ["key", "day", "edit_day", "area_name", "market_name", "category_lv1_name", "code_lv1"]
TOP = {"Desk", "Chair", "Sofa", "Garage Storage Racks", "Bed"}


def part(f, suf):
    p = pd.read_parquet(os.path.join(PRIV, f), columns=COLS)
    for l in LABS + [ORACLE]:
        d = pd.read_csv(os.path.join(PRIV, "labels", f"{l}{suf}.csv")).set_index("key")
        p[l] = p.key.map(d.pred).fillna(-1).astype(int); p["c_" + l] = p.key.map(d.cost_usd)
    return p


p = pd.concat([part("pilot_deid_2024q4.parquet", "_2024q4"), part("pilot_deid.parquet", "")], ignore_index=True)
p = p[p[ORACLE] >= 0].reset_index(drop=True)
big = p.market_name[p.day >= 0].value_counts(); big = set(big[big >= 100].index)   # as in replay.finish
p["market"] = np.where(p.market_name.isin(big), p.market_name, "other-" + p.area_name)
p["cat1"] = np.where(p.category_lv1_name.isin(TOP), p.category_lv1_name, "other")
areas = {a: f"A{i + 1}" for i, a in enumerate(sorted(p.area_name.unique()))}
markets = {m: f"M{i + 1:02d}" for i, m in enumerate(sorted(p.market.unique()))}
cats = {c: (f"C{i + 1}" if c != "other" else "other") for i, c in enumerate(sorted(TOP)) } | {"other": "other"}
cur_days, hist_days = sorted(p.day[p.day >= 0].unique()), sorted(p.day[p.day < 0].unique())
strata = []
for i, ((m, c), g) in enumerate(p.groupby(["market", "cat1"])):
    strata.append({"id": f"S{i + 1:03d}", "area": areas[g.area_name.mode().iloc[0]], "market": markets[m], "category": cats[c],
                   "arrivals_cur": [int((g.day == d).sum()) for d in range(0, 92)],
                   "arrivals_hist": {str(int(d)): int(n) for d, n in g.day[g.day < 0].value_counts().sort_index().items()},
                   "oracle_class_counts": [int((g[ORACLE] == k).sum()) for k in range(4)]})
conf = {l: [[int(((p[ORACLE] == k) & (p[l] == j)).sum()) for j in (-1, 0, 1, 2, 3)] for k in range(4)] for l in LABS}
conf["legacy_code_lv1"] = [[int(((p[ORACLE] == k) & (p.code_lv1 == j)).sum()) for j in (-1, 1, 2, 3, 4)] for k in range(4)]
ed = (p.edit_day[p.edit_day >= 0] - p.day[p.edit_day >= 0]).clip(lower=0)
out = {"note": "Aggregates of the de-identified production RMA table (TopUp paper, Section V). Names replaced by ids; no row-level data.",
       "n_rows": int(len(p)), "classes": ["quality defect", "shipment problem", "customer-side reason", "other"],
       "strata": strata, "labellers": LABS, "oracle": ORACLE,
       "confusion_given_oracle": {"columns": "pred = -1 (no label), 0, 1, 2, 3; legacy code_lv1 = -1, 1, 2, 3, 4", **conf},
       "mean_cost_usd": {l: float(p["c_" + l].mean()) for l in LABS + [ORACLE]},
       "edit_share": float((p.edit_day >= 0).mean()), "edit_delay_hist": {str(int(k)): int(v) for k, v in ed.round().value_counts().sort_index().items()}}
dst = os.path.join(HERE, "..", "results", "rma_synth_stats.json")
json.dump(out, open(dst, "w"), indent=0)
print(len(strata), "strata;", out["n_rows"], "rows ->", dst)
