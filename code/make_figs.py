"""Figures for the TopUp paper from replay JSON files (vector PDF, column width). Style and palette follow the Ladder paper.

Usage: python make_figs.py <label>=<replay_eps05.json> [<label>=<file> ...]
Writes paper/figs/fig_cost.pdf, fig_phase.pdf, fig_coverage.pdf.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "paper", "figs"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "lines.linewidth": 1.4, "lines.markersize": 4, "pdf.fonttype": 42})
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#b05cc6", "#52514e"]  # dataviz reference slots + neutral ink
files = dict(a.split("=", 1) for a in sys.argv[1:])
R = {k: json.load(open(v)) for k, v in files.items()}


def tot(runs, m, price):
    if m in ("semview", "b7", "a_lazy", "p_only"): return np.array([x[m]["cost"][price]["total"] for x in runs])
    return np.array([x[f"{m}_{price}"]["total"] for x in runs])


# Fig. cost: total cost per method, human-oracle price, log scale, one group per dataset (O-lazy from results/olazy_*.json)
OLAZY = {"RMA": "rma", "Amazon-MI": "amazon", "Amazon-IS": "amazon2"}
methods = [("semview", "TopUp"), ("b7", "B7 uniform"), ("b6", "B6 SemViews-ad."), ("b5", "B5 Ladder+cache"),
           ("b5shared", "B5-shared"), ("olazy", "O-lazy (exact)"), ("b4", "B4 Ladder")]
cols = PAL[:5] + ["#c9a227", "#9a9893"]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
YMAX = 2400
w = 0.84 / len(methods); xs = np.arange(len(R))
for i, (m, name) in enumerate(methods):
    vals = []
    for lab, r in R.items():
        if m == "olazy":
            f = os.path.join(HERE, "..", "results", f"olazy_{OLAZY.get(lab, lab)}.json")
            vals.append(json.load(open(f))["touched_mean"] * 0.05 if os.path.exists(f) else np.nan)
        else:
            vals.append(tot(r["runs"], m, "human_oracle").mean())
    xb = xs + i * w - 0.42 + w / 2
    if m == "b4":  # per-query re-planning is 10x higher: clip the bar and print its value, so the other bars stay readable
        ax.bar(xb, [min(v, YMAX) for v in vals], w, label=name, color=cols[i], hatch="////", edgecolor="white", lw=0)
        for x_, v in zip(xb, vals): ax.text(x_, YMAX * 1.01, f"{v / 1000:.1f}k", ha="center", va="bottom", fontsize=5, rotation=90)
    else:
        ax.bar(xb, vals, w, label=name, color=cols[i])
ax.set_ylim(0, YMAX * 1.18); ax.set_xticks(xs); ax.set_xticklabels(list(R)); ax.set_ylabel("labelling cost (\\$)")
ax.legend(ncol=4, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), handlelength=1.2, columnspacing=0.8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cost.pdf"), bbox_inches="tight"); plt.close(fig)

# Fig. phase: TopUp cost relative to labelling everything with the oracle, as a function of kappa = c0 / c1.
# Oracle-label counts do not depend on the price scale (the plan is scale-invariant), so the ratio is analytic:
# TopUp / O-full = (N c1 + n_or c0) / (N c0) = 1 / kappa + n_or / N, with N all tuples any query reads.
fig, ax = plt.subplots(figsize=(3.4, 1.8))
kap = np.logspace(0, 4, 100)
for i, (lab, r) in enumerate(R.items()):
    c = r["runs"][0]["semview"]["counts"]; ncur = r["n_tuples"]  # O-full labels every tuple, history included
    nor = np.mean([x["semview"]["counts"]["oracle_main"] + x["semview"]["counts"]["pilot"] for x in r["runs"]])
    ax.plot(kap, c["cheap_scan"] / (ncur * kap) + nor / ncur, color=PAL[i], label=lab)
ax.axhline(1, color="#52514e", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_ylim(0, 1.5)
for k_, lab_ in ((13, "LLM oracle"), (600, "human review")):  # measured price ratios on RMA
    ax.axvline(k_, color="#9a9893", lw=0.7, ls=":"); ax.text(k_ * 1.15, 1.35, lab_, fontsize=5.5, color="#52514e")
ax.set_xlabel(r"oracle / cheap-bundle price ratio $\kappa=c_0/c_1$"); ax.set_ylabel("TopUp cost / O-full")
ax.legend(fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0)); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_phase.pdf"), bbox_inches="tight"); plt.close(fig)

# Fig. coverage: coverage by slice-size bucket, HT vs floored interval (first dataset)
lab, r = next(iter(R.items()))
buckets = ["50-100", "100-200", "200-500", ">500"]
fig, ax = plt.subplots(figsize=(3.4, 1.6))
for j, (key, name) in enumerate((("groups", "HT"), ("groups2", "floored"))):
    agg = {b: [0, 0] for b in buckets}
    for x in r["runs"]:
        for k, (n, cv) in x["semview"]["quality"][key].items():
            b = k.split("|")[1]; agg[b][0] += n; agg[b][1] += cv
    ax.plot(range(len(buckets)), [agg[b][1] / agg[b][0] for b in buckets], marker="os"[j], color=PAL[j], label=name)
ax.axhline(0.95, color="#52514e", lw=0.8, ls="--"); ax.set_xticks(range(len(buckets))); ax.set_xticklabels(buckets)
ax.set_xlabel("slice size (tuples)"); ax.set_ylabel(f"coverage ({lab})"); ax.set_ylim(0.8, 1.0); ax.legend(fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_coverage.pdf"), bbox_inches="tight"); plt.close(fig)
print("wrote", os.listdir(FIG))
