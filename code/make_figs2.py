"""Figures of the TopUp paper, revised design (11-11): one colour-blind-safe palette (Okabe-Ito) with TopUp always in
vermillion, Times-like text at the size of the IEEEtran body, and a chart type chosen per question.

  fig_pareto.pdf    cost of every method relative to TopUp, grouped by certified share (eps=0.05)         [Fig. 1]
  fig_coverage.pdf  heat map of interval coverage by slice size, HT vs floored, all three datasets            [Fig. 2]
  fig_cdf.pdf       CDF of the paired cost ratio TopUp / matched baseline over all runs                         [RQ1]
  fig_phase.pdf     heat map of TopUp cost / O-full over price ratio kappa and data scale k                     [RQ4]

Inputs: results/table_cells.json (written by make_numbers.py), results/final_*_eps05.json, results/scale_g/*.json.
Usage: python make_figs2.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "paper", "figs")
for f in ("Times New Roman.ttf", "Times New Roman Bold.ttf", "Times New Roman Italic.ttf"):
    p = os.path.join("/System/Library/Fonts/Supplemental", f)
    if os.path.exists(p): font_manager.fontManager.addfont(p)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42})
# Okabe-Ito: TopUp = vermillion; the rest keep one colour each in every figure
TOP = "#D55E00"; BLUE = "#0072B2"; SKY = "#56B4E9"; GREEN = "#009E73"; ORANGE = "#E69F00"; PURPLE = "#CC79A7"; GREY = "#7F7F7F"; INK = "#333333"
DS = (("RMA", "rma"), ("Amazon-MI", "amazon"), ("Amazon-IS", "amazon2"))
C = json.load(open(os.path.join(RES, "table_cells.json")))
cell = lambda lab, e, m: C[f"{lab}|{e}|{m}"]

# ---------------------------------------------------------------- Fig. 1: cost of every method relative to TopUp (dot plot)
# One row per method, one dot per dataset; filled rows certify as many targets as TopUp (within 0.01), hollow rows fewer.
UP = [("a_lazy*", "A-lazy, matched"), ("b5shared*", "B5-shared, matched"), ("b7", "B7 uniform"), ("cong", "Capped stratified"),
      ("olazy", "O-lazy (exact)"), ("ofull", "O-full (exact)")]
LO = [("p_only", "TopUp plan only"), ("a_lazy", "A-lazy"), ("b5shared", "B5-shared"), ("b5", "B5 Ladder+cache"),
      ("b6", "B6 SemViews-ad."), ("b4", "B4 Ladder")]
rowsM = UP + LO
fig, ax = plt.subplots(figsize=(3.5, 2.7))
dcol = {"RMA": TOP, "Amazon-MI": BLUE, "Amazon-IS": GREEN}; off = {"RMA": -0.3, "Amazon-MI": 0.0, "Amazon-IS": 0.3}
for i, (m, name) in enumerate(rowsM):
    y = i + (0.6 if i >= len(UP) else 0)
    for lab, _ in DS:
        sv, c = cell(lab, "05", "semview"), cell(lab, "05", m); x = c["cost"] / sv["cost"]
        ok = c["met"] >= sv["met"] - 0.01
        ax.scatter(x, y + off[lab], s=9, marker="o", facecolor=(dcol[lab] if ok else "white"), edgecolor=dcol[lab], linewidth=0.8,
                   zorder=3, label=(lab if i == 0 else None))
    ax.axhline(y + 0.5, color="#eeeeee", lw=0.4, zorder=0)
ax.axvline(1, color=TOP, lw=1.0, zorder=2); ax.text(1.04, len(rowsM) + 0.15, "TopUp = 1", color=INK, fontsize=6.5, va="center")
ax.axhline(len(UP) - 0.2, color=GREY, lw=0.6, ls=(0, (3, 2)))
ys = [i + (0.6 if i >= len(UP) else 0) for i in range(len(rowsM))]
ax.set_yticks(ys); ax.set_yticklabels([n for _, n in rowsM]); ax.invert_yaxis(); ax.set_ylim(len(rowsM) + 0.3, -1.0)
ax.set_xscale("log"); ax.set_xlim(0.45, 20); ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"]); ax.minorticks_off()
ax.set_xlabel("total labelling cost relative to TopUp (log scale)"); ax.tick_params(axis="y", length=0)
ax.text(19, len(UP) / 2 - 0.5, "as many targets\ncertified (filled)", ha="right", va="center", fontsize=6.3, color=INK, linespacing=1.3)
ax.text(19, len(UP) + 1.6, "fewer targets\ncertified (hollow)", ha="right", va="center", fontsize=6.3, color=INK, linespacing=1.3)
ax.grid(axis="x", lw=0.3, color="#dddddd", zorder=0); ax.spines["left"].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, fontsize=6.5, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.2)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "fig_pareto.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2: coverage heat map (slice size x dataset, HT vs floored)
buckets = ["50-100", "100-200", "200-500", ">500"]
rows, data = buckets + ["all"], []
for lab, key in DS:
    r = json.load(open(os.path.join(RES, f"final_{key}_eps05.json")))
    for g, allk in (("groups", "coverage"), ("groups2", "coverage_cons")):
        agg = {b: [0, 0] for b in buckets}
        for x in r["runs"]:
            for k, (n, cv) in x["semview"]["quality"][g].items():
                b = k.split("|")[1]; agg[b][0] += n; agg[b][1] += cv
        data.append([agg[b][1] / agg[b][0] for b in buckets] + [np.mean([x["semview"]["quality"][allk] for x in r["runs"]])])
A = np.array(data).T  # rows = buckets + all, cols = (dataset, HT/floored)
cmap = LinearSegmentedColormap.from_list("cov", ["#B2182B", "#F4A582", "#F7F7F7", "#92C5DE", "#2166AC"])
fig, ax = plt.subplots(figsize=(3.5, 1.55))
im = ax.imshow(A, cmap=cmap, norm=TwoSlopeNorm(vmin=0.80, vcenter=0.95, vmax=1.0), aspect="auto")
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=6.5, color=("white" if A[i, j] < 0.86 else INK),
                fontweight=("bold" if j % 2 else "normal"))
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r + " tuples" if r != "all" else "all slices" for r in rows])
ax.set_xticks(range(6)); ax.set_xticklabels(["HT", "floored"] * 3)
for j, (lab, _) in enumerate(DS): ax.text(2 * j + 0.5, -0.85, lab, ha="center", va="bottom", fontsize=7.5)
for j in (1.5, 3.5): ax.axvline(j, color="white", lw=2)
ax.axhline(3.5, color="white", lw=2); ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[0.80, 0.95, 1.0]); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(FIG, "fig_coverage.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: CDF of paired cost ratios
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.45), sharey=True)
cols = {"RMA": TOP, "Amazon-MI": BLUE, "Amazon-IS": GREEN}
for ax, (m, title) in zip(axs, (("b5shared*", "vs. B5-shared, matched"), ("a_lazy*", "vs. A-lazy, matched"))):
    for lab, _ in DS:
        for e, ls in (("05", "-"), ("10", (0, (3, 1.5)))):
            x = np.sort(C[f"{lab}|{e}|paired"][m]); ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=cols[lab], ls=ls, lw=1.1,
                                                              label=f"{lab}, $\\epsilon$=0.{e[1:] if e == '05' else '10'}".replace("0.5", "0.05"))
    ax.axvline(1, color=INK, lw=0.7); ax.set_xlim(0.4, 1.05); ax.set_title(title, fontsize=7.5, pad=2)
    ax.grid(lw=0.3, color="#dddddd"); ax.set_xticks([0.4, 0.6, 0.8, 1.0])
axs[0].set_ylabel("share of runs"); fig.supxlabel("paired cost ratio (TopUp / baseline)", fontsize=8, y=0.03)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=6.3, handlelength=1.6, columnspacing=0.8, borderaxespad=0)
fig.tight_layout(pad=0.3, w_pad=0.5); fig.savefig(os.path.join(FIG, "fig_cdf.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: TopUp / O-full over price ratio and data scale
KAP = [1.5, 2, 3, 5, 13, 600]
rows, data = [], []
for lab, key in (("RMA", "rma"), ("Amazon-IS", "amazon2")):
    for k in (1, 2, 4, 8):
        r = json.load(open(os.path.join(RES, "scale_g", f"{key}_k{k}_eps05.json"))); n = r["n_tuples"]
        scan = np.mean([x["semview"]["counts"]["cheap_scan"] for x in r["runs"]])
        phi = np.mean([x["semview"]["counts"]["oracle_main"] + x["semview"]["counts"]["pilot"] for x in r["runs"]]) / n
        rows.append(f"{lab}, $k$={k}"); data.append([scan / (n * kp) + phi for kp in KAP])
A = np.array(data)
cmap2 = LinearSegmentedColormap.from_list("ratio", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"])
fig, ax = plt.subplots(figsize=(3.5, 1.9))
im = ax.imshow(A, cmap=cmap2, norm=TwoSlopeNorm(vmin=0.2, vcenter=1.0, vmax=1.3), aspect="auto")
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=("white" if A[i, j] < 0.45 else INK))
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
ax.set_xticks(range(len(KAP))); ax.set_xticklabels([f"{k:g}" for k in KAP])
ax.set_xlabel("oracle / cheap-bundle price ratio $\\kappa$")
for j, t in ((4, "LLM oracle"), (5, "human")): ax.text(j, -0.65, t, ha="center", va="bottom", fontsize=6.3, color=INK)
ax.axhline(3.5, color="white", lw=2); ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[0.2, 0.6, 1.0, 1.3]); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(FIG, "fig_phase.pdf"), bbox_inches="tight"); plt.close(fig)
json.dump({"phase_rows": rows, "kappa": KAP, "phase": A.round(4).tolist()}, open(os.path.join(RES, "fig_phase_values.json"), "w"), indent=1)
print("wrote fig_pareto, fig_coverage, fig_cdf, fig_phase")
