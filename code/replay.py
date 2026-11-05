"""Offline replay of the SemView/TopUp pilot on the labelled RMA pilot (Oct-Dec 2025, attribute A1).

Every tuple carries one label from every catalog labeler and from the oracle. Methods only "pretend" to buy the
labels in their plan, so all methods and all repetitions reuse the same labels (no API cost here).

Methods
  semview : cheap bundle scanned on arrival (pi_1 = 1) + oracle level with PRN inclusion pi_h per stratum,
            solved once per pilot for the whole workload model (multi-query Neyman), plus lazy query-time top-ups;
            running-max inclusion per (stratum, week) cell, so nothing is ever labelled twice (except after edits)
  b7      : same, but one uniform oracle rate for all strata (full materialisation + shared PPI oracle sample)
  a_lazy  : ablation, PRN + lazy top-ups only (no workload plan; pi starts at the floor)
  b4      : Ladder per query, re-planned for every query; own 300-tuple pilot; labels discarded
  b5      : b4 + label cache (a label bought earlier is free while valid); fresh random samples per query

Usage: python replay.py [--eps 0.05] [--salts 40] [--wseeds 5] [--out results/replay_eps05.json]
"""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PRIV = os.path.join(ROOT, "private")
LABS = os.environ.get("SV_LABS", "ds-flash-b10,kimi-k2.6-b10,glm-flash-b10").split(",")
ORACLE = os.environ.get("SV_ORACLE", "ds-pro-think")
DATA = os.environ.get("SV_DATA", "")  # "" = enterprise RMA pilot; "cfpb" / "amazon" = public datasets with human oracle
C = int(os.environ.get("SV_C", 4))
Z = 1.959964
CODE = {1: 0, 2: 1, 4: 2, 3: 3}
MONTHS = [(0, 30), (31, 60), (61, 91)]  # Oct, Nov, Dec as day ranges
Q_DAYS = range(30, 92)
QPD = 20
_T8W = float(os.environ.get("SV_T8W", 0.15))  # share of T8; T1-T3 keep their 0.35:0.25:0.25 proportions
TEMPLATES = {"T1": 0.35 / 0.85 * (1 - _T8W), "T2": 0.25 / 0.85 * (1 - _T8W), "T3": 0.25 / 0.85 * (1 - _T8W), "T8": _T8W}  # T8: v3 requirement R2
# T8 variants: (window 1, window 2) as day offsets from tau; "mom" uses calendar months
T8_VARIANTS = {"mom": None, "pp30": ((29, 0), (59, 30)), "roll7": ((29, 0), (36, 7))}
if os.environ.get("SV_YOY", "") == "1": T8_VARIANTS["yoy"] = None  # last month vs the same month one year earlier
SLICE_TYPES = [("area",), ("market",), ("cat1",), ("area", "cat1"), ("market", "cat1")]
WINDOWS = ["last7", "last30", "last_month"]
KRES = float(os.environ.get("SV_KRES", 0))  # CLT guardrail: expected sampled non-zero residuals per query (0 = off)
MIN_SUPPORT, M_MIN, PI_MIN, PILOT_N, KAPPA = int(os.environ.get("SV_MINSUP", 50)), int(os.environ.get("SV_MMIN", 30)), 0.01, 400, 10


EPS = 0.05


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


# ------------------------------------------------------------------ data
YOY = os.environ.get("SV_YOY", "") == "1"  # add the 2024 Q4 tuples (negative days) for year-over-year queries


def load_public():
    base = os.path.join(ROOT, "public", DATA)
    cols = ["key", "day", "edit_day", "area_name", "market_name", "category_lv1_name", "code_lv1", "gold"]
    parts = [("", pd.read_parquet(os.path.join(base, "prepared.parquet"), columns=cols))]
    if YOY: parts.insert(0, ("_hist", pd.read_parquet(os.path.join(base, "prepared_hist.parquet"), columns=cols)))
    out = []
    for suf, p in parts:
        for l in LABS:
            d = pd.read_csv(os.path.join(base, "labels", f"{l}{suf}.csv")).set_index("key")
            p[l] = p.key.map(d.pred).fillna(-1).astype(int); p["c_" + l] = p.key.map(d.cost_usd)
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    p[ORACLE] = p.gold.astype(int); p["c_" + ORACLE] = float(os.environ.get("SV_C0", 0.002))  # assumed premium-LLM price
    return p


def load():
    cols = ["key", "day", "edit_day", "area_name", "market_name", "category_lv1_name", "code_lv1"]
    if DATA:
        p = load_public()
        return finish(p)
    p = pd.read_parquet(os.path.join(PRIV, "pilot_deid.parquet"), columns=cols)
    lab = {l: pd.read_csv(os.path.join(PRIV, "labels", f"{l}.csv")).set_index("key") for l in LABS + [ORACLE]}
    if YOY:
        p = pd.concat([pd.read_parquet(os.path.join(PRIV, "pilot_deid_2024q4.parquet"), columns=cols), p], ignore_index=True)
        lab = {l: pd.concat([pd.read_csv(os.path.join(PRIV, "labels", f"{l}_2024q4.csv")), d.reset_index()]).set_index("key")
               for l, d in lab.items()}
    for l, d in lab.items():
        p[l] = p.key.map(d.pred).fillna(-1).astype(int)
        p["c_" + l] = p.key.map(d.cost_usd)
    return finish(p)


def finish(p):
    p = p[p[ORACLE] >= 0].reset_index(drop=True)  # tuples the oracle could not label are out of scope (reported)
    p["code"] = p.code_lv1.map(CODE).fillna(-1).astype(int)
    big = p.market_name[p.day >= 0].value_counts(); big = set(big[big >= 100].index)  # strata from the current year only
    p["market"] = np.where(p.market_name.isin(big), p.market_name, "other-" + p.area_name)
    top = {"Desk", "Chair", "Sofa", "Garage Storage Racks", "Bed"} if not DATA else set(p.category_lv1_name.unique())
    p["cat1"] = np.where(p.category_lv1_name.isin(top), p.category_lv1_name, "other")
    p["area"] = p.area_name
    p["h"] = pd.factorize(p.market + "|" + p.cat1)[0]
    p["week"] = (p.day - p.day.min()) // 7
    return p


def window(w, tau):
    if w == "last7": return tau - 6, tau
    if w == "last30": return tau - 29, tau
    m = [k for k, (a, b) in enumerate(MONTHS) if a <= tau <= b][0]
    return MONTHS[m - 1] if m >= 1 else (tau - 29, tau)  # no previous month in the data yet


def slice_mask(p, st, val):
    m = np.ones(len(p), bool)
    for d, v in zip(st, val): m &= (p[d].values == v)
    return m


def gen_workload(p, seed):
    """Template/slice/window generator from 03_数据与负载.md §2.3 (T1-T3 only); slices Zipf(1.1) by volume."""
    rng = np.random.default_rng(seed)
    vals = {st: [tuple(x) for x in p[list(st)].drop_duplicates().values if not any(str(v).startswith("other-") for v in x)]
            for st in SLICE_TYPES}
    masks = {(st, v): slice_mask(p, st, v) for st in SLICE_TYPES for v in vals[st]}
    day = p.day.values
    qs = []
    for tau in Q_DAYS:
        for _ in range(QPD):
            t = rng.choice(list(TEMPLATES), p=list(TEMPLATES.values()))
            if t == "T8":
                q = gen_t8(rng, tau, vals, masks, day)
                if q is not None:
                    qs.append(q); continue
                t = "T1"
            for _try in range(20):
                st = SLICE_TYPES[rng.integers(len(SLICE_TYPES))]; w = WINDOWS[rng.integers(3)]
                a, b = window(w, tau)
                cand = [(int((masks[(st, v)] & (day >= a) & (day <= b)).sum()), v) for v in vals[st]]
                cand = sorted([c for c in cand if c[0] >= MIN_SUPPORT], reverse=True)
                if cand: break
            pr = np.arange(1, len(cand) + 1) ** -1.1; pr /= pr.sum()
            v = cand[rng.choice(len(cand), p=pr)][1]
            idx = np.flatnonzero(masks[(st, v)] & (day >= a) & (day <= b))
            r = int(rng.integers(C)) if t == "T1" else -1
            qs.append({"tau": tau, "t": t, "st": st, "v": v, "w": w, "idx": idx, "r": r})
    return qs, masks, vals


def t8_windows(var, tau):
    if var in ("mom", "yoy"):
        m = [k for k, (a, b) in enumerate(MONTHS) if a <= tau <= b][0]
        if var == "yoy":  # calendar-year shift (2025 is not a leap year, so no Feb-29 clamp is needed here)
            a, b = MONTHS[m - 1]
            return ((a, b), (a - 365, b - 365)) if m >= 1 else None
        return (MONTHS[m - 1], MONTHS[m - 2]) if m >= 2 else None
    (a1, b1), (a2, b2) = T8_VARIANTS[var]
    return ((tau - a1, tau - b1), (tau - a2, tau - b2)) if tau - a2 >= 0 else None


def gen_t8(rng, tau, vals, masks, day):
    """Comparison query: share of reason r in a slice, window 1 minus window 2 (and their ratio)."""
    avail = [v for v in T8_VARIANTS if t8_windows(v, tau) is not None]
    if not avail: return None
    var = avail[rng.integers(len(avail))]
    (a1, b1), (a2, b2) = t8_windows(var, tau)
    for _try in range(20):
        st = SLICE_TYPES[rng.integers(len(SLICE_TYPES))]
        cand = []
        for v in vals[st]:
            n1 = int((masks[(st, v)] & (day >= a1) & (day <= b1)).sum()); n2 = int((masks[(st, v)] & (day >= a2) & (day <= b2)).sum())
            if min(n1, n2) >= MIN_SUPPORT: cand.append((n1 + n2, v))
        if cand: break
    if not cand: return None
    cand = sorted(cand, reverse=True); pr = np.arange(1, len(cand) + 1) ** -1.1; pr /= pr.sum()
    v = cand[rng.choice(len(cand), p=pr)][1]
    i1 = np.flatnonzero(masks[(st, v)] & (day >= a1) & (day <= b1)); i2 = np.flatnonzero(masks[(st, v)] & (day >= a2) & (day <= b2))
    return {"tau": tau, "t": "T8", "st": st, "v": v, "w": var, "idx": np.union1d(i1, i2), "idx1": i1, "idx2": i2,
            "r": int(rng.integers(C))}


def contrast_weights(q):
    """Signed tuple weights of the linear contrast theta_1 - theta_2 over idx = idx1 U idx2."""
    idx = q["idx"]; in1 = np.isin(idx, q["idx1"]); in2 = np.isin(idx, q["idx2"])
    return in1 / len(q["idx1"]) - in2 / len(q["idx2"]), in1, in2


# ------------------------------------------------------------------ surrogate (Ladder's ridge-on-one-hot bundle)
def onehot(P):
    X = np.zeros((len(P), P.shape[1] * (C + 1)))
    for j in range(P.shape[1]): X[np.arange(len(P)), j * (C + 1) + P[:, j] + 1] = 1
    return np.hstack([np.ones((len(P), 1)), X])


def fit(X, Yoh, lam=1.0):
    R = lam * np.eye(X.shape[1]); R[0, 0] = 0
    return np.linalg.solve(X.T @ X + R, X.T @ Yoh)


def surrogate(X, Yoh, pil, rng):
    W = fit(X[pil], Yoh[pil])
    f = X @ W
    # cross-fitted residuals on the pilot for honest moments
    folds = rng.integers(5, size=len(pil)); E = np.zeros((len(pil), C))
    for k in range(5):
        tr, te = pil[folds != k], pil[folds == k]
        if len(te): E[folds == k] = Yoh[te] - X[te] @ fit(X[tr], Yoh[tr])
    return f, E


def moments(h, pil, E, H):
    """Per-stratum second moments of class residuals e_r and of pair differences e_r - e_s, shrunk to the global mean."""
    hp = h[pil]
    D = np.stack([E[:, r] - E[:, s] for r in range(C) for s in range(C)], 1)  # C*C pair columns
    M = np.zeros((H, C)); Md = np.zeros((H, C * C))
    gM, gD = (E ** 2).mean(0), (D ** 2).mean(0)
    for k in range(H):
        sel = hp == k; n = sel.sum()
        M[k] = (n * ((E[sel] ** 2).mean(0) if n else 0) + KAPPA * gM) / (n + KAPPA)
        Md[k] = (n * ((D[sel] ** 2).mean(0) if n else 0) + KAPPA * gD) / (n + KAPPA)
    return M, Md


# ------------------------------------------------------------------ planner: workload-level convex allocation
def constraints(p, masks, vals, nu, f, M, Md, eps, upto):
    """Steady-state constraints for every (slice, window length, target) the generator can ask.
    Returns A (Q x H) and b (Q,) for sum_h A[q,h] (1/pi_h - 1) <= b[q]."""
    H = len(nu); h = p.h.values; rows, bs = [], []
    hist = p.day.values <= upto
    for st in SLICE_TYPES:
        for v in vals[st]:
            m = masks[(st, v)]
            share = np.bincount(h[m], minlength=H).astype(float)
            share = share / np.maximum(np.bincount(h, minlength=H), 1)  # fraction of stratum in slice (1 or 0 here)
            for wl in (7, 30):
                Nh = wl * nu * share; N = Nh.sum()
                if N < MIN_SUPPORT: continue
                for r in range(C):
                    rows.append(Nh * M[:, r] / N ** 2); bs.append((eps / Z) ** 2)
                fbar = f[m & hist].mean(0) if (m & hist).any() else np.zeros(C)
                o = np.argsort(-fbar); r1, r2 = o[0], o[1]
                gap = max(fbar[r1] - fbar[r2], eps)
                rows.append(Nh * Md[:, r1 * C + r2] / N ** 2); bs.append((gap / Z) ** 2)
            # T8 contrasts: two 30-day windows, disjoint (mom/pp30) or overlapping in 23 days (roll7);
            # per stratum, Var = sum_t w_t^2 e_t^2 (1/pi - 1) with w = 1/N1 on W1 only, -1/N2 on W2 only, 1/N1-1/N2 on overlap
            N1 = 30 * (nu * share).sum()
            if N1 >= MIN_SUPPORT:
                for ov in (0, 23):
                    wgt = nu * share * ((30 - ov) / N1 ** 2 + (30 - ov) / N1 ** 2)  # equal-length windows: overlap term is 0
                    for r in range(C):
                        rows.append(wgt * M[:, r]); bs.append((eps / Z) ** 2)
    return np.array(rows), np.array(bs)


def solve(A, b, cost_h, uniform=False, iters=1500):
    """min sum_h cost_h * pi_h  s.t. A (1/pi - 1) <= b, PI_MIN <= pi <= 1.
    KKT: pi_h = sqrt(sum_q lam_q A_qh / cost_h); multiplicative dual updates, then greedy feasibility repair."""
    H = A.shape[1]
    if uniform:
        lo, hi = PI_MIN, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if ((A @ np.full(H, 1 / mid - 1)) <= b * (1 + 1e-9)).all(): hi = mid
            else: lo = mid
        return np.full(H, hi)
    An = A / b[:, None]
    lam = np.ones(len(b)) / len(b); best, bestc = None, np.inf
    cst = np.maximum(cost_h, 1e-12)
    for it in range(iters):
        pi = np.clip(np.sqrt((An.T @ lam) / cst), PI_MIN, 1.0)
        g = An @ (1 / pi - 1)
        if g.max() <= 1.0 + 1e-6 and (cost_h * pi).sum() < bestc: best, bestc = pi.copy(), (cost_h * pi).sum()
        lam *= np.exp(0.5 * np.clip(g - 1, -1, 3)); lam = np.maximum(lam, 1e-12)
    pi = best if best is not None else pi
    for _ in range(10000):  # repair: scale the strata of the worst violated constraint
        g = An @ (1 / pi - 1); q = int(g.argmax())
        if g[q] <= 1.0 + 1e-9: break
        s = An[q] > 0; k = 1.0
        while (An[q] @ (1 / np.where(s, np.minimum(1, pi * k), pi) - 1)) > 1 and k < 1e6: k *= 1.1
        pi = np.where(s, np.minimum(1, pi * k), pi)
    return pi


def guard_ok(pt, pr):
    """CLT guardrail: either the slice is a census, or at least KRES non-zero residuals are sampled in expectation."""
    return KRES <= 0 or (pt >= 1).all() or (pt * pr).sum() >= KRES


def eval_t8(q, pi_cell, cell, h, f, M, Yoh, U, eps, pres=None, U2=None, lazy=True):
    """Period-over-period contrast on the shared synopsis (coordinated PRN design across both windows)."""
    idx = q["idx"]; r = q["r"]; w, in1, in2 = contrast_weights(q)
    ci = cell[idx]; hi = h[idx]; N1, N2 = len(q["idx1"]), len(q["idx2"])

    def ok(pc):
        pt = pc[ci]
        if (w ** 2 * M[hi, r] * (1 / pt - 1)).sum() > (eps / Z) ** 2: return False
        if pres is not None and not (guard_ok(pt[in1], pres[hi[in1]]) and guard_ok(pt[in2], pres[hi[in2]])): return False
        return pt[in1].sum() >= min(M_MIN, N1) and pt[in2].sum() >= min(M_MIN, N2)

    raised = 0
    if lazy and not ok(pi_cell):
        raised = 1; cs = np.unique(ci); k = 1.0
        while k < 1e4:
            k *= 1.15; trial = pi_cell.copy(); trial[cs] = np.minimum(1, pi_cell[cs] * k)
            if ok(trial): break
        pi_cell = trial
    pt = pi_cell[ci]; I = U[idx] < pt; fe = f[idx, r]; y = Yoh[idx, r]; e = y - fe
    est = (w * fe).sum() + (w * e * I / pt).sum(); truth = (w * y).sum()
    var = (w ** 2 * e ** 2 * I * (1 - pt) / pt ** 2).sum()
    vc = max(var, (w ** 2 * M[hi, r] * (1 / pt - 1)).sum())
    # ratio theta_1 / theta_2 by the delta method; overlap tuples give the covariance term
    def part(m_, n_):
        th = (fe[m_].sum() + (e * I / pt)[m_].sum()) / n_
        return th, (e ** 2 * I * (1 - pt) / pt ** 2)[m_].sum() / n_ ** 2
    t1, v1 = part(in1, N1); t2, v2 = part(in2, N2)
    c12 = (e ** 2 * I * (1 - pt) / pt ** 2)[in1 & in2].sum() / (N1 * N2)
    T1, T2 = y[in1].mean(), y[in2].mean()
    rcov = None
    if T2 >= 0.05 and t2 > 0:
        R = t1 / t2; vr = max(v1 + R ** 2 * v2 - 2 * R * c12, 0) / t2 ** 2
        rcov = bool(abs(R - T1 / T2) <= Z * np.sqrt(vr) + 1e-9)
    # design variance (true residuals) of the coordinated contrast vs. independent designs for the two windows
    d = e ** 2 * (1 / pt - 1)
    vind = (d[in1].sum() / N1 ** 2) + (d[in2].sum() / N2 ** 2); vco = (w ** 2 * d).sum()
    # the same contrast estimated from INDEPENDENT samples of the two windows (window 2 uses a second PRN stream U2),
    # so that Var(coordinated) / Var(independent) can be measured across replays rather than computed from Theorem 2
    err_ind = None
    if U2 is not None:
        I2 = U2[idx] < pt
        e1 = (fe[in1].sum() + (e * I / pt)[in1].sum()) / N1; e2 = (fe[in2].sum() + (e * I2 / pt)[in2].sum()) / N2
        err_ind = float(e1 - e2 - truth)
    rec = {"t": "T8", "N": len(idx), "st": "x".join(q["st"]), "w": q["w"], "n0": int(I.sum()), "err": float(est - truth),
           "err_ind": err_ind,
           "cov": [bool(abs(est - truth) <= Z * np.sqrt(var) + 1e-9)], "cov2": [bool(abs(est - truth) <= Z * np.sqrt(vc) + 1e-9)],
           "hw": [float(Z * np.sqrt(var))], "hw2": [float(Z * np.sqrt(vc))], "rcov": rcov,
           "vratio": float(vco / vind) if vind > 0 else None}
    return pi_cell, rec, raised


# ------------------------------------------------------------------ one replay of a shared-synopsis method
def run_shared(p, qs, X, Yoh, U, V, mode, eps, masks, vals, rng, U2=None):
    lazy = mode != "p_only"  # p_only: the proactive plan alone, no query-time top-ups
    n, H = len(p), p.h.max() + 1
    h, day, week, ed = p.h.values, p.day.values, p.week.values, p.edit_day.values
    NW = int(week.max()) + 1
    cell = h * NW + week
    pi_cell = np.full(H * NW, PI_MIN); pi_plan = np.zeros(H * NW)  # pi_plan: what the proactive plan alone would buy
    # the proactive plan covers the current year; history cells (2024, YoY only) start at the floor and are topped up
    # lazily by the queries that need them (cold-start accounting)
    cur_cell = np.zeros(H * NW, bool); cur_cell[cell[day >= 0]] = True
    stats = {"lazy_raises": 0}

    def plan(upto):
        cur = (day >= 0) & (day <= upto)  # pilot and arrival rates from the current year only
        pil = np.flatnonzero(cur & (V < PILOT_N / max(1, cur.sum()) * (upto + 1) / 30))
        f, E = surrogate(X, Yoh, pil, rng)
        M, Md = moments(h, pil, E, H)
        nu = np.bincount(h[cur], minlength=H) / (upto + 1)
        if mode == "a_lazy":
            pi = np.full(H, PI_MIN)
        else:
            A, b = constraints(p, masks, vals, nu, f, M, Md, eps, upto)
            pi = solve(A, b, nu, uniform=(mode == "b7"))  # oracle price only rescales the objective
        # pilot share of tuples whose cheap-bundle argmax disagrees with the oracle (non-zero residuals), shrunk
        mis = (f[pil].argmax(1) != Yoh[pil].argmax(1)).astype(float); hp = h[pil]
        pres = np.array([((mis[hp == k]).sum() + KAPPA * mis.mean()) / ((hp == k).sum() + KAPPA) for k in range(H)])
        return f, M, Md, pil, pi, pres

    f, M, Md, pil1, pi, pres = plan(29)
    pilots = set(pil1.tolist())
    pi_plan = np.maximum(pi_plan, np.where(cur_cell, np.repeat(pi, NW), 0)); pi_cell = np.maximum(pi_cell, pi_plan)
    out = []
    for q in qs:
        tau = q["tau"]
        if tau == 61:  # monthly pilot refresh (Dec 1): new pilot draws from the independent stream, re-plan, never lower
            f, M, Md, pil2, pi, pres = plan(60); pilots |= set(pil2.tolist())
            pi_plan = np.maximum(pi_plan, np.where(cur_cell, np.repeat(pi, NW), 0)); pi_cell = np.maximum(pi_cell, pi_plan)
        if q["t"] == "T8":
            pi_cell, rec, raised = eval_t8(q, pi_cell, cell, h, f, M, Yoh, U, eps, pres, U2, lazy)
            stats["lazy_raises"] += raised; out.append(rec); continue
        idx = q["idx"]; N = len(idx); ci = cell[idx]; hi = h[idx]
        if q["t"] == "T1": targets = [("r", q["r"])]
        elif q["t"] == "T2": targets = [("r", r) for r in range(C)]
        else:
            fb = f[idx].mean(0); o = np.argsort(-fb); targets = [("d", o[0], o[1], max(fb[o[0]] - fb[o[1]], eps))]

        def pred_ok(pc):
            pt = pc[ci]; inv = 1 / pt - 1
            for tg in targets:
                if tg[0] == "r": v = (M[hi, tg[1]] * inv).sum() / N ** 2; lim = (eps / Z) ** 2
                else: v = (Md[hi, tg[1] * C + tg[2]] * inv).sum() / N ** 2; lim = (tg[3] / Z) ** 2
                if v > lim: return False
            if not guard_ok(pt, pres[hi]): return False
            return pt.sum() >= min(M_MIN, N)

        if lazy and not pred_ok(pi_cell):  # lazy top-up: raise the cells of this query by a common factor (structure-only rule)
            stats["lazy_raises"] += 1
            cs = np.unique(ci); k = 1.0
            while k < 1e4:
                k *= 1.15; trial = pi_cell.copy(); trial[cs] = np.minimum(1, pi_cell[cs] * k)
                if pred_ok(trial): break
            pi_cell = trial
        pt = pi_cell[ci]; I = U[idx] < pt
        fe = f[idx]; Ye = Yoh[idx]
        est = fe.mean(0) + ((Ye - fe) * (I / pt)[:, None]).sum(0) / N
        var = (((Ye - fe) ** 2) * (I * (1 - pt) / pt ** 2)[:, None]).sum(0) / N ** 2
        truth = Ye.mean(0)
        # conservative variant: never below the design variance implied by the pilot moments (guards zero-inflated HT)
        vc = np.maximum(var, (M[hi] * (1 / pt - 1)[:, None]).sum(0) / N ** 2)
        r0 = q["r"] if q["t"] == "T1" else 0
        rec = {"t": q["t"], "N": N, "st": "x".join(q["st"]), "w": q["w"], "n0": int(I.sum()), "err": float(est[r0] - truth[r0])}
        if q["t"] == "T3":
            o = np.argsort(-est); r1, r2 = o[0], o[1]
            e = (Ye[:, r1] - fe[:, r1]) - (Ye[:, r2] - fe[:, r2])
            vd = ((e ** 2) * I * (1 - pt) / pt ** 2).sum() / N ** 2
            sep = (est[r1] - est[r2]) > Z * np.sqrt(vd)
            rec.update(sep=bool(sep), correct=bool(truth.argmax() == r1) if sep else None,
                       cov=[bool(abs(est[r] - truth[r]) <= Z * np.sqrt(var[r]) + 1e-9) for r in range(C)],
                       cov2=[bool(abs(est[r] - truth[r]) <= Z * np.sqrt(vc[r]) + 1e-9) for r in range(C)])
        else:
            rs = [tg[1] for tg in targets]
            rec.update(cov=[bool(abs(est[r] - truth[r]) <= Z * np.sqrt(var[r]) + 1e-9) for r in rs],
                       cov2=[bool(abs(est[r] - truth[r]) <= Z * np.sqrt(vc[r]) + 1e-9) for r in rs], hw2=[float(Z * np.sqrt(vc[r])) for r in rs],
                       hw=[float(Z * np.sqrt(var[r])) for r in rs])
        out.append(rec)
    final_pi = pi_cell[cell]
    lab = U < final_pi
    pil_mask = np.zeros(n, bool); pil_mask[list(pilots)] = True
    edited = (ed >= 0) & (ed <= 91) & (ed > day)
    counts = {"cheap_scan": n, "oracle_main": int((lab & ~pil_mask).sum()), "pilot": int(pil_mask.sum()),
              "edit_cheap": int(edited.sum()), "edit_oracle": int((edited & lab).sum()),
              "plan_only": int(((U < pi_plan[cell]) & ~pil_mask).sum()),
              "oracle_hist": int((lab & ~pil_mask & (day < 0)).sum()), "n_hist": int((day < 0).sum())}
    return out, counts, stats


# ------------------------------------------------------------------ Ladder per query (b4) and with a label cache (b5)
def ladder_need(Yq, fq, targets_fn, eps, N):
    """Oracle sample size n for SRSWOR with FPC: Var = (1-n/N) S^2/n <= V  ->  n = N S^2 / (N V + S^2)."""
    need = {}
    for scan in (False, True):
        n = 0
        for (S2, V) in targets_fn(Yq, fq if scan else np.zeros_like(fq)):
            n = max(n, int(np.ceil(N * S2 / (N * V + S2))) if S2 > 0 else 0)
        need[scan] = min(n, N)
    return need


def run_ladder(p, qs, X, Yoh, rng, eps, c0, c1, cache, ideal=False):
    """ideal=True: the per-query moments are given for free (no pilot cost; surrogate fitted on all of R_q; min 10 labels),
    a lower bound on what any per-query planner can pay."""
    n = len(p); ed, day = p.edit_day.values, p.day.values
    or_day = np.full(n, -1); ch_day = np.full(n, -1)
    total = 0.0; n_or = 0; recs = []
    for q in qs:
        idx = q["idx"]; N = len(idx); tau = q["tau"]
        if q["t"] == "T8":
            c, k, parts = ladder_t8(q, X, Yoh, rng, eps, c0, c1, ideal, cache, or_day, ch_day, ed)
            total += c; n_or += k
            if parts is not None: recs.append(base_rec(q, parts, eps))
            continue
        m = min(10 if ideal else 300, N); pil = rng.choice(idx, m, replace=False)
        fq = X[idx] @ fit(X[idx] if ideal else X[pil], Yoh[idx] if ideal else Yoh[pil])  # the query's own pilot surrogate (Ladder)
        Yq = Yoh[idx]

        def tg(Yq, F):
            if q["t"] == "T1": return [(np.var(Yq[:, q["r"]] - F[:, q["r"]], ddof=1), (eps / Z) ** 2)]
            if q["t"] == "T2": return [(np.var(Yq[:, r] - F[:, r], ddof=1), (eps / Z) ** 2) for r in range(C)]
            fb = fq.mean(0); o = np.argsort(-fb); g = max(fb[o[0]] - fb[o[1]], eps)
            d = (Yq[:, o[0]] - Yq[:, o[1]]) - (F[:, o[0]] - F[:, o[1]])
            return [(np.var(d, ddof=1), (g / Z) ** 2)]

        if cache == "adapt":  # SemViews-style passive reuse adapted to aggregates (B6)
            total_q, k = semviews_adapt(q, idx, X, Yoh, rng, tg, eps, c0, c1, or_day, ch_day, ed, tau)
            total += total_q; n_or += k; continue
        need = ladder_need(Yq, fq, tg, eps, N)
        opt = {s: max(need[s], m) * c0 + (N if s else m) * c1 for s in (False, True)}
        scan = opt[True] < opt[False]
        if not cache:
            total += opt[scan]; n_or += max(need[scan], m); continue
        samp = np.concatenate([pil, rng.choice(np.setdiff1d(idx, pil), max(0, max(need[scan], m) - m), replace=False)])
        vo = (or_day[samp] >= 0) & ~((ed[samp] > or_day[samp]) & (ed[samp] <= tau))
        buy = samp[~vo]; total += len(buy) * c0; n_or += len(buy); or_day[buy] = tau
        cs = idx if scan else pil
        vc = (ch_day[cs] >= 0) & ~((ed[cs] > ch_day[cs]) & (ed[cs] <= tau))
        buyc = cs[~vc]; total += len(buyc) * c1; ch_day[buyc] = tau
        F = fq if scan else np.zeros_like(fq)
        recs.append(base_rec(q, [srs_part(idx, samp, F, Yoh, (Yoh[pil] - F[np.searchsorted(idx, pil)]).var(0, ddof=1))], eps))
    out = {"total": total, "n_oracle": n_or}
    if recs: out["quality"] = summarise_base(recs, eps)
    return out


def semviews_adapt(q, idx, X, Yoh, rng, tg, eps, c0, c1, or_day, ch_day, ed, tau):
    """B6: oracle labels left by earlier queries are exact known values; the remaining tuples of the query are
    certified with a fresh sample (200-tuple shaping sample, then SRS sized for the remainder), as SemViews does
    for filters. theta = (sum_known Y + N_r * mean_hat_r) / N, so the remainder may use variance V (N/N_r)^2."""
    known = (or_day[idx] >= 0) & ~((ed[idx] > or_day[idx]) & (ed[idx] <= tau))
    rem = idx[~known]; N, Nr = len(idx), len(rem)
    if Nr == 0: return 0.0, 0
    m = min(200, Nr); pil = rng.choice(rem, m, replace=False)
    fr = X[rem] @ fit(X[pil], Yoh[pil])
    scale = (N / Nr) ** 2
    need = {}
    for scan in (False, True):
        n = 0
        for S2, V in tg(Yoh[rem], fr if scan else np.zeros_like(fr)):
            Vr = V * scale; n = max(n, int(np.ceil(Nr * S2 / (Nr * Vr + S2))) if S2 > 0 else 0)
        need[scan] = min(n, Nr)
    opt = {s: max(need[s], m) * c0 + (Nr if s else m) * c1 for s in (False, True)}
    scan = opt[True] < opt[False]
    samp = np.concatenate([pil, rng.choice(np.setdiff1d(rem, pil), max(0, max(need[scan], m) - m), replace=False)])
    or_day[samp] = tau
    cs = rem if scan else pil
    vc = (ch_day[cs] >= 0) & ~((ed[cs] > ch_day[cs]) & (ed[cs] <= tau))
    buyc = cs[~vc]; ch_day[buyc] = tau
    return len(samp) * c0 + len(buyc) * c1, len(samp)


def srs_part(ii, samp, F, Yoh, Mfl=None):
    """Difference estimator from an SRS (without replacement) of oracle labels in ii; a census gives the exact value.
    Returns estimate, CLT variance from the sample residuals, the variance floored by the pilot moments Mfl (per class)."""
    N = len(ii); sel = np.isin(ii, samp); n = int(sel.sum()); Y = Yoh[ii]; R = Y - F
    est = F.mean(0) + (R[sel].mean(0) if n else 0)
    fpc = 1 - n / N
    var = fpc * (R[sel].var(0, ddof=1) if n > 1 else np.full(Y.shape[1], np.inf)) / max(n, 1)
    vfl = np.maximum(var, fpc * Mfl / max(n, 1)) if Mfl is not None else var
    return est, var, vfl, Y.mean(0), n


def base_rec(q, parts, eps):
    """Quality record of a per-query plan in the same format as run_shared (cov: CLT interval, cov2: floored interval).
    parts: list of srs_part outputs (one per window; T8 subtracts window 2 from window 1)."""
    rec = {"t": q["t"], "N": len(q["idx"]), "st": "x".join(q["st"]), "w": q["w"], "n0": int(sum(x[4] for x in parts))}
    ok = lambda e, v: bool(abs(e) <= Z * np.sqrt(v) + 1e-9)
    if q["t"] == "T8":
        r = q["r"]; (e1, v1, w1, t1, _), (e2, v2, w2, t2, _) = parts
        err = float(e1[r] - e2[r] - (t1[r] - t2[r])); var = v1[r] + v2[r]; vfl = w1[r] + w2[r]
        rec.update(err=err, cov=[ok(err, var)], cov2=[ok(err, vfl)], hw=[float(Z * np.sqrt(var))], hw2=[float(Z * np.sqrt(vfl))])
        return rec
    est, var, vfl, truth, _ = parts[0]
    r0 = q["r"] if q["t"] == "T1" else 0; rs = [q["r"]] if q["t"] == "T1" else list(range(C))
    rec["err"] = float(est[r0] - truth[r0])
    rec.update(cov=[ok(est[r] - truth[r], var[r]) for r in rs], cov2=[ok(est[r] - truth[r], vfl[r]) for r in rs],
               hw=[float(Z * np.sqrt(var[r])) for r in rs], hw2=[float(Z * np.sqrt(vfl[r])) for r in rs])
    return rec


def summarise_base(recs, eps):
    """Validity of a per-query baseline: CLT and floored coverage (all / worst slice-type x size group with >= 200 targets),
    share of targets with realised error within eps (T1/T2/T8 reported cell), share of T1/T2/T8 targets whose floored
    half-width is within eps (target attainment)."""
    df = pd.DataFrame(recs); out = {}
    for k, col in (("coverage", "cov"), ("coverage_cons", "cov2")):
        out[k] = float(np.concatenate(list(df[col])).mean())
    nb = pd.cut(df.N, [0, 100, 200, 500, 1e9], labels=["50-100", "100-200", "200-500", ">500"])
    out["groups2"] = {f"{st}|{b}": [int(len(c)), int(c.sum())] for (st, b), s in df.groupby([df.st, nb], observed=True)
                      for c in [np.concatenate(list(s["cov2"]))]}
    out["qerr"] = [round(x, 6) for x in df["err"]]
    out["acc"] = float(np.mean(np.abs(df.err.values) <= eps + 1e-12))
    hw2 = [x for r in df[df.t != "T3"].hw2 for x in r]
    out["target_met"] = float(np.mean(np.array(hw2) <= eps + 1e-12))
    out["target_met_110"] = float(np.mean(np.array(hw2) <= 1.1 * eps + 1e-12)); out["hw_ratio_median"] = float(np.median(hw2) / eps)
    for t in ["T1", "T2", "T3", "T8"]:
        s = df[df.t == t]
        if len(s): out[f"coverage_cons_{t}"] = float(np.concatenate(list(s["cov2"])).mean())
    return out


def run_b5shared(p, qs, X, Yoh, V, eps, c0, c1, rng, fallback=0.8):
    """B5-shared: a realistic per-query planner. One shared pilot (same size and refresh as TopUp's) gives surrogates and
    per-stratum residual moments; each query scans the cheap bundle (cached), sizes a fresh SRS of oracle labels from the
    pilot moments, and labels the whole slice exactly when the planned sample exceeds `fallback` of it. Labels are cached."""
    n = len(p); h, day, ed = p.h.values, p.day.values, p.edit_day.values; H = h.max() + 1
    or_day = np.full(n, -1); ch_day = np.full(n, -1); total = 0.0; n_or = 0

    def plan(upto):
        cur = (day >= 0) & (day <= upto)
        pil = np.flatnonzero(cur & (V < PILOT_N / max(1, cur.sum()) * (upto + 1) / 30))
        f, E = surrogate(X, Yoh, pil, rng); M, Md = moments(h, pil, E, H)
        return f, M, Md, pil

    def buy(samp, tau, price, book):
        v = (book[samp] >= 0) & ~((ed[samp] > book[samp]) & (ed[samp] <= tau))
        b = samp[~v]; book[b] = tau; return len(b), len(b) * price

    f, M, Md, pil = plan(29); k, c = buy(pil, 29, c0, or_day); total += c + len(pil) * c1; n_or += k; ch_day[pil] = 29
    recs = []
    for q in qs:
        tau = q["tau"]
        if tau == 61:
            f, M, Md, pil = plan(60); k, c = buy(pil, 60, c0, or_day); total += c; n_or += k
        parts = [(q["idx1"], 0.5), (q["idx2"], 0.5)] if q["t"] == "T8" else [(q["idx"], 1.0)]
        est = []
        for ii, share in parts:
            N = len(ii); hi = h[ii]; V0 = share * (eps / Z) ** 2
            if q["t"] == "T1": S2 = [M[hi, q["r"]].mean()]
            elif q["t"] == "T2": S2 = [M[hi, r].mean() for r in range(C)]
            elif q["t"] == "T8": S2 = [M[hi, q["r"]].mean()]
            else:
                fb = f[ii].mean(0); o = np.argsort(-fb); g = max(fb[o[0]] - fb[o[1]], eps)
                S2 = [Md[hi, o[0] * C + o[1]].mean() * (eps / g) ** 2]  # separation target rescaled to the eps budget
            need = max([int(np.ceil(N * s / (N * V0 + s))) if s > 0 else 0 for s in S2] + [min(10, N)])
            samp = ii if need > fallback * N else rng.choice(ii, need, replace=False)
            k, c = buy(samp, tau, c0, or_day); total += c; n_or += k
            k, c = buy(ii, tau, c1, ch_day); total += c
            est.append(srs_part(ii, samp, f[ii], Yoh, M[hi].mean(0)))
        recs.append(base_rec(q, est, eps))
    return {"total": total, "n_oracle": n_or, "quality": summarise_base(recs, eps)}


def ladder_t8(q, X, Yoh, rng, eps, c0, c1, ideal, cache, or_day, ch_day, ed):
    """Per-query plan for a contrast: one pilot over both windows, then an independent SRS per window,
    each window given half of the variance budget (the optimum for equal-size windows)."""
    idx = q["idx"]; r = q["r"]; tau = q["tau"]; m = min(10 if ideal else 300, len(idx))
    pil = rng.choice(idx, m, replace=False)
    W = fit(X[idx] if ideal else X[pil], Yoh[idx] if ideal else Yoh[pil])
    V = (eps / Z) ** 2 / 2
    plans = {}
    for scan in (False, True):
        ns = []
        for ii in (q["idx1"], q["idx2"]):
            Ni = len(ii); res = Yoh[ii, r] - ((X[ii] @ W)[:, r] if scan else 0)
            S2 = np.var(res, ddof=1); ns.append(min(Ni, int(np.ceil(Ni * S2 / (Ni * V + S2))) if S2 > 0 else 0))
        plans[scan] = ns
    opt = {s: max(sum(plans[s]), m) * c0 + (len(idx) if s else m) * c1 for s in (False, True)}
    scan = opt[True] < opt[False]
    if not cache: return opt[scan], max(sum(plans[scan]), m), None
    ws = [rng.choice(ii, min(len(ii), n_), replace=False) for ii, n_ in zip((q["idx1"], q["idx2"]), plans[scan])]
    samp = np.unique(np.concatenate([pil] + ws))
    vo = (or_day[samp] >= 0) & ~((ed[samp] > or_day[samp]) & (ed[samp] <= tau))
    buy = samp[~vo]; or_day[buy] = tau
    cs = idx if scan else pil
    vc = (ch_day[cs] >= 0) & ~((ed[cs] > ch_day[cs]) & (ed[cs] <= tau))
    buyc = cs[~vc]; ch_day[buyc] = tau
    parts = []
    for ii, s_ in zip((q["idx1"], q["idx2"]), ws):  # each window is estimated from its own SRS (independent designs)
        F = (X[ii] @ W) if scan else np.zeros((len(ii), C))
        Fp = (X[pil] @ W) if scan else np.zeros((len(pil), C))
        parts.append(srs_part(ii, s_, F, Yoh, (Yoh[pil] - Fp).var(0, ddof=1)))
    return len(buy) * c0 + len(buyc) * c1, len(buy), parts


# ------------------------------------------------------------------ driver
def summarise(recs):
    df = pd.DataFrame(recs)
    cov = np.concatenate(list(df["cov"])); out = {"targets": int(len(cov)), "coverage": float(cov.mean())}
    for t in ["T1", "T2", "T3", "T8"]:
        s = df[df.t == t]
        if len(s): out[f"coverage_{t}"] = float(np.concatenate(list(s["cov"])).mean())
    nb = pd.cut(df.N, [0, 100, 200, 500, 1e9], labels=["50-100", "100-200", "200-500", ">500"])
    grp = {}
    for (st, b), s in df.groupby([df.st, nb], observed=True):
        c = np.concatenate(list(s["cov"]))
        grp[f"{st}|{b}"] = [int(len(c)), int(c.sum())]  # pooled across runs later
    out["groups"] = grp
    out["qcov"] = [int(all(c)) for c in df["cov"]]  # per query: every target covered
    out["qerr"] = [round(x, 6) for x in df["err"]]
    out["coverage_cons"] = float(np.concatenate(list(df["cov2"])).mean())
    out["qcov2"] = [int(all(c)) for c in df["cov2"]]
    out["mean_halfwidth_cons"] = float(np.mean([x for r in df[~df.t.isin(["T3", "T8"])].hw2 for x in r]))
    g2 = {}
    for (st, b), s in df.groupby([df.st, nb], observed=True):
        c = np.concatenate(list(s["cov2"])); g2[f"{st}|{b}"] = [int(len(c)), int(c.sum())]
    out["groups2"] = g2
    t3 = df[df.t == "T3"]
    if len(t3):
        out["T3_separated_share"] = float(t3.sep.mean())
        out["T3_correct_when_separated"] = float(t3[t3.sep].correct.astype(float).mean()) if t3.sep.any() else None
    t8 = df[df.t == "T8"]
    if len(t8):
        out["T8"] = {}
        for var, s in t8.groupby("w"):
            rc = s.rcov.dropna()
            vr = s.vratio.dropna()
            out["T8"][var] = {"n": int(len(s)), "cov": [int(len(s)), int(sum(c[0] for c in s["cov"]))],
                              "cov2": [int(len(s)), int(sum(c[0] for c in s["cov2"]))],
                              "rcov": [int(len(rc)), int(rc.astype(bool).sum())],
                              "vratio_mean": float(vr.mean()) if len(vr) else None,
                              "hw_mean": float(np.mean([c[0] for c in s.hw]))}
    if len(t8) and "err_ind" in t8:
        out["t8err"] = [[w_, round(a_, 6), None if b_ is None else round(b_, 6)] for w_, a_, b_ in zip(t8.w, t8.err, t8.err_ind)]
    hw = [x for r in df[~df.t.isin(["T3", "T8"])].hw for x in r]
    out["mean_halfwidth"] = float(np.mean(hw)); out["mean_n0_per_query"] = float(df.n0.mean())
    # target attainment: share of T1/T2/T8 targets whose floored half-width is within eps
    hw2 = [x for r in df[df.t != "T3"].hw2 for x in r]
    out["target_met"] = float(np.mean(np.array(hw2) <= EPS + 1e-12))
    out["target_met_110"] = float(np.mean(np.array(hw2) <= 1.1 * EPS + 1e-12)); out["hw_ratio_median"] = float(np.median(hw2) / EPS)
    return out


def main():
    global EPS
    eps = arg("--eps", 0.05); EPS = eps; salts = arg("--salts", 40, int); wseeds = arg("--wseeds", 5, int)
    out_path = arg("--out", os.path.join(ROOT, "results", f"replay_eps{int(eps * 100):02d}.json"), str)
    p = load()
    c0_llm = float(p["c_" + ORACLE].mean()); c1 = float(sum(p["c_" + l].fillna(0).mean() for l in LABS))
    prices = {"llm_oracle": c0_llm, "human_oracle": 0.05}
    X = onehot(np.stack([p[l].values for l in LABS] + [p.code.values], 1))
    Y = p[ORACLE].values; Yoh = np.eye(C)[Y]
    res = {"eps": eps, "n_tuples": len(p), "c1_bundle": c1, "prices": prices, "runs": []}
    t0 = time.time()
    for ws in range(wseeds):
        qs, masks, vals = gen_workload(p, 1000 + ws)
        for s in range(salts):
            rng = np.random.default_rng(10_000 * ws + s)
            U = rng.random(len(p)); V = rng.random(len(p)); U2 = rng.random(len(p))
            run = {"wseed": ws, "salt": s, "n_queries": len(qs)}
            for mode in ["semview", "b7", "a_lazy", "p_only"]:
                recs, cnt, st = run_shared(p, qs, X, Yoh, U, V, mode, eps, masks, vals, np.random.default_rng(s), U2)
                cost = {}
                for pname, c0 in prices.items():
                    base = cnt["cheap_scan"] * c1 + (cnt["oracle_main"] + cnt["pilot"]) * c0
                    cost[pname] = {"total": base, "pilot": cnt["pilot"] * c0, "cheap_scan": cnt["cheap_scan"] * c1,
                                   "total_with_edits": base + cnt["edit_cheap"] * c1 + cnt["edit_oracle"] * c0}
                run[mode] = {"counts": cnt, "cost": cost, "lazy": st["lazy_raises"], "quality": summarise(recs)}
            for pname, c0 in prices.items():
                run["b4_" + pname] = run_ladder(p, qs, X, Yoh, np.random.default_rng(s), eps, c0, c1, cache=False)
                run["b5_" + pname] = run_ladder(p, qs, X, Yoh, np.random.default_rng(s), eps, c0, c1, cache=True)
                run["b4ideal_" + pname] = run_ladder(p, qs, X, Yoh, np.random.default_rng(s), eps, c0, c1, cache=False, ideal=True)
                run["b5ideal_" + pname] = run_ladder(p, qs, X, Yoh, np.random.default_rng(s), eps, c0, c1, cache=True, ideal=True)
                run["b6_" + pname] = run_ladder(p, qs, X, Yoh, np.random.default_rng(s), eps, c0, c1, cache="adapt")
                run["b5shared_" + pname] = run_b5shared(p, qs, X, Yoh, V, eps, c0, c1, np.random.default_rng(s))
            res["runs"].append(run)
            print(f"ws={ws} salt={s} done {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), default=float)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
