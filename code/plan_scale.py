"""Planner scalability: the workload-level allocation of Theorem 3 on synthetic instances with up to tens of thousands
of strata. The solver is the one of replay.solve (multiplicative dual updates + greedy repair) on a sparse constraint
matrix; at the RMA size it is checked against the dense replay.solve.

Instance: strata = markets x categories; slices = single markets, single categories and market-category pairs;
for every slice, window length (7, 30 days) and value (C=4) one variance constraint, plus one top-k separation and
two comparison constraints per slice, as in replay.constraints. Arrival rates and residual moments are drawn from
gamma laws fitted loosely to RMA (rates of 0.5-50 tuples/day, second moments 0.02-0.2).

Usage: python plan_scale.py  -> results/plan_scale.json
"""
import json, os, sys, time
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = sys.argv[:1]
import replay  # noqa: E402  (dense reference solver, constants PI_MIN, Z)

C, Z, EPS, MIN_SUPPORT = 4, replay.Z, 0.05, 50


def instance(n_mk, n_cat, seed=0):
    rng = np.random.default_rng(seed)
    H = n_mk * n_cat
    mk = np.repeat(np.arange(n_mk), n_cat); cat = np.tile(np.arange(n_cat), n_mk)
    nu = rng.gamma(0.6, 6.0, H) + 0.05                        # tuples per day per stratum (heavy-tailed)
    M = rng.gamma(2.0, 0.04, (H, C)); Md = rng.gamma(2.0, 0.06, H)
    slices = [np.where(mk == m)[0] for m in range(n_mk)] + [np.where(cat == c)[0] for c in range(n_cat)] + [np.array([h]) for h in range(H)]
    rows, cols, vals, b = [], [], [], []
    q = 0
    for s in slices:
        for wl in (7, 30):
            Nh = wl * nu[s]; N = Nh.sum()
            if N < MIN_SUPPORT: continue
            for r in range(C):
                rows += [q] * len(s); cols += list(s); vals += list(Nh * M[s, r] / N ** 2); b.append((EPS / Z) ** 2); q += 1
            rows += [q] * len(s); cols += list(s); vals += list(Nh * Md[s] / N ** 2); b.append((max(rng.uniform(0, .3), EPS) / Z) ** 2); q += 1
        N1 = 30 * nu[s].sum()
        if N1 >= MIN_SUPPORT:
            for ov in (0, 23):
                w = nu[s] * 2 * (30 - ov) / N1 ** 2
                rows += [q] * len(s); cols += list(s); vals += list(w * M[s, 0]); b.append((EPS / Z) ** 2); q += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(q, H))
    cost = nu * 0.05                                               # oracle price x arrival rate (steady-state daily cost)
    return A, np.array(b), cost


def solve_sparse(A, b, cost_h, iters=1500):
    """replay.solve with a sparse A (same updates, same repair)."""
    An = sp.diags(1 / b) @ A; AnT = An.T.tocsr()
    lam = np.ones(len(b)) / len(b); best, bestc = None, np.inf
    cst = np.maximum(cost_h, 1e-12); PI_MIN = replay.PI_MIN
    for it in range(iters):
        pi = np.clip(np.sqrt((AnT @ lam) / cst), PI_MIN, 1.0)
        g = An @ (1 / pi - 1)
        if g.max() <= 1.0 + 1e-6 and (cost_h * pi).sum() < bestc: best, bestc = pi.copy(), (cost_h * pi).sum()
        lam *= np.exp(0.5 * np.clip(g - 1, -1, 3)); lam = np.maximum(lam, 1e-12)
    pi = best if best is not None else pi
    repairs = 0
    for _ in range(100000):
        g = An @ (1 / pi - 1); q = int(g.argmax())
        if g[q] <= 1.0 + 1e-9: break
        row = An.getrow(q); s = row.indices; a = row.data; k = 1.0
        while (a @ (1 / np.minimum(1, pi[s] * k)) - a.sum()) > 1 and k < 1e6: k *= 1.1
        pi[s] = np.minimum(1, pi[s] * k); repairs += 1
    return pi, repairs


def main():
    out = []
    for n_mk, n_cat in ((13, 9), (40, 25), (100, 50), (200, 100), (400, 125)):
        A, b, cost = instance(n_mk, n_cat)
        t = time.perf_counter(); pi, rep = solve_sparse(A, b, cost); dt = time.perf_counter() - t
        viol = float((A @ (1 / pi - 1) / b).max())
        rec = {"strata": A.shape[1], "constraints": A.shape[0], "nnz": int(A.nnz), "seconds": round(dt, 3),
               "repairs": rep, "max_constraint_ratio": round(viol, 6), "daily_cost": float(cost @ pi)}
        if A.shape[1] <= 200:  # dense reference
            t = time.perf_counter(); pd_ = replay.solve(A.toarray(), b, cost); rec["dense_seconds"] = round(time.perf_counter() - t, 3)
            rec["max_abs_diff_vs_dense"] = float(np.abs(pd_ - pi).max())
        print(rec, flush=True); out.append(rec)
    json.dump(out, open(os.path.join(HERE, "..", "results", "plan_scale.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
