"""Build cur.parquet / hist.parquet from streamed intermediates.
usage: build.py DATA_DIR CATEGORY
"""
import sys, json, re, hashlib
import numpy as np, pandas as pd

D, CATEGORY = sys.argv[1], sys.argv[2]
I = f"{D}/intermediate"
SEED = 20261105
rv = pd.read_parquet(f"{I}/reviews_win.parquet")
meta = pd.read_parquet(f"{I}/meta.parquet").drop_duplicates("parent_asin")
st = {"category": CATEGORY, "n_reviews_streamed_in_windows": int(len(rv)), "n_meta_items": int(len(meta))}

def clean(s):
    s = s or ""
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"\[\[VIDEOID:[^\]]*\]\]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

rv = rv.drop_duplicates(["user_id", "parent_asin", "timestamp"])
rv["body"] = rv["text"].map(clean)
rv["ttl"] = rv["title"].map(clean)
rv = rv[rv["body"].str.len() >= 20].copy()
rv["ts"] = pd.to_datetime(rv["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
rv["q"] = rv["ts"].dt.to_period("Q").astype(str)
st["eligible_per_quarter"] = {k: int(v) for k, v in rv["q"].value_counts().sort_index().items()}

def sample(df, n, rng):
    if len(df) <= n:
        return df.copy()
    idx = rng.choice(len(df), n, replace=False)
    s = df.iloc[np.sort(idx)].copy()
    days_all = set(df["ts"].dt.date)
    rest = df.drop(s.index)
    missing = days_all - set(s["ts"].dt.date)
    for d in sorted(missing):  # swap in one row for each missing day from the largest day
        cand = rest[rest["ts"].dt.date == d]
        big = s["ts"].dt.date.value_counts().idxmax()
        drop = s[s["ts"].dt.date == big].sample(1, random_state=SEED).index
        s = pd.concat([s.drop(drop), cand.sample(1, random_state=SEED)])
    st.setdefault("days_swapped_in", 0)
    st["days_swapped_in"] += len(missing)
    return s

rng = np.random.default_rng(SEED)
cur_q, hist_q = "2022Q4", "2021Q4"
cur = sample(rv[rv["q"] == cur_q].sort_values(["timestamp", "user_id", "parent_asin"]).reset_index(drop=True), 20000, rng)
hist = sample(rv[rv["q"] == hist_q].sort_values(["timestamp", "user_id", "parent_asin"]).reset_index(drop=True), 12000, rng)

def price(x):
    try:
        v = float(str(x).replace("$", "").replace(",", ""))
        return v if np.isfinite(v) and v > 0 else np.nan
    except Exception:
        return np.nan
meta["price"] = meta["price_raw"].map(price)
meta["cats"] = meta["categories_json"].map(lambda s: json.loads(s) if s else [])

def prep(df):
    df = df.merge(meta[["parent_asin", "store", "price", "cats"]], on="parent_asin", how="left")
    df["meta_found"] = df["cats"].notna()
    df["cats"] = df["cats"].map(lambda c: c if isinstance(c, list) else [])
    df["key"] = (df["user_id"] + "|" + df["parent_asin"] + "|" + df["timestamp"].astype(str)).map(
        lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
    df["text"] = [clean((t + ". " + b) if t else b) for t, b in zip(df["ttl"], df["body"])]
    df["rating"] = df["rating"].round().astype(int)
    df["gold"] = np.select([df["rating"] <= 2, df["rating"] == 3], [0, 1], 2).astype(int)
    df["categories"] = df["cats"].map(lambda c: " > ".join(c))
    df["area_raw"] = df["cats"].map(lambda c: c[1] if len(c) > 1 else (c[0] if c else "unknown"))
    df["store"] = df["store"].map(lambda s: clean(s) if isinstance(s, str) and clean(s) else None)
    df["cat1"] = pd.cut(df["price"], [0, 10, 25, 60, np.inf], right=False,
                        labels=["<10", "10-25", "25-60", ">=60"]).astype(str).replace("nan", "unknown")
    return df

cur, hist = prep(cur), prep(hist)
both = pd.concat([cur, hist])
top_area = both["area_raw"].value_counts().index[:6].tolist()
top_store = both["store"].dropna().value_counts().index[:25].tolist()
for df in (cur, hist):
    df["area"] = df["area_raw"].where(df["area_raw"].isin(top_area), "other")
    df["market"] = [s if s in top_store else f"other-{a}" for s, a in zip(df["store"], df["area"])]
cols = ["key", "ts", "text", "rating", "verified_purchase", "parent_asin", "store", "price", "categories",
        "gold", "area", "market", "cat1"]
for name, df in (("cur", cur), ("hist", hist)):
    out = df.sort_values("ts")[cols].reset_index(drop=True)
    assert out["key"].is_unique
    out.to_parquet(f"{D}/{name}.parquet", index=False)
    pd_ = out["ts"].dt.date.value_counts()
    st[name] = {
        "rows": int(len(out)), "window": cur_q if name == "cur" else hist_q,
        "population_eligible": int((rv["q"] == (cur_q if name == "cur" else hist_q)).sum()),
        "days": int(len(pd_)), "per_day_min_median_max": [int(pd_.min()), float(pd_.median()), int(pd_.max())],
        "gold_shares": {int(k): round(float(v), 4) for k, v in out["gold"].value_counts(normalize=True).sort_index().items()},
        "median_text_chars": float(out["text"].str.len().median()),
        "meta_found_share": round(float(df["meta_found"].mean()), 4),
        "store_missing_share": round(float(out["store"].isna().mean()), 4),
        "area_counts": {k: int(v) for k, v in out["area"].value_counts().items()},
        "market_top": {k: int(v) for k, v in out["market"].value_counts().head(35).items()},
        "n_market_values": int(out["market"].nunique()),
        "cat1_counts": {k: int(v) for k, v in out["cat1"].value_counts().items()},
        "verified_share": round(float(out["verified_purchase"].mean()), 4),
    }
st["top_areas"] = top_area
st["top_stores"] = top_store
json.dump(st, open(f"{D}/stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(st, indent=1, ensure_ascii=False)[:6000])
