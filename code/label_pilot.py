"""Label the de-identified RMA pilot with one catalog labeler (SemView pilot, attribute A1 = level-1 return reason).

Usage: python label_pilot.py <labeler> [--limit N]
Reads ONLY the memo_deid column of private/pilot_deid.parquet, re-runs the residual scan first and aborts on any hit.
Writes private/labels/<labeler>.csv with key, pred (0..3 or -1), cost_usd, in_tok, out_tok.
Stops when this project's spend on the labeler's model reaches DAILY_CAP today.
"""
import os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm

PRIV = os.path.join(HERE, "..", "private")
OUT = os.path.join(PRIV, "labels")
os.makedirs(OUT, mode=0o700, exist_ok=True)
DAILY_CAP = float(os.environ.get("SV_DAILY_CAP", 18.0))  # USD per model per day for this project (shared cap is $100 across four sessions)
LABELS = ["quality", "shipment", "customer", "others"]
CODE = {1: 0, 2: 1, 4: 2, 3: 3}  # problem_type level-1 id -> label index
QUESTION = (
    "This is a customer-service note about a return/refund/replacement case (RMA) of an office-furniture company "
    "(standing desks, chairs, sofas, beds, storage racks). Classify the PRIMARY reason for the case:\n"
    "- quality: product defect or damage on arrival (broken/damaged/scratched parts, missing or wrong parts in the box, "
    "not working, error codes, noise, wobbling, uneven, cannot install, holes not matching);\n"
    "- shipment: logistics problems (package lost, not delivered, delivered but not found, delay, stuck in transit, "
    "returned to sender, refused delivery, wrong item picked by the warehouse, out of stock);\n"
    "- customer: customer-side reasons with no defect (changed mind, no longer needed, not as expected, size/colour/"
    "comfort preference, ordered by mistake, fraud);\n"
    "- others: price adjustments, discount codes, fees (customs, VAT, return fee), compensation for reviews, promotions, "
    "test orders, or no reason stated.")
LABELERS = {"glm-flash-b10": ("glm-flash", 10), "glm-air-b10": ("glm-air", 10), "ds-flash-b10": ("ds-flash", 10),
            "kimi-k2.6-b10": ("kimi-k2.6", 10), "kimi-k3-b10": ("kimi-k3", 10), "ds-pro-b10": ("ds-pro", 10),
            "ds-pro-think": ("ds-pro-think", 1), "ds-pro-think-b5": ("ds-pro-think", 5), "ds-flash-think-b10": ("ds-flash-think", 10)}
WORKERS = {"glm-flash": 2, "glm-air": 3, "kimi-k2.6": 3, "kimi-k3": 2}


def parse(text):
    t = text.strip().lower().strip(" .\"'*`")
    hits = [(m.start(), i) for i, l in enumerate(LABELS) for m in [re.search(r"(^|\W)" + l + r"($|\W)", t)] if m]
    return min(hits)[1] if hits else -1


def prompt(texts):
    opts = ", ".join(LABELS)
    noun = "texts" if os.environ.get("SV_DATA") else "notes"
    if len(texts) == 1:
        return [{"role": "system", "content": "You are a careful data annotator."},
                {"role": "user", "content": f"{QUESTION}\nAnswer with exactly one of: {opts}.\n\nText:\n\"\"\"{texts[0]}\"\"\"\n\nAnswer:"}]
    body = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    return [{"role": "system", "content": "You are a careful data annotator."},
            {"role": "user", "content": f"{QUESTION}\nFor each of the {len(texts)} {noun} below, answer with exactly one of: {opts}.\n"
                                        f"Return one line per item in the form `[k] answer`, nothing else.\n\n{body}"}]


def main():
    lab = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    model, B = LABELERS[lab]
    suf = os.environ.get("SV_WINDOW", ""); suf = "_" + suf if suf else ""
    ds = os.environ.get("SV_DATA", "")
    global OUT, LABELS, QUESTION
    if ds:  # public dataset: text is already public (no enterprise data), classes and question from classes.json
        import json
        base = os.path.join(HERE, "..", "public", ds)
        spec = json.load(open(os.path.join(base, "classes.json"))); LABELS, QUESTION = spec["labels"], spec["question"]
        df = pd.read_parquet(os.path.join(base, f"prepared{suf}.parquet"), columns=["key", "memo_deid"])
        OUT = os.path.join(base, "labels"); os.makedirs(OUT, exist_ok=True)
    else:
        import deidentify_ext as D  # enterprise path only: residual scan before any external call
        df = pd.read_parquet(os.path.join(PRIV, f"pilot_deid{suf}.parquet"), columns=["key", "memo_deid"])
        res = D.residual_scan(df.memo_deid.tolist())
        assert not any(res.values()), f"residual scan failed, refusing to call any API: {res}"
    if limit: df = df.sample(limit, random_state=0) if limit < len(df) else df
    df = df.reset_index(drop=True)
    texts = df.memo_deid.str.replace(r"\s+", " ", regex=True).str.slice(0, int(os.environ.get("SV_MAXCHARS", 600))).tolist()
    path = os.path.join(OUT, f"{lab}{suf}.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["key", "pred", "cost_usd", "in_tok", "out_tok"])
    good = set(done.key[done.cost_usd.notna()])
    todo = [i for i in range(len(df)) if df.key[i] not in good]
    groups = [todo[i:i + B] for i in range(0, len(todo), B)]
    print(f"{lab}: {len(todo)} tuples to label in {len(groups)} calls; spent today on {model}: ${llm.spent_today(model):.3f}", flush=True)
    stop = [False]

    def work(g):
        if stop[0]: return [(df.key[i], -1, np.nan, np.nan, np.nan) for i in g]
        if llm.spent_today(model) >= DAILY_CAP:
            stop[0] = True; print("daily cap reached", flush=True)
            return [(df.key[i], -1, np.nan, np.nan, np.nan) for i in g]
        try:
            r = llm.chat(model, prompt([texts[i] for i in g]), max_tokens=8000 if "think" in model else 400)
            if B == 1:
                preds = [parse(r["text"])]
            else:
                found = dict((int(k), v) for k, v in re.findall(r"\[(\d+)\]\s*([^\n]+)", r["text"]))
                preds = [parse(found.get(j + 1, "")) for j in range(len(g))]
            u = r["usage"]; c = llm.cost(model, u) / len(g)
            return [(df.key[i], p, c, u.get("prompt_tokens", 0) / len(g), u.get("completion_tokens", 0) / len(g)) for i, p in zip(g, preds)]
        except Exception as e:
            print("ERR", lab, str(e)[:160], flush=True)
            return [(df.key[i], -1, np.nan, np.nan, np.nan) for i in g]

    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", WORKERS.get(model, 16)))) as ex:
        rows = [x for res in ex.map(work, groups) for x in res]
    out = pd.DataFrame(rows, columns=["key", "pred", "cost_usd", "in_tok", "out_tok"])
    out = pd.concat([done[~done.key.isin(out.key)], out]).reset_index(drop=True)
    out.to_csv(path, index=False); os.chmod(path, 0o600)
    ok = out[out.cost_usd.notna()]
    print(f"{lab}: labelled {len(ok)}/{len(out)} unparsed={(ok.pred < 0).mean():.3f} cost=${ok.cost_usd.sum():.3f} "
          f"($/1k={1000 * ok.cost_usd.mean():.3f}) time={time.time() - t0:.0f}s; spent today on {model}: ${llm.spent_today(model):.3f}", flush=True)


if __name__ == "__main__":
    main()
