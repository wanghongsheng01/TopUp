"""Parallel byte-range streaming filter for Amazon Reviews 2023 jsonl files.
Never stores the raw file; keeps only needed fields.
usage: stream_filter.py {reviews|meta} URL OUT_PARQUET LOGFILE [threads]
"""
import sys, json, re, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

mode, url, out, logf = sys.argv[1:5]
nthreads = int(sys.argv[5]) if len(sys.argv) > 5 else 16
KEEP = None
if len(sys.argv) > 6:
    KEEP = set(pd.read_parquet(sys.argv[6], columns=["parent_asin"])["parent_asin"])
CH = 8 * 1024 * 1024

# windows (ms, UTC)
def ms(s):
    return int(pd.Timestamp(s, tz="UTC").value // 10**6)
WIN = [(ms("2021-10-01"), ms("2022-01-01")), (ms("2022-10-01"), ms("2023-01-01")),
       ]
TSRE = re.compile(rb'"timestamp": (\d+)')
PARE = re.compile(rb'"parent_asin": "([^"]+)"')

def head_size():
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])

def fetch(i, size):
    s = i * CH; e = min(size, s + CH) - 1
    for att in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={s}-{e}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            if len(b) == e - s + 1:
                return b
        except Exception as ex:
            err = ex
        time.sleep(2 + 3 * att)
    raise RuntimeError(f"chunk {i} failed")

lock = threading.Lock()
rows = []
quarter_counts = {}

def handle_line(line):
    if not line.strip():
        return
    if mode == "reviews":
        m = TSRE.search(line)
        if not m:
            return
        t = int(m.group(1))
        # count all reviews per quarter (cheap) for sparsity check
        keep = any(a <= t < b for a, b in WIN)
        if not keep:
            return
        d = json.loads(line)
        r = (d.get("user_id"), d.get("parent_asin"), d.get("asin"), int(d["timestamp"]),
             d.get("rating"), d.get("title") or "", d.get("text") or "", bool(d.get("verified_purchase")))
    else:
        if KEEP is not None:
            m = PARE.search(line)
            if m and m.group(1).decode() not in KEEP:
                return
        d = json.loads(line)
        if KEEP is not None and d.get("parent_asin") not in KEEP:
            return
        cats = d.get("categories") or []
        r = (d.get("parent_asin"), d.get("store"), json.dumps(cats, ensure_ascii=False),
             str(d.get("price")) if d.get("price") is not None else None, d.get("main_category"))
    with lock:
        rows.append(r)

def process(i, b, heads, tails):
    first = b.find(b"\n"); last = b.rfind(b"\n")
    if first == -1:
        heads[i] = b; tails[i] = b""; return
    heads[i] = b[:first]          # fragment before first newline (belongs to previous line)
    tails[i] = b[last + 1:]       # fragment after last newline
    for line in b[first + 1:last].split(b"\n"):
        handle_line(line)

def main():
    size = head_size(); n = (size + CH - 1) // CH
    heads, tails = {}, {}
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(nthreads) as ex:
        futs = {ex.submit(fetch, i, size): i for i in range(n)}
        for f in as_completed(futs):
            i = futs[f]
            process(i, f.result(), heads, tails)
            done += 1
            if done % 10 == 0 or done == n:
                el = time.time() - t0
                with open(logf, "a") as L:
                    L.write(f"{done}/{n} chunks {done*CH/1e6/el:.2f} MB/s elapsed {el:.0f}s rows {len(rows)}\n")
    # stitch boundaries: line = heads[0] (start of file) ; tails[i]+heads[i+1]
    handle_line(heads[0])
    for i in range(n - 1):
        handle_line(tails[i] + heads[i + 1])
    handle_line(tails[n - 1])
    if mode == "reviews":
        cols = ["user_id", "parent_asin", "asin", "timestamp", "rating", "title", "text", "verified_purchase"]
    else:
        cols = ["parent_asin", "store", "categories_json", "price_raw", "main_category"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_parquet(out, index=False)
    with open(logf, "a") as L:
        L.write(f"DONE rows {len(df)} size {size} elapsed {time.time()-t0:.0f}s\n")

main()
