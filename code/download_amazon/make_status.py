"""Write STATUS.md from stats.json. usage: make_status.py DATA_DIR DEVIATIONS_TXT"""
import sys, json
D = sys.argv[1]
st = json.load(open(f"{D}/stats.json"))
dev = open(sys.argv[2]).read().strip()
C = st["category"]
B = "https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
L = []
w = L.append
w(f"# Amazon Reviews 2023 -- {C} -- approximate-aggregate benchmark slice\n")
w("Built without any LLM call. Scripts: `../_scripts_amazon/` (stream_filter.py, build.py, make_status.py).\n")
w("## Source\n")
w("- Dataset: Amazon Reviews 2023, McAuley Lab (Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952, 2024).")
w("- Card: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 (read via https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023); site https://amazon-reviews-2023.github.io/")
w(f"- Reviews: {B}/review_categories/{C}.jsonl")
w(f"- Metadata: {B}/meta_categories/meta_{C}.jsonl")
w("- Files were streamed with parallel HTTP range requests and filtered on the fly; the raw files were never stored (disk had ~3 GB free).\n")
w("## Licence / terms\n")
w("- The dataset card (README.md YAML and body) declares **no licence field and no terms-of-use text**; it only asks users to cite Hou et al. 2024 (BibTeX given) and gives a contact for research collaborations. amazon-reviews-2023.github.io likewise shows no licence statement.")
w("- The companion code repository github.com/hyp1231/AmazonReviews2023 is MIT-licensed (applies to the code, not explicitly to the data).")
w("- Review text is Amazon user-generated content; treat as research-use only, cite the paper, and do not redistribute beyond what is needed for the benchmark.\n")
w("## Files\n")
w("- `cur.parquet` -- W_cur, `hist.parquet` -- W_hist; `stats.json` -- all numbers below; intermediates were deleted after the build (disk constraint).")
w("- Columns: key (sha1(user_id|parent_asin|timestamp)[:16]), ts (naive UTC datetime), text (title + '. ' + text, whitespace-normalised), rating (1..5), verified_purchase, parent_asin, store, price, categories (' > '-joined), gold (0 neg 1-2 / 1 neutral 3 / 2 pos 4-5), area (categories[1] else [0] else 'unknown'; top 6, rest 'other'), market (top 25 stores, rest 'other-<area>'), cat1 (price band <10, 10-25, 25-60, >=60, unknown).")
w("- Top-6 areas and top-25 stores are computed on cur+hist combined so the mapping is identical in both files.\n")
w("## Counts\n")
w(f"- Reviews streamed in the kept windows (raw, before cleaning): {st['n_reviews_streamed_in_windows']:,}; metadata items: {st['n_meta_items']:,}")
w(f"- Eligible (text >= 20 chars after cleaning, dedup) per quarter: {st['eligible_per_quarter']}\n")
w("| | cur | hist |\n|---|---|---|")
c, h = st["cur"], st["hist"]
for lab, f in [("window", lambda x: x["window"]), ("population eligible", lambda x: f"{x['population_eligible']:,}"),
               ("rows", lambda x: f"{x['rows']:,}"), ("days covered", lambda x: x["days"]),
               ("per-day min/median/max", lambda x: "/".join(str(v) for v in x["per_day_min_median_max"])),
               ("gold shares neg/neu/pos", lambda x: " / ".join(f"{x['gold_shares'].get(k, 0):.3f}" for k in ("0", "1", "2"))),
               ("median text length (chars)", lambda x: x["median_text_chars"]),
               ("metadata join hit rate", lambda x: x["meta_found_share"]),
               ("store missing share", lambda x: x["store_missing_share"]),
               ("verified share", lambda x: x["verified_share"]),
               ("distinct market values", lambda x: x["n_market_values"]),
               ("cat1 counts", lambda x: x["cat1_counts"])]:
    w(f"| {lab} | {f(c)} | {f(h)} |")
w("\n### Area counts\n")
w("| area | cur | hist |\n|---|---|---|")
for a in list(c["area_counts"]):
    w(f"| {a} | {c['area_counts'].get(a, 0)} | {h['area_counts'].get(a, 0)} |")
w("\n### Top markets (cur)\n")
w("| market | cur | hist |\n|---|---|---|")
for m, v in c["market_top"].items():
    w(f"| {m} | {v} | {h['market_top'].get(m, '-')} |")
w("\n## Deviations from the request\n")
w(dev)
open(f"{D}/STATUS.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
