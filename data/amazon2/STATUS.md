# Amazon Reviews 2023 -- Industrial_and_Scientific -- approximate-aggregate benchmark slice

Built without any LLM call. Scripts: `../_scripts_amazon/` (stream_filter.py, build.py, make_status.py).

## Source

- Dataset: Amazon Reviews 2023, McAuley Lab (Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952, 2024).
- Card: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 (read via https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023); site https://amazon-reviews-2023.github.io/
- Reviews: https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Industrial_and_Scientific.jsonl
- Metadata: https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Industrial_and_Scientific.jsonl
- Files were streamed with parallel HTTP range requests and filtered on the fly; the raw files were never stored (disk had ~3 GB free).

## Licence / terms

- The dataset card (README.md YAML and body) declares **no licence field and no terms-of-use text**; it only asks users to cite Hou et al. 2024 (BibTeX given) and gives a contact for research collaborations. amazon-reviews-2023.github.io likewise shows no licence statement.
- The companion code repository github.com/hyp1231/AmazonReviews2023 is MIT-licensed (applies to the code, not explicitly to the data).
- Review text is Amazon user-generated content; treat as research-use only, cite the paper, and do not redistribute beyond what is needed for the benchmark.

## Files

- `cur.parquet` -- W_cur, `hist.parquet` -- W_hist; `stats.json` -- all numbers below; intermediates were deleted after the build (disk constraint).
- Columns: key (sha1(user_id|parent_asin|timestamp)[:16]), ts (naive UTC datetime), text (title + '. ' + text, whitespace-normalised), rating (1..5), verified_purchase, parent_asin, store, price, categories (' > '-joined), gold (0 neg 1-2 / 1 neutral 3 / 2 pos 4-5), area (categories[1] else [0] else 'unknown'; top 6, rest 'other'), market (top 25 stores, rest 'other-<area>'), cat1 (price band <10, 10-25, 25-60, >=60, unknown).
- Top-6 areas and top-25 stores are computed on cur+hist combined so the mapping is identical in both files.

## Counts

- Reviews streamed in the kept windows (raw, before cleaning): 374,742; metadata items: 84,521
- Eligible (text >= 20 chars after cleaning, dedup) per quarter: {'2021Q4': 162102, '2022Q4': 175713}

| | cur | hist |
|---|---|---|
| window | 2022Q4 | 2021Q4 |
| population eligible | 175,713 | 162,102 |
| rows | 20,000 | 12,000 |
| days covered | 92 | 92 |
| per-day min/median/max | 125/217.5/279 | 86/129.0/177 |
| gold shares neg/neu/pos | 0.185 / 0.072 / 0.743 | 0.250 / 0.070 / 0.679 |
| median text length (chars) | 167.0 | 144.0 |
| metadata join hit rate | 1.0 | 1.0 |
| store missing share | 0.0009 | 0.0007 |
| verified share | 0.8259 | 0.9531 |
| distinct market values | 32 | 32 |
| cat1 counts | {'10-25': 7755, '25-60': 3975, '<10': 3926, '>=60': 2260, 'unknown': 2084} | {'10-25': 4336, '<10': 2344, 'unknown': 2325, '25-60': 1800, '>=60': 1195} |

### Area counts

| area | cur | hist |
|---|---|---|
| other | 9862 | 5324 |
| Professional Medical Supplies | 2032 | 2534 |
| Adhesives, Sealants & Lubricants | 1848 | 853 |
| Additive Manufacturing Products | 1625 | 786 |
| Janitorial & Sanitation Supplies | 1592 | 933 |
| Material Handling Products | 1527 | 760 |
| Test, Measure & Inspect | 1514 | 810 |

### Top markets (cur)

| market | cur | hist |
|---|---|---|
| other-other | 9475 | 5129 |
| other-Professional Medical Supplies | 1688 | 2063 |
| other-Adhesives, Sealants & Lubricants | 1494 | 683 |
| other-Material Handling Products | 1464 | 720 |
| other-Test, Measure & Inspect | 1431 | 774 |
| other-Janitorial & Sanitation Supplies | 1318 | 758 |
| other-Additive Manufacturing Products | 1248 | 621 |
| Gorilla | 180 | 84 |
| iHealth | 145 | 29 |
| SUNLU | 111 | 23 |
| uxcell | 109 | 58 |
| Bissell | 102 | 54 |
| MED PRIDE | 102 | 55 |
| DIYMAG | 96 | 42 |
| Comgrow | 83 | 33 |
| Neosmuk | 80 | 44 |
| Generic | 79 | 18 |
| DEPSTECH | 77 | 33 |
| Rubbermaid | 70 | 50 |
| BEAUTURAL | 70 | 53 |
| ANYCUBIC | 70 | 21 |
| EUREKA | 64 | 44 |
| Loctite | 64 | 37 |
| Command | 63 | 27 |
| OVERTURE | 61 | 47 |
| 3M | 59 | 35 |
| FURemover | 55 | 39 |
| ELEGOO | 51 | 43 |
| Borje | 37 | 80 |
| NNPCBT | 24 | 80 |
| akgk | 23 | 78 |
| Wanwane | 7 | 145 |

## Deviations from the request

- **Second category: Industrial_and_Scientific** (2.35 GB reviews + 1.13 GB metadata), as requested by the coordinator after the first build (amazon/ = Musical_Instruments, which replaced Office_Products for bandwidth reasons: mirror ~1-1.6 MB/s aggregate, disk <1-3 GB free). Office_Products and Arts_Crafts_and_Sewing (3.9+2.2 GB) were infeasible in the time budget.
- Raw files were never stored: parallel HTTP range streaming with an on-the-fly timestamp/parent_asin filter (metadata kept only for parent_asins seen in the windows).
- Text cleaning beyond whitespace normalisation: `<br />` tags and `[[VIDEOID:...]]` tokens replaced by spaces; the >=20-character rule is applied to the cleaned review body (title excluded). Exact duplicates on (user_id, parent_asin, timestamp) dropped.
- ts is stored as naive UTC. Rating rounded to int. Price <= 0 or unparsable -> NaN -> cat1 "unknown". Missing store -> market "other-<area>".
- Day coverage: uniform sample without replacement; a swap-in step would add one row for any uncovered day (see days_swapped_in in stats.json).
- The dataset card states no licence (see above).
