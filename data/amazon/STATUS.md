# Amazon Reviews 2023 -- Musical_Instruments -- approximate-aggregate benchmark slice

Built without any LLM call. Scripts: `../_scripts_amazon/` (stream_filter.py, build.py, make_status.py).

## Source

- Dataset: Amazon Reviews 2023, McAuley Lab (Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952, 2024).
- Card: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 (read via https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023); site https://amazon-reviews-2023.github.io/
- Reviews: https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Musical_Instruments.jsonl
- Metadata: https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Musical_Instruments.jsonl
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

- Reviews streamed in the kept windows (raw, before cleaning): 288,972; metadata items: 55,910
- Eligible (text >= 20 chars after cleaning, dedup) per quarter: {'2021Q4': 67620, '2022Q4': 77184, '2023Q1': 80520, '2023Q2': 32205, '2023Q3': 7934}

| | cur | hist |
|---|---|---|
| window | 2022Q4 | 2021Q4 |
| population eligible | 77,184 | 67,620 |
| rows | 20,000 | 12,000 |
| days covered | 92 | 92 |
| per-day min/median/max | 137/216.5/340 | 75/130.5/213 |
| gold shares neg/neu/pos | 0.167 / 0.073 / 0.760 | 0.200 / 0.072 / 0.728 |
| median text length (chars) | 197.0 | 171.0 |
| metadata join hit rate | 1.0 | 1.0 |
| store missing share | 0.0006 | 0.0007 |
| verified share | 0.8175 | 0.9327 |
| distinct market values | 32 | 32 |
| cat1 counts | {'>=60': 5452, '25-60': 5156, '10-25': 4900, 'unknown': 2719, '<10': 1773} | {'>=60': 3147, '25-60': 2749, '10-25': 2588, 'unknown': 2527, '<10': 989} |

### Area counts

| area | cur | hist |
|---|---|---|
| Instrument Accessories | 6551 | 4100 |
| other | 3816 | 2717 |
| Microphones & Accessories | 3584 | 1860 |
| Live Sound & Stage | 2597 | 1334 |
| Studio Recording Equipment | 1316 | 750 |
| Electronic Music, DJ & Karaoke | 1092 | 579 |
| Drums & Percussion | 1044 | 660 |

### Top markets (cur)

| market | cur | hist |
|---|---|---|
| other-Instrument Accessories | 5146 | 3178 |
| other-Microphones & Accessories | 3117 | 1544 |
| other-other | 3023 | 2243 |
| other-Live Sound & Stage | 2207 | 1134 |
| other-Studio Recording Equipment | 1182 | 692 |
| other-Drums & Percussion | 1008 | 635 |
| other-Electronic Music, DJ & Karaoke | 791 | 462 |
| Pyle | 461 | 289 |
| Fender | 442 | 261 |
| Donner | 358 | 143 |
| D'Addario | 180 | 114 |
| JIM DUNLOP | 151 | 106 |
| JYX | 145 | 28 |
| YAMAHA | 142 | 117 |
| Musiclily | 140 | 80 |
| CAHAYA | 132 | 83 |
| D'Addario Accessories | 119 | 76 |
| Ernie Ball | 116 | 66 |
| Rockville | 106 | 58 |
| Amazon Basics | 103 | 73 |
| Behringer | 101 | 47 |
| MAONO | 97 | 41 |
| LEKATO | 93 | 45 |
| Hola! Music | 83 | 35 |
| Moukey | 78 | 55 |
| GLEAM | 74 | 43 |
| Gator | 73 | 50 |
| InnoGear | 72 | 43 |
| Best Choice Products | 71 | 67 |
| Shure | 66 | 73 |
| TONOR | 65 | 56 |
| Ibanez | 58 | 63 |

## Deviations from the request

- **Category switched: Musical_Instruments instead of Office_Products.** The mirror delivered only ~1-1.6 MB/s aggregate (link-capped; 16-48 parallel range connections did not help) and the disk had ~3 GB (later <1 GB) free. Office_Products is 5.78 GB reviews + 2.15 GB metadata (~80+ min); Musical_Instruments (1.56 GB + 0.63 GB) was the largest catalogue of branded, priced, hierarchically-categorised products that fit the time budget. Arts_Crafts_and_Sewing (3.9+2.2 GB) was also infeasible; Industrial_and_Scientific is built separately in ../amazon2/.
- Raw files were never stored: parallel HTTP range streaming with an on-the-fly timestamp/parent_asin filter (metadata kept only for parent_asins seen in the windows).
- Text cleaning beyond whitespace normalisation: `<br />` tags and `[[VIDEOID:...]]` tokens replaced by spaces; the >=20-character rule is applied to the cleaned review body (title excluded). Exact duplicates on (user_id, parent_asin, timestamp) dropped.
- ts is stored as naive UTC. Rating rounded to int. Price <= 0 or unparsable -> NaN -> cat1 "unknown". Missing store -> market "other-<area>".
- Day coverage: uniform sample without replacement; a swap-in step would add one row for any uncovered day (see days_swapped_in in stats.json).
- The dataset card states no licence (see above).
