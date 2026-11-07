# TopUp: Workload-Aware Semantic Synopses — Artifact

This repository accompanies the ICDE 2027 submission *TopUp: Workload-Aware Semantic Synopses of LLM Labels with Asymptotic Error Guarantees over Evolving Data* (Hongsheng Wang).

## Contents

| Path | What it is |
|---|---|
| `code/replay.py` | Offline replay of all methods: TopUp, B4–B7, A-lazy, per-query lower bounds; workload generator (templates T1–T3, T8); planner (convex allocation by multiplicative dual updates); PRN synopsis with lazy top-ups; HT and floored intervals; contrasts and ratios |
| `code/label_pilot.py`, `code/llm.py` | Labelling with LLM labellers, with an on-disk cache and a per-call USD ledger. Keys are read from a local JSON file (`LLM_KEYS`) and are not included |
| `code/prep_public.py`, `code/download_amazon/` | Download and preparation of the public datasets (Amazon Reviews 2023, Musical Instruments and Industrial & Scientific) |
| `code/analyze_replay.py`, `code/make_numbers.py`, `code/make_figs.py` | Tables, numbers and figures of the paper, regenerated from the replay output |
| `data/<dataset>/labels/*.csv` | Per-review labels of every LLM labeller with token-based cost (`key, pred, cost_usd, in_tok, out_tok`) |
| `data/<dataset>/prepared*_notext.csv` | Review ids, arrival day, slice dimensions and the human oracle (star-rating class). Review text is not redistributed; re-create it with `code/download_amazon/` |

The enterprise return-management dataset used as a case study is private and is not included. All results on it are reported as aggregates only.

## Reproducing the public-data results

1. Re-create the review text (needs network access to the Amazon Reviews 2023 files): see `code/download_amazon/`, then `python code/prep_public.py amazon cur hist`.
2. Labels and costs are included, so no LLM calls are needed to replay. To relabel instead: `SV_DATA=amazon python code/label_pilot.py ds-flash-b10` (needs your own API keys).
3. Replay, e.g. `SV_DATA=amazon SV_C=3 SV_YOY=1 SV_LABS=ds-flash-b10,kimi-k2.6-b10 python code/replay.py --eps 0.05 --out results/final_amazon_eps05.json`. One run of 5 workload seeds × 40 PRN salts takes about 20 minutes on a laptop CPU.
4. Summarise: `python code/analyze_replay.py results/final_amazon_eps05.json`.

## Licence

Code: MIT. The Amazon Reviews 2023 data are by Hou et al.; please cite their paper and follow the dataset's terms.
