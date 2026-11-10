# TopUp: Workload-Aware Semantic Synopses — Artifact

This repository accompanies the ICDE 2027 submission *TopUp: Maintaining Workload-Aware Semantic Synopses of LLM Labels with Error Guarantees over Evolving Data* (Hongsheng Wang and Dazhen Deng).

## Contents

| Path | What it is |
|---|---|
| `code/replay.py` | Offline replay of all methods: TopUp, B4–B7, B5-shared, capped stratified (`cong`), A-lazy, plan-only, per-query references, and the matched-validity variants (B5-shared with sample inflation `SV_GAMMAS`, A-lazy with lazy-rule margin `a_lazy@g`); workload generator (templates T1–T4); data scale-up by tuple replication (`SV_SCALE`); planner (convex allocation by multiplicative dual updates); PRN synopsis with lazy top-ups; HT and floored intervals; contrasts and ratios |
| `code/label_pilot.py`, `code/llm.py` | Labelling with LLM labellers, with an on-disk cache and a per-call USD ledger. Keys are read from a local JSON file (`LLM_KEYS`) and are not included |
| `code/prep_public.py`, `code/download_amazon/` | Download and preparation of the public datasets (Amazon Reviews 2023, Musical Instruments and Industrial & Scientific) |
| `code/analyze_replay.py`, `code/make_numbers.py`, `code/make_extra.py`, `code/make_scale.py`, `code/make_sens.py`, `code/make_figs.py`, `code/text_numbers.py` | Tables, numbers and figures of the paper, regenerated from the replay output |
| `code/run_final.sh`, `code/run_v5_basecov.sh`, `code/run_v6_ratiofloor.sh`, `code/run_v7_matched.sh`, `code/run_scale_g.sh`, `code/run_sens.sh` | The replay configurations behind Tables I–VI: main runs, matched validity (200 runs), scale-up (k = 1, 2, 4, 8) and sensitivity (pilot size, shrinkage) |
| `code/plan_scale.py` | Planner scalability on synthetic instances (up to 50,000 strata) |
| `data/<dataset>/labels/*.csv` | Per-review labels of every LLM labeller with token-based cost (`key, pred, cost_usd, in_tok, out_tok`) |
| `data/<dataset>/prepared*_notext.csv` | Review ids, arrival day, slice dimensions and the human oracle (star-rating class). Review text is not redistributed; re-create it with `code/download_amazon/` |

The enterprise return-management dataset used as a case study is private and is not included. All results on it are reported as aggregates only.

## Reproducing the public-data results

1. Re-create the review text (needs network access to the Amazon Reviews 2023 files): see `code/download_amazon/`, then `python code/prep_public.py amazon cur hist`.
2. Labels and costs are included, so no LLM calls are needed to replay. To relabel instead: `SV_DATA=amazon python code/label_pilot.py ds-flash-b10` (needs your own API keys).
3. Replay, e.g. `SV_DATA=amazon SV_C=3 SV_YOY=1 SV_LABS=ds-flash-b10,kimi-k2.6-b10 python code/replay.py --eps 0.05 --out results/final_amazon_eps05.json`. One run of 5 workload seeds × 40 PRN salts takes about 20 minutes on a laptop CPU.
4. Summarise: `python code/analyze_replay.py results/final_amazon_eps05.json`.
5. Matched-validity comparison, scale-up and sensitivity: `zsh code/run_v7_matched.sh`, `zsh code/run_scale_g.sh`, `zsh code/run_sens.sh` (enterprise cells need the private data and are skipped without it), then `python code/make_scale.py` and `python code/make_sens.py`.

## Licence

Code: MIT. The Amazon Reviews 2023 data are by Hou et al.; please cite their paper and follow the dataset's terms.
