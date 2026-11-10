#!/bin/zsh
# Scale-up with matched-validity B5-shared (gamma 2.5, 3): every tuple replicated k times (same strata and labels, fresh keys and random numbers); shared methods + B5-shared only.
cd "$(dirname "$0")/.."
PY=python3
export SV_LABS=ds-flash-b10,kimi-k2.6-b10 SV_YOY=1 SV_SKIP=1 SV_GAMMAS=2.5,3
for k in 1 2 4 8; do
  SV_C=4 SV_SCALE=$k $PY -I code/replay.py --eps 0.05 --salts 8 --wseeds 5 --out results/scale_g/rma_k${k}_eps05.json > results/scale_g/log_rma_k$k.txt 2>&1 &
  SV_DATA=amazon2 SV_C=3 SV_SCALE=$k $PY -I code/replay.py --eps 0.05 --salts 8 --wseeds 5 --out results/scale_g/amazon2_k${k}_eps05.json > results/scale_g/log_amazon2_k$k.txt 2>&1 &
done
wait; echo ALLDONE
