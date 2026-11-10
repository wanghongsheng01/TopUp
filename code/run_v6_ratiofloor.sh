#!/bin/zsh
# v6: the six main replays again with a floored delta-method ratio interval (rcov2); costs and all other fields unchanged.
cd "$(dirname "$0")/.."
PY=python3
export SV_LABS=ds-flash-b10,kimi-k2.6-b10 SV_YOY=1
for d in rma amazon amazon2; do
  if [ $d = rma ]; then E=(SV_C=4); else E=(SV_DATA=$d SV_C=3); fi
  for e in 05 10; do env $E $PY -I code/replay.py --eps 0.$e --out results/v6/final_${d}_eps$e.json > results/v6/log_${d}_$e.txt 2>&1 & done
done
wait; echo ALLDONE
