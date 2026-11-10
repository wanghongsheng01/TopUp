#!/bin/zsh
# Sensitivity of hand-set constants (40 runs, eps=0.05): pilot size 200/800 (default 400) and moment shrinkage 0/50 (default 10).
cd "$(dirname "$0")/.."
PY=python3
export SV_LABS=ds-flash-b10,kimi-k2.6-b10 SV_YOY=1 SV_SKIP=1
for d in rma amazon; do
  if [ $d = rma ]; then E=(SV_C=4); else E=(SV_DATA=$d SV_C=3); fi
  for v in "SV_PILOT=200" "SV_PILOT=800" "SV_KAPPA=0" "SV_KAPPA=50"; do
    env $E $v $PY -I code/replay.py --eps 0.05 --salts 8 --wseeds 5 --out results/sens/${d}_${v}.json > results/sens/log_${d}_${v}.txt 2>&1 &
  done
done
wait; echo ALLDONE
