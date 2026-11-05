#!/bin/zsh
# Final replays for the ICDE 2027 R2 submission (all fixes in: window bug, float tolerance, B6, cold-start history).
# Usage: zsh code/run_final.sh rma | amazon | amazon2
set -e
cd "$(dirname "$0")/.."
PY=python3
export SV_YOY=1
case "$1" in
  rma)
    export SV_LABS=ds-flash-b10,kimi-k2.6-b10 SV_C=4
    $PY -I code/replay.py --eps 0.05 --out results/final_rma_eps05.json > results/log_final_rma05.txt 2>&1 &
    $PY -I code/replay.py --eps 0.10 --out results/final_rma_eps10.json > results/log_final_rma10.txt 2>&1 &
    SV_MINSUP=200 $PY -I code/replay.py --eps 0.05 --salts 20 --wseeds 2 --out results/final_rma_minsup200_eps05.json > results/log_final_rma_ms.txt 2>&1 &
    wait ;;
  amazon|amazon2)
    export SV_DATA=$1 SV_LABS=ds-flash-b10,kimi-k2.6-b10 SV_C=3
    $PY -I code/replay.py --eps 0.05 --out results/final_${1}_eps05.json > results/log_final_${1}05.txt 2>&1 &
    $PY -I code/replay.py --eps 0.10 --out results/final_${1}_eps10.json > results/log_final_${1}10.txt 2>&1 &
    SV_MINSUP=200 $PY -I code/replay.py --eps 0.05 --salts 20 --wseeds 2 --out results/final_${1}_minsup200_eps05.json > results/log_final_${1}_ms.txt 2>&1 &
    wait ;;
esac
echo done "$1"
