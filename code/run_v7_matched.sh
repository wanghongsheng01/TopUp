#!/bin/zsh
# v7: matched-validity comparisons on the same 200 runs as Table I: B5-shared with sample inflation gamma,
# A-lazy whose lazy rule asks for variance <= limit/g, and a capped stratified (BlinkDB/Congressional-style) design.
cd "$(dirname "$0")/.."
PY=python3
export SV_LABS=ds-flash-b10,kimi-k2.6-b10 SV_YOY=1 SV_SKIP=1 SV_EXTRA="a_lazy@1.5,a_lazy@2,a_lazy@2.5,cong" SV_GAMMAS="1.5,2,2.5,3"
for d in rma amazon amazon2; do
  if [ $d = rma ]; then E=(SV_C=4); else E=(SV_DATA=$d SV_C=3); fi
  for e in 05 10; do env $E $PY -I code/replay.py --eps 0.$e --out results/v7/matched_${d}_eps$e.json > results/v7/log_${d}_$e.txt 2>&1 & done
done
wait; echo ALLDONE
