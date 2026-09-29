#!/bin/bash
# ARM SPEC chain: B's evaluation first (positive control of the path; stop if it fails), then train K, train T,
# then evaluate all three and print the registered verdict. Log: results/spec/chain.log. Powergate-watched.
set -uo pipefail
if [ -z "${SPEC_CHAIN_SNAPSHOT:-}" ]; then
  d=$(mktemp -d); cp "$0" "$d/chain.sh"; SPEC_CHAIN_SNAPSHOT=1 exec /bin/bash "$d/chain.sh" "$@"
fi
cd /Users/andreashad04/money/vesuv
PY=/Users/andreashad04/money/vesuv/env/bin/python3
LOG=results/spec/chain.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
say "chain pid $$ nice $(ps -o nice= -p $$ | tr -d ' ')"
$PY ops/spec/eval.py >> "$LOG" 2>&1; say "eval (B only) rc=$?"
$PY - >> "$LOG" 2>&1 <<'PYCHK'
import json; r = json.load(open("results/spec/spec_scores.json"))["rows"]["B"]
S = {"w00": 0.7680, "ag144": 0.7599, "ag174": 0.7614, "p0500p2": 0.7537}
bad = {k: round(r[k]["real"]["auc"], 4) for k in S if round(r[k]["real"]["auc"], 4) != S[k]}
assert not bad, f"B does not reproduce the stored AUCs: {bad}"
print("B reproduces the stored AUCs on all 4 crops")
PYCHK
rc=$?; say "positive control rc=$rc"; [ $rc = 0 ] || { say "STOP: positive control failed"; exit 1; }
for arm in K T; do
  $PY ops/spec/train_sd.py --arm $arm --out results/spec/$arm >> "results/spec/train_$arm.log" 2>&1
  say "train $arm rc=$?"
done
$PY ops/spec/eval.py >> "$LOG" 2>&1; say "eval (all) rc=$?"
