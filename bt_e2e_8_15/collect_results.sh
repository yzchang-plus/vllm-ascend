#!/bin/bash
# Collect all E2E results into one paste-ready report: results_report.txt
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
REPORT=$DIR/results_report.txt

{
echo "==================== bt_e2e_8_15 E2E results $(date) ===================="
echo "serving stack: $(git -C /vllm-workspace/vllm-ascend log --oneline -1 2>/dev/null)"
pip show openyuanrong-datasystem 2>/dev/null | grep ^Version | sed 's/^/TransferEngine: /'
echo ""

for M in rfork h2d d2d; do
  S=$DIR/summary_${M}.txt
  echo "---------- summary_${M}.txt ----------"
  if [[ -f $S ]]; then
    cat "$S"
    P=$(grep -c "PASS" "$S" || true)
    F=$(grep -c "FAIL" "$S" || true)
    echo "[${M}] PASS=${P} FAIL=${F}"
  else
    echo "(missing - matrix not run yet)"
  fi
  echo ""
done

echo "---------- rfork receiver evidence (per model) ----------"
for L in "$DIR"/logs/R_*_recv.server.log; do
  [[ -f "$L" ]] || continue
  TAG=$(basename "$L" .server.log); TAG=${TAG#R_}; TAG=${TAG%_recv}
  echo "[$TAG]"
  grep -hiE "rfork|transfer" "$L" | head -6
  echo ""
done

echo "---------- h2d restore lines ----------"
grep -h "restore_snapshot rank=" "$DIR"/logs/B_*.server.log 2>/dev/null
echo ""

echo "---------- d2d HCCL recv lines ----------"
grep -h "HCCL recv time" "$DIR"/logs/N_*_client.server.log 2>/dev/null
echo ""

echo "==================== end ===================="
} | tee "$REPORT"

echo ""
echo "report written: $REPORT"
echo "paste it back with: cat $REPORT"
