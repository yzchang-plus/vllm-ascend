#!/bin/bash
# D2D (netloader HCCL P2P) E2E, node-145 model set.
# TP1 only: the multi-rank SOURCE config was never validated in this harness
# (Nov runs were TP1); the W8A8 model fits a single card.
#   q3_30b_w8a8: donor dev0 / client dev1, util 0.60
set -u
DIR=$(cd "$(dirname "$0")" && pwd)

# tag|model_path|dev_donor|dev_client|util
MODELS=(
  "q3_30b_w8a8|/home/cyz/weights/Qwen3-30B-A3B-W8A8|0|1|0.60"
)

mkdir -p "$DIR/logs" "$DIR/out"
SUMMARY=$DIR/summary_d2d.txt
: > "$SUMMARY"

IDX=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r TAG MPATH DEV_D DEV_C UTIL <<< "$entry"
  PORT_D=$((8321 + IDX))
  PORT_C=$((8331 + IDX))
  LP=$((19100 + IDX))
  echo "===== [$TAG] donor=dev$DEV_D client=dev$DEV_C $(date) ====="
  if bash "$DIR/run_netloader.sh" "$TAG" "$MPATH" "$DEV_D" "$DEV_C" "$PORT_D" "$PORT_C" "$LP" "$DIR" "$UTIL" \
      > "$DIR/logs/N_${TAG}.runner.log" 2>&1; then
    RES=PASS
  else
    RES=FAIL
  fi
  RECV=$(grep -h "HCCL recv time" "$DIR/logs/N_${TAG}_client.server.log" 2>/dev/null | head -1)
  echo -e "$TAG\t$RES\tdev_donor=$DEV_D\tdev_client=$DEV_C" >> "$SUMMARY"
  echo "$RECV" >> "$SUMMARY"
  tail -5 "$DIR/logs/N_${TAG}.runner.log"
  IDX=$((IDX + 1))
done

cat "$SUMMARY"
echo "done. summary: $SUMMARY"
