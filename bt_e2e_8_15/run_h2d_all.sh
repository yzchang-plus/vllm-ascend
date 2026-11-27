#!/bin/bash
# H2D E2E, node-145 model set (cards 0-11): fresh convert -> fast restore -> compare.
#   q3_30b_w8a8 : TP1, dev 0,        util 0.60  (quantized MoE)
#   q3_30b_bf16 : TP2, devs 1,2,     util 0.60  (bf16 MoE)
#   dsv4flash   : TP8, devs 4-11,    util 0.70  (DeepseekV4 MoE+MLA, 292G)
# Needs run_phase.sh + compare.py in THIS directory (self-contained harness).
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
# Snapshots go to the 6T weights volume: the /vllm-workspace volume is too small
# for full-weight blobs (12G + 47G + 292G).
CKPT_BASE=/home/cyz/weights/bt_e2e_ckpt
mkdir -p "$CKPT_BASE"

# tag|model_path|dev_list|port|tp|util
MODELS=(
  "q3_30b_w8a8|/home/cyz/weights/Qwen3-30B-A3B-W8A8|0|8311|1|0.60"
  "q3_30b_bf16|/home/cyz/weights/Qwen3-30B-A3B|1,2|8312|2|0.60"
  "dsv4flash|/home/cyz/weights/DeepSeekv4-flash-W8A8|4,5,6,7,8,9,10,11|8313|8|0.70"
)

run_all_phases() {  # $1 = A|B
  local P=$1
  local PIDS=()
  local i=0
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r TAG MPATH DEV PORT TP UTIL <<< "$entry"
    bash "$DIR/run_phase.sh" "${P}_${TAG}" "$MPATH" "$DEV" "$PORT" "$CKPT_BASE/$TAG" "$DIR" "$UTIL" \
      --tensor-parallel-size "$TP" \
      > "$DIR/logs/${P}_${TAG}.runner.log" 2>&1 &
    PIDS+=($!)
    i=$((i+1)); sleep $((i*2))
  done
  local FAILED=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
  return $FAILED
}

mkdir -p "$DIR/logs" "$DIR/out"
echo "===== Phase A (fresh convert + save) $(date) ====="
run_all_phases A
echo "===== Phase B (fast restore) $(date) ====="
run_all_phases B

echo "===== Compare $(date) ====="
SUMMARY=$DIR/summary_h2d.txt
: > "$SUMMARY"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r TAG MPATH DEV PORT TP UTIL <<< "$entry"
  if python3 "$DIR/compare.py" "$DIR/out/A_${TAG}.out" "$DIR/out/B_${TAG}.out" --label-a fresh --label-b restore \
      > "$DIR/logs/compare_${TAG}.log" 2>&1; then
    RES="PASS"
  else
    RES="FAIL"
  fi
  RESTORE_LINE=$(grep -h "restore_snapshot rank=" "$DIR/logs/B_${TAG}.server.log" 2>/dev/null | head -1)
  READY_A=$(grep -oP 'server ready in \K[0-9]+' "$DIR/logs/A_${TAG}.runner.log" 2>/dev/null | head -1)
  READY_B=$(grep -oP 'server ready in \K[0-9]+' "$DIR/logs/B_${TAG}.runner.log" 2>/dev/null | head -1)
  FALLBACK=$(grep -c "falling back" "$DIR/logs/B_${TAG}.server.log" 2>/dev/null)
  echo -e "$TAG\t$RES\tdevs=$DEV tp=$TP\treadyA=${READY_A:-?}s\treadyB=${READY_B:-?}s\tfallback=$FALLBACK" >> "$SUMMARY"
  echo "$RESTORE_LINE" >> "$SUMMARY"
done

cat "$SUMMARY"
echo "done. summary: $SUMMARY"
