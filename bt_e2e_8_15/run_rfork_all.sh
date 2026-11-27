#!/bin/bash
# rfork D2D E2E, node-145 model set (cards 0-11).
# Starts the mock planner once (capacity 8: multi-rank TP instances request
# one lease per rank), then per model: seed (fresh load, publishes) ->
# receiver (TransferEngine transfer) -> probe both -> compare outputs.
#
#   q3_30b_w8a8: TP1, seed dev0 / recv dev1          (processed-layout path, quantized)
#   q3_30b_bf16 : TP2, seed devs 2,3 / recv devs 4,5 (checkpoint-layout path)
#
# Products: logs/R_<tag>_*.log / out/R_<tag>_*.out / summary_rfork.txt
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
PLANNER_PORT=1223
PLANNER_LOG=$DIR/logs/planner.log

mkdir -p "$DIR/logs" "$DIR/out"

# ---- planner (mock, from the v0.18.0 clone's examples; capacity 8 for multi-rank) ----
python3 /vllm-workspace/vllm-ascend/examples/rfork/rfork_planner.py --host 127.0.0.1 --port "$PLANNER_PORT" --default-resource-points 8 \
  > "$PLANNER_LOG" 2>&1 &
PLANNER_PID=$!
PLANNER_UP=0
for _ in $(seq 1 15); do
  if (: < "/dev/tcp/127.0.0.1/${PLANNER_PORT}") 2>/dev/null; then PLANNER_UP=1; break; fi
  sleep 1
done
if [[ $PLANNER_UP != 1 ]]; then
  echo "FATAL: planner did not start, see $PLANNER_LOG"; tail -20 "$PLANNER_LOG"
  kill "$PLANNER_PID" 2>/dev/null; exit 1
fi
echo "planner up (port $PLANNER_PORT, pid $PLANNER_PID)"
trap 'kill "$PLANNER_PID" 2>/dev/null' EXIT

# tag|model_path|dev_seed_list|dev_recv_list|tp|util
MODELS=(
  "q3_30b_w8a8|/home/cyz/weights/Qwen3-30B-A3B-W8A8|0|1|1|0.60"
  "q3_30b_bf16|/home/cyz/weights/Qwen3-30B-A3B|2,3|4,5|2|0.60"
)

SUMMARY=$DIR/summary_rfork.txt
: > "$SUMMARY"

IDX=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r TAG MPATH DEV_S DEV_R TP UTIL <<< "$entry"
  PORT_S=$((8361 + IDX))
  PORT_R=$((8371 + IDX))
  echo "===== [$TAG] seed=devs[$DEV_S] receiver=devs[$DEV_R] tp=$TP $(date) ====="
  if bash "$DIR/run_rfork.sh" "$TAG" "$MPATH" "$DEV_S" "$DEV_R" "$PORT_S" "$PORT_R" "$DIR" "$TP" "$UTIL" \
      > "$DIR/logs/R_${TAG}.runner.log" 2>&1; then
    RES=PASS
  else
    RES=FAIL
  fi
  SEED_LINE=$(grep -h "seed ready in" "$DIR/logs/R_${TAG}.runner.log" 2>/dev/null | head -1)
  RECV_LINE=$(grep -h "receiver ready in" "$DIR/logs/R_${TAG}.runner.log" 2>/dev/null | head -1)
  echo -e "$TAG\t$RES\tdev_seed=$DEV_S\tdev_recv=$DEV_R\ttp=$TP" >> "$SUMMARY"
  echo "$SEED_LINE" >> "$SUMMARY"
  echo "$RECV_LINE" >> "$SUMMARY"
  tail -5 "$DIR/logs/R_${TAG}.runner.log"
  IDX=$((IDX + 1))
done

cat "$SUMMARY"
echo "done. summary: $SUMMARY"
