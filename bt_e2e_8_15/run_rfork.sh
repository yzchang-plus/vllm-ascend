#!/bin/bash
# rfork D2D E2E (YuanRong TransferEngine, planner-mediated seed -> receiver).
#
# Usage: run_rfork.sh <tag> <model_path> <dev_seed_list> <dev_recv_list> \
#            <api_port_seed> <api_port_recv> <workdir> <tp> <gpu_util> [extra vllm flags...]
#
#   dev lists are comma-separated physical chips, e.g. "0,1" for TP2.
#
# Flow: seed starts first (no seed registered yet -> default load -> publishes as seed
#       via start_seed_service), then the receiver starts with the SAME model_url /
#       model_deploy_strategy_name (same seed key) and transfers via TransferEngine.
#
# Verdict: the RECEIVER log must NOT contain "RFork transfer failed" (any failure in the
#       transfer path -- including seed-unavailable -- logs this warning and falls back
#       to the default loader), AND seed vs receiver probe outputs must be identical.
# Note: the SEED instance logging "RFork transfer failed: seed is not available" is
#       EXPECTED (it is the first instance); only the receiver's log decides.
#
# Serving stack: the v0.18.0 port (serving-capable in this container). CANN's PYTHONPATH
# entries are preserved by appending, not overwriting.
set -u

TAG=$1; MODEL=$2; DEV_S=$3; DEV_R=$4; PORT_S=$5; PORT_R=$6; WORKDIR=$7; TP=$8; UTIL=$9; shift 9
EXTRA_ARGS=("$@")

DIR=$(cd "$(dirname "$0")" && pwd)
PLANNER_URL=${RFORK_PLANNER_URL:-http://127.0.0.1:1223}
SEED_SETTLE_SEC=${RFORK_SEED_SETTLE_SEC:-30}

export PYTHONPATH="/vllm-workspace/vllm-ascend${PYTHONPATH:+:$PYTHONPATH}"

LOG_S=$WORKDIR/logs/R_${TAG}_seed.server.log
LOG_R=$WORKDIR/logs/R_${TAG}_recv.server.log
OUT_S=$WORKDIR/out/R_${TAG}_seed.out
OUT_R=$WORKDIR/out/R_${TAG}_recv.out
mkdir -p "$WORKDIR/logs" "$WORKDIR/out"

hbm_mb() {
  npu-smi info 2>/dev/null | awk -v c="$1" '
    $3 == c && $3 ~ /^[0-9]+$/ {
      line = $0; hbm = ""
      while (match(line, /[0-9]+ *\/ *[0-9]+/)) {
        hbm = substr(line, RSTART, RLENGTH)
        line = substr(line, RSTART + RLENGTH)
      }
      gsub(/ /, "", hbm); split(hbm, a, "/"); print a[1]; exit
    }'
}

wait_hbm_free() {  # $1=comma dev list $2=label
  for _ in $(seq 1 36); do
    ALL_FREE=1
    for d in ${1//,/ }; do
      HBM=$(hbm_mb "$d")
      [[ -z "$HBM" || "$HBM" -ge 10000 ]] && ALL_FREE=0
    done
    [[ $ALL_FREE == 1 ]] && break
    sleep 5
  done
  echo "[$2] devs=$1 HBM(last)=${HBM:-?}MB"
}

wait_ready() {  # $1=port $2=pgid $3=timeout_x10s
  for _ in $(seq 1 "${3:-120}"); do
    sleep 10
    if curl -sf --noproxy '*' "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 -- -$2 2>/dev/null; then return 1; fi
  done
  return 1
}

kill_server() {  # $1=pgid
  kill -TERM -- -$1 2>/dev/null; sleep 8; kill -KILL -- -$1 2>/dev/null
}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
# No VLLM_SLEEP_WHEN_IDLE: sleep mode discards/reallocates weight storage, which would
# break the seed's registered memory.

MODEL_URL="bt-e2e-${TAG}"
STRATEGY="v018-port"
EXTRA_CFG="{\"model_url\": \"${MODEL_URL}\", \"model_deploy_strategy_name\": \"${STRATEGY}\", \"rfork_scheduler_url\": \"${PLANNER_URL}\"}"

COMMON_ARGS=(
  --load-format rfork
  --served-model-name test-model
  --max-model-len 2048
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$UTIL"
)

cleanup() {
  [[ -n "${PID_R:-}" ]] && kill_server "$PID_R"
  [[ -n "${PID_S:-}" ]] && kill_server "$PID_S"
}
trap cleanup EXIT

# ---- 1) seed (fresh default load, then publishes as seed) ----
T0=$(date +%s)
ASCEND_RT_VISIBLE_DEVICES=$DEV_S setsid vllm serve "$MODEL" \
  "${COMMON_ARGS[@]}" --port "$PORT_S" \
  --model-loader-extra-config "$EXTRA_CFG" \
  "${EXTRA_ARGS[@]}" > "$LOG_S" 2>&1 &
PID_S=$!

if ! wait_ready "$PORT_S" "$PID_S" 180; then
  echo "FATAL[$TAG]: seed not ready, see $LOG_S"; tail -30 "$LOG_S"; exit 1
fi
T1=$(date +%s)
echo "[$TAG] seed ready in $((T1-T0))s (fresh load, devs=$DEV_S tp=$TP)"
# Let the seed service register with the planner (start_seed_service runs after
# load_model; allow the registration/heartbeat to land before the receiver asks).
sleep "$SEED_SETTLE_SEC"

# ---- 2) receiver (TransferEngine transfer from the seed) ----
T2=$(date +%s)
ASCEND_RT_VISIBLE_DEVICES=$DEV_R setsid vllm serve "$MODEL" \
  "${COMMON_ARGS[@]}" --port "$PORT_R" \
  --model-loader-extra-config "$EXTRA_CFG" \
  "${EXTRA_ARGS[@]}" > "$LOG_R" 2>&1 &
PID_R=$!

if ! wait_ready "$PORT_R" "$PID_R" 180; then
  echo "FATAL[$TAG]: receiver not ready, see $LOG_R"; tail -40 "$LOG_R"; exit 1
fi
T3=$(date +%s)
echo "[$TAG] receiver ready in $((T3-T2))s (devs=$DEV_R tp=$TP)"

# ---- 3) verdict from the receiver log ----
TRANSFER_OK=1
if grep -q "RFork transfer failed" "$LOG_R"; then
  TRANSFER_OK=0
  echo "[$TAG] receiver fell back to default loader:"
  grep "RFork transfer failed" "$LOG_R" | head -3
else
  echo "[$TAG] rfork transfer path OK (no fallback marker in receiver log)"
fi
echo "[$TAG] receiver rfork evidence lines:"
grep -hiE "rfork|transfer|seed" "$LOG_R" | grep -viE "transfer failed" | head -10

# ---- 4) probes ----
python3 "$DIR/probe.py" --port "$PORT_S" --model test-model --out "$OUT_S" || true
python3 "$DIR/probe.py" --port "$PORT_R" --model test-model --out "$OUT_R" || true

# ---- 5) cleanup + compare ----
kill_server "$PID_R"; PID_R=""
kill_server "$PID_S"; PID_S=""
wait_hbm_free "$DEV_R" "$TAG recv"
wait_hbm_free "$DEV_S" "$TAG seed"

if [[ $TRANSFER_OK != 1 ]]; then
  echo "[$TAG] RESULT: FAIL (receiver did not use rfork transfer)"
  exit 1
fi
if python3 "$DIR/compare.py" "$OUT_S" "$OUT_R" --label-a seed --label-b receiver \
    > "$WORKDIR/logs/compare_R_${TAG}.log" 2>&1; then
  echo "[$TAG] RESULT: PASS (rfork transfer + seed vs receiver outputs identical)"
  exit 0
else
  echo "[$TAG] RESULT: FAIL (outputs differ), see $WORKDIR/logs/compare_R_${TAG}.log"
  exit 1
fi
