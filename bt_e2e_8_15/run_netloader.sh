#!/bin/bash
# D2D E2E (netloader HCCL P2P weight transfer) - self-contained in bt_e2e_8_15.
#
# Usage: run_netloader.sh <tag> <model_path> <dev_donor> <dev_client> \
#          <api_port_donor> <api_port_client> <listen_port> <workdir> <gpu_util> [extra vllm flags...]
#
# Flow: donor starts first (default loader + weight listener) -> client pulls weights
#       from donor -> both run the fixed probe (temperature=0) -> outputs must match.
# Success marker (client log): "[netloader_p2p] HCCL recv time"
# Fallback markers: "use DefaultModelLoader" / "Netloader fallback"
set -u

TAG=$1; MODEL=$2; DEV_D=$3; DEV_C=$4; PORT_D=$5; PORT_C=$6; LP=$7; WORKDIR=$8; UTIL=$9; shift 9
EXTRA_ARGS=("$@")

DIR=$(cd "$(dirname "$0")" && pwd)

# Key: pin vllm_ascend resolution to the v0.18.0-lineage clone (serving-capable here).
# Preserve CANN's PYTHONPATH entries by appending.
export PYTHONPATH="/vllm-workspace/vllm-ascend${PYTHONPATH:+:$PYTHONPATH}"

LOG_D=$WORKDIR/logs/N_${TAG}_donor.server.log
LOG_C=$WORKDIR/logs/N_${TAG}_client.server.log
OUT_D=$WORKDIR/out/N_${TAG}_donor.out
OUT_C=$WORKDIR/out/N_${TAG}_client.out
ADDR_FILE=$WORKDIR/logs/${TAG}.addr0.txt
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

wait_hbm_free() {  # $1=dev $2=label
  for _ in $(seq 1 36); do
    HBM=$(hbm_mb "$1")
    [[ -n "$HBM" && "$HBM" -lt 10000 ]] && break
    sleep 5
  done
  echo "[$2] dev$1 HBM=${HBM:-?}MB"
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

COMMON_ARGS=(
  --load-format netloader
  --served-model-name test-model
  --max-model-len 2048
  --gpu-memory-utilization "$UTIL"
  --tensor-parallel-size 1
)
export VLLM_SLEEP_WHEN_IDLE=1
# Weights and the donor address are local/intranet; unset proxies so the
# client->donor HTTP discovery requests are not hijacked by the proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="$no_proxy"

# ---- 1) donor ----
rm -f "$ADDR_FILE"
T0=$(date +%s)
ASCEND_RT_VISIBLE_DEVICES=$DEV_D setsid vllm serve "$MODEL" \
  "${COMMON_ARGS[@]}" --port "$PORT_D" \
  --model-loader-extra-config "{\"LISTEN_PORT\": $LP, \"OUTPUT_PREFIX\": \"$WORKDIR/logs/${TAG}.addr\"}" \
  "${EXTRA_ARGS[@]}" > "$LOG_D" 2>&1 &
PID_D=$!

if ! wait_ready "$PORT_D" "$PID_D" 120; then
  echo "FATAL[$TAG]: donor not ready, see $LOG_D"; tail -30 "$LOG_D"
  kill_server "$PID_D"; exit 1
fi
T1=$(date +%s)
echo "[$TAG] donor ready in $((T1-T0))s (dev=$DEV_D port=$PORT_D listen=$LP)"

# Wait for the listener address file (written after the donor model load completes).
for _ in $(seq 1 60); do
  [[ -f "$ADDR_FILE" ]] && break
  sleep 5
done
if [[ ! -f "$ADDR_FILE" ]]; then
  echo "FATAL[$TAG]: donor address file missing: $ADDR_FILE, see $LOG_D"
  kill_server "$PID_D"; exit 1
fi
ADDR=$(cat "$ADDR_FILE" | tr -d '[:space:]')
echo "[$TAG] donor weight-listener addr: $ADDR"

# ---- 2) client (pull weights from donor) ----
T2=$(date +%s)
ASCEND_RT_VISIBLE_DEVICES=$DEV_C setsid vllm serve "$MODEL" \
  "${COMMON_ARGS[@]}" --port "$PORT_C" \
  --model-loader-extra-config "{\"SOURCE\":[{\"device_id\":0,\"sources\":[\"$ADDR\"]}]}" \
  "${EXTRA_ARGS[@]}" > "$LOG_C" 2>&1 &
PID_C=$!

if ! wait_ready "$PORT_C" "$PID_C" 120; then
  echo "FATAL[$TAG]: client not ready, see $LOG_C"; tail -40 "$LOG_C"
  kill_server "$PID_C"; kill_server "$PID_D"; exit 1
fi
T3=$(date +%s)
echo "[$TAG] client ready in $((T3-T2))s (dev=$DEV_C port=$PORT_C)"

if grep -q "HCCL recv time" "$LOG_C"; then
  echo "[$TAG] D2D transfer success marker: $(grep 'HCCL recv time' "$LOG_C" | head -1)"
else
  echo "WARN[$TAG]: no HCCL recv marker in client log, may have fallen back:"
  grep -E "DefaultModelLoader|Netloader fallback" "$LOG_C" | head -3
fi

# ---- 3) probes on both sides ----
python3 "$DIR/probe.py" --port "$PORT_D" --model test-model --out "$OUT_D" || true
python3 "$DIR/probe.py" --port "$PORT_C" --model test-model --out "$OUT_C" || true

# ---- 4) cleanup ----
kill_server "$PID_C"; kill_server "$PID_D"
wait_hbm_free "$DEV_C" "$TAG client"
wait_hbm_free "$DEV_D" "$TAG donor"

if python3 "$DIR/compare.py" "$OUT_D" "$OUT_C" --label-a donor --label-b client \
    > "$WORKDIR/logs/compare_N_${TAG}.log" 2>&1; then
  echo "[$TAG] RESULT: PASS (donor vs client outputs identical)"
  exit 0
else
  echo "[$TAG] RESULT: FAIL, see $WORKDIR/logs/compare_N_${TAG}.log"
  exit 1
fi
