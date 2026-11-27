#!/bin/bash
# 问题 1：多 P、多 D 共用 checkpoint_path 时 rank 文件冲突
# 验证：让 P 和 D 使用**同一个** checkpoint_path（不再像测试人员那样手工分 prefill/decode），
#       指纹应自动按 kv_role 区分目录，避免写入冲突。
#
# 用法：
#   work-19: bash 01_pd_shared_path.sh prefill
#   work-11: bash 01_pd_shared_path.sh decode
#
# 退出码：0=PASS，非 0=FAIL

set -u
ROLE="${1:-}"
if [[ "$ROLE" != "prefill" && "$ROLE" != "decode" ]]; then
    echo "用法: $0 {prefill|decode}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# 安装退出/中断兜底清理，并清理上次运行残留的引擎
install_cleanup_trap
show_vllm_processes
kill_stale_vllm

# ---- 关键改动：P 和 D 用**同一个**父目录 ----
SHARED_CKPT="${SNAPSHOT_ROOT}/pd_shared"
mkdir -p "$SHARED_CKPT"

MODEL="/mnt/share/weight/DeepSeek-V4-Flash-0731-w4a4c8-new"
LOG="${LOG_DIR}/01_pd_${ROLE}.log"

# ---- 启动前 dump ----
echo "=============================================="
log_info "问题 1 验证：PD 共用 checkpoint_path"
log_info "角色: $ROLE"
log_info "共享目录: $SHARED_CKPT"
echo "=============================================="
dump_snapshot_tree "$SHARED_CKPT" "启动前目录结构"
BEFORE_COUNT=$(count_fingerprint_dirs "$SHARED_CKPT")
log_info "启动前 fingerprint 子目录数: $BEFORE_COUNT"

# ---- 构造启动命令（严格复制测试人员参数，仅改 checkpoint_path）----
if [[ "$ROLE" == "prefill" ]]; then
    KV_ROLE="kv_producer"
    KV_PORT="30000"
    ENGINE_ID="3"
    # work-19
    HOST="$HOST_IP"
    DP=1
    PCP=4
    MAX_SEQS=16
    MAX_TOKENS=8192
    ADDITIONAL_CFG='{"enable_cpu_binding":true,"enable_fused_mc2":0,"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true}'
    ENFORCE_EAGER_FLAG="--enforce-eager"
    SPEC_ENFORCE_EAGER="true"
    COMPILATION_FLAG=""
    ASYNC_FLAG=""
else
    KV_ROLE="kv_consumer"
    KV_PORT="30100"
    ENGINE_ID="1"
    HOST="${HOST_IP}"
    DP=4
    PCP=1
    MAX_SEQS=56
    MAX_TOKENS=1024
    ADDITIONAL_CFG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"enable_shared_expert_dp":true,"combine_quant_mode":4}'
    ENFORCE_EAGER_FLAG=""
    SPEC_ENFORCE_EAGER="false"
    COMPILATION_FLAG='--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}'
    ASYNC_FLAG="--async-scheduling"
fi

# shellcheck disable=SC2089
CMD=(
    vllm serve
    --data-parallel-size "$DP"
    --tensor-parallel-size 1
    --pipeline-parallel-size 1
    --prefill-context-parallel-size "$PCP"
    --data-parallel-rpc-port "$([ "$ROLE" = prefill ] && echo 12321 || echo 12325)"
    --served-model-name dsv4
    --model "$MODEL"
    --seed 1024
    --enable-expert-parallel
    --max-model-len 1048576
    --max-num-batched-tokens "$MAX_TOKENS"
    --max-num-seqs "$MAX_SEQS"
    $ASYNC_FLAG
    --block-size 32
    --kv-cache-dtype fp8
    --attention-config '{"indexer_kv_dtype":"fp8"}'
    --no-disable-hybrid-kv-cache-manager
    --load-format bigtensorloader
    --model-loader-extra-config "{\"fast_restore\":true,\"checkpoint_path\":\"${SHARED_CKPT}\",\"verify\":\"none\",\"bulk_h2d\":true}"
    --trust-remote-code
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --gpu-memory-utilization 0.9
    --speculative-config "{\"num_speculative_tokens\":5,\"method\":\"dspark\",\"enforce_eager\":${SPEC_ENFORCE_EAGER}}"
    $COMPILATION_FLAG
    --kv-transfer-config "{\"kv_connector\":\"MooncakeHybridConnector\",\"kv_role\":\"${KV_ROLE}\",\"kv_port\":\"${KV_PORT}\",\"kv_load_failure_policy\":\"fail\",\"engine_id\":\"${ENGINE_ID}\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":1,\"pp_size\":1},\"decode\":{\"dp_size\":4,\"tp_size\":1,\"pp_size\":1}}}"
    --additional-config "$ADDITIONAL_CFG"
    --disable-access-log-for-endpoints /health,/metrics,/snapshot/health,/v1/fault_tolerance/status
    --cp-kv-cache-interleave-size 1
)
# prefill 为 DP=1：不加 data-parallel-address/rank（否则 vLLM 判定 external LB 要求 DP>1）
if [[ "$ROLE" == "prefill" ]]; then
    CMD+=(--host "$HOST" --port 10000 --no-enable-prefix-caching "$ENFORCE_EAGER_FLAG")
fi

log_info "启动命令（已用共享 checkpoint_path）:"

# ---- 启动引擎（后台）；decode 按原始日志启动 rank 0..3 四个独立进程 ----
ENGINE_LOGS=()
if [[ "$ROLE" == "decode" ]]; then
    start_dp_engines CMD "${LOG_DIR}/01_pd_${ROLE}" 4 "$HOST" 10000
else
    log_info "启动 $ROLE 引擎（单进程，device 0），日志：$LOG"
    printf '  %s\n' "${CMD[*]}"
    MASTER_ADDR="$HOST" MASTER_PORT=9100 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_DEVICE_ID=0 \
        setsid "${CMD[@]}" >"$LOG" 2>&1 &
    ENGINE_PID=$!
    record_engine_pid "$ENGINE_PID"
    ENGINE_LOGS=("$LOG")
    log_info "引擎 PID=$ENGINE_PID"
fi

# ---- 等待所有 API 端口启动（默认 BT_START_TIMEOUT=1800s）----
STARTUP_OK=0
if [[ "$ROLE" == "decode" ]]; then
    wait_for_dp_ports 4 10000 && STARTUP_OK=1
else
    wait_for_port 10000 && STARTUP_OK=1
fi
if (( STARTUP_OK == 0 )); then
    for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
        show_log_excerpt "$ENGINE_LOG" "ERROR|Error|Traceback|FAILED" 3
    done
    kill_all_engines
    fail "引擎启动失败"
fi
pass "$ROLE 引擎的所有 API 端口均已监听"

# ---- 等待快照写入完成（再等 60 秒让异步保存落地）----
log_info "等待 60s 让快照异步写入完成..."
sleep 60

# ---- 验证点 1：checkpoint_path 下应出现指纹子目录 ----
dump_snapshot_tree "$SHARED_CKPT" "启动后目录结构"
AFTER_COUNT=$(count_fingerprint_dirs "$SHARED_CKPT")
log_info "启动后 fingerprint 子目录数: $AFTER_COUNT"

# 单角色启动后应新增至少 1 个 hash 目录；P+D 都跑完后应等于 2
if (( AFTER_COUNT <= BEFORE_COUNT )); then
    for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
        show_log_excerpt "$ENGINE_LOG" "snapshot save|snapshot restore|BigTensor" 2
    done
    kill_all_engines
    fail "未产生新的 fingerprint 子目录（前 $BEFORE_COUNT → 后 $AFTER_COUNT）"
fi
pass "新增了 fingerprint 子目录（$BEFORE_COUNT → $AFTER_COUNT）"

# ---- 验证点 2：日志中应出现 snapshot save 成功或 restore 成功 ----
if grep -qE "snapshot restore" "${ENGINE_LOGS[@]}"; then
    pass "日志含 snapshot restore（命中已有快照）"
elif grep -qE "snapshot save|save_snapshot disk write complete" "${ENGINE_LOGS[@]}"; then
    pass "日志含 snapshot save（首次写入）"
else
    log_warn "未找到 snapshot save/restore 关键字，请人工检查日志"
fi

# ---- 验证点 3：日志中不应有保存失败/服务退出 ----
if grep -qE "snapshot save FAILED|snapshot background disk write FAILED|snapshot restore FAILED" "${ENGINE_LOGS[@]}"; then
    for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
        show_log_excerpt "$ENGINE_LOG" "FAILED" 3
    done
    kill_all_engines
    fail "日志含 FAILED 标记"
fi
pass "无 FAILED 标记"

# ---- 关闭引擎 ----
kill_all_engines

# ---- 最终判定 ----
echo ""
echo "=============================================="
log_info "角色 $ROLE 验证结束"
log_info "预期：P 和 D 都跑完后，$SHARED_CKPT 下应有 2 个不同的 hash 子目录"
log_info "当前 hash 子目录数: $(count_fingerprint_dirs "$SHARED_CKPT")"
echo "=============================================="
pass "$ROLE 端验证通过"
