#!/bin/bash
# 问题 4：投机推理 target/drafter 同进程同 rank 写入冲突
# 验证：单进程启用 --speculative-config，target 和 drafter 共用同一个 checkpoint_path，
#       指纹应自动按 model_role 区分目录，避免 drafter 覆盖 target。
#
# 用法：bash 02_mtp_target_drafter.sh
# 退出码：0=PASS，非 0=FAIL

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# 安装退出/中断兜底清理，并清理上次运行残留的引擎
install_cleanup_trap
show_vllm_processes
kill_stale_vllm

SHARED_CKPT="${SNAPSHOT_ROOT}/mtp_shared"
mkdir -p "$SHARED_CKPT"
MODEL="/mnt/share/weight/DeepSeek-V4-Flash-0731-w4a4c8-new"
LOG="${LOG_DIR}/02_mtp.log"
HOST_IP="${HOST_IP}"   # 由 common.sh 自动探测本机 IP

echo "=============================================="
log_info "问题 4 验证：MTP target/drafter 共用 checkpoint_path"
log_info "共享目录: $SHARED_CKPT"
echo "=============================================="
dump_snapshot_tree "$SHARED_CKPT" "启动前目录结构"
BEFORE_COUNT=$(count_fingerprint_dirs "$SHARED_CKPT")
log_info "启动前 fingerprint 子目录数: $BEFORE_COUNT"

# ---- 严格复制测试人员 decode 端原始命令（w4a4c8 + dsv4 + DP=4），仅改 checkpoint_path ----
# NodeManager 实际行为：DP=4 用 4 个独立 vllm serve（rank 0..3，端口 10000/10002/10004/10006）
# 默认走 bigtensorloader；LOAD_FORMAT=auto 可做 A/B 对照，用默认加载器跑同一模型，
# 用来区分「快照恢复引入的问题」和「模型/量化配置本身的问题」。
# （必须在 CMD 数组展开前赋默认值：set -u 下引用未定义变量会直接中止脚本。）
LOAD_FORMAT="${LOAD_FORMAT:-bigtensorloader}"
CMD=(
    vllm serve
    --data-parallel-size 4
    --tensor-parallel-size 1
    --pipeline-parallel-size 1
    --prefill-context-parallel-size 1
    --data-parallel-rpc-port 12325
    --served-model-name dsv4
    --model "$MODEL"
    --seed 1024
    --enable-expert-parallel
    --max-model-len 1048576
    --max-num-batched-tokens 1024
    --max-num-seqs 56
    --async-scheduling
    --block-size 32
    --kv-cache-dtype fp8
    --attention-config '{"indexer_kv_dtype":"fp8"}'
    --no-enable-prefix-caching
    --no-disable-hybrid-kv-cache-manager
    --load-format "$LOAD_FORMAT"
    --trust-remote-code
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --gpu-memory-utilization 0.9
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":false}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"30100","kv_load_failure_policy":"fail","engine_id":"1","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1,"pp_size":1},"decode":{"dp_size":4,"tp_size":1,"pp_size":1}}}'
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"enable_shared_expert_dp":true,"combine_quant_mode":4}'
    --disable-access-log-for-endpoints /health,/metrics,/snapshot/health,/v1/fault_tolerance/status
    --cp-kv-cache-interleave-size 1
)

if [[ "$LOAD_FORMAT" == "bigtensorloader" ]]; then
    CMD+=(--model-loader-extra-config "{\"fast_restore\":true,\"checkpoint_path\":\"${SHARED_CKPT}\",\"verify\":\"none\",\"bulk_h2d\":true}")
fi
log_info "load_format=$LOAD_FORMAT"

# ---- 启动 4 个 DP rank 进程 ----
start_dp_engines CMD "${LOG_DIR}/02_mtp" 4 "$HOST_IP" 10000

# ---- 等待所有 rank 监听 ----
if ! wait_for_dp_ports 4 10000; then
    for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
        show_log_excerpt "$ENGINE_LOG" "ERROR|Error|Traceback|FAILED" 3
    done
    kill_all_engines
    fail "MTP 引擎启动失败"
fi
pass "MTP 引擎的所有 DP rank 均已监听"

log_info "等待 90s 让 target 和 drafter 快照都落地..."
sleep 90

# ---- 验证点 1：每个 rank 都应产生两个不同的 hash 子目录（target 一个，drafter 一个）----
dump_snapshot_tree "$SHARED_CKPT" "启动后目录结构"
AFTER_COUNT=$(count_fingerprint_dirs "$SHARED_CKPT")
NEW_DIRS=$(( AFTER_COUNT - BEFORE_COUNT ))
log_info "新增 fingerprint 子目录数: $NEW_DIRS"

# 4 个 rank，每个 rank 有 target+drafter = 8 个 hash 目录（rank 间因 kv_role/model_role 一致会共享）
# 保守判定：至少应有 2 个新目录（target 和 drafter 各 1 个）
if (( NEW_DIRS < 2 )); then
    for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
        show_log_excerpt "$ENGINE_LOG" "snapshot|BigTensor|drafter|target" 2
    done
    kill_all_engines
    fail "target 和 drafter 应各产生独立 hash 目录（期望 ≥2，实际 $NEW_DIRS）"
fi
pass "target 和 drafter 各产生了独立的 hash 目录（新增 $NEW_DIRS）"

# ---- 验证点 2：每个 hash 目录下应有 .json + .snapshot 文件 ----
MISSING_FILES=0
while IFS= read -r -d '' subdir; do
    for ext in json snapshot; do
        if ! find "$subdir" -name "*.$ext" -print -quit | grep -q .; then
            log_error "  $subdir 下缺少 .$ext 文件"
            MISSING_FILES=$(( MISSING_FILES + 1 ))
        fi
    done
done < <(find "$SHARED_CKPT" -maxdepth 1 -mindepth 1 -type d -print0)

if (( MISSING_FILES > 0 )); then
    kill_all_engines
    fail "$MISSING_FILES 个 hash 目录缺少 .json 或 .snapshot 文件"
fi
pass "每个 hash 目录下都有 .json 和 .snapshot 文件"

# ---- 验证点 3：日志无 FAILED 标记 ----
if grep -qE "snapshot save FAILED|snapshot background disk write FAILED|snapshot restore FAILED" "${ENGINE_LOGS[@]}"; then
    for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
        show_log_excerpt "$ENGINE_LOG" "FAILED" 3
    done
    kill_all_engines
    fail "日志含 FAILED 标记"
fi
pass "无 FAILED 标记"

# ---- 验证点 4（关键）：重启一次，应命中各自的 fast restore，不冲突 ----
log_info "关闭引擎，做第二次启动以验证 fast restore 命中"
kill_all_engines
sleep 10

start_dp_engines CMD "${LOG_DIR}/02_mtp" 4 "$HOST_IP" 10000 "_2nd"

if ! wait_for_dp_ports 4 10000; then
    for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
        show_log_excerpt "$ENGINE_LOG" "ERROR|Error|Traceback|FAILED" 3
    done
    kill_all_engines
    fail "第二次启动失败"
fi

# 统计第二次启动（rank0 日志）中 'snapshot restore' 出现次数
RESTORE_LOG="${LOG_DIR}/02_mtp_rank0.log_2nd"
if [[ ! -f "$RESTORE_LOG" ]]; then
    # 回退：看所有日志
    RESTORE_COUNT=$(grep -cE "snapshot restore" "${ENGINE_LOGS[@]}" || true)
else
    RESTORE_COUNT=$(grep -cE "snapshot restore" "$RESTORE_LOG" || true)
fi
log_info "第二次启动 'snapshot restore' 出现次数: $RESTORE_COUNT"
if (( RESTORE_COUNT < 2 )); then
    for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
        show_log_excerpt "$ENGINE_LOG" "snapshot|drafter|target" 2
    done
    kill_all_engines
    fail "第二次启动应至少 2 次 snapshot restore（target 和 drafter 各一次），实际 $RESTORE_COUNT"
fi
pass "第二次启动 target 和 drafter 各自命中 fast restore（$RESTORE_COUNT 次）"

kill_all_engines

echo ""
echo "=============================================="
pass "MTP target/drafter 隔离验证通过"
echo "=============================================="
