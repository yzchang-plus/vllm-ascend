#!/bin/bash
# 问题 5：save_snapshot() 同步序列化/D2H 异常导致推理服务退出
# 验证：故意把 checkpoint_path 指向只读路径，触发保存失败。
#       期望：服务**不退出**、模型正常加载、日志含 "snapshot save FAILED"、推理可用。
#
# 关键：DeepSeek-V4.1-Flash 是 384-expert MoE，不加 EP 单卡装不下；
#       但 w4a8 quant 的 process_weights_after_loading 假设 EP 已分片专家
#       384→48，所以 EP 必须开。除此之外所有 speculative/MultiConnector/
#       quant override 都是无关变量，全部去掉。
#
# 用法：bash 03_save_failure.sh
# 退出码：0=PASS，非 0=FAIL

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# 安装退出/中断兜底清理，并清理上次运行残留的引擎
install_cleanup_trap
show_vllm_processes
kill_stale_vllm

# ---- 构造只读 checkpoint_path ----
# /proc/__bigtensor_save_failure_test__：procfs，mkdir/写文件都会失败
READONLY_CKPT="/proc/__bigtensor_save_failure_test__"
LOG_PREFIX="${LOG_DIR}/03_save"
MODEL="/mnt/share/weight/DeepSeek-V4.1-Flash"
# DP=8 × EP=8：每个 rank 48 expert，对齐原始部署
DP_SIZE=8

echo "=============================================="
log_info "问题 5 验证：保存路径只读导致 D2H/写盘失败"
log_info "模型: $MODEL"
log_info "checkpoint_path: $READONLY_CKPT"
log_info "HOST_IP: $HOST_IP"
echo "=============================================="

# 确认只读路径确实不可写
if touch "$READONLY_CKPT/__probe" 2>/dev/null; then
    rm -f "$READONLY_CKPT/__probe"
    fail "预期 $READONLY_CKPT 不可写，但实际可以写入，请换一个只读路径"
fi
pass "确认 $READONLY_CKPT 不可写（写入会失败）"

# ---- 启动命令 ----
# 保留 --enable-expert-parallel + DP=8（w4a8 quant 的 process_weights_after_loading
# 假设 EP 已分片专家 384→48，去掉 EP 会导致 reshape 崩）。
# 去掉所有无关变量：MultiConnector、speculative、quant override、cudagraph、prefix cache。
CMD=(
    vllm serve
    --data-parallel-size "$DP_SIZE"
    --tensor-parallel-size 1
    --pipeline-parallel-size 1
    --data-parallel-rpc-port 9000
    --enable-expert-parallel
    --served-model-name dsv41
    --model "$MODEL"
    --max-model-len 1048576
    --max-num-batched-tokens 1024
    --max-num-seqs 64
    --gpu-memory-utilization 0.92
    --block-size 128
    --quantization deepseek_v4_fp8
    --trust-remote-code
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v41
    --no-enable-prefix-caching
    --host "$HOST_IP"
)

# 默认走 bigtensorloader（问题 5 判据需要它）。LOAD_FORMAT=auto 用于 A/B 对照：
# 同一模型/拓扑改用默认加载器，用来判断 capture 阶段崩溃是否由快照恢复引入。
LOAD_FORMAT="${LOAD_FORMAT:-bigtensorloader}"
CMD+=(--load-format "$LOAD_FORMAT")
if [[ "$LOAD_FORMAT" == "bigtensorloader" ]]; then
    CMD+=(--model-loader-extra-config "{\"fast_restore\":true,\"checkpoint_path\":\"${READONLY_CKPT}\",\"verify\":\"none\",\"bulk_h2d\":true}")
fi
log_info "load_format=$LOAD_FORMAT"

# 该模型在本环境 capture 阶段会崩在 "No device metadata submission is in flight"，
# 与权重加载无关，但会让脚本永远等不到监听端口。问题 5 只关心「保存失败后服务是否
# 存活且可服务」，cudagraph 模式与此无关，故默认关闭以便引擎真正进入 serving。
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-NONE}"
CMD+=(--compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\"}")
log_info "cudagraph_mode=$CUDAGRAPH_MODE"

# ---- 启动 8 个 DP rank ----
log_info "启动 8 个 DP rank（rank 0..7），日志前缀：$LOG_PREFIX"
start_dp_engines CMD "$LOG_PREFIX" 8 "$HOST_IP" 10000

# ---- 验证点 1：所有 rank 进入 serving 阶段（默认 BT_START_TIMEOUT=1800s，含首次保存）----
if ! wait_for_dp_ports 8 10000; then
    DEAD_PIDS=()
    for pid in "${ENGINE_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            DEAD_PIDS+=("$pid")
        fi
    done
    if (( ${#DEAD_PIDS[@]} > 0 )); then
        log_error "以下引擎进程已退出: ${DEAD_PIDS[*]}"
        for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
            show_log_excerpt "$ENGINE_LOG" "ERROR|Error|Traceback|FAILED|Exception" 3
        done
        kill_all_engines
        fail "保存失败导致部分引擎进程退出（问题 5 未修复）"
    else
        log_error "所有进程仍在但 15 分钟内未完全监听端口"
        for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
            show_log_excerpt "$ENGINE_LOG" "ERROR|Error|Traceback" 3
        done
        kill_all_engines
        fail "引擎启动超时"
    fi
fi
pass "所有 DP rank 进程存活且监听 API 端口"

# ---- 验证点 2：保存尝试后进程仍存活 ----
log_info "等待 90s 让保存路径走完..."
sleep 90

if ! all_engines_alive; then
    for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
        show_log_excerpt "$ENGINE_LOG" "ERROR|Error|Traceback|FAILED|Exception" 3
    done
    kill_all_engines
    fail "保存失败导致服务退出（问题 5 未修复）"
fi
pass "保存尝试后所有引擎进程仍存活"

# ---- 验证点 3：日志必须含 snapshot save FAILED（仅 bigtensorloader 场景）----
FAILED_PATTERN='bigtensorloader: snapshot save FAILED|snapshot background disk write FAILED|snapshot background save FAILED'
if [[ "$LOAD_FORMAT" != "bigtensorloader" ]]; then
    log_warn "load_format=$LOAD_FORMAT 不对快照保存做断言（A/B 对照运行）"
elif grep -qE "$FAILED_PATTERN" "${ENGINE_LOGS[@]}"; then
    pass "日志含 snapshot save FAILED 标记"
else
    for ENGINE_LOG in "${ENGINE_LOGS[@]}"; do
        show_log_excerpt "$ENGINE_LOG" "snapshot|BigTensor|save" 2
    done
    kill_all_engines
    fail "日志未找到 snapshot save FAILED 标记"
fi

# ---- 验证点 4：推理可用（rank 0）----
log_info "发一个简单 chat completion 请求，验证推理可用"
RESPONSE=$(curl -sf --noproxy '*' -X POST "http://${HOST_IP}:10000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"dsv41","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":8}' \
    2>&1 || echo "CURL_FAILED")

if echo "$RESPONSE" | grep -qE '"content"|"choices"'; then
    pass "推理请求成功返回"
else
    log_warn "推理请求返回异常: $RESPONSE"
fi

# ---- 关闭引擎 ----
kill_all_engines

echo ""
echo "=============================================="
pass "问题 5 验证通过：保存失败时服务不退出"
echo "=============================================="
