#!/bin/bash
# 公共函数库：被 01/02/03 脚本 source

set -u

# ---- 全局变量（被调用脚本可覆盖）----
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/mnt/share/l00943473/bigtensor_validation}"
LOG_DIR="${LOG_DIR:-/tmp/bt_e2e_logs}"
mkdir -p "$LOG_DIR"

# ---- 本机 IP 自动探测 ----
# 优先显式 export HOST_IP；否则用 hostname -I 取第一个非 loopback IP，
# 这样同一套脚本在 work-11（141.61.33.11）和 work-19（141.61.33.19）上都能自适配，
# 不必硬编码机器相关的默认值。
if [[ -z "${HOST_IP:-}" ]]; then
    _candidate=""
    if command -v hostname >/dev/null 2>&1; then
        _candidate=$(hostname -I 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i !~ /^127\./) {print $i; exit}}')
    fi
    # 容器里 hostname -I 可能只返回 127.0.0.1，再兜底一把
    if [[ -z "$_candidate" ]] && command -v ip >/dev/null 2>&1; then
        _candidate=$(ip -o -4 addr show scope global 2>/dev/null | awk '{print $4}' | head -n1 | cut -d/ -f1)
    fi
    HOST_IP="${_candidate:-}"
    unset _candidate
fi
if [[ -z "$HOST_IP" ]]; then
    echo "[ERROR] 无法探测本机 IP，请显式 export HOST_IP=<ip>" >&2
fi

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- 失败即退出 ----
fail() {
    log_error "FAIL: $*"
    exit 1
}

pass() {
    log_info "PASS: $*"
}

# ---- 打印快照目录结构 ----
dump_snapshot_tree() {
    local dir="$1"
    local label="${2:-snapshot tree}"
    log_info "$label: $dir"
    if [[ ! -d "$dir" ]]; then
        echo "  (directory does not exist yet)"
        return
    fi
    find "$dir" -maxdepth 3 -printf "  %y %p %s bytes\n" 2>/dev/null | sort
}

# ---- 统计目录数（应等于 hash 子目录数）----
count_fingerprint_dirs() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo 0
        return
    fi
    find "$dir" -maxdepth 1 -mindepth 1 -type d | wc -l
}

# ---- 等待进程监听端口（最长 N 秒）----
wait_for_port() {
    local port="$1"
    local timeout="${2:-${BT_START_TIMEOUT:-1800}}"
    local elapsed=0
    while (( elapsed < timeout )); do
        if ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$"; then
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    return 1
}

# ---- 检查日志中是否含关键标记 ----
check_log_contains() {
    local log_file="$1"
    local pattern="$2"
    local description="${3:-log marker}"
    if grep -qE "$pattern" "$log_file"; then
        pass "$description found: '$pattern'"
        return 0
    else
        log_warn "$description NOT found: '$pattern'"
        return 1
    fi
}

# ---- 提取关键日志片段 ----
show_log_excerpt() {
    local log_file="$1"
    local pattern="$2"
    local context="${3:-2}"
    log_info "excerpt from $log_file matching '$pattern':"
    grep -nE -C "$context" "$pattern" "$log_file" | tail -50 || echo "  (no match)"
}

# ---- 引擎进程记录：用于本次清理与脚本中断后的残留清理 ----
# ENGINE_PIDS: 本次运行启动的 PID（退出 trap 使用）
# ENGINE_PIDFILE: 持久化 PID，供下次运行清理上次被 Ctrl+C 中断的残留
ENGINE_PIDS=()
ENGINE_PIDFILE="${LOG_DIR}/bt_e2e_engine.pid"

record_engine_pid() {
    local pid="$1"
    [[ -z "$pid" ]] && return 0
    ENGINE_PIDS+=("$pid")
    echo "$pid" >>"$ENGINE_PIDFILE" 2>/dev/null || true
}

ENGINE_LOGS=()

# ---- 按 NodeManager 方式启动 external-LB DP：每个 rank 一个独立 vllm serve ----
# 用法：start_dp_engines <cmd_array_var_name> <log_prefix> <dp_size> <host> <base_port> [log_suffix]
start_dp_engines() {
    local cmd_name="$1"
    local log_prefix="$2"
    local dp_size="$3"
    local host="$4"
    local base_port="$5"
    local log_suffix="${6:-}"
    local rank port pid log_file
    local rank_cmd
    local -a base_cmd

    # bash 3.x 兼容：用 eval 间接取数组，替代 bash 4.3+ 的 local -n
    eval "base_cmd=(\"\${${cmd_name}[@]}\")"

    # 物理卡映射：默认 rank r -> NPU r；BT_DEVICES="0 1 6 7" 可指定每个 rank
    # 实际落在哪些物理卡上（跳过被其他容器占用的卡）。
    local -a devices
    if [[ -n "${BT_DEVICES:-}" ]]; then
        read -ra devices <<<"$BT_DEVICES"
    else
        for ((rank = 0; rank < dp_size; rank++)); do devices+=("$rank"); done
    fi
    if ((${#devices[@]} < dp_size)); then
        log_error "BT_DEVICES='$BT_DEVICES' 只提供 ${#devices[@]} 张卡，少于 dp_size=$dp_size"
        return 1
    fi

    for ((rank = 0; rank < dp_size; rank++)); do
        port=$(( base_port + rank * 2 ))
        log_file="${log_prefix}_rank${rank}.log${log_suffix}"
        rank_cmd=("${base_cmd[@]}"
            --host "$host"
            --port "$port"
            --data-parallel-address "$host"
            --data-parallel-rank "$rank")
        log_info "启动 DP rank=$rank，端口=$port，device=${devices[rank]}，日志：$log_file"
        printf '  %s\n' "${rank_cmd[*]}"
        # 注入 torch.distributed 标准 env var。vLLM 用 --data-parallel-address 做 DP 协调，
        # 但 EP 的 HCCL process group 初始化走的是 torch.distributed 标准路径，必须读
        # 到 MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK / LOCAL_RANK。NodeManager 起
        # 进程时这些环境变量由调度层注入，我们手工起必须自己补。
        # MASTER_PORT 用 dp_size * 1000 + 9000 避免跟服务端口/DP rpc port 冲突。
        #
        # 引擎子进程内禁用 http(s) 代理：本机镜像设了 http_proxy 用于 git 拉取，但引擎
        # 的 API server bind / DP 内部 RPC / HCCL 初始化若走代理解析本机地址，会以
        # OSError [Errno 99] Cannot assign requested address 失败。git 代理不受影响，
        # 只在启动子进程里清掉并把所有本机地址加入 NO_PROXY。
        # HCCL_BUFFSIZE：MoE dispatch 在 profile_run 阶段会因通信窗口不足报
        # "Get WinSize failed / Tiling check window size failed"（561002）。默认
        # 200MB 不够，这里放开到 1024MB，可通过同名环境变量覆盖。
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
            -u all_proxy -u ALL_PROXY \
            NO_PROXY="localhost,127.0.0.1,${host},${HOST_IP},172.27.8.193,174.27.1.19,141.61.33.11,141.61.33.19" \
            no_proxy="localhost,127.0.0.1,${host},${HOST_IP},172.27.8.193,174.27.1.19,141.61.33.11,141.61.33.19" \
        MASTER_ADDR="$host" MASTER_PORT=$((dp_size * 1000 + 9000)) \
        WORLD_SIZE="$dp_size" RANK="$rank" LOCAL_RANK="$rank" \
        HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}" \
        ASCEND_RT_VISIBLE_DEVICES="${devices[rank]}" ASCEND_DEVICE_ID=0 \
            setsid "${rank_cmd[@]}" >"$log_file" 2>&1 &
        pid=$!
        record_engine_pid "$pid"
        ENGINE_LOGS+=("$log_file")
        log_info "DP rank=$rank PID=$pid"
    done
}

wait_for_dp_ports() {
    local dp_size="$1"
    local base_port="$2"
    local timeout="${3:-${BT_START_TIMEOUT:-1800}}"
    local rank port elapsed=0
    local -a remaining=()
    for ((rank = 0; rank < dp_size; rank++)); do
        remaining+=("$(( base_port + rank * 2 ))")
    done
    while ((elapsed < timeout)); do
        local -a still=()
        local p f
        for p in "${remaining[@]}"; do
            if ! ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE ":${p}$"; then
                still+=("$p")
            fi
        done
        if ((${#still[@]} == 0)); then
            log_info "全部 $dp_size 个 DP rank 端口已就绪（${elapsed}s）"
            return 0
        fi
        remaining=("${still[@]}")
        # 每 60s 报告一次进度和每个 rank 日志的最后一行：既证明进程还活着，
        # 也能直接看出卡在加载、process_weights 还是快照保存。
        if ((elapsed % 60 == 0)); then
            log_info "等待端口 ${still[*]}（${elapsed}s/${timeout}s）"
            for f in ${ENGINE_LOGS[@]+"${ENGINE_LOGS[@]}"}; do
                [[ -f "$f" ]] || continue
                log_info "  $(basename "$f"): $(tail -n 1 "$f" | cut -c1-150)"
            done
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    for port in "${remaining[@]}"; do
        log_error "DP 端口 $port 未在 ${timeout}s 内监听"
    done
    return 1
}

# ---- 递归列出 pid 的所有后代 ----
# vLLM 的 Worker（VLLM::Worker_DPx_EPx）会脱离父进程的进程组，只按 pgid 杀会留下
# 占着 NPU 显存的孤儿进程，导致下一轮启动失败。
list_descendants() {
    local parent="$1" child
    for child in $(ps -o pid= --ppid "$parent" 2>/dev/null); do
        echo "$child"
        list_descendants "$child"
    done
}

remove_from_pidfile() {
    local pid="$1"
    [[ -z "$pid" || ! -f "$ENGINE_PIDFILE" ]] && return 0
    grep -vx "$pid" "$ENGINE_PIDFILE" >"${ENGINE_PIDFILE}.tmp" 2>/dev/null || true
    mv -f "${ENGINE_PIDFILE}.tmp" "$ENGINE_PIDFILE" 2>/dev/null || true
}

# ---- 杀掉指定 PID 所在进程组（setsid 后 pgid == pid）及其所有后代 ----
kill_engine() {
    local pid="$1"
    [[ -z "$pid" ]] && return 0
    local -a victims=()
    local v line
    if kill -0 "$pid" 2>/dev/null; then
        while read -r line; do
            [[ -n "$line" ]] && victims+=("$line")
        done < <(list_descendants "$pid")
        log_info "停止引擎进程组 PID=$pid（后代 ${#victims[@]} 个）"
        # 先记进 pidfile：即使本次清理被 Ctrl+C 打断，下一次启动也能兜底清掉
        for v in ${victims[@]+"${victims[@]}"}; do
            echo "$v" >>"$ENGINE_PIDFILE" 2>/dev/null || true
        done
        for v in ${victims[@]+"${victims[@]}"}; do
            kill -TERM "$v" 2>/dev/null || true
        done
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        local i
        for i in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 2
        done
        if kill -0 "$pid" 2>/dev/null; then
            log_warn "PID=$pid 未响应 TERM，强制 KILL"
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
        for v in ${victims[@]+"${victims[@]}"}; do
            if kill -0 "$v" 2>/dev/null; then
                log_warn "子进程 PID=$v 未退出，强制 KILL"
                kill -KILL "$v" 2>/dev/null || true
            fi
        done
    fi
    remove_from_pidfile "$pid"
    for v in ${victims[@]+"${victims[@]}"}; do
        remove_from_pidfile "$v"
    done
}

kill_all_engines() {
    local pid
    for pid in "${ENGINE_PIDS[@]}"; do
        kill_engine "$pid"
    done
    # 兜底：kill -KILL 杀掉引擎父进程后，vllm 的 worker 子进程可能被
    # re-parent 到 PID 1，不在上面的后代树里，残留后会占住 NPU 显存
    # （npu-smi 里显示为 VLLMWorker_*，几十 GB 不释放）。按进程名补杀一次。
    local orphan
    orphan=$(ps -ef | awk '/VLLM(WORKER)?[:_]/ && !/awk/ {print $2}')
    if [[ -n "$orphan" ]]; then
        log_info "补杀残留 worker 进程: $(echo $orphan | tr '\n' ' ')"
        echo "$orphan" | xargs -r kill -9 2>/dev/null || true
        sleep 3
    fi
}

all_engines_alive() {
    local pid
    for pid in "${ENGINE_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log_error "引擎进程 PID=$pid 已退出"
            return 1
        fi
    done
}

# ---- 启动前清理上次运行残留的引擎（仅按 pidfile 精确清理，不误杀其他部署）----
kill_stale_vllm() {
    if [[ ! -f "$ENGINE_PIDFILE" ]]; then
        log_info "未发现上次运行的残留引擎进程"
        return 0
    fi
    local pid
    local cleaned=0
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            log_warn "清理上次运行残留的引擎进程 PID=$pid"
            kill_engine "$pid"
            cleaned=$(( cleaned + 1 ))
        fi
    done <"$ENGINE_PIDFILE"
    rm -f "$ENGINE_PIDFILE"
    if (( cleaned == 0 )); then
        log_info "未发现上次运行的残留引擎进程"
    fi
}

# ---- 注册退出/中断兜底清理 ----
install_cleanup_trap() {
    trap '_bt_cleanup_on_exit $?' EXIT
    trap '_bt_cleanup_on_exit 130' INT
    trap '_bt_cleanup_on_exit 143' TERM
}

_bt_cleanup_on_exit() {
    local code="${1:-0}"
    trap - EXIT INT TERM
    if (( ${#ENGINE_PIDS[@]} > 0 )); then
        log_warn "脚本退出(code=$code)，清理本次启动的引擎进程"
        local pid
        for pid in "${ENGINE_PIDS[@]}"; do
            kill_engine "$pid" >/dev/null 2>&1 || true
        done
    fi
    exit "$code"
}

# ---- 显示当前残留的 vllm 进程（只读诊断，不杀进程）----
show_vllm_processes() {
    local found
    found=$(pgrep -af "vllm serve" 2>/dev/null || true)
    if [[ -n "$found" ]]; then
        log_info "当前机器上的 vllm serve 进程（仅展示，未清理）:"
        echo "$found" | sed 's/^/  /'
    else
        log_info "当前无 vllm serve 进程"
    fi
}

# ---- 确认 vllm_ascend 安装来源 ----
check_installation() {
    python -c "import vllm_ascend; print('vllm_ascend:', vllm_ascend.__file__)"
    log_warn "请确认上面打印的路径指向当前包含修复的源码目录（应为 editable 安装）"
}
