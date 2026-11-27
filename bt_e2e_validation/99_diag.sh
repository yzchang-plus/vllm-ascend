#!/bin/bash
# 一键采集排查所需的全部信息，输出可直接整段贴回分析。
# 只读：不启动/停止引擎，不修改快照。
#
# 用法：
#   bash bt_e2e_validation/99_diag.sh                 # 默认 02_mtp，4 个 rank
#   bash bt_e2e_validation/99_diag.sh 03_save 8       # 指定日志前缀和 rank 数
#   LOG_DIR=/tmp/bt_e2e_logs bash bt_e2e_validation/99_diag.sh 02_mtp 4

set -u

PREFIX="${1:-02_mtp}"
NPROC="${2:-4}"
LOG_DIR="${LOG_DIR:-/tmp/bt_e2e_logs}"
SNAP_ROOTS="${SNAP_ROOTS:-/vllm-workspace/bt_snap /mnt/share/l00943473/bigtensor_validation}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

hdr() {
    echo
    echo "==================== $* ===================="
}

hdr "基本信息"
echo "time      : $(date '+%F %T')"
echo "host      : $(hostname)"
echo "repo      : $REPO_DIR"
echo "git head  : $(git -C "$REPO_DIR" log -1 --oneline 2>/dev/null || echo '?')"
echo "git status: $(git -C "$REPO_DIR" status --porcelain 2>/dev/null | head -5)"

hdr "环境变量"
for v in HCCL_BUFFSIZE http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY SNAPSHOT_ROOT LOAD_FORMAT BT_START_TIMEOUT BT_DEVICES; do
    echo "$v=${!v-<unset>}"
done

hdr "残留 vllm 进程"
pgrep -af "vllm serve" | head -10 || echo "(none)"

hdr "磁盘"
df -h /vllm-workspace /tmp 2>/dev/null
for root in $SNAP_ROOTS; do
    [ -d "$root" ] || continue
    echo "--- $root : $(du -sh "$root" 2>/dev/null | cut -f1) ---"
    find "$root" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | head -5
    find "$root" -maxdepth 2 -name "*.snapshot" -printf '%p %s\n' 2>/dev/null | head -10
    find "$root" -maxdepth 2 -name ".bigtensor-*" -printf '%p %s\n' 2>/dev/null | head -10
done

# 各 rank 的错误签名：签名一致的 rank 不必逐份贴日志，只展开有差异的。
# 用函数 + 普通变量而非关联数组，兼容 bash 4 以下。
sig_of() {
    grep -oE "(RuntimeError|OSError|ChildProcessError|ValueError|TypeError|AssertionError): .{0,110}" "$1" 2>/dev/null | tail -1
}

hdr "各 rank 错误签名"
SIG0=""
for ((rank = 0; rank < NPROC; rank++)); do
    log_file="${LOG_DIR}/${PREFIX}_rank${rank}.log"
    if [[ ! -f "$log_file" ]]; then
        echo "rank${rank}: (missing)"
        continue
    fi
    sig="$(sig_of "$log_file")"
    (( rank == 0 )) && SIG0="$sig"
    echo "rank${rank}: ${sig:-<no error>}"
done

for ((rank = 0; rank < NPROC; rank++)); do
    log_file="${LOG_DIR}/${PREFIX}_rank${rank}.log"
    if [[ ! -f "$log_file" ]]; then
        hdr "rank${rank} 日志"
        echo "(missing: $log_file)"
        continue
    fi
    if (( rank > 0 )) && [[ "$(sig_of "$log_file")" == "$SIG0" ]]; then
        echo "(rank${rank} 错误签名与 rank0 一致，略)"
        continue
    fi
    hdr "rank${rank} 日志"
    echo "file: $log_file  size: $(stat -c%s "$log_file") bytes"

    echo
    echo "--- snapshot 相关 ---"
    grep -n "snapshot" "$log_file" | tail -12 || echo "(none)"

    echo
    echo "--- 非连续 NZ 权重 ---"
    grep -nE "non-contiguous NZ weight|base permute=" "$log_file" | head -10 || echo "(none)"

    echo
    echo "--- snapshot 失败完整栈（从首条 save FAILED 起 55 行）---"
    sn_line="$(grep -nE "snapshot save FAILED|snapshot background disk write FAILED" "$log_file" | head -1 | cut -d: -f1)"
    if [[ -n "$sn_line" ]]; then
        sed -n "$((sn_line)),$((sn_line + 55))p" "$log_file"
    else
        echo "(none)"
    fi

    echo
    echo "--- 内部错误完整栈（aclnn/Inner error 前 8 行起到 40 行）---"
    in_line="$(grep -nE "The Inner error is reported as above|WorkerProc hit an exception" "$log_file" | head -1 | cut -d: -f1)"
    if [[ -n "$in_line" ]]; then
        in_start=$(( in_line - 8 ))
        (( in_start < 1 )) && in_start=1
        sed -n "${in_start},$((in_line + 40))p" "$log_file"
    else
        echo "(none)"
    fi

    echo
    echo "--- 加载进度 ---"
    tail -c 4000 "$log_file" | tr '\r' '\n' | grep -E "Loading safetensors|Loading weights took|Model loading took" | tail -3 || echo "(none)"

    echo
    echo "--- 首个 Worker/Engine 异常栈 ---"
    awk '/WorkerProc hit an exception|Worker failed with error|Engine core initialization failed/{f=1} f{print; n++} n>=32{exit}' "$log_file" | head -40 || echo "(none)"

    echo
    echo "--- 关键错误摘要 ---"
    grep -nE "RuntimeError|PermissionError|Read-only file system|aclrtMemcpy|error code is [0-9]+|Get WinSize|axis K of x|expanded size|507899|561002|161002|No space left" "$log_file" | tail -30 || echo "(none)"

    echo
    echo "--- 最后 8 行 ---"
    tail -8 "$log_file" | tr '\r' '\n' | cut -c1-200
done

hdr "采集完成"
echo "把以上全部输出贴回即可。"
