#!/bin/bash
# 清理验证脚本产生的所有快照目录和日志

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log_warn "将执行以下清理（不影响测试人员原有快照）:"
echo "  - 清理本目录脚本启动的残留引擎进程（按 pidfile）"
echo "  - 删除 ${SNAPSHOT_ROOT}"
echo "  - 删除 ${LOG_DIR}"
read -r -p "确认删除？[y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    log_info "取消"
    exit 0
fi

# 先清理残留引擎进程，再删目录
show_vllm_processes
kill_stale_vllm

if [[ -d "$SNAPSHOT_ROOT" ]]; then
    log_info "删除 $SNAPSHOT_ROOT"
    rm -rf "$SNAPSHOT_ROOT"
fi

if [[ -d "$LOG_DIR" ]]; then
    log_info "删除 $LOG_DIR"
    rm -rf "$LOG_DIR"
fi

# /proc 那个只读测试路径不会真创建，无需清理
pass "清理完成"
