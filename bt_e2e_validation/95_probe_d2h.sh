#!/bin/bash
# 探针：定位 w13_weight_scale 的 D2H 失败到底是 dtype(float8_e8m0fnu) 还是布局(非连续) 引起
#
# 每个 case 独立进程、独立卡，避免一次 aicpu 故障污染后续结论。
# 用法：bash 95_probe_d2h.sh
set -u

# 与真实权重一致的形状：逻辑 (48, 80, 4608, 2)，stride (737280, 2, 160, 1)
# 等价于 contiguous (48, 4608, 80, 2) 做 permute(0, 2, 1, 3)。
CASES="A B C D E F G"

echo "torch/npu 版本"
python -c "import torch, torch_npu; print('torch', torch.__version__, 'torch_npu', torch_npu.__version__)"

dev=0
for CASE_ID in $CASES; do
    echo "=============================================="
    echo "case ${CASE_ID} (device=${dev})"
    ASCEND_RT_VISIBLE_DEVICES="$dev" CASE_ID="$CASE_ID" python - <<'PY'
import os
import traceback

import torch
import torch_npu

case = os.environ["CASE_ID"]
E8 = torch.float8_e8m0fnu
SHAPE = (48, 4608, 80, 2)  # 连续 base 的形状


def make(dtype=E8, permute=None):
    t = torch.empty(SHAPE, dtype=dtype, device="npu")
    return t.permute(0, 2, 1, 3) if permute else t


try:
    if case == "A":
        # 基线：e8m0 连续张量直接 D2H
        t = make()
    elif case == "B":
        # 真实布局：e8m0 的 permute 视图
        t = make(permute=True)
    elif case == "C":
        # 对照：同样布局但 bf16，验证布局本身没问题
        t = make(dtype=torch.bfloat16, permute=True)
    elif case == "D":
        # e8m0 permute 视图 -> 设备侧 continuous -> D2H
        t = make(permute=True).contiguous()
    elif case == "E":
        # e8m0 permute 视图 -> 还原到连续 base 顺序 -> D2H
        t = make(permute=True).permute(0, 2, 1, 3)
    elif case == "F":
        # e8m0 permute 视图 -> bitcast uint8 -> D2H
        t = make(permute=True).view(torch.uint8)
    elif case == "G":
        # e8m0 permute 视图 -> 先 D2H 到固定内存再拷贝（模拟分步）
        t = make(permute=True)
        host = torch.empty(t.shape, dtype=t.dtype, pin_memory=False)
        host.copy_(t)
        torch.npu.synchronize()
        print("RESULT=OK", tuple(host.shape), host.dtype)
        raise SystemExit(0)

    print("shape", tuple(t.shape), "stride", tuple(t.stride()), "dtype", t.dtype, "contig", t.is_contiguous())
    cpu = t.to("cpu")
    torch.npu.synchronize()
    print("RESULT=OK", tuple(cpu.shape), cpu.dtype)
except SystemExit:
    raise
except Exception as exc:  # noqa: BLE001 - 探针需要把任何错误都打出来
    print("RESULT=FAIL", type(exc).__name__, str(exc)[:400])
    traceback.print_exc()
PY
    echo "--- case ${CASE_ID} 结束 (exit=$?) ---"
    dev=$((dev + 1))
done

echo "=============================================="
echo "判定方法："
echo "  A 失败            -> dtype 本身不支持 D2H"
echo "  A 成功 B 失败 C 成功 -> 非连续 e8m0 视图的 D2H 有问题"
echo "  D/F/G 成功        -> 对应的规避方式可用"
