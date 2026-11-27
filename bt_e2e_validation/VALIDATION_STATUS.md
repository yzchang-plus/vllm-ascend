# BigTensorLoader 验证进度（交接文档）

- 分支：`origin/bigtensors`（fork: yzchang-plus/vllm-ascend），上游 PR #16734
- 目标：修复测试人员提出的 5 个问题，且不回归已验证合入的功能
- 最近验证时间：2026-11-26，机器 work-19（Ascend950DT ×8，torch 2.10.0+cpu / torch_npu 2.10.0.post7）

## 一、总体状态

| # | 问题 | 状态 | 结论依据 |
|---|---|---|---|
| 1 | 多 P/D 共用 checkpoint_path 冲突 | **待验证**（需 8 卡） | 代码就绪 + UT 覆盖；E2E 未跑 |
| 2 | 快照保存主机内存随模型线性增长 | **代码已修，待采内存曲线** | UT 52 passed；02 跑时采 `/tmp/mem_02.log` 即可闭环 |
| 3 | `FRACTAL_NZ_C0_16`（format 50）不被识别 | **保存侧已验**；恢复侧 UT 已验 | 真实模型 47GB/rank 序列化成功；恢复侧待 02 二次启动确认 |
| 4 | 投机推理 target/drafter 同 rank 写入冲突 | **待验证**（需 4 卡） | 代码就绪 + UT 覆盖；E2E 未跑 |
| 5 | 保存异常导致服务退出 | **已验证 PASS** | E2E 两轮：服务存活 + chat completion 正常返回 |

UT 现状：`pytest -sv tests/ut/model_loader/bigtensorloader` → **52 passed**（含 ND、format 29、packed NZ、NZ C0 变体、4D permuted MX scale、别名、容器、快照复用、损坏回退、有界缓冲、只读路径 fail-fast）。

## 二、各问题验证细节

### 问题 5（已 PASS，可复验）
场景：`03_save_failure.sh`，DP8 + checkpoint_path 指向 `/proc`（只读）。
已观察到的行为：
- 保存失败仅打日志：`snapshot save FAILED; continuing with the freshly loaded model`（同步序列化路径）
  与 `snapshot background disk write FAILED; the loaded model is unaffected`（后台写盘路径）
- 引擎继续启动直至 serving，curl chat completion 正常返回
- 写盘失败会 fail-fast 回传生产者，不再白白做几十 GB 的 D2H
- 当前脚本默认 `cudagraph_mode=NONE`（capture 阶段存在与本功能无关的崩溃，见第四节）

### 问题 3（保存侧已 PASS）
场景：`03_save_failure.sh` 的模型 `/mnt/share/weight/DeepSeek-V4.1-Flash`（含 format 50 的 NZ 专家权重、
4D permuted `float8_e8m0fnu` MX scale）。
已观察到的行为：每 rank `tensors=3120`、约 47GB 全部序列化成功，无 aicpu 异常。
关键技术点（修复后）：
- NZ 权重 permute 回连续 base 再 D2H（aicpu Transpose kernel 对 e8m0 等内格式直接崩溃，507018）
- manifest 记录 `format`/`permute`/`save_shape`，恢复时重放视图与 NZ cast recipe
- 恢复侧 UT：`test_nz_c0_variant_preserves_cast_spec` 覆盖 bulk_h2d 两条路径

### 问题 2（代码已修，差一次采样）
修复：`_SnapshotBlobWriter` 流式写盘，生产者-消费者经 512MB 有界队列，主机内存恒定；
sha256 增量计算。02 运行时另开终端采样：
```bash
while true; do date +%T; free -g | sed -n 2p; sleep 10; done | tee /tmp/mem_02.log
```
判据：available 不出现与模型体量（约 47GB/rank）相当的骤降。

### 问题 1 / 4（待跑）
- 问题 1：`01_pd_shared_path.sh`，prefill（DP1/PCP4）与 decode（DP4）共用同一 checkpoint_path，
  需 **8 卡**。
- 问题 4：`02_mtp_target_drafter.sh`，DP4 + `--speculative-config dspark`，需 **4 卡**。
  卡不够时可用 `BT_DEVICES="0 1 6 7" bash ...` 跳过被占卡（如 NPU 2-5 被其他容器占用时）。
  判据：首轮 target/drafter 各产生独立 hash 目录（≥2 个新增）；二次启动 ≥2 次 `snapshot restore`。

## 三、复验步骤（work-19）

```bash
cd /vllm-workspace/vllm-ascend
git fetch && git reset --hard origin/bigtensors

# 1) UT（约 4 秒）
pytest -sv tests/ut/model_loader/bigtensorloader        # 期望 52 passed

# 2) 问题 5（DP8，8 卡全空时）
bash bt_e2e_validation/03_save_failure.sh               # 期望末段 PASS

# 3) 问题 2/3恢复侧/4（DP4；卡不足时加 BT_DEVICES）
BT_DEVICES="0 1 6 7" bash bt_e2e_validation/02_mtp_target_drafter.sh

# 4) 问题 1（DP8 拓扑，需 8 卡）
bash bt_e2e_validation/01_pd_shared_path.sh
```

环境注意：
- 快照目录默认 `SNAPSHOT_ROOT=/mnt/share/l00943473/bigtensor_validation`（NAS），
  02 一轮约 230GB，跑前确认空间；可 `export SNAPSHOT_ROOT=/vllm-workspace/bt_snap` 换本地 NVMe。
- 本机镜像设了 http_proxy：脚本已在引擎子进程自动清除，无需手工处理。
- 启动超时默认 1800s，可 `BT_START_TIMEOUT=3600` 放宽。
- 出问题时先跑 `bash bt_e2e_validation/99_diag.sh <日志前缀> <进程数>` 采集诊断
  （如 `99_diag.sh 03_save 8`），把输出整段贴回即可。

## 四、已知无关问题（不阻塞本 PR 判定）

1. **02 的启动配置**：`--max-num-batched-tokens 1024` 与 dspark speculative 的 2048 buffer 需求冲突，
   表现为 `build_sas_metadata` 处 `expanded size (2048) must match existing size (1024)`。
   这是复制测试人员原始命令带来的既有问题，`LOAD_FORMAT=auto` 基线同样失败；
   若复现，把该参数改 2048 后重跑（不影响问题 4 的判定逻辑）。
2. **cudagraph capture 崩溃**：`No device metadata submission is in flight`（03 曾在 capture 阶段崩），
   属 v2 前向 metadata 队列问题，与权重加载无关；03 脚本已默认 `cudagraph_mode=NONE` 绕开。
3. **强杀进程残留**：`kill -KILL` 后 worker 可能被 re-parent 逃过清理、占住几十 GB 显存
   （npu-smi 显示 `VLLMWorker_*`）。脚本已加按名补杀；手工清理：`pkill -9 -f "VLLMWorker"`。
