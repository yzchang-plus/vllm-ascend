# BigTensorLoader 端到端验证脚本

针对 PR #16734 修复的 5 个问题做端到端验证。基于测试人员 2026-11-22 提供的真实启动命令改造。

## 前置条件

1. Linux/NPU 机器，CANN 已安装（验证机 work-11 / work-19）
2. 已 `pip install -e .` 安装包含本修复的 `vllm_ascend`
3. 模型已就位：
   - `/mnt/share/weight/DeepSeek-V4-Flash-0731-w4a4c8-new`
   - `/mnt/share/weight/DeepSeek-V4.1-Flash`
4. NAS 共享目录可写：`/mnt/share/l00943473/`
5. Mooncake / KV connector 依赖已就绪（复用现有 PD 部署）

## 问题与脚本对应

| # | 问题 | 脚本 | 验证核心 |
|---|---|---|---|
| 1 | 多 P、多 D 共用 checkpoint_path 时 rank 文件冲突 | `01_pd_shared_path.sh` | P 和 D 用**同一个** checkpoint_path，验证指纹自动按 kv_role 分目录 |
| 4 | 投机推理 target/drafter 同进程同 rank 冲突 | `02_mtp_target_drafter.sh` | 单进程 MTP 启动，target/drafter 用同一个 checkpoint_path，验证指纹按 model_role 分目录 |
| 5 | 保存异常导致服务退出 | `03_save_failure.sh` | 注入只读 checkpoint_path，验证服务不退出且日志含 `snapshot save FAILED` |
| 2 | 主机内存随模型大小线性增长 | 待代码修复后验证 | 当前实现仍是全量 D2H + bytes 累积，已知问题 |
| 3 | FRACTAL_NZ_C0_16 格式 | 待代码修复后验证 | 当前 `_is_packed_nz` 不识别此 format |

## 执行顺序

**重要**：PD 场景需要 work-11 和 work-19 两台机器协同，分别在对应机器执行 `01_pd_shared_path.sh` 的 P 和 D 角色。

```bash
# === 单机 work-11：先跑问题 5（最简单，验证容错）===
cd /path/to/bt_e2e_validation
bash 03_save_failure.sh

# === 单机 work-11：跑问题 4（MTP）===
bash 02_mtp_target_drafter.sh

# === 跨机 work-19 + work-11：跑问题 1（PD 共享路径）===
# 终端 A（work-19，prefill）：
bash 01_pd_shared_path.sh prefill
# 终端 B（work-11，decode）：
bash 01_pd_shared_path.sh decode
```

## 通过判据

每个脚本退出码 0 = 通过，非 0 = 失败。脚本会打印：
- 启动前/后的快照目录结构
- 关键日志片段（snapshot save/restore）
- PASS/FAIL 判定与失败原因

## 清理

```bash
bash 99_cleanup.sh
```

会先按 pidfile 清理本目录脚本启动的残留引擎进程，再删除本次验证产生的快照目录（`/mnt/share/l00943473/bigtensor_validation/`）和日志目录，不影响测试人员原有数据。

## 进程清理机制

脚本内置三层清理，避免残留引擎占用 NPU：

1. **正常结束**：每个脚本验证完成后自动 `kill_engine` 停止引擎进程组（含 DP worker）。
2. **异常/中断兜底**：脚本安装 `EXIT/INT/TERM` trap，Ctrl+C 或失败退出时自动清理本次启动的引擎。
3. **跨次运行残留**：启动的 PID 会记录到 `${LOG_DIR}/bt_e2e_engine.pid`，下次运行同一脚本开始时自动清理上次残留；`99_cleanup.sh` 也会处理。

注意：清理**只针对本目录脚本记录的 PID**（pidfile 精确匹配），不会 `pkill` 掉机器上其他人或测试人员自己的部署。

若需要只读查看当前机器上的 vllm 进程：

```bash
pgrep -af "vllm serve"
```

## 注意事项

- 不要在生产部署上跑这些脚本，会重启引擎
- `01_pd_shared_path.sh` 故意把 P 和 D 的 checkpoint_path 改成同一个，**与测试人员原始命令不同**——这是修复后的目标用法
- 首次启动会产生新快照；再次启动应命中 fast restore，能验证快照复用
- `03_save_failure.sh` 会启动一个真实引擎实例，跑完后用 SIGTERM 关闭
