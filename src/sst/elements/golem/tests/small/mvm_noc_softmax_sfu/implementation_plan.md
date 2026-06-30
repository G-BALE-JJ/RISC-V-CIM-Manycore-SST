# MVM NoC Softmax SFU 实现计划

## 总目标

在 Golem SST 组件中新增独立 `golem.SFU` 子组件，并在
`mvm_noc_softmax_sfu` small test 中用 SFU 完成跨 tile full row-wise
online softmax。

第一版目标不是追求最复杂的 NoC 建模，而是先把功能边界、数值正确性和可调试性做稳：

- SFU 是独立子组件，挂在 RoCC 下，类似 `array`、`global_memory`。
- 名称只叫 `SFU`，不叫 `SoftmaxSFU`。
- softmax 必须是完整行 softmax，不接受 tile-local 结果。
- 跨 tile 合并使用 online softmax：
  - tile 本地计算 `(tile_m, tile_l)`。
  - row-owner/reducer 合并为 `(global_m, global_l)`。
  - 所有 tile 用全行统计做 normalize。
- 第一版使用 SFU-managed reducer state，后续再扩展为显式 SimpleNetwork 消息。
- 第一版沿用现有 workload DMA 机制搬运 C tile，SFU 负责数学计算、跨 tile 合并和同步。

## 兼容性边界：不得影响原始 GEMM 功能

所有后续代码修改都必须遵守一条硬边界：**不能影响项目原始 GEMM 功能实现和已有
GEMM small tests 的默认行为**。

具体规则：

- `GOLEM_SFU_ENABLE` 默认必须为 `0` 或 false。
- 不设置 SFU 相关环境变量时，`cpu_builder.py` 不挂载 `golem.SFU`，RoCC 行为保持原样。
- 原有 GEMM RoCC 指令编码、array 调用路径、DMA 调度路径、global_memory 默认行为不能改语义。
- SFU 的新 func7 编码只能新增，不能复用或改变已有 `0x11` 到 `0x16` 的含义。
- 新 workload 必须放在 `mvm_noc_softmax_sfu`，不修改 `mvm_noc_softmax_cpu` 的默认执行逻辑。
- 对共享文件的修改必须是可回退、默认关闭、向后兼容的：
  - `golem.cc` 只新增 include / 注册入口。
  - `Makefile.am` 只新增 SFU 源文件。
  - `rocc/roccAnalog.h` 只在新 func7 和 `sfu_enable` 分支下进入 SFU。
  - `cpu_builder.py` 只在 `GOLEM_SFU_ENABLE=1` 时挂载 SFU。
- 每次实现阶段验证时，除了新 SFU softmax case，还要至少跑一个原始 GEMM smoke test，
  证明默认 GEMM 路径未被破坏。

## 目标文件结构

新增组件文件：

```text
src/sst/elements/golem/sfu/sfu.h
src/sst/elements/golem/sfu/sfu.cc
```

修改组件注册和挂载：

```text
src/sst/elements/golem/golem.cc
src/sst/elements/golem/Makefile.am
src/sst/elements/golem/rocc/roccAnalog.h
src/sst/elements/golem/tests/architecture/cpu_builder.py
```

新增测试目录内容：

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/ex_instr.h
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/golem_softmax_sfu_runtime.h
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/golem_softmax_sfu_runtime.cpp
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_noc_dma_softmax_sfu.cpp
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/Makefile
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_noc_dma_softmax_sfu_pipeline.sh
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/verify_softmax_sfu_against_golden.py
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_verify_softmax_sfu_against_golden.py
```

## Phase 1: 先写 checker 和最小回归

目标：先固定正确性定义，避免实现过程中退回 tile-local softmax。

- [ ] 在 `verify_softmax_sfu_against_golden.py` 中实现完整 row-wise golden：
  - 从 GEMM 输入或输出 dump 重建 `C = A @ B`。
  - 对每一行计算 `softmax(C[row, :])`。
  - 检查 SFU 输出：
    - 所有元素 finite。
    - 每行和接近 1。
    - 与 golden softmax 在容差内一致。
- [ ] 在 `test_verify_softmax_sfu_against_golden.py` 中加入纯 Python 单元测试：
  - `64x64, block_n=64`：单 tile 行。
  - `128x128, block_n=64`：每行跨 2 个 tile。
  - `256x256, block_n=64`：每行跨 4 个 tile。
  - 包含大正数/大负数输入，验证数值稳定性。
- [ ] 运行：

```bash
python3 src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_verify_softmax_sfu_against_golden.py
```

预期结果：

- checker 明确以 full row-wise softmax 为准。
- 跨 tile case 如果人为输入 tile-local softmax，应能失败。

## Phase 2: 新增 SFU 子组件骨架

目标：让 `golem.SFU` 能被 SST 注册、构建、挂载，但先不接完整 softmax 行为。

- [ ] 新建 `src/sst/elements/golem/sfu/sfu.h`：
  - 定义 `SFUAPI : public SST::SubComponent`。
  - 定义 `SFU : public SFUAPI`。
  - 暴露方法：

```cpp
virtual bool issueSoftmaxTile(uint64_t descAddr, uint64_t tag) = 0;
virtual bool wait(uint64_t tag, uint64_t* status) = 0;
virtual void bindGlobalMemory(GlobalMemoryAPI* globalMem) = 0;
virtual void setCoreInfo(uint32_t coreId, uint32_t activeWorkerCores) = 0;
```

- [ ] 新建 `src/sst/elements/golem/sfu/sfu.cc`：
  - 注册 `golem.SFU`。
  - 读取参数：
    - `core_id`
    - `active_worker_cores`
    - `max_inflight`
    - `stats_latency`
    - `merge_latency`
    - `normalize_latency`
    - `verbose`
  - 注册统计项：
    - `sfu_ops_issued`
    - `sfu_softmax_rows`
    - `sfu_softmax_tiles`
    - `sfu_partial_submits`
    - `sfu_partial_done`
    - `sfu_credit_stalls`
    - `sfu_cross_tile_wait_cycles`
    - `sfu_retry_events`
- [ ] 修改 `golem.cc`：
  - include `sst/elements/golem/sfu/sfu.h`。
- [ ] 修改 `Makefile.am`：
  - 把 `sfu/sfu.h`、`sfu/sfu.cc` 加入 `libgolem_la_SOURCES`。
- [ ] 运行 golem 组件构建。

推荐命令：

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST
./scripts/build_and_install_local.sh --reconfigure --jobs 16
```

如果当前环境没有该脚本，则改用项目已有的局部构建入口；添加新源文件后需要确保
`Makefile.am` 重新生效。

预期结果：

- SST 能看到 `golem.SFU`。
- 还不要求 workload 调用 SFU。

## Phase 3: 定义 SFU descriptor 和 online reducer

目标：实现跨 tile online softmax 的核心状态机。

- [ ] 在 `sfu.h` 中定义 descriptor。RISC-V workload 先把 descriptor 写入本 core
  local GM，RoCC 指令通过 `rs1` 传 descriptor GM 地址。

```cpp
struct SFUSoftmaxTileDesc {
    uint64_t job_id;
    uint64_t local_input_gm_addr;
    uint64_t local_output_gm_addr;
    uint32_t global_m;
    uint32_t global_n;
    uint32_t block_m;
    uint32_t block_n;
    uint32_t m_tile;
    uint32_t n_tile;
    uint32_t valid_m;
    uint32_t valid_n;
    uint32_t n_tiles_per_row;
    uint32_t elem_bytes;
    uint32_t flags;
};
```

- [ ] 第一版数据边界：
  - workload 使用现有 DMA 把 HBM/remote C tile 搬到 `local_input_gm_addr`。
  - SFU 从 local GM 读 tile。
  - SFU 把归一化结果写到 `local_output_gm_addr`。
  - workload 在 `sfu.wait` 后用现有 DMA 把 local output 写回 HBM/remote C tile。
- [ ] 在 `sfu.cc` 中实现 tile 本地统计：
  - 对每个有效 row fragment 计算 `tile_m`。
  - 计算 `tile_l = sum(exp(x - tile_m))`。
  - 支持 `float32` 第一版；`elem_bytes` 先要求为 4。
- [ ] 实现 SFU-managed reducer state：
  - key 使用 `(job_id, global_row)`。
  - state 保存：
    - `m_acc`
    - `l_acc`
    - `partials_seen`
    - `n_tiles_expected`
    - `ready`
  - owner 映射：

```text
owner_core(row) = row % active_worker_cores
```

- [ ] 合并公式：

```text
m_new = max(m_old, tile_m)
l_new = l_old * exp(m_old - m_new)
      + tile_l * exp(tile_m - m_new)
```

- [ ] 当 `partials_seen == n_tiles_expected`：
  - 标记该 row 的 `(global_m, global_l)` ready。
  - 等待该 row 的所有 producer tile normalize 完成后清理 state。
- [ ] 实现 normalize：

```text
y = exp(x - global_m) / global_l
```

- [ ] 加入基本限流：
  - 每个 SFU `max_inflight`。
  - reducer row state 数量上限。
  - 超限时 `issueSoftmaxTile` 返回 false，由 RoCC 下次 tick retry。

预期结果：

- 单进程 SST 中多个 core 的 SFU 可以通过共享 reducer state 完成跨 tile softmax。
- 不使用 HBM 非原子 read-modify-write reduction buffer。

## Phase 4: 接入 RoCC 指令

目标：RISC-V workload 能通过 RoCC 触发 SFU。

- [ ] 在 `rocc/roccAnalog.h` 中加入 include：

```cpp
#include "sst/elements/golem/sfu/sfu.h"
```

- [ ] 增加 func7 编码，避开已有 `0x11` 到 `0x16`：

```cpp
static constexpr uint32_t GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE = 0x17;
static constexpr uint32_t GOLEM_ROCC_FUNC7_SFU_WAIT         = 0x18;
```

- [ ] 在 RoCC 类成员中加入：

```cpp
Golem::SFUAPI* sfu = nullptr;
```

- [ ] 在 RoCC 构造函数中加载：

```cpp
sfu = loadUserSubComponent<Golem::SFUAPI>("sfu", ComponentInfo::SHARE_NONE);
```

- [ ] 如果 `sfu_enable=1` 但 `sfu == nullptr`，直接 fatal，避免静默走 CPU fallback。
- [ ] 给 SFU 绑定：
  - `globalMem`
  - `coreID`
  - `active_worker_cores`
- [ ] 在 tick 的非阻塞分支中处理：
  - `GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE`
    - `rs1 = descriptor local GM address`
    - `rs2 = tag`，第一版可传 0。
    - 如果 `sfu->issueSoftmaxTile(...)` 成功，pop command 并写 response。
    - 如果 SFU inflight 满，保留 command，下一 tick retry。
  - `GOLEM_ROCC_FUNC7_SFU_WAIT`
    - 等待 tag 或当前 core 所有 SFU 请求完成。
    - 完成后 `completeRoCC(status)`。
- [ ] 在 `finish()` 中调用 SFU 的 finish/flush 逻辑。

预期结果：

- RoCC 能识别 SFU 指令。
- SFU 忙时不会丢命令，而是像现有 batch/DMA 路径一样 retry。

## Phase 5: 在 cpu_builder.py 挂载 SFU

目标：只在新测试里打开 SFU，不影响旧测试和原始 GEMM 默认行为。

- [ ] 修改 `src/sst/elements/golem/tests/architecture/cpu_builder.py`：
  - 读取环境变量：

```python
sfu_enable = int(os.getenv("GOLEM_SFU_ENABLE", "0")) != 0
```

  - 当 `sfu_enable` 为真时：

```python
sfu = cpu_rocc.setSubComponent("sfu", "golem.SFU")
```

  - 给 SFU 传参：
    - `core_id`
    - `active_worker_cores`
    - `max_inflight`
    - `stats_latency`
    - `merge_latency`
    - `normalize_latency`
    - `verbose`
  - 给 RoCC 传 `sfu_enable`，用于缺失子组件 fatal。

预期结果：

- 旧 workload 不设置 `GOLEM_SFU_ENABLE` 时行为不变。
- 新 workload 设置 `GOLEM_SFU_ENABLE=1` 时挂载 `golem.SFU`。
- 原始 GEMM 测试不需要知道 SFU 的存在。

## Phase 6: 新增 RISC-V workload runtime

目标：把原 CPU softmax fallback 替换成 SFU 指令路径。

- [ ] 新建 `ex_instr.h`：
  - 复用现有 `.insn r 0x0b, 7, func7, rd, rs1, rs2` 风格。
  - 增加：

```cpp
uint64_t sfu_softmax_tile_async(uint64_t desc_gm_addr, uint64_t tag);
uint64_t sfu_wait(uint64_t tag);
```

- [ ] 新建 `golem_softmax_sfu_runtime.h/.cpp`：
  - 定义与 C++/SFU descriptor 对齐的 `SFUSoftmaxTileDesc`。
  - 提供 `run_softmax_sfu_tiles(...)`。
- [ ] 每个 core 按自己负责的 C tile 执行：
  - 用现有 DMA helper 把 remote/HBM C tile 搬到本 core local GM buffer。
  - 写 `SFUSoftmaxTileDesc` 到本 core local GM。
  - 调用 `sfu_softmax_tile_async(desc_gm_addr, tag)`。
  - 调用 `sfu_wait(tag)`。
  - 用现有 DMA helper 把 `local_output_gm_addr` 写回原 C tile 地址。
- [ ] 同一行跨 tile 的所有 core 必须都能并发 issue SFU command。
- [ ] 新建 `test_noc_dma_softmax_sfu.cpp`：
  - 复用现有 `mvm_noc_softmax_cpu` 的 GEMM 输入生成和 MVM pipeline。
  - 删除 CPU fallback softmax 计算路径。
  - GEMM 完成后调用 SFU runtime。

预期结果：

- softmax 数学和跨 tile同步由 SFU 完成。
- tile 数据搬运继续复用已有 DMA 机制，风险较低。

## Phase 7: 新增运行脚本和构建入口

目标：新测试目录可以独立构建和运行。

- [ ] 新建 `Makefile`：
  - 参考 `mvm_noc_softmax_cpu` 或 `mvm_noc_int_array`。
  - 生成 RISC-V ELF：`test_noc_dma_softmax_sfu`。
- [ ] 新建 `run_noc_dma_softmax_sfu_pipeline.sh`：
  - 设置：

```bash
export GOLEM_SFU_ENABLE=1
export VANADIS_EXE=.../test_noc_dma_softmax_sfu
```

  - 复用现有 NoC/GEMM 参数：
    - `block_m=64`
    - `block_n=64`
    - `active_worker_cores=16`
  - 先跑小规模：
    - `64x64`
    - `128x128`
  - 再跑：
    - `256x256`
    - `512x512`
- [ ] 输出 dump 文件供 checker 使用。

预期结果：

- 用户可以在新目录下单独运行 SFU softmax 测试。
- 旧 `mvm_noc_softmax_cpu` 不受影响。

## Phase 8: 验证顺序

目标：按风险从低到高验证，同时确认原始 GEMM 默认路径没有被 SFU 改动破坏。

- [ ] Python checker 单测：

```bash
python3 src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_verify_softmax_sfu_against_golden.py
```

- [ ] golem 组件构建：

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST
./scripts/build_and_install_local.sh --reconfigure --jobs 16
```

- [ ] 原始 GEMM smoke test：
  - 使用一个现有 GEMM small test。
  - 不设置 `GOLEM_SFU_ENABLE`。
  - 期望结果与修改前一致。

- [ ] RISC-V workload 构建：

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu
make clean
make
```

- [ ] 单 tile 行 correctness：

```bash
./run_noc_dma_softmax_sfu_pipeline.sh 64 64
python3 verify_softmax_sfu_against_golden.py --m 64 --n 64 --block-n 64
```

- [ ] 跨 tile correctness：

```bash
./run_noc_dma_softmax_sfu_pipeline.sh 128 128
python3 verify_softmax_sfu_against_golden.py --m 128 --n 128 --block-n 64

./run_noc_dma_softmax_sfu_pipeline.sh 256 256
python3 verify_softmax_sfu_against_golden.py --m 256 --n 256 --block-n 64
```

- [ ] 压力验证：

```bash
./run_noc_dma_softmax_sfu_pipeline.sh 512 512
python3 verify_softmax_sfu_against_golden.py --m 512 --n 512 --block-n 64
```

## 构建边界

- 只改 `src/sst/elements/golem/sfu/`、`rocc/`、`golem.cc`、`Makefile.am` 时：
  - 需要重构建 golem element。
  - 因为新增源文件进 `Makefile.am`，第一次需要 reconfigure。
- 只改 `mvm_noc_softmax_sfu` workload：
  - 只需要在该目录 `make clean && make`。
- 只改 Python checker：
  - 不需要重编 SST。
- 如果后续修改 `globalmemory` 或 SST core 架构脚本的接口：
  - 需要完整重构建并重新跑至少 `64x64`、`128x128`。

## 关键风险与处理

| 风险 | 处理方式 |
| --- | --- |
| 多核同时更新同一行统计导致丢更新 | 不使用 HBM read-modify-write，改用 SFU-managed reducer state |
| 某些 tile 等不到 global stats | 加入 `n_tiles_expected`、ready 状态、wait retry 和 timeout 统计 |
| tile 数据地址不在 local GM | 第一版 workload 显式 DMA 到 local GM 后再交给 SFU |
| 旧测试被 SFU 修改影响 | `GOLEM_SFU_ENABLE` 默认关闭，只在新测试开启 |
| 原始 GEMM 路径被共享文件修改误伤 | 所有共享文件改动保持默认不变，并增加原始 GEMM smoke test |
| descriptor C++/RISC-V 两边布局不一致 | 使用固定宽度整数，添加 static_assert 检查 size/offset |
| 误实现成 tile-local softmax | checker 必须包含 `N > block_n` case，并对 tile-local 结果失败 |

## 完成标准

- `golem.SFU` 可以被 SST 注册并通过 `cpu_builder.py` 挂载。
- 新 RoCC SFU 指令能从 RISC-V workload 发出并完成。
- 不设置 `GOLEM_SFU_ENABLE` 时，原始 GEMM smoke test 通过，默认 GEMM 路径行为不变。
- `64x64`、`128x128`、`256x256` 至少通过 golden checker。
- `512x512` 能运行并给出数值正确结果或明确性能/资源瓶颈诊断。
- `design.md`、`task_plan.md`、`progress.md`、`findings.md` 与实际实现状态一致。
