# MVM NoC Softmax SFU 实现计划

## 当前状态

- 已实现独立 `golem.SFU` 子组件，并通过 `GOLEM_SFU_ENABLE` 默认关闭地挂载到 RoCC。
- 当前对 workload 暴露的是 fused full row-wise softmax，不是 standalone primitive。
- fused softmax 已通过 `64x64`、`128x128`、`256x256` golden checker。
- 旧 GEMM smoke 在不设置 `GOLEM_SFU_ENABLE` 时已验证通过。
- 下一步是 `512x512` 压力验证，以及 Phase 9 通用 SFU standalone primitive ABI。

## 总目标

在 Golem SST 组件中新增独立 `golem.SFU` 子组件，并在
`mvm_noc_softmax_sfu` small test 中用 SFU 完成跨 tile full row-wise
online softmax。

第一版目标不是追求最复杂的 NoC 建模，而是先把功能边界、数值正确性和可调试性做稳：

- SFU 是独立子组件，挂在 RoCC 下，类似 `array`、`global_memory`。
- 名称只叫 `SFU`，不叫 `SoftmaxSFU`。
- SFU 长期是通用 Special Function Unit，应保留扩展到 `exp`、`log`、
  `reciprocal`、`rsqrt`、`sigmoid`、`tanh`、`layernorm`、`gelu` 等特殊函数
  或 fused op 的空间。
- 当前第一版只实现 fused softmax operation，不单独暴露 standalone primitive
  RoCC 指令；softmax 内部按 `max/reduction -> exp -> sum/reduction ->
  reciprocal/divide -> normalize` 的 SFU primitive 思路组织。
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

本阶段实现的是 SFU 的第一个 fused op：softmax。接口命名可以先使用
`issueSoftmaxTile` 和 `SFUSoftmaxTileDesc`，但实现边界不要把 SFU 组件本身写成
softmax-only：统计、参数、状态机命名应允许后续加入 standalone `exp`、`log`、
`reciprocal`、`rsqrt` 或其他 fused op。

- [x] 在 `sfu.h` 中定义 descriptor。RISC-V workload 先把 descriptor 写入本 core
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

- [x] 第一版数据边界：
  - workload 使用现有 DMA 把 HBM/remote C tile 搬到 `local_input_gm_addr`。
  - SFU 从 local GM 读 tile。
  - SFU 把归一化结果写到 `local_output_gm_addr`。
  - workload 在 `sfu.wait` 后用现有 DMA 把 local output 写回 HBM/remote C tile。
- [x] 在 `sfu.cc` 中实现 tile 本地统计：
  - 对每个有效 row fragment 计算 `tile_m`。
  - 计算 `tile_l = sum(exp(x - tile_m))`。
  - 支持 `float32` 第一版；`elem_bytes` 先要求为 4。
- [x] 实现 SFU-managed reducer state：
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

- [x] 合并公式：

```text
m_new = max(m_old, tile_m)
l_new = l_old * exp(m_old - m_new)
      + tile_l * exp(tile_m - m_new)
```

- [x] 当 `partials_seen == n_tiles_expected`：
  - 标记该 row 的 `(global_m, global_l)` ready。
  - 等待该 row 的所有 producer tile normalize 完成后清理 state。
- [x] 实现 normalize：

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

- [x] 在 `rocc/roccAnalog.h` 中加入 include：

```cpp
#include "sst/elements/golem/sfu/sfu.h"
```

- [x] 增加 func7 编码，避开已有 `0x11` 到 `0x16`：

```cpp
static constexpr uint32_t GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE = 0x17;
static constexpr uint32_t GOLEM_ROCC_FUNC7_SFU_WAIT         = 0x18;
```

- [x] 在 RoCC 类成员中加入：

```cpp
Golem::SFUAPI* sfu = nullptr;
```

- [x] 在 RoCC 构造函数中加载：

```cpp
sfu = loadUserSubComponent<Golem::SFUAPI>("sfu", ComponentInfo::SHARE_NONE);
```

- [x] 如果 `sfu_enable=1` 但 `sfu == nullptr`，直接 fatal，避免静默走 CPU fallback。
- [x] 给 SFU 绑定：
  - `globalMem`
  - `coreID`
  - `active_worker_cores`
- [x] 在 tick 的非阻塞分支中处理：
  - `GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE`
    - `rs1 = descriptor local GM address`
    - `rs2 = tag`，第一版可传 0。
    - 如果 `sfu->issueSoftmaxTile(...)` 成功，pop command 并写 response。
    - 如果 SFU inflight 满，保留 command，下一 tick retry。
  - `GOLEM_ROCC_FUNC7_SFU_WAIT`
    - 等待 tag 或当前 core 所有 SFU 请求完成。
    - 完成后 `completeRoCC(status)`。
- [x] 在 `finish()` 中调用 SFU 的 finish/flush 逻辑。

预期结果：

- RoCC 能识别 SFU 指令。
- SFU 忙时不会丢命令，而是像现有 batch/DMA 路径一样 retry。

## Phase 5: 在 cpu_builder.py 挂载 SFU

目标：只在新测试里打开 SFU，不影响旧测试和原始 GEMM 默认行为。

- [x] 修改 `src/sst/elements/golem/tests/architecture/cpu_builder.py`：
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

- [x] 新建 `ex_instr.h`：
  - 复用现有 `.insn r 0x0b, 7, func7, rd, rs1, rs2` 风格。
  - 增加：

```cpp
uint64_t sfu_softmax_tile_async(uint64_t desc_gm_addr, uint64_t tag);
uint64_t sfu_wait(uint64_t tag);
```

- [x] 新建 `golem_softmax_sfu_runtime.h/.cpp`：
  - 定义与 C++/SFU descriptor 对齐的 `SFUSoftmaxTileDesc`。
  - 提供 `run_softmax_sfu_tiles(...)`。
- [x] 每个 core 按自己负责的 C tile 执行：
  - 用现有 DMA helper 把 remote/HBM C tile 搬到本 core local GM buffer。
  - 写 `SFUSoftmaxTileDesc` 到本 core local GM。
  - 调用 `sfu_softmax_tile_async(desc_gm_addr, tag)`。
  - 调用 `sfu_wait(tag)`。
  - 用现有 DMA helper 把 `local_output_gm_addr` 写回原 C tile 地址。
- [x] 同一行跨 tile 的所有 core 必须都能并发 issue SFU command。
- [x] 新建 `test_noc_dma_softmax_sfu.cpp`：
  - 复用现有 `mvm_noc_softmax_cpu` 的 GEMM 输入生成和 MVM pipeline。
  - 删除 CPU fallback softmax 计算路径。
  - GEMM 完成后调用 SFU runtime。

预期结果：

- softmax 数学和跨 tile同步由 SFU 完成。
- tile 数据搬运继续复用已有 DMA 机制，风险较低。

## Phase 7: 新增运行脚本和构建入口

目标：新测试目录可以独立构建和运行。

- [x] 新建 `Makefile`：
  - 参考 `mvm_noc_softmax_cpu` 或 `mvm_noc_int_array`。
  - 生成 RISC-V ELF：`test_noc_dma_softmax_sfu`。
- [x] 新建 `run_noc_dma_softmax_sfu_pipeline.sh`：
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
- [x] 输出 dump 文件供 checker 使用。

预期结果：

- 用户可以在新目录下单独运行 SFU softmax 测试。
- 旧 `mvm_noc_softmax_cpu` 不受影响。

当前验证状态（2026-06-30）：

- `--verify-softmax` 会在 SST 运行后自动从 HBM output 解包 C tensor，并调用
  `verify_softmax_sfu_against_golden.py` 对比 `softmax(A@B)`。
- `64x64, block=64x64` 已通过完整 SFU 路径 golden checker。
- `128x128, block=64x64` 已通过跨 N tile online reducer golden checker。

## Phase 8: 验证顺序

目标：按风险从低到高验证，同时确认原始 GEMM 默认路径没有被 SFU 改动破坏。

- [x] Python checker 单测：

```bash
python3 src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_verify_softmax_sfu_against_golden.py
```

- [x] golem 组件构建：

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST
./scripts/build_and_install_local.sh --reconfigure --jobs 16
```

- [x] 原始 GEMM smoke test：
  - 使用一个现有 GEMM small test。
  - 不设置 `GOLEM_SFU_ENABLE`。
  - 期望结果与修改前一致。

验证记录（2026-06-30）：

```bash
env -u GOLEM_SFU_ENABLE \
  GOLEM_ARCH_SCRIPT=small/mvm_noc_softmax_cpu/ncores_selfcom_dma_softmax_archive.py \
  GOLEM_GROUP_MANAGER_ENABLE=0 \
  GOLEM_CTRL_LINK_ENABLE=0 \
  GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0 \
  ./src/sst/elements/golem/tests/run_noc_dma_pipeline.sh \
    --gemm-m 64 --gemm-n 64 --gemm-k 64 \
    --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
    --dtype fp32 --tensor-source sample --verify-c
```

结果：`[VERIFY-C] PASS dtype=fp32 sampled=64 mismatches=0 max_abs_diff=0`。
运行时需要显式使用当前 build `.libs` 中的 `libgolem.so`，避免旧 install
library 与当前源码/build tree 不一致。

- [x] RISC-V workload 构建：

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu
make clean
make
```

- [x] 单 tile 行 correctness：

```bash
./run_noc_dma_softmax_sfu_pipeline.sh \
  --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --verify-softmax
```

- [x] 跨 tile correctness（128x128 已完成）：

```bash
./run_noc_dma_softmax_sfu_pipeline.sh \
  --gemm-m 128 --gemm-n 128 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --verify-softmax
```

- [x] 跨 tile correctness（256x256 已完成）：

```bash
./run_noc_dma_softmax_sfu_pipeline.sh \
  --gemm-m 256 --gemm-n 256 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --verify-softmax
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

## Phase 9: 通用 SFU standalone primitive 规划

目标：在当前 fused softmax 已闭环的基础上，把 `golem.SFU` 演进成通用
Special Function Unit。当前 fused softmax 不删除、不降级；后续 standalone
primitive 作为新的 opcode/descriptor 类型加入，供其他 workload 直接调用。

### 设计原则

- fused softmax 是第一类 SFU fused op，继续保留：
  - 对外接口仍是 `issueSoftmaxTile` / `SFU_SOFTMAX_TILE`。
  - 内部继续按 `max/reduction -> exp -> sum/reduction -> reciprocal/divide -> normalize`
    组织。
- standalone primitive 是第二条接口线，不应复用 softmax descriptor：
  - 每个 primitive 明确输入、输出、元素数、dtype、stride、近似模式。
  - primitive 不隐式做跨 tile reduction，除非显式定义 reduce primitive。
- 通用 SFU ABI 要能支持：
  - unary elementwise：`exp`、`log`、`reciprocal`、`rsqrt`、`sqrt`、`tanh`、`sigmoid`。
  - binary elementwise：`add`、`mul`、`div` 可选，优先级低于特殊函数。
  - reduction：`reduce_max`、`reduce_sum`，用于后续把 softmax 拆成 primitive pipeline。
  - fused op：`softmax`、`layernorm`、`gelu`，作为更高层组合操作。
- 默认行为必须仍然保持兼容：
  - `GOLEM_SFU_ENABLE=0` 时旧 GEMM 路径不挂载 SFU。
  - 不新增 standalone primitive workload 时，当前 softmax 测试行为不变。

### 建议 ABI

新增通用 primitive descriptor，例如：

```cpp
struct SFUPrimitiveDesc {
    uint64_t job_id;
    uint64_t input0_gm_addr;
    uint64_t input1_gm_addr;
    uint64_t output_gm_addr;
    uint32_t op;
    uint32_t dtype;
    uint32_t elem_count;
    uint32_t input0_stride_bytes;
    uint32_t input1_stride_bytes;
    uint32_t output_stride_bytes;
    uint32_t flags;
    uint32_t approx_mode;
};
```

初始 `op` 枚举建议：

```text
0x01 = EXP
0x02 = LOG
0x03 = RECIPROCAL
0x04 = RSQRT
0x05 = SQRT
0x06 = TANH
0x07 = SIGMOID
0x20 = REDUCE_MAX
0x21 = REDUCE_SUM
0x40 = GELU
0x41 = LAYERNORM
0x80 = FUSED_SOFTMAX
```

RoCC 指令建议新增独立 func7，避免改变当前 softmax 指令：

```cpp
GOLEM_ROCC_FUNC7_SFU_PRIMITIVE = 0x19;
GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_WAIT = 0x1a;
```

### 分阶段实施建议

- [ ] Phase 9A：只加 ABI 和测试，不改数学实现。
  - 在 `sfu.h` 中定义 `SFUPrimitiveDesc` 和 `SFUPrimitiveOp`。
  - 在 RISC-V `ex_instr.h` 中新增 primitive wrapper。
  - 加 static/assert 测试，固定 descriptor size、offset、op 编码。
- [ ] Phase 9B：实现 unary primitive 最小集合。
  - 先实现 `EXP`、`RECIPROCAL`、`LOG`。
  - 输入输出都在 local GM，元素类型先只支持 fp32。
  - checker 使用 Python `math`/NumPy golden 比较。
- [ ] Phase 9C：实现 `RSQRT`、`TANH`、`SIGMOID`。
  - 允许后续通过 `approx_mode` 切换精确数学库和近似 LUT/多项式模型。
  - 统计项区分 exact/approx primitive latency。
- [ ] Phase 9D：实现 reduction primitive。
  - `REDUCE_MAX` 和 `REDUCE_SUM` 先支持单 local buffer。
  - 后续再考虑跨 core reducer，与 fused softmax 的 reducer state 复用。
- [ ] Phase 9E：把 fused softmax 可选拆解为 primitive pipeline 对照测试。
  - 保留 fused softmax 快路径。
  - 新增一个 debug workload：`reduce_max -> exp -> reduce_sum -> reciprocal -> normalize`。
  - 用它对照 fused softmax 的数值和事件统计。

### 测试计划

- 每个 standalone primitive 先做 host-side 单元测试：
  - descriptor ABI。
  - fp32 正常值、极大/极小值、NaN/Inf 策略。
- 再做 RISC-V small workload：
  - local GM 输入。
  - SFU primitive 指令。
  - local GM 输出。
  - HBM dump 或 stdout checker 对比 golden。
- 最后做 fused 对照：
  - 当前 `64x64`、`128x128`、`256x256` fused softmax 保持通过。
  - primitive pipeline softmax 与 fused softmax 在容差内一致。

## 完成标准

- `golem.SFU` 可以被 SST 注册并通过 `cpu_builder.py` 挂载。
- 新 RoCC SFU 指令能从 RISC-V workload 发出并完成。
- 不设置 `GOLEM_SFU_ENABLE` 时，原始 GEMM smoke test 通过，默认 GEMM 路径行为不变。
- `64x64`、`128x128`、`256x256` 至少通过 fused softmax golden checker。
- `512x512` 能运行并给出数值正确结果或明确性能/资源瓶颈诊断。
- 通用 SFU 后续阶段应至少完成 `EXP`、`LOG`、`RECIPROCAL` 三个 standalone
  fp32 primitive，再考虑 `RSQRT`、`TANH`、`SIGMOID` 和 reduction primitive。
- `design.md`、`task_plan.md`、`progress.md`、`findings.md` 与实际实现状态一致。
