# SFU Softmax Row Engine NoC 总体架构设计

**状态：** 已实施并通过 `1024x4096`：真实 NoC 逐行 DMA、三阶段计算、输出 ACK 和唯一 band completion 已形成完整因果链

**日期：** 2026-07-16

**主验证负载：** FP32 `1024x4096` row-wise Softmax

**目标平台：** 16 个计算 tile、4 列 mesh、4 个 data HBM node、2.3 GHz 模拟时钟

## 0. 2026-07-21 最终实施状态

当前实现已经落地以下设计要点：

- guest 只提交一个 tensor job；controller 通过 explicit NoC 向 16 个物理
  SFU/GlobalMemory endpoint 分发 16 个连续 row band；
- 每个 endpoint 使用 4 个物理 row context，一次只驻留一行；
- 每行严格执行 `input DMA -> MAX -> EXP/SUM -> NORMALIZE -> output DMA ACK`；
- MAX/Normalize 共享 16-lane vector resource，EXP/SUM 使用 4-lane EXP resource，
  阶段时延通过 SST self-link event 推进；
- worker 仅在输出 DMA ACK 成功后发送 success completion；controller 校验
  `jobId/row/worker/shape/band` 并对 completion 去重；
- unsafe transport、scratch 超限、band 过订阅和并发 tensor job 会失败或
  backpressure，不再静默悬挂或复用同一 scratch；
- 四个 HBM node 按连续 64-row band 条带化，input/output payload 各为 16 MiB；
- 主实验产生 1,024 次输入 DMA、1,024 次输出 DMA ACK、每个计算阶段 1,024 个事件，
  reduction request 为 0；
- `1024x4096` golden 检查 4,194,304 个元素，mismatch 为 0，最大绝对误差
  `5.72476014e-11`。

最终严格完成边界为 descriptor acceptance 到 accelerator ready，实测 `66,958 cycles`。
其中非重叠路径为 `11 + 256 + 66,549 + 88 + 54` cycles。分析模型给出的
`66,061 cycles` 仅作为 compute reference，不能代替完成时间。clean guest kernel 为
`73,309 cycles`，整个 SST/NoC 窗口为 `278.661 us / 640,921 cycles`。

默认 1200 GB/s 配置下最大 NoC port utilization 为 1.257%，EXP/SUM 占聚合 active
service 的 66.7%，因此下一计算优化目标是 EXP throughput。但 DMA 已真实接入时序链：
将 NoC/DirCtrl 降至 64 GB/s 会把 `16x4096` latency 从 2,076 提高到 4,294 cycles。
第 7 节超宽行 pair collective 仍是后续工作；它不影响当前 `dim=4096` 的 row-local
验收结论。

## 1. 决策摘要

本设计不再把“增加参与同一行的 worker 数量”作为 Softmax 的默认扩展方式。主路径改为：

> 每个计算 tile 是一个独立 Softmax Row Engine；对于 `dim=4096`，一个 tile 完整处理一行，
> 16 个 tile 同时处理 16 行。只有单行超过本地容量或吞吐边界时，才启用 2/4-tile
> 分层 collective。

最终架构由五部分组成：

1. 每 tile 一个带本地 scratchpad 的 Softmax Row Engine；
2. 16-lane FP32 vector ALU、16-input reduction tree 和 4-lane EXP pipeline；
3. 一个 tensor 级 hardware row scheduler，负责把行分发到 16 个 tile；
4. 按 mesh 列对 4 个 data HBM node 做 row striping；
5. 仅供超宽行使用的 `(m,l)` pair collective，以及与 DMA 隔离的发送队列。

`1024x4096` 的默认映射为 `1 tile/row x 16 row groups`。固定 `4 tiles/row x 4 groups`
是过渡/后备映射，不是该 shape 的最终主路径。现有 `16 workers/row x 1 group` 保留为
legacy regression，不再作为性能默认值。

## 2. 设计依据

### 2.1 GPU 参考原则

GPU reduction 是 thread、warp、block、device 的分层协作，而不是所有计算单元集中到一个
全局 reduction ALU。NVIDIA CUB 明确将 reduction 分为 ThreadReduce、WarpReduce、
BlockReduce 和 DeviceReduce；CUDA warp shuffle 提供 lane 间交换和树形 reduction/broadcast。

- [NVIDIA CUB developer overview](https://nvidia.github.io/cccl/unstable/cub/developer_overview.html)
- [CUDA warp reduce/shuffle functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-reduce-functions)
- [NVIDIA Ampere warp-level reduction support](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#warp-level-support-for-reduction-operations)

本设计映射关系如下：

| GPU 层级 | Golem 对应层级 |
|---|---|
| thread local values | Row Engine vector lanes |
| warp reduction | tile-local reduction tree |
| block/shared memory | tile-local scratchpad + row contexts |
| block scheduling | hardware row scheduler |
| cross-block/device reduction | 仅超宽行启用的 multi-tile collective |

GPU 经验在这里用于选择并行层级，不用于声称 Golem 已具有 GPU 相同的物理执行单元。

### 2.2 当前实测事实

以下是已验证结果，不是估算：

| 负载/配置 | Simulated time | Transport timestamp delta | 结论 |
|---|---:|---:|---|
| `16x4096`, 4 workers | `422.029 us` | `9438.813 ticks` | 最低 fan-in |
| `16x4096`, 8 workers | `423.385 us` | `11516.695 ticks` | 增加 worker 无加速 |
| `16x4096`, 16 workers | `427.053 us` | `15582.348 ticks` | fan-in 和等待继续增加 |
| `1024x4096`, 16 workers | `7236.07 us` | `15580.885 ticks` | 当前最大真实 PASS 点 |

这里的 manifest 字段历史名称是 `latency_avg_cycles`，但实现使用 `getCurrentSimCycle()`，当前
SST timebase 为 1 ps。因此 `15580.885` 对应约 `15.58 ns`，不是 15,580 个 2.3 GHz CPU
周期。Phase 0 必须修正字段命名或显式换算，旧字段不得继续直接用于 cycle breakdown。

证据来源：

- [Phase 4F manifest](../../../src/sst/elements/golem/tests/artifacts/sweeps/sfu_phase4f_large_scale_explicit_noc_20260715/large_scale_manifest.csv)
- [Capacity manifest](../../../src/sst/elements/golem/tests/artifacts/sweeps/sfu_4096x4096_capacity_explicit_noc_20260716/capacity_manifest.csv)
- [Durable findings](../../../src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/findings.md)

当前 `1024x4096` 还产生：

- `65,536 = 4 x rows x workers` 个 max/sum request/response transport event；
- input/output 各 `16,384` 次 DMA operation，平均只有 `1 KiB/operation`；
- `4,096` 次 SFU job 和约 `5.98M` 次累计 wait poll；
- `76,053` 次 NoC xbar stall；
- 约 `41.9M` 条 Vanadis retired instruction；
- `sfu_job()` 后立即 `sfu_job_wait()`，SFU 状态推进依赖 wait 重试。

按 2.3 GHz 换算，整个 `7236.07 us` 区间约为 `16,642,961` 个 CPU 等效周期。从首个
input DMA 到最后 output write ACK 的后验 envelope 仍约为 `15,944,285` 个等效周期，且包含
guest、DMA、NoC、polling 和控制。当前仓库还没有可直接作为纯 SFU latency 的实测字段。

当前逻辑 tensor DMA 字节数已经是 input/output 各 `16 MiB`，不能把瓶颈描述为额外 HBM
数学 pass。当前 NoC 最大端口利用率约 `0.20%`、output stall 为 0、memory queue delay 约
1 tick，也没有证据证明 NoC/HBM 已经饱和。可证明的问题是错误的行/列并行层级、细粒度 DMA、
大量 job/wait 控制、local GlobalMemory 中间数据往返和 reduction transport 组织方式。
现有 `stats_latency/merge_latency/normalize_latency` 参数也没有参与状态调度；历史
`sfu_cross_tile_wait_cycles` 是 pending wait 时的累计加一，跨 core 求和不能作为系统
critical-path cycles。

## 3. 性能口径与目标

### 3.1 必须使用两个周期口径

1. `softmax_accelerator_latency_cycles`：hardware row scheduler 接受 descriptor，到最后一个
   Row Engine 完成 output DMA；这是主要架构指标。
2. `softmax_issue_to_completion_cycles`：第一个 RoCC job issue，到 guest 观察到完成；用于衡量
   RoCC、Vanadis wait 和完成通知开销。

SST 启动、guest 进程初始化、输入生成和仿真结束时间不得混入 GPU kernel cycle 对比。
当前 `7236.07 us` 是整个 SST 区间，不能直接等同为纯 Softmax accelerator latency。

### 3.2 `1024x4096` 初始周期预算

每 tile 初始配置：16 个普通 vector lane、4 个 EXP lane。忽略 pipeline drain 时：

```text
MAX pass        ceil(4096 / 16) =  256 cycles/row
EXP/SUM pass    ceil(4096 /  4) = 1024 cycles/row
NORMALIZE pass  ceil(4096 / 16) =  256 cycles/row
                                  ----------------
                                  1536 cycles/row

16 tiles 并行，1024 / 16 = 64 row waves
compute occupancy lower bound = 64 x 1536 = 98,304 cycles
```

这是设计预算，不是当前模型的测量值。加入 pipeline drain、DMA、队列和调度后，目标为：

| 指标 | Acceptance target | Stretch target |
|---|---:|---:|
| `softmax_accelerator_latency_cycles` | `<= 150,000` | `<= 120,000` |
| `softmax_issue_to_completion_cycles` | `<= 200,000` | `<= 150,000` |
| row reduction messages (`dim=4096`) | `0` | `0` |
| HBM data traffic | 1 input read + 1 output write | 同左 |
| DMA read/write operations | `<= 1024` each | 2D/batched 后更低 |
| retry/rejected/stale | `0` | `0` |

GPU p50 的约 `115,718 cycles` 是依据 nominal SM clock 换算的参考值，不是 GPU 硬件 cycle
counter。最终报告必须同时给 cycles、time 和 elements/cycle，避免只比较不同频率下的 cycle 数。

### 3.3 带宽约束

FP32 `1024x4096` 输入和输出各 `16,777,216 B`，不可消除的数据量合计 `32 MiB`。
在 2.3 GHz 下：

- `120k cycles` 对应约 `52.2 us`，需要约 `643 GB/s` 有效端到端带宽；
- `150k cycles` 对应约 `65.2 us`，需要约 `515 GB/s` 有效端到端带宽。

`1200GB/s` NoC 参数只是链路配置上限，不证明 HBM backend、DMA 和端点已经达到上述有效带宽。
当前单 data-node HBM 的 tCCD_L roofline 约为 `512 GB/s`；只搬运 32 MiB 的理想下界已经约为
`150,733 cycles`。因此 `<=150k` 最终 gate 在 single-node layout 下不自洽，四节点 row
striping 是最终性能目标的必要条件，而不是可选优化。

该 roofline 来自当前 [HBM configuration](../../../src/sst/elements/golem/tests/architecture/dram/HBM_4Gb_x128.ini)
的 channel 和 tCCD_L 参数。它仍是理想上限，不包含 turnaround、refresh 和 pipeline fill/drain。

## 4. 总体架构

```text
                         one SFUJobDesc
                               |
                    Tensor Softmax Job Controller
                 row allocation / credits / completion
                               |
        +----------------------+----------------------+
        |        16 independent Softmax Row Engines  |
        |                                             |
        |  Tile 0  Tile 1  ...                 Tile15 |
        |    |       |                              |  |
        |  local   local                          local |
        |  SRAM    SRAM                           SRAM  |
        |    |       |                              |  |
        | vec/reduce/EXP pipelines on every tile       |
        +----------------------+----------------------+
                               |
               row-striped HBM0 / HBM1 / HBM2 / HBM3

Multi-tile fallback only:
local (m,l) partials -> isolated collective queue -> group reducer
                    <- global (m,l) response -------
```

完整 SST topology 仍包含 OS memory row；“4x4”指 16 个计算 tile 平面。四个 data HBM node
位于 data-memory row，和四个 mesh column 对齐。

现有架构已经在每个物理 core/RoCC 下挂载一个 `golem.SFU`。这里的“16 个 Row Engine”指
16 个物理 SFU 各增加一个 Row Engine 状态机，不是在单个 SFU component 内再复制 16 个引擎。

## 5. Softmax Row Engine

### 5.1 初始硬件模型参数

| 参数 | 初始值 | 语义 |
|---|---:|---|
| `vector_lanes` | `16` | FP32 max/add/sub/mul 吞吐 |
| `reduction_lanes` | `16` | 每拍输入宽度 |
| `reduction_tree_latency` | `4 cycles` | `log2(16)` pipeline depth |
| `exp_lanes` | `4` | 每 tile 每周期 EXP 接受数 |
| `exp_latency` | configurable, 初始 `8 cycles` | latency；吞吐与 latency 分开建模 |
| `reciprocal_lanes` | `1` | 每行一个 reciprocal |
| `scratchpad_bytes` | `64 KiB` | 4 个 4096-FP32 row buffer |
| `row_contexts` | `4` | DMA/compute/store 重叠 |
| `job_queue_depth` | `8` | 与现有 `max_inflight=8` 对齐 |

Scratchpad data array 至少提供 `64 B/cycle` vector read。EXP/SUM 阶段至少需要
`16 B/cycle` read 和 `16 B/cycle` in-place write；bank conflict/stall 必须单独统计，不能假设
scratchpad 永远单周期完成。

这些值是首版建模点。只有完成 Phase 0 cycle breakdown 后，才允许做小范围吞吐 DSE。

### 5.2 Scratchpad 数据流

4096 个 FP32 元素占 16 KiB。每个 row context 使用一个 16 KiB buffer，并原地复用：

```text
DMA input -> buffer contains x
MAX pass  -> buffer remains x, scalar local_max produced
EXP/SUM   -> overwrite x with exp(x - local_max), scalar local_sum produced
NORMALIZE -> multiply buffer by reciprocal(local_sum)
DMA output
```

主路径不把中间 exp 写入 GlobalMemory/HBM，也不为 normalize 重新读取输入。四个 context
允许 input DMA、MAX、EXP/SUM、NORMALIZE 和 output DMA 在不同 row 上重叠。

### 5.3 资源解耦

Vector ALU、reduction tree、EXP pipeline、reciprocal 和 DMA 分别维护 resource reservation。
EXP 处理 row N 时，vector/reduction 可以处理 row N+1，DMA 可以预取 row N+2。

禁止继续把 `REDUCE_MAX`、`REDUCE_SUM` 和 `EXP` 当作一个串行、互斥的 monolithic SFU
issue slot。外部 opcode 可以保持不变，内部必须路由到独立资源。

### 5.4 仿真推进方式

为降低 SST wall time，Row Engine 不采用“每 lane、每元素、每周期一个 SST event”的实现。
推荐使用 event-driven resource reservation：

1. stage 开始时根据元素数、吞吐、pipeline latency 和资源 `free_cycle` 计算 `ready_cycle`；
2. 只为最近的 stage completion 安排 self event；
3. completion 时批量执行功能计算并推进状态；
4. `wait(tag)` 只查询状态，不再承担硬件状态推进。

功能数据仍通过 C++ vector 计算以保持 golden 正确；周期由显式吞吐/队列模型决定。该模型同时
避免当前 wait polling 对硬件进度和 simulation wall time 的耦合。

## 6. 行映射策略

本文映射记号固定为 `tiles_per_row x concurrent_groups`。对应现有 runner 参数时，
`band_cores=16` 固定，`worker_cores=tiles_per_row`，因此实验点分别是：

```text
1x16  -> workers/bands = 1/16
2x8   -> workers/bands = 2/16
4x4   -> workers/bands = 4/16
8x2   -> workers/bands = 8/16
16x1  -> workers/bands = 16/16
```

### 6.1 AUTO policy

```text
dim <= 4096       1 tile/row, 16 independent groups
4096 < dim <= 8192    2 tiles/row, 8 groups
8192 < dim <= 16384   4 tiles/row, 4 groups
dim > 16384           explicit policy; first validate capacity and bandwidth
```

阈值必须由 scratchpad capacity 和实测 throughput gate，不能仅按维度硬编码。首版先实现
`ROW_LOCAL` 和 legacy `DISTRIBUTED_COLUMNS`，AUTO 在两条路径稳定后启用。

### 6.2 4096 维主路径

行 `r` 分配到 `tile = r mod 16`。每个 tile 完整拥有该行，不产生 max/sum NoC collective。
所有 16 个 tile 同时工作，消除“16 worker 等待同一行 global max/global sum”的两次 barrier。

### 6.3 Multi-tile 物理分组

2/4-tile group 优先使用同一 mesh column 内相邻 tile。4-tile group 对应一列，owner/reducer
选择该列的中间 tile，避免固定在列端点。只有 8/16-tile group 才允许跨列。

## 7. 超宽行 `(m,l)` Pair Collective

多 tile 模式不再执行独立的 global-max collective 和 global-sum collective。每个 tile 对自己的
column slice 计算：

```text
m_i = max(x_i)
l_i = sum(exp(x_i - m_i))
```

reducer 使用可结合的 merge：

```text
m = max(m_a, m_b)
l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
```

当前 `SFU::mergeTileStats()` 已包含相同数学形式，应作为功能参考，但需要新的显式时序、队列和
transport contract。owner 返回全局 `(m,l)`；worker 使用本地 `m_i` 计算 correction：

```text
y = exp(x - m_i) * exp(m_i - m) / l
```

消息 contract：

- 新增 `StatsPairRequest` 和 `StatsPairResponse`；
- 一个 request 同时携带 `m_i` 和 `l_i`；
- group size 1 必须走 local fast path，发送 `0` 个 reduction message；
- group size 2/4 首版允许 unicast response；router multicast 仅在证据证明值得时实现；
- legacy Max/Sum request/response 保留到新路径完成回归，不原地改变历史计数语义。

在无 multicast 的情况下，pair collective transport 为：

```text
2 * rows * tiles_per_row
```

相比旧路径的 `4 * rows * tiles_per_row` 减半，并且不再有 max response 后的全组 barrier。

## 8. NoC 与 HBM 设计

### 8.1 HBM Row Striping

当前 standalone Softmax 将 row-major input/output 固定在 data node 1。新布局按计算 tile 的
mesh column 把行分布到四个 data HBM node：

```text
tile_column = assigned_tile % 4
data_node   = 1 + tile_column
local_row   = floor(row / 4)   # 具体 packing 由 layout helper 唯一定义
```

要求：

- input 和 output 使用同一 row-to-node mapping；
- HBM generator、runtime address calculation 和 golden dumper 共用一个结构化 layout helper；
- 不复制完整 tensor 到四个节点；
- 每个 tile 优先访问同列 HBM，减少横向 hop 并平衡四个 backend；
- 保留 `single_node` legacy layout 作为回归模式。

### 8.2 DMA Contract

主路径每行是一个连续 16 KiB transfer：

- input：HBM -> tile scratchpad；
- output：tile scratchpad -> HBM；
- 每行不再拆成 16 个 worker slice DMA；
- 支持至少两个 context 的 double buffering；
- 后续可增加 2D/strided descriptor，把多个同列 row 合并为一个 hardware descriptor。

### 8.3 Collective 与 DMA 隔离

当前 reduction 和 DMA 共用 GlobalMemory NIC 的 `send_retry_queue`。首版不立即增加 router local
port，而是在现有 NIC 中提供独立 `collective_send_queue`、独立 credit/high-water 统计和明确
仲裁，继续使用固定 reduction VN。

只有满足以下任一条件，才升级为独立 `SoftmaxCollectiveEndpoint` 物理端点：

- pair collective queue wait 仍占 accelerator cycles 的 10% 以上；
- DMA head-of-line blocking 可由事件证据复现；
- 增加 local port 不破坏 L2、GlobalMemory 和 scheduler endpoint wiring。

这避免在 `dim=4096` 主路径已经没有 reduction message 的情况下提前扩大 NoC 修改面。

## 9. Job、ABI 与完成通知

### 9.1 保持 `SFUJobDesc` 128-byte ABI

guest 和 SST 两侧均有 `static_assert(sizeof(SFUJobDesc) == 128)`。不得扩展或重排现有结构。
新配置通过现有 `params_addr` 指向版本化参数块：

```cpp
struct SFUSoftmaxJobParamsV1 {
    uint32_t magic;
    uint16_t version;
    uint16_t size_bytes;
    uint32_t mapping_policy;      // AUTO, ROW_LOCAL, CLUSTER, LEGACY
    uint32_t tiles_per_row;       // 0 means AUTO
    uint32_t row_contexts_hint;
    uint32_t hbm_layout;          // SINGLE_NODE or ROW_STRIPED
    uint32_t data_node_mask;
    uint32_t flags;
    uint64_t completion_addr;
    uint64_t reserved[3];
};
```

精确布局在实现阶段由 ABI test 锁定。规则：

- 体系结构 lane 数、latency 和 scratchpad size 是 SST component 参数，不是 job 参数；
- unknown version/size 必须返回 `InvalidDescriptor`；
- `reserved0` 的 legacy worker-slot 语义保持不变；
- 新旧 guest/runtime 可以通过 flag 和 `params_addr==0` 明确区分；
- production library 和 guest ELF SHA 必须进入新 sweep signature，禁止复用旧 cached PASS。

### 9.2 单次 tensor job

新主路径只由 coordinator guest 发出一个 tensor-level `sfu_job`。Job Controller 在硬件模型内：

1. 解析 descriptor 和 params；
2. 选择 mapping；
3. 按 tile credit 分发 row context；
4. 汇总每 tile completion；
5. 在最后一个 output DMA 完成后设置 tensor completion。

现有每个 worker、每 4 行反复构造 descriptor 并同步 wait 的路径保留为 legacy，不用于目标周期。

### 9.3 Event-driven Completion

`sfu_job_wait(tag)` 继续作为兼容接口，但它只读取 pending/success/error。SFU 状态由 self event、
DMA callback 和 collective receive 自动推进。首版 guest 可以低频 polling；后续再决定是否由 RoCC
completion event 唤醒，不能把改变 wait 语义和 Row Engine 数据通路放在同一个不可回退补丁中。

## 10. 周期与统计 Contract

必须新增以下 tensor-level 统计：

```text
softmax_system_start_cycle
softmax_system_end_cycle
softmax_accelerator_latency_cycles
softmax_issue_to_completion_cycles
softmax_rows_dispatched
softmax_rows_completed
softmax_active_row_contexts_high_water
```

必须新增以下资源 active/stall 统计：

```text
softmax_input_dma_cycles
softmax_input_dma_wait_cycles
softmax_vector_active_cycles
softmax_reduction_tree_active_cycles
softmax_exp_active_cycles
softmax_exp_queue_wait_cycles
softmax_normalize_active_cycles
softmax_output_dma_cycles
softmax_output_dma_wait_cycles
softmax_collective_active_cycles
softmax_collective_queue_wait_cycles
softmax_job_queue_wait_cycles
softmax_local_gm_read_bytes
softmax_local_gm_write_bytes
softmax_hbm_payload_read_bytes
softmax_hbm_payload_write_bytes
softmax_transport_latency_ps
```

所有新增 `*_cycles` 统一使用 2.3 GHz accelerator clock domain。SST timebase tick 只允许写入
显式带 `_ps`/`_ticks` 后缀的字段；parser 不得把旧 `latency_avg_cycles` 静默并入新 cycle 指标。

同时记录：

- rows/cycle、elements/cycle 和 EXP elements/cycle；
- 每 data HBM node 的 useful bytes、active cycles 和 imbalance；
- 每 tile 分配/完成行数，最大值与最小值差；
- collective request/response、fan-in、latency 和 queue high-water；
- DMA read/write operations、bytes、RTT、retry/exhaustion；
- NoC xbar/output stalls 和 hotspot router/port。

phase active cycles 可以重叠，不能简单相加等于总 latency。报告必须区分 occupancy、stall 和
critical-path wall interval。

## 11. 实现修改面

### 11.1 Production component

| 文件 | 计划修改 |
|---|---|
| `src/sst/elements/golem/sfu/sfu.h` | ABI flags、Job Controller 状态、参数/统计声明 |
| `src/sst/elements/golem/sfu/sfu.cc` | 新 job dispatch、legacy/local 路由、event-driven completion |
| `src/sst/elements/golem/sfu/softmax_row_engine.h/.cc` | 新增 scratchpad、row contexts、resource reservation、功能计算 |
| `src/sst/elements/golem/sfu/softmax_collective.h/.cc` | 新增 pair merge 和 group state；首版可后置 |
| `src/sst/elements/golem/globalmemory/globalmemory.h/.cc` | row DMA callback/布局支持、collective 独立队列和统计 |
| `src/sst/elements/golem/Makefile.am` | 注册新增 production source |

### 11.2 Architecture wiring

| 文件 | 计划修改 |
|---|---|
| `src/sst/elements/golem/tests/architecture/cpu_builder.py` | Row Engine 参数、clock、scratchpad/context 配置 |
| `src/sst/elements/golem/tests/architecture/ncores_selfcom_dma_ctrl.py` | row-to-column/HBM mapping；仅在需要时增加 endpoint wiring |
| `src/sst/elements/golem/tests/architecture/noc_builder.py` | 默认不改；独立 endpoint gate 通过后再扩 local port |

### 11.3 Guest/runtime/data layout

| 文件 | 计划修改 |
|---|---|
| `golem_softmax_sfu_runtime.h/.cpp` | 镜像参数 ABI、单次 tensor job、兼容 legacy path |
| `test_noc_dma_softmax_sfu.cpp` | coordinator dispatch、row-striped HBM address、周期证据输出 |
| `tests/tools/gen_hbm_init.py` | 四节点 row-striped input 初始化和 output region |
| `run_sfu_unified_job_distributed_scaling.sh` | 新 mapping/layout/signature，不改变历史 artifact |

### 11.4 Tests/reporting

计划新增：

```text
test_sfu_softmax_cycle_accounting.py
test_sfu_softmax_row_engine.py
test_sfu_softmax_job_params.py
test_sfu_softmax_row_striping.py
test_sfu_softmax_pair_collective.py
run_sfu_softmax_row_engine_sweep.sh
plot_sfu_softmax_row_engine.py
```

现有 descriptor、RoCC、primitive、workload、Phase 4F、capacity 和 golden tests 必须继续运行。

## 12. 分阶段实施计划

### 实施前置条件

先完成现有 wall-time 计划中的 W0 build isolation：custom Softmax 运行不得重建或覆盖未使用的
GEMM guest。关闭 Vanadis trace、逐事件文本和 quiet guest 属于独立的 simulator-efficiency
profile；这类改动必须证明 accelerator simulated cycles、output 和 lifecycle counters 不变，
不得和 Row Engine 架构收益混合计数。

### Phase 0：建立可信周期基线

**行为变化：** 无 Softmax 数学和 mapping 变化。

1. 增加 accelerator/issue-to-completion 周期窗口；
2. 增加阶段 active/wait 和 per-tile row 统计；
3. 把 git commit、production `libgolem.so`、guest ELF、runner、parser、verifier、input 和
   output SHA 纳入 signature；
4. 新 artifact root 依次运行 `64x4096`、`256x4096`，通过后再运行 `1024x4096` baseline；
5. 证明统计窗口从首个 tensor issue 到最后 output DMA，而不是 SST 全程。

**退出条件：** 相同 run 的 start/end/latency 自洽；rows/bytes/counter 完整；golden PASS；旧 cached
artifact 不会被 production 改动误复用。

### Phase 1：验证 Row-local Mapping

**行为变化：** 新增 `ROW_LOCAL`，legacy 保持默认可选。

1. 复用现有 child runner 能力运行 `1x16, 2x8, 4x4, 8x2, 16x1` mapping matrix，实际
   `worker_cores/band_cores` 为 `1/16, 2/16, 4/16, 8/16, 16/16`；
2. 为 `worker_cores==1` 增加 local reduction fast path，禁止 self request/response；
3. shape、总 tile、HBM/NoC 配置、chunk 和 correctness gate 保持固定；
4. 选择 `1024x4096` 最低 accelerator cycles 的 mapping。

**退出条件：** `1 tile/row` 的 reduction transport 为 0；16 个 tile 都完成非零行数；golden 和
DMA lifecycle PASS；结果支持或否定主设计假设。若 `1x16` 不是最优，不得继续把它写死为 AUTO。

### Phase 2：实现 Event-driven Row Engine Timing Model

**行为变化：** 功能数学不变，完成时刻改由显式资源模型决定。

1. 新增 `softmax_row_engine.*` 和 SST self event；
2. 实现 scratchpad/context 状态机和 resource reservation；
3. `wait()` 改为只读状态，硬件进度与 guest polling 解耦；
4. 参数化 vector/EXP/reduction throughput 和 latency；
5. 使用 batch functional compute，禁止 per-element SST event。

**退出条件：** fixed seed 下输出通过 golden；修改 throughput 参数会按解析模型改变 cycles；减少
guest poll 次数不改变 accelerator cycles；SST wall time 不因 per-cycle事件膨胀。

### Phase 3：本地 Scratchpad 与一次读/一次写

**行为变化：** 中间 exp 不再写回 GlobalMemory。

1. 实现 64 KiB scratchpad 和 4 row contexts；
2. input/max/exp-sum/normalize/output 使用 in-place buffer；
3. 重叠 input DMA、compute 和 output DMA；
4. 保持 scoped tensor DMA read/write 各 `16 MiB`，同时消除中间 exp 的 local GlobalMemory
   write/read；
5. 对 32/64 KiB 和 2/4 contexts 做有限 A/B。

**退出条件：** `1024x4096` scoped payload read/write 各等于 `16 MiB`；中间 local GM bytes
归零；无 scratch overflow；context high-water 非零；output 与 golden 一致。

### Phase 4：单次 Tensor Job 与 Hardware Row Scheduler

**行为变化：** 由 coordinator 发出一个 job，硬件内部派发行。

1. 定义并锁定 `SFUSoftmaxJobParamsV1`；
2. 实现 tensor Job Controller、tile credits 和 completion aggregation；
3. legacy per-worker issue 路径继续存在；
4. coordinator-only guest 路径使用一个 descriptor/issue；
5. 将 RoCC command count、retired instructions 和 wait polls 纳入报告。

**退出条件：** tensor job 只 issue 一次；每行只完成一次；最后 output DMA 前不得报告 success；
`softmax_issue_to_completion_cycles - softmax_accelerator_latency_cycles` 可解释且稳定。

### Phase 5：四 HBM Node Row Striping

**行为变化：** standalone row-major tensor 从单 node 改为四 node 条带化。

1. 新增唯一 layout helper 和版本化 layout id；
2. generator/runtime/dumper 共享 row-to-node contract；
3. 每 tile 优先访问同列 data node；
4. 保留 `SINGLE_NODE` A/B；
5. 对比 per-node bytes、backend active time、NoC hop/hotspot 和 accelerator cycles。

**退出条件：** 四个 data node useful bytes 基本均衡；总 bytes 不因 striping 增加；golden PASS；
固定 topology signature 与实际 runtime 一致。

### Phase 6：超宽行 Pair Collective

**行为变化：** 新增 cluster mapping；legacy max/sum collective 保留。

1. 实现 `StatsPairRequest/Response` 和 `(m,l)` merge；
2. 实现 2/4-tile group 和中央 owner 选择；
3. 在 GlobalMemory NIC 中隔离 collective queue；
4. 验证 `2 * rows * tiles_per_row` transport contract；
5. 仅在 queue wait gate 失败时评估独立 physical endpoint/multicast。

**退出条件：** adversarial max、全相等值、大幅值和普通随机输入均通过 golden；transport、fan-in、
owner cleanup、abort/stale/duplicate contract 完整；pair path 快于相同 group size 的 legacy 两阶段路径。

### Phase 7：性能收口与可选 GEMM+Softmax Fusion

1. 在固定 `1024x4096` 上逐项归因 Phase 1--6 的 cycle 改善；
2. 达到 `<=150k accelerator cycles` 后恢复 `2048x4096/4096x4096` capacity ladder；
3. standalone 达标后，单独设计 GEMM output -> Row Engine scratchpad streaming；
4. fusion 不得覆盖 standalone baseline，也不得把 GEMM compute cycles算成 Softmax 加速。

NVIDIA CUTLASS 历史优化包含 GEMM+Softmax reduction fusion，可作为后续方向，而不是当前
standalone correctness 的前置条件：[CUTLASS changelog](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)。

## 13. 验证矩阵

### 13.1 每次 production 变更的 focused gate

```bash
TMPDIR=/data4/jjgong/tmp PYTHONDONTWRITEBYTECODE=1 \
  /data4/jjgong/.venvs/golem-plot/bin/python -m unittest discover \
  -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu -p 'test_*.py' -v
```

这些测试主要是 ABI/source contract 和 synthetic artifact 测试，不能代替真实 SST component
execution。状态机或时序变更必须至少补一个真实 SST small smoke。

### 13.2 Runner 语法与 dry-run

```bash
bash -n src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_unified_job_distributed_scaling.sh
TMPDIR=/data4/jjgong/tmp GOLEM_SFU_CAPACITY_DRY_RUN=1 \
  bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh
```

Row-group mapping dry-run：

```bash
TMPDIR=/data4/jjgong/tmp GOLEM_DRY_RUN_SWEEP=1 \
GOLEM_SWEEP_ROOT=/data4/jjgong/tmp/sfu_row_groups_dryrun \
GOLEM_SFU_DISTRIBUTED_POINT_LIST='16:512:16:16 16:512:8:16 16:512:4:16 16:512:2:16 16:512:1:16' \
bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_unified_job_distributed_scaling.sh
```

### 13.3 Real SST 分级 gate

每个 Phase 按以下顺序推进，任一点失败即停止扩大规模：

```text
4x64 component smoke
16x512 mapping/lifecycle smoke
64x4096 performance anchor, fixed watchdog 600s
256x4096 confirmation
1024x4096 final target
```

真实 SST 必须使用新的 artifact root，不能复用 Phase 4F/capacity 的 historical PASS marker。

### 13.4 Guest build 与默认 GEMM 回归

Guest ABI/source 变化后重新构建：

```bash
make -C src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu ARCH=riscv64
```

任何 SFU、RoCC、GlobalMemory、architecture wiring 或共享 runner 变化都必须运行默认 GEMM：

```bash
TMPDIR=/data4/jjgong/tmp \
GOLEM_ARTIFACT_ROOT=/data4/jjgong/tmp/sfu_arch_gemm_regression \
env -u GOLEM_SFU_ENABLE -u GOLEM_SFU_STANDALONE_SOFTMAX \
  -u GOLEM_SFU_JOB_SOFTMAX -u GOLEM_SFU_PRIMITIVE_SOFTMAX \
  -u GOLEM_SFU_REDUCTION_VN -u GOLEM_DMA_RESPONSE_VN -u GOLEM_ARCH_SCRIPT \
  -u GOLEM_GROUP_MANAGER_ENABLE -u GOLEM_CTRL_LINK_ENABLE \
  -u GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE \
  bash src/sst/elements/golem/tests/run_noc_dma_pipeline.sh \
  --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --dtype fp32 --tensor-source sample --verify-c
```

要求 exit 0、simulation complete、VERIFY-C PASS、DMA lifecycle 完整且 retry 为 0，同时没有
SFU/reduction activity。真实 SST 和该 GEMM runner 均串行执行。

### 13.5 Correctness/lifecycle gate

每个 PASS 点必须同时满足：

- full-output golden checked，mismatch 为 0；
- input/output DMA issue、completion、bytes 完整；
- retry、exhausted、write retry、rejected、stale 均为 0；
- rows dispatched/completed 与 shape 一致；
- scratch/context 无 overflow、double-completion 或 tag reuse；
- runtime mapping/layout/lanes/clock signature 与 manifest 一致；
- legacy mode 保持历史计数 contract；新模式使用新的 schema/version；
- GEMM 输出和原有 RoCC func7/descriptor ABI 回归通过。

数值计算顺序改变时，new-mode output SHA 可以和 legacy 不同，但必须通过既有容差 golden，并在
同一新配置下保持确定性。不得把 legacy SHA 当作 pair reduction 的逐位相等要求。

## 14. 风险与回退

| 风险 | 检测证据 | 回退/处置 |
|---|---|---|
| 1 tile EXP 吞吐不足 | EXP active/queue wait 占 critical path | 增加 EXP throughput 前先验证 2-tile mapping |
| HBM 带宽低于预算 | 四 node backend active 且 accelerator memory-bound | 优化 striping/DMA；不得用虚高 NoC 参数掩盖 |
| event model 与 polling 耦合 | 改 poll interval 后 accelerator cycles 改变 | 阻止 Phase 2 退出，修复自推进状态机 |
| scratch 容量不足 | overflow/capacity reject | AUTO 切换 2/4-tile cluster，不静默覆盖数据 |
| pair reduction 数值差异 | adversarial golden failure | 保留 legacy max/sum，修正 merge precision/order |
| collective 与 DMA 队首阻塞 | collective queue wait >10% | 独立队列；证据仍失败才增加物理 endpoint |
| 单 tensor controller 成热点 | dispatch/queue wait 高 | 分层 scheduler 或每列 controller，不回退到 guest 多 issue |
| simulation wall time上升 | event count/host time增大但 cycles不变 | 合并 self event，禁止 per-cycle/per-element event |
| 历史 cache 误命中 | production SHA 与 marker 不一致 | 新 artifact root + lib/guest SHA 强制签名 |

所有新路径必须由 flag/schema 隔离。每个阶段均可切回 legacy unified job，而不是删除已验证路径后
再调试新模型。

## 15. 最终验收定义

本架构只有在以下条件同时满足时，才可以声明“`1024x4096` Softmax 周期逼近 GPU”：

1. `softmax_accelerator_latency_cycles <= 150,000`，并报告 `<=120,000` stretch 是否达到；
2. `softmax_issue_to_completion_cycles <= 200,000`；
3. FP32 full-output golden mismatch 为 0；
4. `dim=4096` 主路径 row reduction NoC message 为 0；
5. HBM 数据流为一次 input read 和一次 output write，四 data node 无严重失衡；
6. NoC/DMA lifecycle 无 retry、reject、stale 或未完成事务；
7. 真实 SST 小规模和 `1024x4096` 均来自包含 production/guest SHA 的新 artifact；
8. GEMM、legacy SFU job、RoCC ABI 和 descriptor tests 全部保持通过；
9. 报告同时给 cycles、time、elements/cycle 和 wall time，不混淆模拟性能与模拟器运行效率。

在这些证据产生前，`120k--150k cycles` 只能称为 architecture target，不能称为实测结果。
