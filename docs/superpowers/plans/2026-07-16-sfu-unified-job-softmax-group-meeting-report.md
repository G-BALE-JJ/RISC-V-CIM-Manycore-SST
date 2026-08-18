# Unified SFU Softmax 组会汇报建议

## 1. 先解释 guest 是什么

本项目中的 guest 是运行在 Vanadis 模拟 RISC-V CPU 上的 workload 程序。它不是宿主机上的 Python 脚本，也不是直接运行在真实 RISC-V 芯片上的程序。

可以用下面的分层解释：

```text
宿主机 shell/Python runner
    -> 启动 SST 和生成输入/HBM/artifact
SST architecture Python
    -> 创建 Vanadis CPU、RoCC、GlobalMemory、SimpleNetwork
Vanadis 模拟 CPU
    -> 逐条执行 guest RISC-V 指令
guest RISC-V binary
    -> test_noc_dma_softmax_sfu
    -> 通过 RoCC 提交 SFU job、等待完成、检查状态
SST Golem components
    -> 模拟 SFU、GlobalMemory、DMA、SimpleNetwork 和 reduction
```

Softmax guest 的源文件主要是 `test_noc_dma_softmax_sfu.cpp`，它使用统一 SFU wrapper 提交 job。SST 中的 SFU C++ component 才是被 guest 调用的硬件/架构模型。宿主机 shell 负责构建和收集证据，但不替代 guest 的执行过程。

因此，guest 输出的 `printf`、guest 的 polling/wait 指令和 guest 的 RoCC command 都会成为 Vanadis 模拟执行的一部分，并可能影响 simulated time；宿主机日志输出则主要影响 SST wall time。

## 2. 汇报主结论

建议用一句话贯穿整场汇报：

> 本阶段将 Softmax 从 guest-side primitive 和 mailbox 实验收敛为 unified SFU job，并在真实 SST explicit-NoC 路径上完成了正确性、跨 worker reduction 和规模验证；当前最大真实容量点为 `1024x4096`，更大容量验证和仿真效率问题是下一阶段边界。

需要明确区分：

- Phase 4F 的八个规模点已经通过真实 SST correctness/lifecycle gate；
- 容量阶梯中的 `512x4096` 和 `1024x4096` 已通过；
- `2048x4096` 和 `4096x4096` 目前延期，不能写成已支持或已通过。

## 3. 推荐汇报结构

建议使用 10 到 12 页、15 到 20 分钟的结构。

### 第 1 页：问题与目标

说明本阶段要解决的不是“再实现一个 Softmax kernel”，而是验证统一 SFU job 是否能够：

- 替代 guest-side primitive/stage/mailbox 控制流；
- 支持 row-band/chunk streaming；
- 让多个 worker 分担 column slice；
- 在 SFU 内部完成 max/sum reduction；
- 通过真实 SST SimpleNetwork 产生 explicit-NoC request/response traffic；
- 在大规模矩阵下保持 golden、DMA、reduction 和 transport lifecycle 正确。

### 第 2 页：统一 SFU Job 设计

展示以下结构：

```text
guest
  -> issue SFUJobDesc
  -> SFU job queue
  -> executor dispatch
  -> completion/status
  -> wait
```

重点解释：

- `SFUJobDesc` 是正式统一 ABI；
- Softmax 使用 `op_type=SOFTMAX_ROW`；
- `chunk_elems` 是 SFU 内部流式粒度，不是 guest issue 数量；
- guest 不再逐行发布 local max/local sum mailbox；
- primitive/batch/stage 路径降级为 legacy/debug/reference。

### 第 3 页：一次 Softmax job 的工作机制

建议用数据流图：

```text
HBM row-major input
    -> 每个 worker 读取自己的 column slice
    -> local row max
    -> max request/response
    -> exp(x - global_max)
    -> local row sum
    -> sum request/response
    -> reciprocal / normalize
    -> HBM row-major output
```

对于每行和每个 worker，理论上会产生：

```text
max request       1
max response      1
sum request       1
sum response      1
```

因此 transport 总量为 `4 * rows * worker_cores`。

### 第 4 页：Explicit-NoC reduction transport

说明 SFU 不直接连接独立 NoC port，而是复用每个 core 的 GlobalMemory NIC：

```text
worker SFU
  -> GlobalMemory transport bridge
  -> SimpleNetwork request
  -> owner reducer
  -> SimpleNetwork response
  -> worker SFU response inbox
```

固定实验配置：

```text
transport=explicit_noc
num_vns=3
reduction_vn=0
dma_response_vn=0
link/xbar/highlink=1200GB/s
NoC input/output buffer=512KB
GlobalMemory buffer=1024KB
flit=128B
```

强调这些值与成熟 GEMM 的实际生效配置一致，不是 Softmax 单独调参得到的性能配置。

### 第 5 页：正确性和 lifecycle gate

建议放一张总表：

| Shape | Golden checked/mismatch | 每类 reduction | Transport total | DMA bytes | Child wall |
|---|---:|---:|---:|---:|---:|
| `512x4096` | `2,097,152 / 0` | `8,192` | `32,768` | `8,388,608` | `1462 s` |
| `1024x4096` | `4,194,304 / 0` | `16,384` | `65,536` | `16,777,216` | `2890 s` |

每个 PASS 点还必须满足：

- active worker/band 数量正确；
- max/sum 四类 request/response 总数正确；
- explicit transport receive 完整；
- GlobalMemory immediate+queued 等于 transport total；
- rejected/stale reduction message 为 0；
- DMA issue/completion 完整；
- DMA retry、exhaustion、write retry 为 0；
- output size/hash、网络/VN runtime 回显和 manifest 一致。

### 第 6 页：Phase 4F 规模结果

建议用三个小图或三个 panel：dimension scaling、worker scaling、row scaling。

关键数据：

- `16x4096` 下 4/8/16 workers 的 simulated time 约为 `422.029/423.385/427.053 us`；增加 worker 没有明显收益，reduction 和同步成本抵消了 column split 的收益；
- `dim=4096`、workers=16 时，rows 从 16 增加到 256，normalized time per row 从 `26.691 us` 降到 `7.995 us`；
- 这说明固定控制和 reduction 开销可以通过更多 rows 摊薄，但不说明 worker 越多越快。

### 第 7 页：重要技术问题和工程纠错

建议采用“问题 -> 影响 -> 纠正”的形式：

| 问题 | 影响 | 纠正 |
|---|---|---|
| guest primitive/stage/mailbox 控制流过多 | guest 指令和同步事件膨胀 | 收敛到 unified SFU job |
| modeled reduction 不是真实网络负载 | 无法观察 NoC transport pressure | 改为 explicit SimpleNetwork request/response |
| VN 和网络配置容易漂移 | 实验不可比较 | 固定参数并验证 runtime signature |
| direct row-major 大 tensor 可能超过 HBM backing | 可能在 Softmax 前容量失败 | 明确 memory-node capacity 并使用 direct HBM path |
| 较大 job_rows 造成 DMA retry/exhaustion | 把容量压力误判为数学错误 | 固定 clean profile，记录边界，不静默改参数 |
| Softmax custom guest 触发未使用 GEMM guest 重编译 | 构建污染和额外开销 | 下一阶段修复 build isolation |
| `GOLEM_BENCH_QUIET_LOGS` 只在 shell 层生效 | quiet benchmark 不是真正 quiet | 增加 guest 侧显式 quiet 语义 |

这页体现工程判断：不仅验证了功能，也修正了实验路径中会误导结论的构建、网络和日志问题。

### 第 8 页：wall time 与 simulated time 的区别

建议单独说明：

- `1024x4096` simulated time 约 `7236.07 us`；
- child wall time 约 `2890 s`；
- 约 94.5% 的 parent wall time 在 SST 主仿真阶段；
- 16 个 Vanadis core 合计约 41.9M retired instructions；
- pipeline trace、GlobalMemory/DMA/band 逐事件文本和 guest polling 是主要怀疑热点。

结论应表述为：

> 当前主要限制已经从 Softmax 数值正确性转移到仿真执行成本。wall time 问题不能被解释为 Softmax 算法失败，也不能通过修改网络、VN 或 wait 语义来直接掩盖。

### 第 9 页：当前状态边界

建议用状态表：

```text
unified SFU Job architecture       PASS
row-band/chunk streaming            PASS
distributed columns                 PASS
cooperative workers                 PASS
explicit-NoC reduction              PASS
Phase 4F eight-point matrix         PASS
capacity 512x4096                   PASS
capacity 1024x4096                  PASS
capacity 2048x4096                  DEFERRED
capacity 4096x4096                  DEFERRED
GEMM regression                     PRESERVED
```

### 第 10 页：下一步计划

下一阶段不直接恢复 `2048x4096` 和 `4096x4096` 容量阶梯，而是先把
`1024x4096` 周期过长作为性能问题处理。主目标是在 correctness、DMA、
reduction 和 explicit-NoC lifecycle gate 不变的前提下，定位长周期的主要来源，
降低 Softmax 纯硬件任务周期数和 SST simulated time。宿主机 wall time 只作为仿真
效率的辅助指标，不作为架构优化是否成功的判据。

第一阶段先建立可比较的周期口径：

1. 增加 `softmax_system_start_cycle`、`softmax_system_end_cycle` 和
   `softmax_system_latency_cycles`，范围固定为第一个 unified SFU job 发出到最后一个
   worker 的输出 DMA 完成；
2. 将总周期拆分为 input DMA、guest/RoCC control、local max、max reduction wait、
   exp/local sum、sum reduction wait、normalize、output DMA，避免只依赖 SST 结束时间；
3. 冻结当前 `1024x4096`、16 total cores、`job_rows=4`、`chunk=256`、
   `explicit_noc` 和固定 GEMM 网络配置，生成新的 cycle baseline；
4. baseline 必须同时记录 cycles/row、cycles/element、NoC packet latency、reduction
   transport latency、xbar stalls、DMA RTT 和 active/inflight job 数。

第二阶段验证 row-level parallelism 是否为首要瓶颈。保持总核数为 16，依次运行：

```text
16 workers x 1 group
 8 workers x 2 groups
 4 workers x 4 groups
 2 workers x 8 groups
 1 worker  x 16 groups
```

每次只改变 workers/groups 映射，保持 shape、总核数、NoC、VN、DMA、chunk 和
`job_rows` 不变。重点检查更多 row groups 是否能减少顺序 row waves，并观察
reduction fan-in/fan-out、owner-router hotspot 和 DMA 小事务数量是否同步下降。

第三阶段在最佳 workers/groups 映射上逐项优化流水：

1. 对 `job_rows` 和 `staging_rows` 做小范围 A/B，判断减少 job 数是否能降低
   descriptor、RoCC issue/wait 和 batch barrier 开销；
2. 测试多 tag/multi-job inflight，使下一批 input DMA、当前 SFU job 和上一批 output
   DMA 能够重叠；
3. 将 cycle-polled SFU wait 与硬件状态推进解耦，验证 event-driven completion 是否能
   消除重复 RoCC wait 和队首阻塞；
4. 只有在 breakdown 证明 reduction 仍占主导后，才评估多 owner、分层 reduction 或
   reduction/DMA VN 隔离，不提前修改网络带宽和 buffer 掩盖问题。

每个 A/B 点必须通过 full-output golden、四类 reduction counter、transport、DMA、
output hash 和固定运行时配置 gate。主要排序指标为
`softmax_system_latency_cycles`，`simulated_time_us` 为第二指标；任一优化若只降低
wall time 而不降低 simulated cycles，不计为架构性能改进。第一阶段优化目标是在
固定 `1024x4096` workload 下，将纯 Softmax 系统周期相对新 baseline 至少降低 50%，
之后再决定是否恢复大容量阶梯以及是否进入 GEMM+Softmax fusion。

## 4. 汇报中应避免的表述

- 不要说“已经支持 `4096x4096`”；应说“目标是 `4096x4096`，当前真实 PASS 到 `1024x4096`”。
- 不要把 simulated time 和 wall time 混成同一个性能指标。
- 不要把 `modeled_noc` 与 `explicit_noc` 放在主性能曲线中比较。
- 不要把 worker 数量增加自动解释成加速；当前 `16x4096` 数据并不支持这个结论。
- 不要把 timeout 通过扩大 retry、改变 VN 或降低规模后的结果称为同一配置 PASS。
- 不要把 guest 输出、宿主日志和 SST stats 统称为一种“日志开销”；它们影响的层次不同。

## 5. 推荐开场与结尾

开场：

> 本阶段的核心不是再增加一个 Softmax 测试，而是把 Softmax 收敛成 unified SFU job，并验证它在真实 SST explicit-NoC 环境下的 correctness、跨 worker reduction 和规模行为。

结尾：

> 当前 unified SFU Softmax 的功能和通信路径已经得到真实 SST 证据支持，最大真实容量点达到 `1024x4096`。下一阶段将先建立纯 Softmax 全硬件周期口径，定位 row parallelism、reduction、DMA 和 wait 路径中的主要周期来源，并在 correctness gate 不变的前提下降低 simulated cycles；达到阶段目标后再恢复更大容量验证。
