# SFU 4096x4096 Explicit-NoC Softmax 容量与正确性验证设计

## 1. 背景

Phase 4F 已经完成 unified-job `explicit_noc` softmax 的真实 SST 大规模矩阵。当前
最大通过点为：

```text
rows=256
dim=4096
worker_cores=16
band_cores=16
golden_checked=1,048,576
golden_mismatches=0
```

该结果证明 `256x4096` 可以正常运行，但不能证明 `4096x4096` 已经受支持。用户将
现阶段最终目标明确限定为 `rows=4096, dim=4096`；不继续扩展到 `dim=8192`，也不在
本阶段开展设计空间搜索。

本阶段的任务是证明固定配置下的容量、正确性和生命周期完整性，而不是寻找最优参数。

## 2. 目标

在 Phase 4F 已验证的同一套 softmax 和 GEMM 网络配置下，逐级运行真实 SST，最终完成：

```text
shape=4096x4096
transport=explicit_noc
worker_cores=16
band_cores=16
golden_checked=16,777,216
golden_mismatches=0
```

最终结论只能是以下两类之一：

1. `4096x4096` 通过全部 gate，可以声明该固定配置已完成真实 SST 验证；
2. 在某一级出现明确的 `FAIL`、`TIMEOUT` 或 `ARTIFACT_FAIL`，记录容量边界和首个失败
   证据，不通过改变参数绕过。

## 3. 非目标

本阶段明确不做：

- worker、band、chunk、job-row、staging-row 或 retry DSE；
- VN、bandwidth、buffer、flit、topology 或 memory-node-size sweep；
- `modeled_noc` 对照；
- `dim=8192` 或 `rows>4096`；
- 性能排名、最优配置推断或统计显著性分析；
- GEMM+softmax fusion；
- primitive/batch softmax 主线；
- SFU 数学、reduction protocol、guest ABI、GlobalMemory、SimpleNetwork 或 GEMM 修改；
- 为减少生成时间而重写共享 tensor/HBM/golden 工具。

四个逐级点用于故障定位和风险控制，不构成 DSE。

## 4. 固定执行配置

所有点必须使用以下 softmax 配置：

```text
distributed_reduction_transport=explicit_noc
num_vns=3
request_vn=0
ordinary_response_vn=1
dma_response_vn=0
reduction_vn=0
worker_cores=16
band_cores=16
cooperative_groups=1
chunk_elems=256
staging_rows=4
job_rows=4
retry_ticks=1024
max_retries=8
distributed_columns=1
direct_rowmajor_hbm=1
mem_node_size=268435456 bytes
```

网络参数继续与成熟 GEMM 的实际 preset 完全一致：

```text
GOLEM_NOC_LINK_BW=1200GB/s
GOLEM_NOC_XBAR_BW=1200GB/s
GOLEM_DIRCTRL_HIGHLINK_BW=1200GB/s
GOLEM_NOC_INPUT_BUF_SIZE=512KB
GOLEM_NOC_OUTPUT_BUF_SIZE=512KB
GOLEM_NOC_FLIT_SIZE=128B
GOLEM_GM_BUFFER_LENGTH=1024KB
GOLEM_NOC_INTER_ROUTER_NO_CUT=0
GOLEM_NOC_LOCAL_NO_CUT=0
```

所有值都必须写入 point signature，并从 child manifest、SST log、stats 和
`run_summary.csv` 重新验证。外部环境中与固定值冲突的变量必须在创建 child artifact
前被拒绝。

## 5. 容量阶梯

固定 `dim=4096`、16 workers 和 16 bands，只增加 rows：

```text
512x4096
1024x4096
2048x4096
4096x4096
```

执行必须严格串行。只有前一点为完整 `PASS` 时才启动下一点；首个非 PASS 状态立即停止。
不得自动降低 rows，不得回退到 modeled-NoC，不得调整网络、worker、chunk 或 retry。

## 6. 容量审计

### 6.1 Tensor、计数和流量

FP32 元素为 4 bytes。每点的确定性期望如下：

| rows | elements | 单个 rows x dim tensor | 单类 reduction | transport total | DMA read/write bytes |
|---:|---:|---:|---:|---:|---:|
| 512 | 2,097,152 | 8,388,608 B | 8,192 | 32,768 | 8,388,608 B |
| 1,024 | 4,194,304 | 16,777,216 B | 16,384 | 65,536 | 16,777,216 B |
| 2,048 | 8,388,608 | 33,554,432 B | 32,768 | 131,072 | 33,554,432 B |
| 4,096 | 16,777,216 | 67,108,864 B | 65,536 | 262,144 | 67,108,864 B |

最终点的 A、B、logits 和 output 各为 64 MiB，逻辑 tensor payload 合计 256 MiB。
这不包含 HBM backing file、SST log、stats 和临时文件。

### 6.2 HBM 布局

使用现有 `gen_hbm_init.py` 的真实布局公式，在与 child runner 相同的 fp32、block、
reuse、core 和 memory-node 参数下得到：

| rows | direct-rowmajor region end | 自动最小 node size | 本阶段固定 node size |
|---:|---:|---:|---:|
| 512 | 37,748,736 B | 64 MiB | 256 MiB |
| 1,024 | 58,720,256 B | 64 MiB | 256 MiB |
| 2,048 | 100,663,296 B | 128 MiB | 256 MiB |
| 4,096 | 184,549,376 B | 256 MiB | 256 MiB |

固定 256 MiB/node 时，bias base 为 `268,419,072` bytes。最终点仍有
`83,869,696` bytes 地址余量，因此无需改变 Phase 4F 的 256 MiB 配置。

每个 child root 会产生 4 个 256 MiB HBM init 和 4 个 256 MiB HBM out 文件，表观
空间约 2 GiB。四点运行前目标文件系统必须至少有 16 GiB 可用；每次启动新点前再次
检查。禁止自动删除任何旧 artifact 来腾出空间。

### 6.3 Host memory 和整数范围

最终 `elem_count=16,777,216`，安全落在当前 `uint32_t` descriptor 范围内。rows、cols、
reduction counter、transport total 和 DMA bytes 也均未接近现有 32/64-bit 计数上限。

现有 HBM generator 和 logits golden verifier 会将较大的 tensor 展开为 Python 对象。
本阶段不修改它们；启动前记录 `/proc/meminfo`，要求 available host memory 至少 8 GiB。
当前主机满足该条件，但正式执行必须重新检查。

### 6.4 临时目录

根分区 `/tmp` 已接近满容量。所有 focused test、dry-run、HBM generation、真实 SST 和
报告命令必须显式使用：

```text
TMPDIR=/data4/jjgong/tmp
```

不得清理或覆盖用户已有 artifact。

## 7. Timeout 策略

Phase 4F 的 `256x4096` wall time 为 845 秒。仅用于调度预算的线性外推约为：

```text
512 rows   ~1,690 s
1024 rows  ~3,380 s
2048 rows  ~6,760 s
4096 rows ~13,520 s
```

这不是性能预测，也不进入实验结论。为了保留足够余量，固定 timeout 为：

| shape | timeout |
|---|---:|
| 512x4096 | 3,600 s |
| 1024x4096 | 7,200 s |
| 2048x4096 | 10,800 s |
| 4096x4096 | 14,400 s |

timeout 是 canonical signature 的一部分。child runner 使用 GNU `timeout` 包裹完整
pipeline；底层 pipeline 的退出清理会终止 SST 进程组。发生 exit code 124 后，parent
必须记录正式 `TIMEOUT`、实际 wall time、point signature、child root 和日志位置，立即
停止后续更大点。不得以扩大 timeout/retry window、改变网络配置或减少 worker 的方式
继续同一阶段；超时原因留到后续独立诊断。

## 8. Runner 和证据边界

新增独立 parent runner：

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  run_sfu_4096x4096_capacity.sh
```

它逐点调用现有：

```text
run_sfu_unified_job_distributed_scaling.sh
```

Phase 4F 的 parent runner、8 点矩阵和报告模块保持不变。新阶段可复用
`plot_sfu_phase4f_large_scale.py` 中与矩阵无关的 `PointSpec`、`parse_child_point()`、
`upsert_parent_manifest()` 和 `load_parent_manifest()`，但必须在新模块中实现自己的四点
resolver、完整矩阵 gate 和容量报告，不能调用 Phase 4F 的 8 点 `resolve_point()`。

新增容量控制和证据模块：

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  sfu_4096x4096_capacity.py
```

公开控制仅包括：

```text
GOLEM_SFU_CAPACITY_ROOT=<fresh absolute root>
GOLEM_SFU_CAPACITY_DRY_RUN=0|1
GOLEM_SFU_CAPACITY_POINT_LIST="rows:4096:16:16 [more ordered prefix points]"
GOLEM_SFU_CAPACITY_STOP_ON_FAIL=1
```

默认 point list 必须严格等于四点阶梯。override 只允许默认矩阵的有序前缀，用于 focused
恢复；禁止跳过较小点直接启动更大点。真实运行强制 `STOP_ON_FAIL=1`。

## 9. Artifact 和恢复规则

canonical root：

```text
src/sst/elements/golem/tests/artifacts/sweeps/
  sfu_4096x4096_capacity_explicit_noc_20260716
```

parent root 必须包含：

```text
parent_schema
capacity_preflight.csv
capacity_status.csv
capacity_manifest.csv
children/<point>/attempt-NNNN/
completed/<point>.marker
report/sfu_4096x4096_capacity_source_data.csv
report/sfu_4096x4096_capacity_summary.md
```

每次 attempt 不可变。PASS marker 必须绑定完整 signature、child runner、容量模块、
Phase 4F generic parser、HBM generator 和 golden verifier 的 SHA-256，以及 pipeline args
hash、child root 和 output hash。只有 marker、signature、hash、完整 artifact 和重新解析
的证据完全一致时才能 resume；失败 attempt 必须保留。

本阶段只生成可重建 CSV 和中文 Markdown 表，不生成 DSE 图或性能排名图。

## 10. 每点验收公式

对每个 `rows=R, dim=4096, workers=16` 的 PASS 点，必须满足：

```text
golden_checked                       = R * 4096
golden_mismatches                    = 0
active workers/bands                 = 16 / 16
max reduction requests               = R * 16
max reduction responses              = R * 16
sum reduction requests               = R * 16
sum reduction responses              = R * 16
explicit transport received          = 4 * R * 16
GlobalMemory immediate + queued      = 4 * R * 16
GlobalMemory received                = 4 * R * 16
GlobalMemory rejected                = 0
stale reduction messages             = 0
DMA read/write issue and completion  = R * 16
DMA read bytes                       = R * 4096 * 4
DMA write bytes                      = R * 4096 * 4
DMA retry/exhaustion/write retry     = 0 / 0 / 0
output size                          = R * 4096 * 4
```

queueing 和 latency 可以记录，但不作为失败条件；rejected、stale、retry 和 exhaustion
必须为 0。网络/VN/memory/timeout/profile 必须与固定 signature 完全一致。

## 11. 测试和隔离

新增：

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  test_sfu_4096x4096_capacity.py
```

focused tests 必须覆盖：

- 四点顺序、只允许有序前缀和禁止 `dim=8192`；
- 容量表、布局边界、计数公式和 timeout；
- disk、host memory、TMPDIR preflight；
- inherited conflict rejection、root lock、schema、signature 和 immutable attempt；
- dry-run 不启动 SST、不生成大 HBM 文件；
- 首个非 PASS 立即停止；
- cached PASS 必须重新验证 output hash 和全部 child artifact；
- 四点 source CSV/summary 的完整重建；
- Phase 4F 默认矩阵和已有 focused suite 不变。

实现只允许新增 softmax test tooling 和更新 softmax 文档。不得修改 GEMM runner、GEMM
architecture、GEMM guest binary 或 production component。若实现意外触及任一 shared/GEMM
文件，必须停止并重新评审范围，同时执行原有真实 GEMM 回归。

## 12. 完成判据

只有以下条件同时满足，才能声明 `4096x4096` 已完成：

- preflight 对四点全部 PASS；
- 四点按顺序真实 SST PASS，无参数漂移；
- 最终点 golden `16,777,216/0`；
- 最终点四类 reduction totals 各 `65,536`，transport total `262,144`；
- 最终点 DMA read/write bytes 各 `67,108,864`，全部 retry 为 0；
- parent resume 重新验证四点且不启动新 SST；
- source CSV 和 Markdown summary 可从 child artifact 确定性重建；
- softmax focused suite、shell syntax、Python syntax 和 diff hygiene 全部通过；
- Phase 4F 历史矩阵、GEMM 路径和 production component 没有被修改。

在上述条件完成前，只能表述为“目标是 4096x4096”，不能表述为“已经支持
4096x4096”。

## 13. 2026-07-16 范围更新：暂停容量阶梯并优化 wall time

真实 SST 已完成并通过 `512x4096` 和 `1024x4096`；当前最大验证 shape 为
`1024x4096`。用户决定本轮到此停止，因此 `2048x4096` 和 `4096x4096` 延期，既不算
失败也不算 timeout，且不得把 dry-run marker 误报为真实结果。

后续先调查 SST wall-clock。优化不得改变本设计冻结的数学、explicit-NoC 网络参数、
VN、worker/band、chunk、retry 或 memory 配置。优先使用较小代表点做单变量 A/B，依次
验证 Vanadis pipeline trace、NoC/GlobalMemory/DMA 逐事件文本、统计加载范围和 SST host
thread 配置；任何可能改变 guest polling 或模拟时序语义的方案必须另行设计和批准。

### 13.1 已确认的耗时证据

`1024x4096` 的约 `2914s` parent wall time 可拆为：输入/HBM/启动约 `113s`、SST 主仿真
约 `2755s`、输出/golden/marker 约 `26s`，主仿真占约 94.5%。16 个 Vanadis core 合计
执行 `115,777,120` core cycles、退休 `41,858,634` 条指令，并发出约 `301,434` 个 RoCC
command。当前 CPU builder 默认开启同路径 pipeline trace，每次 retire 都执行 `fprintf`；
runner 还开启 NoC、GlobalMemory、DMA 和 band 逐事件文本。另一方面，mailbox
`adaptive_wait_eq` 和 RoCC SFU wait 的逐周期重试也是指令/事件热点。因此日志是高优先级
低风险假设，但不是唯一根因。

### 13.2 优化层级

1. **构建隔离**：custom Softmax guest 已生成时，禁止共享 pipeline 重编未使用的 GEMM
   guest。默认 GEMM 路径行为必须完全不变，并以原 GEMM pipeline 回归证明。
2. **宿主可观测性**：关闭 Vanadis pipeline trace 和逐事件 NoC/GM/DMA 文本；保留一次性
   网络/VN 配置、golden、reduction/transport/DMA/NoC 汇总 stats。该层必须保持 simulated
   time、output SHA 和全部验收计数不变。
3. **Benchmark quiet 模式**：只移除 guest 成功路径的 band/debug `printf`，保留错误和
   最终 PASS 摘要。由于打印本身是被模拟的 RISC-V 指令，该层会改变 simulated time，
   必须使用新口径并禁止与旧数据直接比较。
4. **Selective stats 与 host threads**：先只启用验收所需 stats，再分别测试 SST
   `--num-threads=2/4`。线程收益不得假设，必须验证确定性。
5. **事件驱动 wait**：若前四层收益仍不足，再单独设计 SFU 完成事件唤醒 RoCC/CPU，
   取代逐周期 wait 重试。该层修改 production 时序模型，不属于本轮低风险优化。

### 13.3 A/B 合同

首轮使用已有 `64x4096, workers=16` 路径，单点 watchdog 固定 `600s`，一次只改变一个
变量。每点必须比较 wall time、simulated time、retired instructions、日志字节数、output
SHA、golden、四类 reduction、transport、DMA retry 和 NoC counters。任何 mismatch、
timeout 或参数漂移立即停止。低风险层在 `64x4096` 通过后用 `256x4096` 复验；在确认
收益前不重跑 `1024x4096`，也不恢复 2048/4096 容量阶梯。
