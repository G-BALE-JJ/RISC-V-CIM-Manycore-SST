# SFU Phase 4F 大规模 Explicit-NoC Softmax 设计

## 目的

在成熟 GEMM 测试已经使用的同一套标准 NoC 配置下，评估 unified-job softmax 随矩阵维度、worker 数量和行数扩展时的性能。Phase 4F 只使用已经完成的 `explicit_noc` reduction transport，不扫描带宽或 VN。

带宽压力实验降级为可选诊断，不再属于下一阶段 softmax 主线。Phase 4E 已经在与标准 GEMM 相同的网络配置下验证了真实 reduction request/response 流量，因此正常运行性能评估不依赖带宽扫描。

## 标准 GEMM 网络配置

网络配置的权威来源是 `src/sst/elements/golem/tests/configs/30_network.env`。`run_noc_dma_pipeline.sh` 启动时先通过 `configs/default.env` 自动加载该文件，然后才应用脚本内部的兜底值。

Phase 4F 必须显式固定以下实际生效值：

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

这些值不是 `run_noc_dma_pipeline.sh` 中的紧急兜底值。脚本里的 `25GB/s` 和 `8KB` 只在 preset 缺失时使用；显式环境变量或 CLI 参数仍可覆盖 preset 和兜底值。

标准网络 preset 和已完成的默认 GEMM 回归均解析为：link、xbar 和 directory highlink 为 `1200GB/s`，router 输入/输出 buffer 为 `512KB`，flit 为 `128B`。现有 4096-shaped SST artifact 也记录了相同网络配置，但它只作为配置证据，不能代替对纯 GEMM workload mode 的识别。

## Softmax 架构约束

每个 Phase 4F 实验点必须使用：

```text
distributed_reduction_transport=explicit_noc
num_vns=3
request_vn=0
ordinary_response_vn=1
dma_response_vn=0
reduction_vn=0
chunk_elems=256
staging_rows=4
job_rows=4
cooperative_groups=1
retry_ticks=1024
max_retries=8
```

VN0/VN1/VN2 兼容性已经在 Phase 4E 完成。`modeled_noc` 不进入 Phase 4F 实验矩阵。除容纳更大 tensor 所必需的 memory-node capacity 外，不允许在性能点之间改变 bandwidth、xbar、flit、buffer、topology、retry 或 VN 参数。

Phase 4F 不修改 SFU 数学、reduction message、GlobalMemory、SimpleNetwork、GEMM、guest ABI 或 primitive/batch softmax。

## 实验矩阵

实验按阶段组织，每个阶段只改变一个规模轴。重复 anchor identity 只执行一次。

### 阶段 A：维度扩展

固定 `rows=16`、`worker_cores=16` 和 `band_cores=16`：

```text
16x512
16x1024
16x2048
16x4096
```

该阶段在最大 column cooperation 下测量每行列方向工作量增加的影响。

### 阶段 B：大维度下的并行度收益与效率验证

固定 `rows=16` 和 `dim=4096`：

```text
worker_cores/band_cores = 4/4
worker_cores/band_cores = 8/8
worker_cores/band_cores = 16/16
```

该阶段不测试单 worker，也不再论证“大维度是否应该使用多 worker”。多 worker
column cooperation 已经是目标架构；这里要量化的是 4、8、16 workers 之间的实际
加速、并行效率和收益递减，并确认旧 primitive 路径中观察到的 16-worker 优势在
unified-job + explicit-NoC reduction 路径下是否仍然成立。

`16/16` 点与阶段 A 的 `16x4096` anchor 相同。当 signature 和 artifact 仍然有效时
不得重复执行。因此阶段 B 只新增 `4/4` 和 `8/8` 两次真实 SST 运行，再与复用的
`16/16` anchor 组成三点并行度曲线。

### 阶段 C：大维度下的行数扩展

固定 `dim=4096`、`worker_cores=16` 和 `band_cores=16`：

```text
rows=16
rows=64
rows=256
```

`rows=16` 是 Stage A/B 共用的 anchor。Stage C 串行执行，在首个无效或超时点停止；不得静默减少 rows，也不得修改 retry 或网络参数。

默认矩阵最终包含 8 个唯一真实 SST 点：

```text
16:512:16:16
16:1024:16:16
16:2048:16:16
16:4096:16:16
16:4096:4:4
16:4096:8:8
64:4096:16:16
256:4096:16:16
```

## 内存容量和超时策略

内存容量只用于保证实验可执行，不作为性能搜索轴。runner 必须记录解析后的值，并使用：

```text
dim <= 1024: mem_node_size=134217728 bytes
dim >= 2048: mem_node_size=268435456 bytes
```

timeout 按 shape 固定，并写入 artifact：

```text
16x512: 900 seconds
16x1024: 1800 seconds
16x2048: 2400 seconds
16x4096: 3600 seconds
64x4096: 7200 seconds
256x4096: 14400 seconds
```

超时是状态为 `TIMEOUT` 的正式实验结果，不代表可以改变网络或正确性约束。只有完整 point signature 与已有 marker 一致时，才允许在同一个 root 中恢复并复用已完成点。

## 运行器架构

新增专用 parent runner：

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  run_sfu_phase4f_large_scale_explicit_noc.sh
```

parent runner 每次在独立 child root 中调用现有 `run_sfu_unified_job_distributed_scaling.sh` 执行一个点。通用 GEMM runner 和 architecture 文件保持不变。

parent runner 必须：

- 固定标准 GEMM 网络配置和 explicit-NoC/VN0 约束；
- 拒绝继承环境中冲突的 transport、VN、network、buffer、retry、chunk、staging 或 job-row 值；
- 使用防并发冲突的 parent root lock；
- 在 child root 名称和 signature 中包含 stage 与完整 point identity；
- 对已经验证有效的重复 anchor 直接复用，不重新执行；
- 拒绝旧 manifest schema、旧 marker、hash 不一致和不完整 child artifact；
- 支持 dry-run、focused point-list override、resume 和 stop-on-fail；
- 串行执行真实实验点。

支持以下公开控制：

```text
GOLEM_PHASE4F_LARGE_SCALE_ROOT=<fresh absolute root>
GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN=1
GOLEM_PHASE4F_LARGE_SCALE_STOP_ON_FAIL=1
GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="rows:dim:workers:bands [more-points]"
```

默认 point list 必须严格等于上述 8 点矩阵。override 只用于 focused recovery 和测试；解析后的点仍必须完整签名。

## 产物和正确性检查

每个可用点必须满足：

- child manifest 为 `PASS/PASS` 且 exit code 为 0；
- 独立 full-row logits golden 的 `checked = rows * dim`，mismatches 为 0；
- active SFU worker/band 数量正确；
- Max/Sum request 和 response 四类总数各等于 `rows * worker_cores`；
- explicit transport receive 等于 `4 * rows * worker_cores`；
- GlobalMemory immediate 与 queued send 之和等于同一个 transport total；
- rejected 和 stale reduction message 均为 0；
- 运行时 VN 和 network profile 与固定约束完全一致；
- DMA issue/completion 和 bytes 与 rows、dim、worker partition 一致；
- DMA retry、exhaustion 和 write-timeout retry 均为 0；
- output size、output hash、signature、log、stats、NoC summary 和 DMA summary 全部存在且相互一致。

当其他 transport 和 lifecycle gate 均通过时，reduction queueing 只作为观测结果，不判定为失败。即使大规模点出现 queueing，也不得改变固定的 `1200GB/s` 网络配置。

## 父级 Manifest 和指标

父 `large_scale_manifest.csv` 对每个唯一 point 保存一行 canonical record：

```text
run_id,stage,rows,dim,chunk_elems,worker_cores,band_cores,
transport,reduction_vn,num_vns,dma_response_vn,noc_link_bw,noc_xbar_bw,
dirctrl_highlink_bw,noc_input_buffer,noc_output_buffer,gm_buffer,flit_size,
mem_node_size,retry_ticks,max_retries,timeout_sec,status,exit_code,
artifact_validation,golden_checked,golden_mismatches,transport_events,
transport_immediate,transport_queued,transport_rejected,transport_stale,
inbox_high_water,latency_avg_cycles,latency_max_cycles,total_send_packets,
total_send_bits,total_xbar_stalls,simulated_time_us,wall_time_sec,dma_timeout_retry,
dma_timeout_exhausted,dma_write_timeout_retry,output_sha256,child_root
```

主要性能指标包括：

- simulated time 和 wall time；
- reduction transport 平均/最大 latency；
- transport event 和 queued send；
- total packet、bit 和 xbar stall；
- 每行和每元素归一化时间；
- `dim=4096` 下 4/8/16 workers 的加速、并行效率和收益递减；
- DMA issue/completion、bytes、retry 和 round-trip 指标。

这些数据是单次确定性 SST 结果，不允许添加误差棒、置信区间或统计显著性结论。

## 分析和图表产物

新增：

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  plot_sfu_phase4f_large_scale.py
```

该脚本重新解析父/子证据、再次执行全部 gate，并生成：

```text
tests/artifacts/sweeps/sfu_phase4f_large_scale_explicit_noc_20260715/report/
  sfu_phase4f_large_scale_source_data.csv
  sfu_phase4f_large_scale.svg
  sfu_phase4f_large_scale.pdf
  sfu_phase4f_large_scale.png
  sfu_phase4f_large_scale_qa.md
```

英文 16:9 结果图包含：

- dimension scaling 的 runtime 和 reduction latency；
- `dim=4096` 下 4/8/16 workers 的 speedup/efficiency 和收益递减；
- row scaling 的 total time 和 normalized time per row；
- NoC pressure 指标和紧凑的 correctness/lifecycle 区域。

图中只使用 explicit-NoC，并标注固定的标准 GEMM network profile。不得加入 modeled-NoC、bandwidth comparison、未来 fusion 计划或推断统计。

## 测试策略

新增：

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  test_sfu_phase4f_large_scale.py
```

focused tests 必须覆盖：

- 标准 GEMM 网络值及其 preset 来源；
- 8 点默认矩阵和 duplicate-anchor elimination；
- 阶段 B 不包含单 worker，只新增 `4/4`、`8/8` 并复用 `16/16` anchor；
- 按维度解析 memory/timeout；
- explicit-NoC/VN0-only 环境和 inherited-conflict rejection；
- signature、manifest、child environment 和 runtime log 中的 network profile；
- root lock、dry-run、resume、旧 schema、旧 marker 和损坏 output；
- 按 shape 推导的 golden、transport、DMA、byte 和 output-size gate；
- latency、queueing、packet/bit、xbar、runtime 和 DMA 指标解析；
- invalid/timeout 时停止且不修改参数；
- CSV/SVG/PDF/PNG 的确定性生成和完整 source reconstruction；
- SVG editable text、PDF TrueType text、300 dpi PNG 和视觉无重叠。

synthetic fixture 和 dry-run 用于验证 runner/analyzer 行为。只有 focused tests 通过后才能启动真实 SST 点。

## GEMM 隔离约束

专用 Phase 4F runner 不得修改 `src/sst/elements/golem/tests/run_noc_dma_pipeline.sh`、GEMM architecture 文件或 GEMM guest binary。

如果实现只新增 softmax runner、analyzer 和 tests，则现有标准 GEMM artifact 与 focused isolation test 足够。如果不得不修改 shared runner、preset、architecture 或 production component，则 Phase 4F 验收前必须重新运行现有默认 GEMM 回归。

## 执行顺序

1. 记录标准 GEMM 网络证据和 Phase 4E explicit-NoC anchor。
2. 在 focused TDD 下实现 artifact parser 和专用 parent runner 约束。
3. 在 focused TDD 下实现报告生成器。
4. 执行完整 8 点 dry-run，检查解析后的 signature。
5. 串行执行 Stage A，在首个无效点停止。
6. 复用有效 `16x4096` anchor，再执行阶段 B 的 `4/4` 和 `8/8` 两个新增点，
   计算三点并行加速与效率。
7. 再次复用同一 anchor，然后串行执行 Stage C。
8. 生成并视觉检查确定性报告产物。
9. 运行完整 softmax focused suite 和 GEMM isolation gate。

## 完成判据

Phase 4F 完成必须满足：

- 所有可执行点均有 canonical parent/child artifact；
- 每个可用点均通过 golden、transport、DMA、topology 和 artifact gate；
- invalid 或 timeout 点被明确报告，且没有改变 network 或 correctness contract；
- source CSV 和可编辑确定性图表能够重建所有报告数值；
- 阶段 B 明确报告 4/8/16 workers 的加速、效率和收益递减，不包含单 worker；
- 主矩阵不包含 modeled-NoC 或 bandwidth sweep；
- 固定 network 值与标准 GEMM preset 和 runtime log 一致；
- GEMM 路径未改变；如果不可避免修改 shared 文件，则现有 GEMM 回归重新 PASS。

## Phase 4F 后续范围

完成大规模 softmax 特征分析后，再根据主要瓶颈决定是优化 softmax，还是开始 GEMM+softmax fusion。带宽压力实验继续作为可选诊断，不进入默认路线图。
