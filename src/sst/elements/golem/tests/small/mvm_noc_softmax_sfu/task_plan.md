# Task Plan: MVM NoC Softmax SFU

## Goal

实现独立 `golem.SFU` 子组件，并在 `mvm_noc_softmax_sfu` small test 中用
跨 tile online softmax 替代 CPU fallback softmax。

## Current Phase

Phase 9 standalone softmax-only benchmark bring-up:

- Phase 8 fused softmax 功能验证基本完成，`64x64`、`128x128`、`256x256`
  已通过 golden checker，`512x512` no-ctrl 基线也已通过。
- `1024x1024` fused softmax no-ctrl 压力项已暴露 DMA retry/exhausted 与 golden
  失败；原始 GEMM `1024x1024x64` 对照已通过。
- A1 `GOLEM_SFU_INTERLEAVE_GEMM=1` 实验证实 C tile reload 可以被绕过，但在
  `512x512` 中只减少 `64` 次 DMA read 和 `1 MiB` 读流量，主要瓶颈不是 C tile reload，
  而是 no-ctrl GEMM A/B DMA 数据路径及其 retry 压力。
- 已明确 standalone softmax 与 fused GEMM+softmax 是两条不同评估路线：
  standalone softmax 必然从 HBM 读 logits，优化目标是必要 HBM read/write 的效率；
  fused 路径才优化 GEMM C 的中间 HBM write/read。
- standalone softmax-only benchmark 已建立：HBM init 可将 logits 预置到 GEMM C tile
  区域，guest 只执行 `HBM C tile -> SFU -> HBM C tile`，不跑 GEMM。
- `128x128` 和 `512x512` standalone 真实 SST 已通过 logits golden；`512x512`
  DMA 统计只剩 64 次读、64 次写，总读写各 1 MiB，且 `timeout_retry=0`、
  `timeout_exhausted=0`。这说明 standalone SFU 必要 HBM read/write 路径当前不是
  主要阻塞。
- `1024x1024` standalone 曾暴露 SFU issue/wait 节流问题：默认
  `GOLEM_SFU_MAX_INFLIGHT=8` 时每 core 需要一次性 issue 16 个 tile，超过 SFU
  credit 后在进入 wait 前阻塞；`GOLEM_SFU_MAX_INFLIGHT=32` 对照可完成 SST，
  但 verifier 失败且 `write_completion=241/256`。
- 已实现按完整 `m_tile` row-band 的有界 issue/wait 窗口。修复后默认
  `GOLEM_SFU_MAX_INFLIGHT=8` 的 `1024x1024` standalone 已通过 logits golden：
  wall time `159s`、simulated time `525.847 us`、`read/write_issue=256/256`、
  `write_completion=256/256`、`timeout_retry=4`、`timeout_exhausted=0`。
- Phase 9A standalone primitive ABI 骨架已落地：新增 `SFUPrimitiveDesc`、
  `SFUPrimitiveOp`、RoCC func7 预留 `0x19/0x1a` 和 guest primitive wrapper。
- Phase 9B standalone unary primitive 最小路径已落地：SFUAPI/RoCC 支持
  `EXP`、`LOG`、`RECIPROCAL`，当前实现从 local GM 读写 fp32 buffer，并通过
  `GOLEM_SFU_PRIMITIVE_SMOKE=1` 的 RISC-V/SST smoke 验证。
- primitive smoke 已扩展到 `1024x1024` 逻辑规模：
  `GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS=1048576`、默认 `chunk_elems=4`。
  当前 guest 每个 op 只发一次 primitive，SFU 组件计算真实小 chunk 用于校验，并通过
  `sfu_primitive_elems` 统计逻辑处理量。最新实跑 `run_20260702_162003_2304427`
  通过，`processed_elems=3145728`、`sfu_ops_issued=3`、simulated time
  `226.537 us`、wall time 约 `52.6s`。
- 当前 primitive 数学仍是 SST SFU 组件内的 host C++ 功能模型：
  `std::exp`、`std::log` 和 `1.0f / value`，不是 RTL 或周期精确数学硬件。
- 已明确后续 SFU 目标不是一步到位 RTL，而是先升级为 hardware-like
  architecture timing model：保留 host C++ 数学结果计算，同时增加 per-op latency、
  issue bandwidth、queue depth、resource occupancy、backpressure 和 stall/wait
  统计，使 SFU 在 SST 中表现得更像受资源约束的硬件流水线。
- Phase 9C standalone unary primitive 扩展已落地：新增 `RSQRT`、`TANH`、
  `SIGMOID`，并通过小规模真实 SST smoke：
  `run_20260703_125355_3868668`，`processed_elems=96`、`sfu_ops_issued=6`、
  simulated time `276.651 us`。
- Phase 9D 最小 HBM streaming primitive benchmark 已落地：新增
  `GOLEM_SFU_PRIMITIVE_HBM_STREAM` 路径，执行 HBM C region read -> local GM ->
  `sfu_primitive(EXP)` -> HBM writeback，并通过真实 SST smoke
  `run_20260703_131521_3969240`，`hbm_read_bytes=256`、
  `hbm_write_bytes=256`、`sfu_ops_issued=1`、`sfu_primitive_elems=64`、
  simulated time `294.77 us`。
- HBM streaming primitive 已支持可配置 op list：`GOLEM_SFU_PRIMITIVE_HBM_OPS=ALL`
  展开为 `EXP,LOG,RECIPROCAL,RSQRT,TANH,SIGMOID`，并通过真实 SST smoke
  `run_20260703_132634_4023666`，`processed_elems=96`、
  `hbm_read_bytes=384`、`hbm_write_bytes=384`、`sfu_ops_issued=6`、
  `sfu_primitive_elems=96`，simulated time `382.396 us`。
- HBM streaming primitive all-op sweep 已按当前实验要求收敛到
  `16/1024/4096` elements/op 三档，并生成 CSV、notes、SVG、PNG、PDF 组会材料。
  三档 HBM stream bytes 均与理论值一致；DMA/SFU issue count 均为 `6`，原因是
  本轮 `chunk_elems == total_elems`，每个 op 只发一个 chunk。
- 固定 `chunk_elems=1024` 的 event-scaling sweep 已完成两档 PASS：
  `1024` elems/op -> `chunks=6`、DMA read/write `6/6`、wait `12`；
  `2048` elems/op -> `chunks=12`、DMA read/write `12/12`、wait `24`。
  `4096` all-op 与 `65536` single-op `EXP` 诊断均显示长 wall time 来自 per-chunk
  guest/SST 执行开销随 chunk 数放大，而不是全量统计或多 op 数学本身。
- `65536` single-op `EXP` 已用大 chunk 完成真实 SST PASS 诊断：
  `chunk_elems=4096` -> `chunks=16`、DMA read/write `16/16`、wait `32`、
  simulated time `2.15259 ms`、wall `422s`；`chunk_elems=8192` ->
  `chunks=8`、DMA read/write `16/16`、wait `32`、simulated time
  `2.09856 ms`、wall `398s`。对应 chunk 诊断 CSV/notes/SVG/PNG/PDF/TIFF
  已生成。

## Active Checklist

- [x] 明确 SFU 是通用 Special Function Unit，不是 softmax-only 组件。
- [x] 新增并编译 `src/sst/elements/golem/sfu/sfu.h/.cc`。
- [x] 在 RoCC 中接入默认关闭的 SFU func7：`0x17/0x18`。
- [x] 在 `cpu_builder.py` 中用 `GOLEM_SFU_ENABLE` 控制 SFU 挂载。
- [x] 新增 RISC-V workload、SFU runtime、pipeline wrapper 和 golden checker。
- [x] 验证旧 GEMM smoke 默认路径未被 SFU 修改破坏。
- [x] 验证 fused softmax：`64x64`、`128x128`、`256x256`。
- [x] 验证 fused softmax no-ctrl 基线：`512x512`。
- [x] 验证原始 GEMM 对照：`1024x1024x64 fp32 --verify-c`。
- [x] 诊断 fused softmax no-ctrl 压力项：`1024x1024` 仿真时间长且 golden 失败。
- [x] 建立 standalone softmax-only benchmark：预置 logits 到 HBM，只运行 SFU softmax，不跑 GEMM。
- [x] standalone `128x128` 真实 SST + logits golden 验证通过。
- [x] standalone `512x512` 真实 SST + logits golden 验证通过，HBM->SFU 路径低 retry。
- [x] standalone `1024x1024` 默认 inflight=8 阻塞复现，inflight=32 性能对照完成。
- [x] 修复 standalone SFU 调度：有界 issue/wait 窗口，并验证 `1024x1024` 默认 inflight=8 的 HBM writeback 完成。
- [x] Phase 9A standalone primitive ABI 骨架：descriptor、op enum、RoCC func7 预留和 guest wrapper。
- [x] Phase 9B 实现 standalone unary primitive：`EXP`、`LOG`、`RECIPROCAL` 优先。
- [x] primitive smoke 扩展到 `1024x1024` 逻辑规模，并修复长仿真问题。
- [x] 新增 `sfu_primitive_elems` 统计，区分真实 chunk 校验和逻辑 processed elems。
- [x] Phase 9C 扩展 standalone unary primitive：`RSQRT`、`TANH`、`SIGMOID`。
- [x] Phase 9D 新增最小 HBM streaming primitive benchmark：HBM read/write +
  SFU `EXP`，并导出 DMA/SFU/NoC 统计。
- [x] HBM streaming primitive 支持 `GOLEM_SFU_PRIMITIVE_HBM_OPS` 多 op 列表和
  HBM generator 正输入预置。
- [x] 完成 HBM streaming primitive all-op 三档 sweep：`16/1024/4096` elements/op。
- [x] 新增 Python 汇总/绘图脚本并生成 CSV、notes、SVG、PNG、PDF 组会材料。
- [x] 安装隔离 Python 绘图环境：`/data4/jjgong/.venvs/golem-plot`。
- [x] 完成固定 `chunk_elems=1024` 的 event-scaling sweep：`1024/2048` PASS。
- [x] 对 `4096` all-op 和 `65536` single-op `EXP` 做 timeout 诊断。
- [x] 用大 chunk 完成 `65536` single-op `EXP` PASS 诊断：`chunk_elems=4096/8192`。
- [x] 新增 `plot_sfu_hbm_chunk_diag.py` 并生成 `65536` EXP chunk 诊断组会图。
- [x] 设计并实现 batched primitive descriptor，减少 per-chunk guest-side 调度开销。
- [x] 完成 batch smoke：`1024` elems、`chunk_elems=256`、4 chunks 合并为
  1 次 `sfu_ops_issued`，`sfu_primitive_elems=1024`。
- [x] 做 batch vs non-batch event-scaling 对照，并生成组会图。
- [ ] fused GEMM+softmax 后续优化：接回/复用 ctrl-link/WCP 数据通路，并评估 local-accum handoff 是否值得进入正式路径。

## Key Documents

- 设计：`design.md`
- 主实现计划：`implementation_plan.md`
- 决策、问题、验证记录：`findings.md`
- 历史进展摘要：`progress.md`

## Open Questions

1. 是否要把 SFU-managed reducer state 替换为显式 SimpleNetwork 消息？
2. HBM streaming primitive 下一步是先设计 batched primitive descriptor，还是先
   做 `chunk_elems=4096/8192` 的 all-op 大规模 sweep？
3. `512x512` 应作为常规回归，还是仅作为压力测试按需运行？
4. standalone softmax 的 HBM->SFU 数据路径应复用现有 GlobalMemory DMA，还是新增
   SFU 专用 HBM streaming/read descriptor？
5. fused GEMM+softmax 是否需要作为单独性能路线保留，还是只作为 attention/GEMM 后处理
   场景的专项优化？

## Next Plan

1. 固化 primitive smoke 和 batch primitive 语义：
   - 文档中明确 `GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS` 是逻辑规模；
   - `GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS` 是真实 local GM 工作集；
   - 当前 smoke 用于 ABI/功能/统计验证，不作为 HBM 带宽 benchmark。
   - `SFUPrimitiveBatchDesc` 只聚合已有 `SFUPrimitiveDesc`，不引入新数学语义；
   - batch 统计应满足：多个 child chunk 只产生一次 `sfu_ops_issued`，
     `sfu_primitive_elems` 等于 child 元素数之和。
2. 固化本轮 HBM streaming primitive 与 batch smoke 报告：
   - 保留 `16/1024/4096` 源 CSV 和 SVG/PNG/PDF 图；
   - README 中明确 `65536` all-op 未完成，`65536` single-op `EXP` 已在
     `chunk_elems=4096/8192` 下完成 PASS 诊断；
   - README 中记录 `sfu_hbm_batch_exp_elems_1024_chunk256_allstats`：
     `chunks=4`、DMA read/write `4/4`、`sfu_ops_issued=1`、
     `sfu_primitive_elems=1024`；
   - 组会汇报中强调本轮验证的是 HBM bytes 线性和 ABI/DMA/SFU event 正确性。
3. 已完成 batch vs non-batch event-scaling sweep：
   - 固定小 chunk，例如 `chunk_elems=256` 或 `1024`；
   - 已跑 `1024/4096`，`EXP`，`chunk_elems=256`；
   - 已采集 PASS 行、DMA summary、`sfu_ops_issued`、`sfu_primitive_elems`、
     wall time 和 simulated time；
   - non-batch 的 `sfu_ops_issued` 随 chunk 数增长为 `4/16`，batch 均为 `1`；
   - DMA read/write 与 HBM bytes 保持一致，证明 batch 降低的是控制 issue 数。
4. 已更新绘图脚本：
   - 新增 `plot_sfu_hbm_batch_compare.py`；
   - 输出 issue count、DMA issue count、wall time、simulated time 四组图；
   - 图和 notes 明确 batch 降低的是 guest/RoCC issue overhead，不改变 HBM bytes。
5. 下一轮 batch sweep 扩展：
   - 可加 `16384` single-op `EXP`，但建议先设 `timeout` 并观察 wall time；
   - 可把 op list 从 `EXP` 扩展到 `ALL`，验证多 op 下 batch 是否继续压缩 issue；
   - 优先补一个 SFU stats CSV exporter，避免后续手动 grep `stats_selfcom.txt`。
6. SFU timing model 设计：
   - 目标层级是 SST C++ architecture timing model，不是 RTL；
   - primitive 结果仍可由 host C++ `std::exp`/`std::log` 等函数计算；
   - 为每类 primitive 增加可配置 latency、throughput/issue bandwidth、queue depth、
     pipeline occupancy 和 backpressure；
   - 新增或整理 per-op issue/complete/stall/wait/latency 统计，用于 softmax 阶段
     瓶颈分析；
   - 先在 standalone primitive/HBM streaming benchmark 上验证，再接回 fused
     softmax 路径。
7. 单独诊断 `65536` 长运行：
   - 已复现单 op `EXP` + `chunk_elems=1024` + 减统计仍然超时；
   - 已验证 `chunk_elems=4096/8192` 可完成 `65536` single-op PASS；
   - 后续不建议继续用 `chunk_elems=1024` 跑 65536 完整 PASS；
   - batch 路径可用于重新评估 `65536` single-op `EXP` 是否能在小 chunk 下
     显著降低 wall time，但仍需设置 timeout 和阶段性日志检查。
8. 评估是否继续扩展 primitive op：
   - `SQRT` 可直接复用 `std::sqrt`；
   - `GELU`、`LAYERNORM` 属于更高阶/fused primitive，建议等 HBM streaming
     benchmark 设计清楚后再做。
9. 保持 row-band issue/wait 作为 standalone/fused bulk softmax 的默认调度；
   后续更大规模若再次出现 `write_completion < write_issue`，再补
   `remote_store` 显式 ack/fence 或复用 GlobalMemory DMA write completion token。
10. 对比 standalone `512x512/1024x1024` 与 fused/no-ctrl `512x512/1024x1024`，
   固化性能结论：
   - standalone 必要 C/logits read/write；
   - fused 混合路径中的 GEMM A/B DMA；
   - interleaved local-accum 只消掉的 C reload。
11. fused GEMM+softmax 单独排期：在 standalone softmax 基线稳定后，再决定是否把
   A1 local-accum handoff 正式化。
