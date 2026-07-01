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
- [ ] 设计并实现 standalone primitive ABI：`EXP`、`LOG`、`RECIPROCAL` 优先。
- [ ] fused GEMM+softmax 后续优化：接回/复用 ctrl-link/WCP 数据通路，并评估 local-accum handoff 是否值得进入正式路径。

## Key Documents

- 设计：`design.md`
- 主实现计划：`implementation_plan.md`
- 决策、问题、验证记录：`findings.md`
- 历史进展摘要：`progress.md`

## Open Questions

1. 是否要把 SFU-managed reducer state 替换为显式 SimpleNetwork 消息？
2. standalone primitive 是先只支持 local GM fp32，还是同步设计 HBM/stride 模式？
3. `512x512` 应作为常规回归，还是仅作为压力测试按需运行？
4. standalone softmax 的 HBM->SFU 数据路径应复用现有 GlobalMemory DMA，还是新增
   SFU 专用 HBM streaming/read descriptor？
5. fused GEMM+softmax 是否需要作为单独性能路线保留，还是只作为 attention/GEMM 后处理
   场景的专项优化？

## Next Plan

1. 开始 standalone primitive ABI 设计与实现：
   - descriptor 独立于 `SFUSoftmaxTileDesc`；
   - local GM fp32 input/output 先行；
   - op 枚举优先 `EXP`、`LOG`、`RECIPROCAL`。
2. 保持 row-band issue/wait 作为 standalone/fused bulk softmax 的默认调度；
   后续更大规模若再次出现 `write_completion < write_issue`，再补
   `remote_store` 显式 ack/fence 或复用 GlobalMemory DMA write completion token。
3. 对比 standalone `512x512/1024x1024` 与 fused/no-ctrl `512x512/1024x1024`，
   固化性能结论：
   - standalone 必要 C/logits read/write；
   - fused 混合路径中的 GEMM A/B DMA；
   - interleaved local-accum 只消掉的 C reload。
4. fused GEMM+softmax 单独排期：在 standalone softmax 基线稳定后，再决定是否把
   A1 local-accum handoff 正式化。
