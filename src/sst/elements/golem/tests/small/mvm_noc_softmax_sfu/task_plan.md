# Task Plan: MVM NoC Softmax SFU

## Goal

实现独立 `golem.SFU` 子组件，并在 `mvm_noc_softmax_sfu` small test 中用
跨 tile online softmax 替代 CPU fallback softmax。

## Current Phase

Phase 8/9 boundary:

- Phase 8 fused softmax 功能验证基本完成，`64x64`、`128x128`、`256x256`
  已通过 golden checker。
- `512x512` 仍作为压力/性能验证项保留。
- Phase 9 是下一阶段：把当前 fused softmax 背后的 SFU 扩展成通用
  standalone primitive 单元。

## Active Checklist

- [x] 明确 SFU 是通用 Special Function Unit，不是 softmax-only 组件。
- [x] 新增并编译 `src/sst/elements/golem/sfu/sfu.h/.cc`。
- [x] 在 RoCC 中接入默认关闭的 SFU func7：`0x17/0x18`。
- [x] 在 `cpu_builder.py` 中用 `GOLEM_SFU_ENABLE` 控制 SFU 挂载。
- [x] 新增 RISC-V workload、SFU runtime、pipeline wrapper 和 golden checker。
- [x] 验证旧 GEMM smoke 默认路径未被 SFU 修改破坏。
- [x] 验证 fused softmax：`64x64`、`128x128`、`256x256`。
- [ ] 验证或诊断 fused softmax：`512x512`。
- [ ] 设计并实现 standalone primitive ABI：`EXP`、`LOG`、`RECIPROCAL` 优先。

## Key Documents

- 设计：`design.md`
- 主实现计划：`implementation_plan.md`
- 决策、问题、验证记录：`findings.md`
- 历史进展摘要：`progress.md`

## Open Questions

1. 是否要把 SFU-managed reducer state 替换为显式 SimpleNetwork 消息？
2. standalone primitive 是先只支持 local GM fp32，还是同步设计 HBM/stride 模式？
3. `512x512` 应作为常规回归，还是仅作为压力测试按需运行？
