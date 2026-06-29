# Task Plan: MVM NoC Softmax SFU

## Goal

实现独立 `golem.SFU` 子组件，并在 `mvm_noc_softmax_sfu` 小测试中用跨 tile online softmax 替代 CPU fallback softmax。

## Current Phase

Phase 3: Implementation Plan

## Phases

### Phase 1: Requirements & Discovery

- [x] 明确模块命名为 `SFU`，不叫 `SoftmaxSFU`。
- [x] 明确项目文件放在 `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/`。
- [x] 明确第一版必须支持跨 tile full row-wise softmax。
- [x] 确认应借鉴 GEMM DMA 的 inflight、credit、submit/done、retry 机制。
- [x] 确认 softmax 协议应采用 online softmax 统计合并。
- [x] 确认所有 SFU 修改必须默认关闭且不得影响原始 GEMM 功能。
- **Status:** complete

### Phase 2: Design Documentation

- [x] 编写中文 `design.md`。
- [x] 将 GEMM DMA 竞争/死锁控制借鉴方案写入设计。
- [x] 将 online softmax 统计合并协议写入设计。
- [x] 创建 `task_plan.md`、`findings.md`、`progress.md`。
- **Status:** complete

### Phase 3: Implementation Plan

- [x] 拆分 SFU 子组件实现步骤。
- [x] 定义 SFU API、RoCC 指令编码、参数格式。
- [x] 定义 `mvm_noc_softmax_sfu` 测试入口和 wrapper 复用策略。
- [x] 明确局部构建与全量构建边界。
- **Status:** complete

### Phase 4: Component Implementation

- [ ] 新增 `src/sst/elements/golem/sfu/sfu.h`。
- [ ] 新增 `src/sst/elements/golem/sfu/sfu.cc`。
- [ ] 修改 `golem.cc` 和 `Makefile.am` 注册 SFU。
- [ ] 修改 `rocc/roccAnalog.h` 加载 `sfu` slot 并处理 SFU 指令。
- [ ] 修改 `tests/architecture/cpu_builder.py` 挂载 `golem.SFU`。
- **Status:** pending

### Phase 5: Workload and Verification

- [ ] 在 `mvm_noc_softmax_sfu` 下创建 RISC-V workload。
- [ ] 复用现有 GEMM pipeline 参数和构建方式。
- [ ] 新增 full row-wise `softmax(A @ B)` checker。
- [ ] 验证 64x64、128x128、256x256、512x512 配置。
- **Status:** pending

## Key Questions

1. 后续是否需要把 SFU-managed reducer state 替换为显式 SimpleNetwork 消息？
2. 何时把单 tile RoCC 命令扩展为 batch tile 命令？
3. SFU credit manager 后续是否需要接入 request scheduler 控制路径？
4. 后续是否需要支持 output base 与 input base 分离？

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 组件名使用 `SFU` | SFU 后续不只服务 softmax，也可支持 exp、sigmoid、tanh、layernorm 等 |
| 第一版直接做跨 tile full row-wise softmax | 当前 `N > block_n` 时 tile-local softmax 不正确 |
| 使用 online softmax 统计合并 | 更适合跨 tile 流式合并，减少独立 max/sum reduction 阶段 |
| 借鉴 GEMM DMA 的限流和完成回收机制 | 避免多核同时提交 partial stats 造成竞争或死锁 |
| 不采用 HBM 非原子 read-modify-write reduction | 多核并发会丢更新，DMA retry 不能修复数值原子性 |
| 第一版使用 SFU-managed reducer state | 保证正确性和可调试性，后续再接显式网络建模 |
| 第一版一个 RoCC 命令描述一个 C tile | 语义简单，便于逐 tile 调试和定位同步问题 |
| 第一版沿用 workload DMA 做 tile 数据搬运 | 复用现有可靠路径，SFU 专注数学计算和跨 tile 同步 |
| 所有 SFU 修改默认关闭且不得影响原始 GEMM | 保护已有 GEMM 功能和 small tests 的可复现性 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| 无 | 1 | 当前阶段只进行设计和文档更新 |

## Notes

- 详细实现计划见 `implementation_plan.md`。
- 每次发现构建、同步、数值正确性问题都要同步更新 `findings.md` 和 `progress.md`。
