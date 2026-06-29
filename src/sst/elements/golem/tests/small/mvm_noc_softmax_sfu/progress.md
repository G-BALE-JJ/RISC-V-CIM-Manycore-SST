# Progress Log

## Session: 2026-06-29

### Phase 1: Requirements & Discovery

- **Status:** complete
- Actions taken:
  - 确认用户希望 SFU 作为独立子组件，类似 `array`、`global_memory` 挂在 RoCC 下。
  - 确认模块名称只叫 `SFU`，不叫 `SoftmaxSFU`。
  - 分析当前 tile 形状：`block_n=64` 时，`128/256/512` 都存在跨 N tile 行。
  - 确认 tile-local softmax 对大维度矩阵不满足 full row-wise softmax 正确性。
  - 确认第一版直接做跨 tile full row-wise softmax。
  - 确认多核竞争和死锁控制应参考 GEMM DMA / request scheduler。
  - 确认 softmax 方案应更新为 online softmax 统计合并。
- Files created/modified:
  - `design.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 2: Design Documentation

- **Status:** complete
- Actions taken:
  - 将设计文档改为中文。
  - 将传统三阶段 softmax 改为 `tile_m/tile_l -> online merge -> normalize`。
  - 增加“借鉴 GEMM DMA 的竞争与死锁控制”章节。
  - 创建 planning-with-files 要求的持久规划文件。
- Files created/modified:
  - `design.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 3: Implementation Plan

- **Status:** complete
- Actions taken:
  - 新增 `implementation_plan.md`，把后续实现拆成 checker、SFU 子组件骨架、online reducer、RoCC 指令、`cpu_builder.py` 挂载、RISC-V workload、运行脚本和验证八个阶段。
  - 明确第一版数据边界：workload 继续用现有 DMA 把 C tile 搬到 local GM，SFU 从 local GM 读写 tile，`sfu.wait` 后 workload 再写回 HBM/remote C tile。
  - 明确第一版 RoCC 指令编码使用 `0x17` 和 `0x18`，descriptor 地址通过 `rs1` 传入。
  - 明确第一版 reducer 采用 SFU-managed shared state，key 为 `(job_id, global_row)`，用 online softmax 公式合并 partial stats。
  - 明确构建边界：新增 SFU 源文件后需要 golem element reconfigure；只改 workload 时只在新目录构建。
  - 增加兼容性硬边界：所有 SFU 相关改动必须默认关闭，不设置 `GOLEM_SFU_ENABLE` 时不能影响原始 GEMM 功能和已有 GEMM small tests。
- Files created/modified:
  - `implementation_plan.md`
  - `task_plan.md`
  - `progress.md`

## Test Results

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 文档自检 | 阅读 `design.md` | 包含 SFU 命名、online softmax、GEMM DMA 借鉴方案 | 已写入 | pass |
| 实现计划自检 | 阅读 `implementation_plan.md` | 包含路径、descriptor、RoCC 编码、DMA 边界、构建与验证顺序 | 已写入 | pass |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-06-29 | 无 | 1 | 当前阶段仅更新文档 |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 2 已完成，下一步是 Phase 3 implementation plan |
| Where am I going? | 下一步进入 Phase 4，先写 checker，再实现 SFU 子组件骨架 |
| What's the goal? | 用独立 `golem.SFU` 子组件实现跨 tile online full row-wise softmax |
| What have I learned? | 见 `findings.md` |
| What have I done? | 已创建中文设计文档、持久规划文件和详细实现计划 |
