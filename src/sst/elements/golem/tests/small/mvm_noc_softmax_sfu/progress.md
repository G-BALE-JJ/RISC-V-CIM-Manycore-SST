# Progress Log

This file is intentionally compact. Detailed design and implementation state
live in `design.md`, `implementation_plan.md`, and `findings.md`.

## 2026-06-29

- 明确 SFU 是通用 Special Function Unit，第一版只暴露 fused softmax。
- 确认 softmax 必须是跨 tile full row-wise softmax，不能停留在 tile-local。
- 将方案从传统三阶段 softmax 更新为 online softmax：
  `tile_m/tile_l -> online merge -> normalize`。
- 创建持久规划文档：`design.md`、`implementation_plan.md`、`task_plan.md`、
  `findings.md`、`progress.md`。

## 2026-06-30

- 新增 `golem.SFU` 子组件骨架，并接入 `golem.cc`、`Makefile.am`。
- 在 RoCC 中新增默认关闭的 SFU 指令路径：
  `SFU_SOFTMAX_TILE=0x17`、`SFU_WAIT=0x18`。
- 在 `cpu_builder.py` 中新增 `GOLEM_SFU_ENABLE` 挂载开关，保持旧 GEMM 默认路径不变。
- 新增 `SFUSoftmaxTileDesc`、tagged status、tile stats、online reducer 和 normalize。
- 新增 RISC-V workload、runtime、Makefile、pipeline wrapper 和 golden checker。
- 修复单 array GEMM baseline 读取 B 向量地址的问题，GEMM-only golden 已通过。
- 完成 fused softmax 数值验证：
  - `64x64`: PASS，`checked=4096`，`mismatches=0`
  - `128x128`: PASS，`checked=16384`，`mismatches=0`
  - `256x256`: PASS，`checked=65536`，`mismatches=0`
- 完成旧 GEMM smoke 验证：
  - 不设置 `GOLEM_SFU_ENABLE`
  - `64x64 fp32 --verify-c`: PASS，`mismatches=0`
- 整理项目目录：
  - 删除运行产物、HBM dump、RISC-V 编译产物、Python cache 和临时 tensor 文件。
  - 补充 `.gitignore`，避免生成物再次污染工作区。
  - 新增 `README.md` 作为目录入口。

## Next

- 跑或诊断 `512x512` fused softmax 压力 case。
- 开始 Phase 9A：standalone primitive ABI 和 descriptor/static tests。
