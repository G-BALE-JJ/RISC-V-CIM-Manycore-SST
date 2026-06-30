# Findings & Decisions

## Requirements

- 新增 SFU 硬件建模路径，替代当前单核 CPU softmax fallback。
- SFU 是独立 SST 子组件，像 `array`、`global_memory` 一样挂在 RoCC 下。
- 组件名使用 `SFU`，不要在组件名中绑定 `softmax`。
- SFU 长期应是通用 Special Function Unit，后续可扩展 `exp`、`log`、`reciprocal`、
  `rsqrt`、`sigmoid`、`tanh`、`layernorm`、`gelu` 等 primitive 或 fused op。
- 项目文档和 small test 放在 `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/`。
- 第一版 softmax 必须是跨 tile full row-wise softmax。
- 需要考虑多核竞争和死锁问题。
- 设计应符合 online softmax 思路。

## Research Findings

- 当前 softmax CPU 配置基本使用 `block_m=64`、`block_n=64`、`GEMM worker cores=16`。
- `256x256` 配置下 C tile 网格是 `4x4`，每个 core 一个 tile，但每一行跨 4 个 N tile。
- `512x512` 配置下 C tile 网格是 `8x8`，每个 core 四个 tile，每一行跨 8 个 N tile。
- 只要 `N > block_n`，tile-local softmax 就不等价于完整 row-wise softmax。
- 旧 CPU cross-tile prototype 使用 HBM reduction buffer 的非原子 read-modify-write，存在多核 race。
- GEMM DMA / request scheduler 的 inflight、credit、submit/done、retry、batch 思路适合借鉴给 SFU 控制路径。
- DMA retry 能解决传输完成问题，不能解决多个 core 对同一 sum/max 的数值原子更新问题。
- online softmax 可以用每个 tile 的 `(tile_m, tile_l)` 统计量进行稳定合并：
  `m_new=max(m_old,tile_m)`，`l_new=l_old*exp(m_old-m_new)+tile_l*exp(tile_m-m_new)`。

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 使用 row-owner/reducer | 避免多个 core 直接竞争写同一个 HBM reduction entry |
| 使用 online softmax merge | 减少独立 max reduction 与 sum reduction 的分离，表达更接近现代 softmax/attention 硬件 |
| 第一版建议 SFU-managed reducer state | 先保证正确性和可调试性，再考虑显式 NoC 流量建模 |
| 第一版建议内部轻量 credit/inflight 控制 | 借鉴 GEMM DMA 机制，但避免过早耦合 request scheduler |
| 第一版建议原地覆盖 C | 降低 workload 和地址管理复杂度 |
| SFU 骨架已作为独立子组件注册 | `golem.cc` include + `Makefile.am` 源文件列表足够让 `sfu/sfu.cc` 编进 `libgolem.la` |
| RoCC SFU 指令接入默认关闭 | `roccAnalog.h` 只在 `sfuEnable=1` 时加载 `"sfu"` slot；新增 func7 为 `0x17/0x18`，已有 GEMM batch/WCP func7 `0x11` 到 `0x16` 保持不变 |
| cpu_builder SFU 挂载默认关闭 | `GOLEM_SFU_ENABLE` 默认 `0`；仅在设置为 `1` 时挂载 `cpu_rocc.setSubComponent("sfu", "golem.SFU")`，并给 RoCC 传 `sfuEnable` |
| 第一版先做 fused softmax，不暴露 standalone primitive 指令 | 当前迁移目标是替代 CPU fallback softmax；softmax 内部按 `max/reduction -> exp -> sum/reduction -> reciprocal/divide -> normalize` 组织，后续再抽出 standalone primitive |
| Phase 5A descriptor/status 骨架已完成 | `SFUSoftmaxTileDesc` 固定 72 字节 ABI；`issueSoftmaxTile` 记录 `descAddr/tag`，`wait(tag)` 返回并 retire tagged status；尚未读取 GM descriptor/C tile，也尚未实现 tile stats、online reducer 或 normalize |
| Phase 5B/5C SFU 数学路径已实现 | `issueSoftmaxTile` 从 local GM 读取 descriptor 和 fp32 C tile，计算每行 `tile_m/tile_l`，提交到 `(job_id, global_row)` 共享 reducer；`wait(tag)` 在 reducer 未 ready 时返回 false，ready 后 normalize 到 local output GM |
| SFU workload 使用 GEMM tile-packed C layout | 现有 GEMM 输出地址来自 `GemmTaskDescriptor::c_base_mm`，每个 C tile 是 tile-packed 连续块；SFU runtime 不按完整矩阵 row-major 地址推导，而是复用 `gemm_task_desc_for_task` 读取/写回每个 tile |
| SFU Phase 5 数值闭环已通过 64x64/128x128 | `64x64, block=64x64` full path softmax 通过 `softmax(A@B)` golden；`128x128, block=64x64` 跨两个 N tile 的 online reducer 也通过 golden |
| GEMM-only 诊断路径保留 | `GOLEM_SFU_SKIP_SOFTMAX=1` 会在 SFU workload 中只跑 GEMM 并跳过 softmax，便于隔离 GEMM 输出和 SFU 输出问题 |
| SFU 应继续演进为通用数学单元 | 当前 fused softmax 保留为第一类 fused op；后续 standalone primitive 通过独立 `SFUPrimitiveDesc` 和新 RoCC func7 加入，优先支持 `exp`、`log`、`reciprocal`、`rsqrt`、`tanh`、`sigmoid` |
| standalone primitive 不应复用 softmax descriptor | primitive descriptor 需要显式描述 op、dtype、elem_count、输入/输出 local GM 地址和 stride；跨 tile reduction 只在明确的 reduce primitive 或 fused op 中处理 |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 旧 cross-tile prototype 的 HBM counter/sum 更新非原子 | 正式设计改为 row-owner/reducer，不让多个 core 直接写同一个 reduction 值 |
| tile-local softmax 对大 N 不正确 | 第一版直接设计 full row-wise softmax |
| 传统三阶段 softmax 不是 online softmax | 设计更新为 tile stats + online merge + normalize |
| `scripts/build_and_install_local.sh --reconfigure --jobs 16` 在 rsync/chgrp 阶段失败 | 不是 SFU 编译错误；脚本在 `prepare_local_build.sh` 使用 `rsync -a` 保留 group 属性，当前文件系统对部分文件 chgrp 返回 `Invalid argument`。已在复制出的 build tree 中手动 `autogen/configure/make -C src/sst/elements/golem -j16` 验证 SFU 骨架编译通过 |
| build tree 不是源码目录 symlink | 修改 `src/sst/elements/golem/sfu/sfu.h/.cc` 后，必须同步到 `build/sst-elements/src/sst/elements/golem/sfu/` 再运行 golem 局部 make，才能真正编译新 SFU 源码 |
| Vanadis 逻辑 core id 与实际 RoCC/global-memory executor core id 不一致 | workload 区分 `requested_core_id` 与 `executor_core_id`：任务分配使用 requested core，local GM/RoCC 操作使用 executor core |
| wrapper 一度解包错布局 | SFU wrapper 显式导出 `GOLEM_GEMM_OUT_LAYOUT=colmajor_tile` 和 `GOLEM_MATMUL_*`，保证 checker 按 GEMM tile-packed C layout 解包 |
| GEMM baseline 曾只产生第 0 列正确输出 | 单 array baseline 每列只 DMA 一个 B 向量到 `rt.local_vec_in`，但 compute 曾读 `vec_col_addr(rt,n_col)`；修复为每列 compute 读取 `rt.local_vec_in`，GEMM-only golden 已通过 |

## Verification Log

| Date | Command / Case | Result |
|------|----------------|--------|
| 2026-06-30 | `python3 test_sfu_workload_scaffold.py`、`test_sfu_online_softmax_core.py`、`test_run_noc_dma_softmax_sfu_pipeline.py`、`test_verify_softmax_sfu_against_golden.py` | 全部通过 |
| 2026-06-30 | `GOLEM_SFU_SKIP_SOFTMAX=1 ... --gemm-m 64 --gemm-n 64 --gemm-k 64` 后解包 C 并运行 `verify_c_against_golden.py` | PASS，`max_abs_diff=0` |
| 2026-06-30 | `run_noc_dma_softmax_sfu_pipeline.sh ... --gemm-m 64 --gemm-n 64 --gemm-k 64 --verify-softmax` | PASS，`checked=4096`，`mismatches=0`，`max_abs_diff=1.09131064e-08` |
| 2026-06-30 | `run_noc_dma_softmax_sfu_pipeline.sh ... --gemm-m 128 --gemm-n 128 --gemm-k 64 --verify-softmax` | PASS，`checked=16384`，`mismatches=0`，`max_abs_diff=6.38621372e-09` |
| 2026-06-30 | `run_noc_dma_softmax_sfu_pipeline.sh ... --gemm-m 256 --gemm-n 256 --gemm-k 64 --verify-softmax` | PASS，`checked=65536`，`mismatches=0`，`max_abs_diff=3.19310681e-09` |
| 2026-06-30 | 不设置 `GOLEM_SFU_ENABLE`，运行旧 GEMM pipeline `64x64 fp32 --verify-c` | PASS，`sampled=64`，`mismatches=0`，`max_abs_diff=0` |

## Run Notes

- 旧 GEMM smoke test 使用 `run_noc_dma_pipeline.sh` 时，当前 shell 需要显式提供
  RISC-V toolchain PATH，否则 `riscv64-linux-musl-g++` 可能找不到。
- 当前验证应优先加载 `build/sst-elements/src/sst/elements/golem/.libs/libgolem.so`。
  若误用 `install/lib/sst-elements-library/libgolem.so`，可能混入旧组件实现。
- 无 ctrl-link 的默认 smoke 路径使用
  `small/mvm_noc_softmax_cpu/ncores_selfcom_dma_softmax_archive.py`，并显式关闭
  `GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE`，与本轮 SFU/GEMM 数值验证环境保持一致。

## Cleanup Notes

- 运行产物不进入源码管理：
  - `src/sst/elements/golem/tests/artifacts/`
  - `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/riscv64/`
  - `__pycache__/`
  - 临时 `data/` 和 `tests/data/*.bin`
- `mvm_noc_softmax_sfu/bin/sst` 不是普通生成物，而是 pipeline wrapper 通过 PATH
  使用的本地 SST shim；它需要保留并进入源码管理。
- `mvm_noc_softmax_sfu/Makefile` 是 workload 构建入口，已在 `.gitignore` 中显式取消忽略。
- 文档入口是 `README.md`；主计划看 `implementation_plan.md`，问题和验证看
  `findings.md`，过程历史只保留在压缩后的 `progress.md`。

## Future Primitive Plan

- 保留当前 fused softmax 路径，作为性能和功能基线。
- 新增 standalone primitive ABI：
  - descriptor 独立于 `SFUSoftmaxTileDesc`。
  - fp32 local-GM 输入输出先行。
  - op 枚举优先覆盖 `EXP`、`LOG`、`RECIPROCAL`，再扩展 `RSQRT`、`TANH`、`SIGMOID`。
- reduction primitive 后置：
  - 先支持 local buffer 的 `REDUCE_MAX`、`REDUCE_SUM`。
  - 后续再评估是否复用 fused softmax 的 row reducer state。
- 验证顺序：
  - ABI/static tests。
  - host-side primitive golden tests。
  - RISC-V primitive small workload。
  - primitive pipeline softmax 与 fused softmax 对照。

## Resources

- 设计文档：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/design.md`
- 旧 CPU fallback 目录：`src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/`
- 旧 cross-tile prototype：`mvm_noc_softmax_cpu/golem_softmax_cross_tile.cpp`
- GEMM DMA 编译边界：`src/sst/elements/golem/tests/doc/compile_boundaries.md`
- Request scheduler：`src/sst/elements/golem/requestscheduler/`
- Global memory DMA：`src/sst/elements/golem/globalmemory/`

## Visual/Browser Findings

- 未使用浏览器或图像工具；当前发现均来自本地代码和文档阅读。
