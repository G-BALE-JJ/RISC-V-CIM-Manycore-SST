# Findings & Decisions

## Requirements

- 新增 SFU 硬件建模路径，替代当前单核 CPU softmax fallback。
- SFU 是独立 SST 子组件，像 `array`、`global_memory` 一样挂在 RoCC 下。
- 组件名使用 `SFU`，不要在组件名中绑定 `softmax`。
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

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 旧 cross-tile prototype 的 HBM counter/sum 更新非原子 | 正式设计改为 row-owner/reducer，不让多个 core 直接写同一个 reduction 值 |
| tile-local softmax 对大 N 不正确 | 第一版直接设计 full row-wise softmax |
| 传统三阶段 softmax 不是 online softmax | 设计更新为 tile stats + online merge + normalize |
| `scripts/build_and_install_local.sh --reconfigure --jobs 16` 在 rsync/chgrp 阶段失败 | 不是 SFU 编译错误；脚本在 `prepare_local_build.sh` 使用 `rsync -a` 保留 group 属性，当前文件系统对部分文件 chgrp 返回 `Invalid argument`。已在复制出的 build tree 中手动 `autogen/configure/make -C src/sst/elements/golem -j16` 验证 SFU 骨架编译通过 |

## Resources

- 设计文档：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/design.md`
- 旧 CPU fallback 目录：`src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/`
- 旧 cross-tile prototype：`mvm_noc_softmax_cpu/golem_softmax_cross_tile.cpp`
- GEMM DMA 编译边界：`src/sst/elements/golem/tests/doc/compile_boundaries.md`
- Request scheduler：`src/sst/elements/golem/requestscheduler/`
- Global memory DMA：`src/sst/elements/golem/globalmemory/`

## Visual/Browser Findings

- 未使用浏览器或图像工具；当前发现均来自本地代码和文档阅读。
