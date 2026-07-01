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
| A1 采用 GEMM local accumulator 直连 SFU | `GlobalMemory::rd_from_globalmem` 只能从本 core local GM 地址读，不能直接把 HBM `task.c_base_mm` 当作 SFU local input；因此 interleaved 实验复用 `rt.local_accum`，避免 softmax 阶段重新从 HBM 读 C tile |
| `GOLEM_SFU_INTERLEAVE_GEMM=1` 只是诊断开关，不是最终优化 | 它保留 GEMM 将 C 写回 HBM 的行为，只跳过 SFU 的 C tile HBM reload；用于量化 C reload 占比 |
| no-ctrl 是调试基线，不是最终性能标准 | no-ctrl 指 `GROUP_MANAGER=0`、`CTRL_LINK=0`、`WCP=0` 和 archive selfcom DMA 架构；它用于隔离 SFU 数值/descriptor/online reducer 问题，但会放大 worker-side DMA 压力 |
| standalone softmax 必然需要 HBM read | 若输入 logits 已经在 HBM，独立 softmax 的最低必要流量是读 logits、写 output；优化目标不是消除 HBM read，而是让 HBM->SFU 读写稳定、连续、低 retry |
| fused GEMM+softmax 的优化目标不同 | fused 路径才应该优先避免中间 C 先写 HBM 再被 SFU 读回；这与 standalone softmax 的必要 HBM read 不是同一个问题 |
| standalone softmax-only benchmark 复用 GEMM C tile HBM 区 | HBM init 将 logits 预置到 `OFF_GEMM_OUT_BASE` 对应的 C tile-packed 区域，guest 通过 `golemRunStandaloneSoftmaxSfuForCore` 只执行 C/logits tile read、SFU softmax、C tile writeback，不跑 GEMM |
| standalone golden 使用 logits 文件 | wrapper 在 `GOLEM_SFU_STANDALONE_SOFTMAX=1` 时默认 `GOLEM_SOFTMAX_VERIFY_REFERENCE=logits`，checker 用 `--logits-file` 计算 full row-wise softmax golden |
| standalone `512x512` 暂不暴露 HBM->SFU retry 瓶颈 | `512x512` standalone 只产生 64 次 read 和 64 次 write，总读写各 1 MiB，`timeout_retry=0`、`timeout_exhausted=0`；优先级应转向 primitive ABI，而不是先做 SFU 专用 HBM streaming |

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
| `GOLEM_SFU_INTERLEAVE_GEMM` 最初没有进入 Vanadis guest 环境 | no-ctrl archive shim 的 `process_env_keys` 只包含旧 SFU env；已补 `GOLEM_SFU_INTERLEAVE_GEMM`，`stdout-100` 可见 `mode=sfu-interleaved-local-accum` |
| 512 interleaved run 被窗口关闭中断 wrapper 收尾 | SST 主仿真已完成并写出 HBM output/DMA stats，但 stdout 目录为空、`run_summary.csv` 未追加；通过保存的 HBM output 执行 `unpack_c_from_hbm.py` 和 golden checker 补做 correctness 验证 |
| standalone SST 初次运行遇到 `libpython3.13.so.1.0` 加载失败 | `bin/sst` shim 当前不是 executable，shell 走到真实 SST；wrapper 已显式导出 `LD_LIBRARY_PATH=$SST_SOFTMAX_LD_LIBRARY_PATH`，即使不走 shim 也能加载 conda Python/SST/golem libraries |
| `--softmax-logits-file data/...` 相对路径在 HBM init 与 verifier 中解析不一致 | wrapper 现在用 `normalize_path_under_script_dir` 把相对 logits 路径归一化到 `mvm_noc_softmax_sfu/` 目录下，保证 HBM init 和 verifier 使用同一文件 |

## Verification Log

| Date | Command / Case | Result |
|------|----------------|--------|
| 2026-06-30 | `python3 test_sfu_workload_scaffold.py`、`test_sfu_online_softmax_core.py`、`test_run_noc_dma_softmax_sfu_pipeline.py`、`test_verify_softmax_sfu_against_golden.py` | 全部通过 |
| 2026-06-30 | `GOLEM_SFU_SKIP_SOFTMAX=1 ... --gemm-m 64 --gemm-n 64 --gemm-k 64` 后解包 C 并运行 `verify_c_against_golden.py` | PASS，`max_abs_diff=0` |
| 2026-06-30 | `run_noc_dma_softmax_sfu_pipeline.sh ... --gemm-m 64 --gemm-n 64 --gemm-k 64 --verify-softmax` | PASS，`checked=4096`，`mismatches=0`，`max_abs_diff=1.09131064e-08` |
| 2026-06-30 | `run_noc_dma_softmax_sfu_pipeline.sh ... --gemm-m 128 --gemm-n 128 --gemm-k 64 --verify-softmax` | PASS，`checked=16384`，`mismatches=0`，`max_abs_diff=6.38621372e-09` |
| 2026-06-30 | `run_noc_dma_softmax_sfu_pipeline.sh ... --gemm-m 256 --gemm-n 256 --gemm-k 64 --verify-softmax` | PASS，`checked=65536`，`mismatches=0`，`max_abs_diff=3.19310681e-09` |
| 2026-06-30 | 不设置 `GOLEM_SFU_ENABLE`，运行旧 GEMM pipeline `64x64 fp32 --verify-c` | PASS，`sampled=64`，`mismatches=0`，`max_abs_diff=0` |
| 2026-07-01 | `512x512` 默认 ctrl-link/WCP 配置，补 `LD_LIBRARY_PATH` 后运行 | 未完成；30 分钟超时，SST log 停在主仿真阶段，无 verifier 结果 |
| 2026-07-01 | `512x512` no-ctrl 基线：`GROUP_MANAGER=0 CTRL_LINK=0 WCP=0` | PASS，`checked=262144`，`mismatches=0`，`max_abs_diff=1.43280396e-09`，SST wall time `788s`，simulated time `3.34121 ms` |
| 2026-07-01 | `1024x1024` no-ctrl 基线，`GOLEM_SFU_MAX_INFLIGHT=32` | SST 完成但 verifier FAIL，`checked=1048576`，`mismatches=62044`，`max_abs_diff=238`，SST wall time `5314s`，simulated time `23.888 ms` |
| 2026-07-01 | 原始 GEMM pipeline：`1024x1024x64 fp32 --verify-c`，默认 ctrl-link/WCP，`GOLEM_SFU_ENABLE=0` | PASS，`sampled=1024`，`mismatches=0`，`max_abs_diff=0`，SST wall time `85s`，simulated time `233.434 us` |
| 2026-07-01 | 单元测试：`python3 -m unittest test_sfu_workload_scaffold.py test_run_noc_dma_softmax_sfu_pipeline.py` | PASS，17 tests |
| 2026-07-01 | 全量 small-test Python tests：`python3 -m unittest discover -s . -p 'test_*.py'` | PASS，43 tests |
| 2026-07-01 | SFU workload rebuild：`make clean ARCH=riscv64 && make ARCH=riscv64 CFLAGS=...` | PASS，生成 `riscv64/test_noc_dma_softmax_sfu` |
| 2026-07-01 | `128x128` interleaved：`GOLEM_SFU_INTERLEAVE_GEMM=1 ... --verify-softmax` | PASS，`mode=sfu-interleaved-local-accum`，SST wall time `77s`，simulated time `419.961 us`，`dma_read_issue_count_sum=512` |
| 2026-07-01 | `512x512` interleaved：窗口中断 wrapper，但 SST completed；离线 unpack + golden verify | PASS，`checked=262144`，`mismatches=0`，`max_abs_diff=1.43280396e-09`；DMA totals `read_issue=8192`、`write_issue=128`、`read_bytes=68157440`、`write_bytes=2097152`、`timeout_retry=774`、`timeout_exhausted=0`，simulated time `3.16903 ms` |
| 2026-07-01 | standalone 单元测试：`python3 -m unittest test_sfu_workload_scaffold.py test_run_noc_dma_softmax_sfu_pipeline.py test_verify_softmax_sfu_against_golden.py` | PASS，28 tests |
| 2026-07-01 | standalone HBM layout smoke：生成 `8x8` logits 到 HBM C tile 区，复制 init->out 后用 `unpack_c_from_hbm.py` 解包 | PASS，`unpacked_logits.bin` 与 `logits.bin` 最大差 `0.0` |
| 2026-07-01 | SFU workload clean rebuild after standalone changes | PASS，生成 `riscv64/test_noc_dma_softmax_sfu` |
| 2026-07-01 | standalone `128x128`：`GOLEM_SFU_STANDALONE_SOFTMAX=1 ... --verify-softmax` | PASS，stdout `mode=sfu-standalone-softmax`，`checked=16384`，`mismatches=0`，`max_abs_diff=9.29062844e-10`；DMA totals `read_issue=4`、`write_issue=4`、`read_bytes=65536`、`write_bytes=65536`、`timeout_retry=0`、`timeout_exhausted=0`，SST wall time `46s`，simulated time `204.533 us` |
| 2026-07-01 | wrapper regression：relative `--softmax-logits-file` path normalization | RED then GREEN；`test_wrapper_normalizes_relative_softmax_logits_file_to_script_dir` PASS |
| 2026-07-01 | standalone `512x512`：`GOLEM_SFU_STANDALONE_SOFTMAX=1 ... --gemm-m 512 --gemm-n 512 --gemm-k 64 --verify-softmax` | PASS，stdout `mode=sfu-standalone-softmax`，`checked=262144`，`mismatches=0`，`max_abs_diff=2.30018935e-10`；DMA totals `read_issue=64`、`write_issue=64`、`read_bytes=1048576`、`write_bytes=1048576`、`timeout_retry=0`、`timeout_exhausted=0`，SST wall time `57s`，simulated time `224.299 us` |
| 2026-07-01 | targeted wrapper/checker tests after path fix | PASS，17 tests |
| 2026-07-01 | standalone `1024x1024` 默认 `GOLEM_SFU_MAX_INFLIGHT=8` | 未完成；SST 阶段 30 分钟超时，日志停在 SST 执行阶段并出现 `DMA READ retry`。源码显示每 core 需 issue `16` 个 tile，但 SFU 默认 credit 只有 `8`，当前 runtime 又是先 issue 全部 tile 再 wait，形成 credit/inflight 阻塞 |
| 2026-07-01 | standalone `1024x1024` 对照：`GOLEM_SFU_MAX_INFLIGHT=32` | SST 完成，wall time `88s`，simulated time `327.308 us`；DMA totals `read_issue=256`、`write_issue=256`、`read_bytes=4194304`、`write_bytes=4194304`、`timeout_retry=6`、`timeout_exhausted=0`；verifier FAIL，`checked=1048576`、`mismatches=57344`、`max_abs_diff=1.87508551`，`write_completion=241/256` |
| 2026-07-01 | row-band issue/wait 调度修复：新增静态 TDD 测试并改 `golem_softmax_sfu_runtime.cpp` | RED then GREEN；目标测试组 `python3 -m unittest test_sfu_workload_scaffold.py test_run_noc_dma_softmax_sfu_pipeline.py test_verify_softmax_sfu_against_golden.py` PASS，`30 tests`；`make clean ARCH=riscv64 && make ARCH=riscv64` PASS |
| 2026-07-01 | standalone `512x512` row-band 回归，默认 `GOLEM_SFU_MAX_INFLIGHT=8` | PASS，`checked=262144`、`mismatches=0`、`max_abs_diff=2.30018935e-10`；wall time `71s`，simulated time `270.521 us`；DMA totals `read_issue=64`、`write_issue=64`、`read_bytes=1048576`、`write_bytes=1048576`、`timeout_retry=0`、`timeout_exhausted=0`、`write_completion=64/64` |
| 2026-07-01 | standalone `1024x1024` row-band 回归，默认 `GOLEM_SFU_MAX_INFLIGHT=8` | PASS，`checked=1048576`、`mismatches=0`、`max_abs_diff=1.15859469e-10`；wall time `159s`，simulated time `525.847 us`；DMA totals `read_issue=256`、`write_issue=256`、`read_bytes=4194304`、`write_bytes=4194304`、`timeout_retry=4`、`timeout_exhausted=0`、`write_completion=256/256` |

## 2026-07-01 Scale Findings

- no-ctrl 基线用于隔离 SFU fused softmax 数值路径：关闭 group manager、ctrl-link
  和 worker command processor，使用 16 个 worker core 和 archive 架构脚本，避免把
  控制链路/调度器集成问题混入 SFU 数值正确性判断。
- `1024x1024` 相比 `512x512`：
  - C tile 从 `64` 增至 `256`，每 core tile 从 `4` 增至 `16`。
  - 每 core DMA read issue 从 `516` 增至 `2064`，write issue 从 `8` 增至 `32`。
  - 总读字节从 `69.2 MB` 增至 `276.8 MB`，总写字节从 `2.1 MB` 增至 `8.4 MB`。
  - DRAMSim3 backend read count 从 `1,348,906` 增至 `11,491,453`，超过简单 4x
    数据规模增长。
  - DMA timeout retry 从 `1,298` 增至 `32,433`，并出现
    `timeout_exhausted=3,969`；这是 1024 运行时间和数值失败的首要诊断线索。
- 原始 GEMM `1024x1024x64` 默认 ctrl-link/WCP 路径与 SFU no-ctrl 路径不是同一条
  数据通路：
  - 原始 GEMM 使用 `GOLEM_GROUP_MANAGER_ENABLE=1`、`GOLEM_CTRL_LINK_ENABLE=1`、
    `GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=1`，`20` total/GEMM cores，其中 4 个
    是 manager 相关 core。
  - 原始 GEMM run summary 中 `dma_read_issue_count_sum=0`、
    `dma_write_issue_count_sum=256`、`dma_timeout_retry_sum=0`。
  - SFU no-ctrl `1024x1024` 需要在 softmax 阶段通过 worker-side DMA 读取 C tile
    并写回，统计为 `dma_read_issue_count_sum=33024`、
    `dma_write_issue_count_sum=512`、`dma_timeout_retry_sum=32433`。
  - 因此本轮超长运行不是 GEMM 计算本身导致，而是 SFU no-ctrl 包装路径的大量
    worker-side DMA，以及由此产生的 retry/exhausted。

## 2026-07-01 Phase A Bottleneck Exploration

- A1 实验目标：让 GEMM 与 SFU 交错执行，SFU 直接消费 GEMM 的 `rt.local_accum`，
  跳过默认 SFU 路径中的 C tile HBM reload。
- 128 对照：
  - 默认 SFU path：`dma_read_issue_count_sum=516`，`read_bytes_total_sum=4325376`。
  - interleaved local-accum：`dma_read_issue_count_sum=512`，`read_bytes_total_sum=4259840`。
  - 差值正好是 4 个 C tile reload，即 `4 * 64 * 64 * 4 = 65536 B`。
- 512 对照：
  - no-ctrl baseline：`read_issue=8256`，`read_bytes=69206016`，
    `write_issue=128`，`write_bytes=2097152`，SST simulated time `3.34121 ms`。
  - interleaved local-accum：`read_issue=8192`，`read_bytes=68157440`，
    `write_issue=128`，`write_bytes=2097152`，SST simulated time `3.16903 ms`。
  - 差值正好是 64 个 C tile reload，即 `64 * 64 * 64 * 4 = 1048576 B`。
- 结论：之前“softmax 阶段额外 C tile DMA read/write 是主瓶颈”的判断过粗。
  C tile reload 确实存在且可被 A1 消掉，但在 512 中只占 read issue 的 `64/8256`
  和读字节的约 `1.5%`。主要瓶颈来自 no-ctrl GEMM A/B worker-side DMA：
  `8192` 次 read issue 和约 `65 MiB` 读流量仍然保留。
- 方案 B 应优先考虑：
  - 接回或复用原始 GEMM ctrl-link/WCP 数据通路，降低 no-ctrl A/B DMA 压力。
  - 将 GEMM C 的 final write 与 SFU consume 融合，避免 C 先写 HBM、再被后续阶段使用。
  - 对 `1024x1024` 的 correctness 风险，优先处理 DMA timeout/exhausted，而不是只增加
    SFU inflight 或只消除 C reload。

## 2026-07-01 Path Clarification

- `group manager / ctrl-link / WCP` 是 GEMM 加速路径中的控制面/调度面，不是 SFU
  数学单元本身：
  - group manager：为一组 worker 分配 tile/task，管理 worker slot 和执行顺序。
  - ctrl-link：在 manager、scheduler、worker、RoCC 子组件之间传递控制消息。
  - WCP：worker 本地命令处理器，负责推进 descriptor、DMA prefetch、array compute、
    reuse/window 和 writeback。
- 早期 SFU 测试关闭它们，是为了减少变量，先验证 SFU descriptor、RoCC func7、
  local GM 读写、online reducer 和 golden checker 是否正确。
- 这个选择只适合作为 correctness/debug baseline。性能评估时，no-ctrl 路径里的
  worker-side DMA 不能代表最终架构：
  - 原始 GEMM `1024x1024x64` ctrl-link/WCP 路径：`85s` wall time，
    `dma_read_issue_count_sum=0`，`dma_timeout_retry_sum=0`。
  - SFU no-ctrl `1024x1024` 混合路径：`5314s` wall time，
    `dma_read_issue_count_sum=33024`，`dma_timeout_retry_sum=32433`，
    `timeout_exhausted=3969`。
- 后续必须区分两条路线：
  1. standalone softmax：输入 tensor 已在 HBM，必须读 HBM。优化目标是构建
     softmax-only benchmark 和专用 HBM->SFU 数据路径，衡量必要 HBM read/write 的效率。
  2. fused GEMM+softmax：softmax 紧跟 GEMM 时，优化目标是 local-accum handoff，
     避免中间 C 写回 HBM 再读回。

## 2026-07-01 Standalone Softmax Bring-up

- standalone 模式定义：
  - `GOLEM_SFU_STANDALONE_SOFTMAX=1`。
  - HBM init 在仿真前把 logits 写进 GEMM C tile 区。
  - guest workload 在 GEMM 之前截获执行，不调用 `run_gemm_for_core`。
  - SFU runtime 仍复用当前 tile 地址/online reducer 路径：每个 C tile 一次 HBM read，
    一次 SFU issue/wait，一次 HBM writeback。
- `128x128` standalone 的 DMA 统计与理论值吻合：
  - tile 数：`2x2=4`。
  - tile bytes：`64*64*4 = 16384 B`。
  - 总 read/write：`4*16384 = 65536 B`。
  - 实测 `read_issue=4`、`write_issue=4`、`read_bytes=65536`、`write_bytes=65536`。
- 这说明 standalone softmax 路径已经隔离掉 mixed GEMM+softmax 中的 A/B DMA。
  后续 `512x512` standalone 如果仍出现高 retry，才说明 SFU 必要 HBM->SFU 数据路径
  需要进一步优化；如果 retry 低，则 mixed 路径的主要问题继续锁定为 GEMM A/B no-ctrl DMA。
- `512x512` standalone 已完成：
  - 配置中的 `K=64` 只是复用 GEMM/HBM layout 生成器所需的占位维度；standalone
    softmax 的实际输入是 `512x512` logits。
  - tile 数：`8x8=64`。
  - tile bytes：`64*64*4 = 16384 B`。
  - 总 read/write：`64*16384 = 1048576 B`。
  - 实测 `read_issue=64`、`write_issue=64`、`read_bytes=1048576`、
    `write_bytes=1048576`、`timeout_retry=0`、`timeout_exhausted=0`。
- 结论更新：standalone SFU 的必要 HBM read/write 数据路在 512 规模不是当前瓶颈。
  mixed `GEMM+softmax` no-ctrl 的性能问题主要仍来自 GEMM A/B worker-side DMA 和
  retry 压力；standalone 方向可以优先推进 primitive ABI。

## 2026-07-01 Standalone 1024 Bottleneck Recheck

- 这次重新跑 `1024x1024` standalone softmax-only，配置中 `K=64` 仍只是复用
  GEMM/HBM layout 的占位维度；softmax 输入实际是 `1024x1024` logits。
- 默认 `GOLEM_SFU_MAX_INFLIGHT=8` 时，SST 运行 30 分钟未完成：
  - 该规模 tile 数为 `16x16=256`，每 core 分到 `16` 个 tile。
  - 当前 `golemRunSoftmaxSfuForCore` 对本 core 的 tile 先全部
    `dma_remote_load_to_gm -> sfu_softmax_tile`，之后才统一 `sfu_wait -> remote_store`。
  - SFU 组件在 `inflight_ >= maxInflight_` 时拒绝第 9 个 issue；RoCC 会把该命令
    push 回队头重试，但 guest 代码还没有进入 wait，因此没有机会释放 SFU credit。
  - 这解释了为什么 `512x512` 可完成：`512` 时每 core 只有 `4` 个 tile，低于默认
    inflight `8`。
- 将 `GOLEM_SFU_MAX_INFLIGHT=32` 后，SST 阶段恢复到合理时间：
  - wall time `88s`，simulated time `327.308 us`。
  - 理论必要 HBM 流量为 `256 * 64 * 64 * 4 = 4194304 B` read 和同等 write；
    实测 `read_bytes=4194304`、`write_bytes=4194304`，说明 standalone 路径没有
    混入 GEMM A/B DMA。
  - `timeout_retry=6`、`timeout_exhausted=0`，相比 mixed no-ctrl `1024x1024` 的
    `timeout_retry=32433`、`timeout_exhausted=3969`，必要 HBM read/write 不是
    本轮半小时级仿真时间的根因。
- 但 `GOLEM_SFU_MAX_INFLIGHT=32` 只是绕过 credit 阈值，不是完整修复：
  - verifier FAIL，`mismatches=57344`，失败样例显示部分输出仍是原始 logits。
  - DMA summary 中 `write_issue=256` 但 `write_completion=241`，说明 guest 退出时
    仍可能有 HBM writeback 未完成。
- 已实现 row-band issue/wait 后，默认 `GOLEM_SFU_MAX_INFLIGHT=8` 不再需要靠
  放大 inflight 绕过问题：
  - runtime 按完整 `m_tile` row-band 提交 tile；每批提交后立即 wait/store，再进入
    下一批。
  - `1024x1024`、`block_n=64` 时 `n_tiles=16`，row-band 为 `1` 个完整 M tile；
    每个 row 的所有 N partial 仍可由 16 个 worker 并行提交，符合 online reducer
    normalize 前必须看到完整 row partial 的约束。
  - `1024x1024` standalone 回归 PASS，wall time `159s`，simulated time
    `525.847 us`，`write_completion=256/256`。
- 这次修复把“半小时级仿真卡住”的根因收敛为 SFU issue/wait 调度，而不是
  standalone 必要 HBM read/write 流量。与 `GOLEM_SFU_MAX_INFLIGHT=32` 对照相比，
  row-band simulated time 更长，但正确性完整，且不依赖调大 SFU credit。
- `remote_store` 当前仍没有独立 ack/fence 指令；本轮 512/1024 已观察到
  `write_completion=write_issue`。如果后续更大规模或不同 NoC 配置再次出现
  末尾写回未完成，再补显式写回 completion token。

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
