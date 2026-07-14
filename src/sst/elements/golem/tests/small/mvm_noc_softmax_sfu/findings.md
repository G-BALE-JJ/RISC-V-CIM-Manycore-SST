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
| SFU primitive softmax 必须是多核协同路线 | 当前 `requested_core_id==0` 单核整行 primitive softmax 只是 deprecated 原型；正式架构必须让多个 worker core 分摊同一行的 column slice，并通过跨 core reduction 合并全局 `row_max` 和 `row_sum` |
| unified SFU job standalone 大维度输入应按 row-band streaming | `GOLEM_SFU_JOB_SOFTMAX` 不再把完整 `M*N` logits 一次性 staging 到 local GM；使用 `GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS` 分 band 从 HBM C tile-packed 区读取、执行 unified softmax job、再 patch 回 HBM |
| unified SFU job 的 band descriptor 和 runtime config 必须同形状 | 每个 band 的 `band_desc.outer` 和传给 `golemRunStandaloneSoftmaxSfuJobForCore` 的 `cfg.m` 必须都等于 `row_band_rows`；否则 `validate_sfu_softmax_request` 会在 job 执行前失败 |
| unified SFU job row-band 可以按 band 分摊到多个 requested cores | `GOLEM_SFU_JOB_SOFTMAX_BAND_CORES` 让多个 cores 以 `band_index % band_core_count` 分摊 row-band；这是真正脱离 GEMM 的 HBM logits read + unified job + HBM writeback cooperative path |
| cooperative row-band 需要完整 m-tile band 才能避免 patch race | 当多个 cores 并行写 HBM C tile-packed 区时，`staging_rows` 必须覆盖完整 m-tile；否则两个 band 可能 patch 同一个 C tile 的不同行，引入 read-modify-write race |
| 完整 row-band store 不应 reload HBM tile | 当 band 覆盖完整 C tile 行时，输出 tile 可直接由 row-band buffer pack 后写回；只有 partial band 需要 reload+patch。该 fast path 将 128x128 row-band DMA read issue 从 `8` 降到 `4` |
| staging-band 内 sub-job 可以切短单条 `SFU_JOB`，但不能改变 tile ownership | `GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS` 在 local GM row-major staging buffer 内按行拆多个 unified job；外层仍用完整 m-tile staging/store，避免 multi-core cooperative path 的 partial tile write race |
| sub-job runtime config 的 `block_m` 必须匹配 sub-job rows | 真实 SST 暴露 `m=16, block_m=64` 会被 `validate_sfu_softmax_request` 拒绝；修复为 `sub_cfg.block_m=sub_job_rows` 后 `128x128,BAND_CORES=2,JOB_ROWS=16` PASS |
| completed-run recovery 应作为大维度 correctness 收尾入口 | `GOLEM_SFU_RECOVER_COMPLETED_RUN=1` / `--recover-completed-run` 跳过 build 和 base SST pipeline，复用 wrapper 的 unpack + full-row verifier；适合外层 timeout 但 SST 已完成并留下 HBM dump 的 512+ run |
| unified job column-coop 必须可观测 | `SFU::executeSoftmaxRowJob` 的内部 max/sum/norm chunk pass 现在分别记录 `sfu_job_softmax_max_chunks`、`sfu_job_softmax_sum_chunks`、`sfu_job_softmax_norm_chunks`，后续真实 SST probe 可以确认是否走到 SFU-side chunk cooperative path |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 旧 cross-tile prototype 的 HBM counter/sum 更新非原子 | 正式设计改为 row-owner/reducer，不让多个 core 直接写同一个 reduction 值 |
| tile-local softmax 对大 N 不正确 | 第一版直接设计 full row-wise softmax |
| 传统三阶段 softmax 不是 online softmax | 设计更新为 tile stats + online merge + normalize |
| `scripts/build_and_install_local.sh --reconfigure --jobs 16` 在 rsync/chgrp 阶段失败 | 不是 SFU 编译错误；脚本在 `prepare_local_build.sh` 使用 `rsync -a` 保留 group 属性，当前文件系统对部分文件 chgrp 返回 `Invalid argument`。已在复制出的 build tree 中手动 `autogen/configure/make -C src/sst/elements/golem -j16` 验证 SFU 骨架编译通过 |
| build tree 不是源码目录 symlink | 修改 `src/sst/elements/golem/sfu/sfu.h/.cc` 后，必须同步到 `build/sst-elements/src/sst/elements/golem/sfu/` 再运行 golem 局部 make，才能真正编译新 SFU 源码 |
| 新增 SFU 虚函数后只重编 `sfu.lo` 会导致 SST wire-up 崩溃 | `roccAnalog.h`/`sfu.h` 被 `golem.cc` 包含；新增 `issuePrimitiveBatch` 后旧 `golem.o` 仍按旧 vtable 调用，曾在 `SFU::wait` 崩溃。同步 `sfu.h/.cc`、`roccAnalog.h` 到 build tree 后，必须重编 `golem.lo` 和 `sfu.lo` 并重新链接 `libgolem.so` |
| Vanadis 逻辑 core id 与实际 RoCC/global-memory executor core id 不一致 | workload 区分 `requested_core_id` 与 `executor_core_id`：任务分配使用 requested core，local GM/RoCC 操作使用 executor core |
| wrapper 一度解包错布局 | SFU wrapper 显式导出 `GOLEM_GEMM_OUT_LAYOUT=colmajor_tile` 和 `GOLEM_MATMUL_*`，保证 checker 按 GEMM tile-packed C layout 解包 |
| GEMM baseline 曾只产生第 0 列正确输出 | 单 array baseline 每列只 DMA 一个 B 向量到 `rt.local_vec_in`，但 compute 曾读 `vec_col_addr(rt,n_col)`；修复为每列 compute 读取 `rt.local_vec_in`，GEMM-only golden 已通过 |
| `GOLEM_SFU_INTERLEAVE_GEMM` 最初没有进入 Vanadis guest 环境 | no-ctrl archive shim 的 `process_env_keys` 只包含旧 SFU env；已补 `GOLEM_SFU_INTERLEAVE_GEMM`，`stdout-100` 可见 `mode=sfu-interleaved-local-accum` |
| 512 interleaved run 被窗口关闭中断 wrapper 收尾 | SST 主仿真已完成并写出 HBM output/DMA stats，但 stdout 目录为空、`run_summary.csv` 未追加；通过保存的 HBM output 执行 `unpack_c_from_hbm.py` 和 golden checker 补做 correctness 验证 |
| standalone SST 初次运行遇到 `libpython3.13.so.1.0` 加载失败 | `bin/sst` shim 当前不是 executable，shell 走到真实 SST；wrapper 已显式导出 `LD_LIBRARY_PATH=$SST_SOFTMAX_LD_LIBRARY_PATH`，即使不走 shim 也能加载 conda Python/SST/golem libraries |
| `--softmax-logits-file data/...` 相对路径在 HBM init 与 verifier 中解析不一致 | wrapper 现在用 `normalize_path_under_script_dir` 把相对 logits 路径归一化到 `mvm_noc_softmax_sfu/` 目录下，保证 HBM init 和 verifier 使用同一文件 |
| unified job row-band 初版 512 形状立即失败 | 根因是 `band_desc.outer=64` 但仍传全局 `cfg.m=512`，导致 unified job 请求校验失败；修复为派生 `band_cfg` 并设置 `band_cfg.m=row_band_rows` |
| unified job row-band `512x512,staging_rows=64` 修复后仍超过 900s wall | 该 run 不再出现 shape mismatch 失败，但 executor core 未输出 PASS，stats/stdout 未收尾；下一步需要加 per-band progress/timeout instrumentation 区分是 SFU job wait、HBM patch-back，还是单 executor band loop 固定开销 |
| unified job row-band stdout 在 timeout 时不一定可见 | 512 timeout 中未退出的 pid 不落 stdout 文件；`GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS` 对已完成规模有用，但 512 timeout 还需要 emergency dump 或更底层 progress counter 才能在线观察 |
| `512x512,BAND_CORES=8` 仍超过 600s wall | log 中 pid `100-107` 未退出、`108-119` 已退出，证明 8 个 band cores 参与了；但每个 512-wide band 自身仍很重，下一步应继续降低 per-band HBM/SFU job 固定开销或外显 column chunk 协作 |
| `512x512,BAND_CORES=8,JOB_ROWS=16` 可离线验证 PASS，但 wrapper wall-time 收尾仍需处理 | run id `run_20260708_205112_3064723` 的外层命令 720s 超时；随后 log 显示 `all process have exited`、simulated `2.00541 ms`，离线 unpack + logits verifier PASS。问题转为 wrapper timeout/recovery 与长 run stdout 落盘，而不是 correctness |
| recovery 从 repo 根目录启动时可能找错 standalone logits | `configs/50_tensor_verify.env` 使用 `${PWD}/data` 作为默认 `GOLEM_TENSOR_DIR`；从 repo 根目录调用 wrapper 会默认找 `/data4/jjgong/RISC-V-CIM-Manycore-SST/data/softmax_logits_512x512.bin`。恢复 standalone SFU run 时显式设置 `GOLEM_TENSOR_DIR=.../small/mvm_noc_softmax_sfu/data` 或 `--softmax-logits-file` |

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
| 2026-07-04 | ALL-op batch limited validation: `1024` elems/op, `chunk_elems=256`, `GOLEM_SFU_PRIMITIVE_HBM_BATCH=1` | PASS，run id `run_20260704_150513_1404172`；`chunks=24`、`processed_elems=6144`、HBM read/write `24576/24576` bytes、DMA read/write `24/24`、wait `48`、`sfu_ops_issued=6`、`sfu_primitive_elems=6144`、timeout retry/exhausted `0/0`、simulated time `2.7661 ms`、wall `583s` |
| 2026-07-04 | ALL-op batch limited validation: `4096` elems/op, `chunk_elems=4096`, `GOLEM_SFU_PRIMITIVE_HBM_BATCH=1` | PASS，run id `run_20260704_151536_1475667`；`chunks=6`、`processed_elems=24576`、HBM read/write `98304/98304` bytes、DMA read/write `6/6`、wait `12`、`sfu_ops_issued=6`、`sfu_primitive_elems=24576`、timeout retry/exhausted `0/0`、simulated time `1.05165 ms`、wall `218s` |
| 2026-07-07 | multi-core primitive softmax smoke：`dim=512, worker_cores=4, chunk=64, rows=1, verify=1` | PASS，真实 SST；`worker_cores=4`、`dim_per_core=128`、`chunks=8`、`batches=16`、`cross_core_reduce_stages=2`、`max_abs_diff=1.90714e-10`、`max_rel_diff=6.92901e-08`、`max_row_sum_error=4.19792e-09` |
| 2026-07-07 | multi-core primitive softmax sweep：`dim=512`，`chunk=64/128`，`worker_cores=4/8/16`，perf profile，真实 SST | PASS `6/6`，timeout `0`；输出目录 `artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_multicore_sweep_d512_c64_128_w4_8_16_v4/figures/`；simulated time 随 worker 增加约 `348 us -> 323 us -> 311 us`，wall time 每点约 `83-89s` |
| 2026-07-07 | multi-core primitive softmax probe：`dim=1024/2048/4096`，`chunk=256`，`worker_cores=16`，perf profile，真实 SST | PASS `3/3`，timeout `0`；`4096` 已跑通，`dim_per_core=256`、`chunks=16`、`batches=64`、simulated `415.888 us`、wall `123s`；输出目录 `artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_multicore_probe_d1024_2048_4096_c256_w16/figures/` |
| 2026-07-07 | multi-core primitive softmax correctness 点：`dim=512`、`chunk=256`、`worker_cores=16`、`verify=1`，真实 SST | PASS；`dim_per_core=32`、`chunks=16`、`batches=64`、`max_abs_diff=1.90714e-10`、`max_rel_diff=6.92901e-08`、`max_row_sum_error=6.14326e-10`、simulated `330.679 us`、wall `88s`；输出目录 `artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_multicore_verify_d512_c256_w16/` |
| 2026-07-07 | 4096 focused DSE：`dim=4096`，worker sweep `c256,w=4/8/16`，chunk sweep `w16,c=128/256/512`，perf profile，真实 SST | PASS `5/5`，timeout `0`；worker 增加使 simulated time `731.988 -> 518.271 -> 415.888 us`；`w16` 下 chunk sweep 为 `c128=412.406 us`、`c256=415.888 us`、`c512=415.888 us`；输出目录 `artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_multicore_dse_d4096_worker_chunk_v1/figures/`，新增 `softmax_primitive_dse.svg` |
| 2026-07-08 | multi-row primitive softmax 代表点：`chunk=256`、`worker_cores=16`、perf profile，真实 SST | PASS `4/5`，timeout `1`；`rows=4,dim=1024/2048/4096` simulated `542.939/680.226/919.973 us`，wall `163/203/270s`；`rows=8,dim=2048` simulated `1.01583 ms`，wall `310s`；`rows=16,dim=1024` 在 `360s` timeout，emergency simulated time `1.04688 ms`；输出目录 `artifacts/sweeps/sfu_softmax_primitive_sweep_20260708_multicore_multirow_v1/figures/`，新增 `softmax_primitive_multirow.svg` |
| 2026-07-08 | `rows=16,dim=1024` timeout 诊断：解析 emergency log 中可见 per-core DMA 统计 | 可见未完成 core median read/write issue 为 `28/29`，估算推进到约 `14/16` 行、完成约 `13/16` 行输出；可见 core 的 DMA timeout retry/exhausted 为 `0/0`，说明不是 HBM retry 卡死，而是 per-row 控制/同步成本导致 wall time 超限；新增 `softmax_primitive_timeout_diagnosis.md` |
| 2026-07-08 | coordinator-only reciprocal correctness：`rows=1,dim=512,chunk=256,worker_cores=16,verify=1`，真实 SST | PASS；`batches=49`，`max_abs_diff=1.94327e-10`，`max_rel_diff=6.61088e-08`，`max_row_sum_error=7.84031e-10`，simulated `329.337 us`，wall `84s`；证明 coordinator 广播 `inv_sum` 数学正确 |
| 2026-07-08 | coordinator-only reciprocal rows=16 probe：`rows=16,dim=1024,chunk=256,worker_cores=16,perf_profile=1`，真实 SST | `360s` TIMEOUT；emergency simulated time `1.08284 ms`；可见未完成 core median read/write issue `27/29`，DMA retry/exhausted `0/0`；说明去掉重复 scalar reciprocal 还不足以解决 rows=16 wall-time 瓶颈 |
| 2026-07-08 | row-block softmax correctness：`rows=4,dim=512,chunk=256,worker_cores=16,row_block=4,verify=1`，真实 SST | PASS；`row_block=4,row_blocks=1,block_syncs=2`，`batches=196`，`max_abs_diff=1.94327e-10`，`max_row_sum_error=7.84031e-10`，simulated `513.946 us`，wall `164s`；证明 block 内多行 max/sum/inv_sum 数学正确 |
| 2026-07-08 | row-block rows=16 probe：`rows=16,dim=1024,chunk=256,worker_cores=16,row_block=4,perf_profile=1`，真实 SST | `360s` TIMEOUT；emergency simulated time `1.03686 ms`；可见未完成 core median read/write issue `24/26`，估算推进到约 `12/16` 行、完成约 `10/16` 行输出；DMA retry/exhausted `0/0`；简单 row-block 化没有改善 rows=16 wall-time |
| 2026-07-08 | unified SFU job row-band 静态回归：`test_sfu_workload_scaffold.py -v`、`test_run_noc_dma_softmax_sfu_pipeline.py -v` | PASS，分别为 `41 tests OK` 和 `23 tests OK`；覆盖 `GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS`、band-local `band_cfg.m=row_band_rows`、wrapper/architecture env 传递 |
| 2026-07-08 | unified SFU job row-band build：`make clean ARCH=riscv64`、`make ARCH=riscv64` | PASS，生成 `riscv64/test_noc_dma_softmax_sfu` |
| 2026-07-08 | standalone unified job row-band `128x128`：`GOLEM_SFU_STANDALONE_SOFTMAX=1 GOLEM_SFU_JOB_SOFTMAX=1 GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS=64 ... --verify-softmax` | PASS，`reference=logits checked=16384 mismatches=0 max_abs_diff=1.64943281e-09`；stdout 确认 `dispatch=sfu-standalone-unified-job-softmax`、`worker_cores=16`、`staging_rows=64`；DMA totals `read_issue=8`、`write_issue=4`、`read_bytes=131072`、`write_bytes=65536`、`timeout_retry=0`、`timeout_exhausted=0` |
| 2026-07-08 | standalone unified job row-band `512x512,staging_rows=64` | `900s` TIMEOUT；不再出现修复前的 row-band shape mismatch 立刻失败，但 executor core 未输出 PASS，stats/stdout 未收尾 |
| 2026-07-08 | unified SFU job row-band cooperative 静态回归：`test_sfu_workload_scaffold.py -v`、`test_run_noc_dma_softmax_sfu_pipeline.py -v` | PASS，分别为 `42 tests OK` 和 `23 tests OK`；覆盖 `GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS`、`GOLEM_SFU_JOB_SOFTMAX_BAND_CORES`、band modulo 分发、full-tile store fast path、wrapper/architecture env 传递 |
| 2026-07-08 | unified SFU job row-band cooperative build：`make ARCH=riscv64` | PASS |
| 2026-07-08 | standalone unified job row-band cooperative `128x128,BAND_CORES=2`，真实 SST | PASS，`reference=logits checked=16384 mismatches=0 max_abs_diff=1.64943281e-09`；core0/core1 均输出 `band_cores=2 PASS`；DMA totals `read_issue=4`、`write_issue=4`、`read_bytes=65536`、`write_bytes=65536`、`timeout_retry=0`、`timeout_exhausted=0`；wall `156s`、simulated `728.187 us` |
| 2026-07-08 | standalone unified job row-band cooperative `512x512,BAND_CORES=8`，真实 SST | `600s` TIMEOUT；pid `100-107` 未退出、`108-119` 已退出，说明 8 个 active band cores 进入 cooperative path，但 512-wide band 仍未在窗口内收尾 |
| 2026-07-08 | unified job sub-job row streaming 静态回归：`test_sfu_workload_scaffold.py -v`、`test_run_noc_dma_softmax_sfu_pipeline.py -v` | PASS，分别为 `43 tests OK` 和 `23 tests OK`；覆盖 `GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS`、sub-job GM offset、`sub_cfg.block_m=sub_job_rows`、wrapper/architecture env 传递 |
| 2026-07-08 | standalone unified job row-band cooperative `128x128,BAND_CORES=2,JOB_ROWS=16`，真实 SST | PASS，`reference=logits checked=16384 mismatches=0 max_abs_diff=1.64943281e-09`；run id `run_20260708_204743_3059548`，wall `155s`、simulated `739.614 us` |
| 2026-07-08 | standalone unified job row-band cooperative `512x512,BAND_CORES=8,JOB_ROWS=16`，真实 SST + offline verify | 外层命令 `720s` timeout，但 SST log 随后显示 `all process have exited`、simulated `2.00541 ms`；离线 verifier PASS，`checked=262144 mismatches=0 max_abs_diff=3.90064624e-10` |
| 2026-07-08 | wrapper completed-run recovery：`--recover-completed-run --verify-softmax --softmax-reference logits` 复用 512 HBM dump | PASS，自动 unpack 到 `softmax_sfu_c_out_512_recovered.bin`，`checked=262144 mismatches=0 max_abs_diff=3.90064624e-10`；同时 `test_run_noc_dma_softmax_sfu_pipeline.py -v` 为 `24 tests OK`、`test_sfu_workload_scaffold.py -v` 为 `43 tests OK` |
| 2026-07-08 | unified job Phase 2B column-coop observability | RED/GREEN 新增 SFU-side chunk-pass stats；`test_sfu_primitive_core.py -v` 为 `16 tests OK`，`test_sfu_workload_scaffold.py -v` 为 `43 tests OK`，`test_run_noc_dma_softmax_sfu_pipeline.py -v` 为 `24 tests OK`；`make -C build/sst-elements/src/sst/elements/golem -j16` PASS，512 recovery PASS |
| 2026-07-08 | unified job Phase 2B stats probe：`64x64,chunk=16,workers=4,job_rows=16,band_cores=1`，真实 SST | PASS，run id `run_20260708_214124_3115525`，simulated `605.098 us`，`checked=4096 mismatches=0 max_abs_diff=3.77493008e-09`；executor core3 stats：`max/sum/norm_chunks=256/256/256`，匹配 `64 rows * 4 workers * 1 chunk` |
| 2026-07-08 | unified job Phase 2B stats probe：`512x512,chunk=256,workers=16,job_rows=16,band_cores=8`，真实 SST | PASS，run id `run_20260708_214416_3119600`，simulated `1.95484 ms`，`checked=262144 mismatches=0 max_abs_diff=3.90064624e-10`；executor core3..10 各 `max/sum/norm_chunks=1024/1024/1024`，全局每 pass `8192` chunks；DMA read/write issue `64/64`，timeout retry/exhausted `0/0` |

## 2026-07-01 Scale Findings

## 2026-07-07 Multi-core Primitive Softmax DSE

- `4096` 维 multi-core cooperative primitive softmax 已从单点 probe 扩展为
  focused DSE，并且 5 个真实 SST 点全部 PASS、无 timeout。
- 固定 `chunk=256` 时，`worker_cores=4/8/16` 分别对应
  `dim_per_core=1024/512/256`，simulated time 为
  `731.988/518.271/415.888 us`。这说明在当前模型里，增加参与 core 数可以显著
  缩短每 core 局部 softmax 工作量，是 4096 维性能最明显的调节旋钮。
- 固定 `worker_cores=16` 时，`chunk=128/256/512` 的 simulated time 为
  `412.406/415.888/415.888 us`。`chunk=128` 使 chunks 和 DMA issue 翻倍
  (`64/64` read/write issue)，但 simulated time 没有明显变差，说明当前
  batch-default 功能模型尚未对每个 chunk 的 SFU pipeline occupancy/issue latency
  做周期精确惩罚。
- 因此组会汇报中应把本轮结论表述为：多核协同路线已经跑通到 `4096`，worker 数
  对性能有清晰影响；chunk 对 event/issue 数有清晰影响，但对模拟硬件时间的影响
  需要后续 hardware-like SFU timing model 才能进一步放大和解释。
- DSE 图表和中文 notes 位于
  `artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_multicore_dse_d4096_worker_chunk_v1/figures/`。

## 2026-07-08 Multi-row Primitive Softmax Findings

- multi-row 真实 SST 代表点已生成在
  `artifacts/sweeps/sfu_softmax_primitive_sweep_20260708_multicore_multirow_v1/figures/`。
- 本轮固定 `chunk=256`、`worker_cores=16`，主要观察 rows 与 dim 扩展，不是
  chunk/worker DSE。
- PASS 点：
  - `rows=4, dim=1024`: simulated `542.939 us`，wall `163s`，
    `dim_per_core=64`，HBM stream `49152 B`。
  - `rows=4, dim=2048`: simulated `680.226 us`，wall `203s`，
    `dim_per_core=128`，HBM stream `98304 B`。
  - `rows=4, dim=4096`: simulated `919.973 us`，wall `270s`，
    `dim_per_core=256`，HBM stream `196608 B`。
  - `rows=8, dim=2048`: simulated `1.01583 ms`，wall `310s`，
    HBM stream `196608 B`。
- `rows=16, dim=1024` 在 `360s` timeout；日志 emergency simulated time 为
  `1.04688 ms`，说明仿真仍在推进，但真实 wall time 已超过可接受阈值。
- 对 emergency log 的可见 per-core DMA 统计进一步解析：
  - 可见未完成 core 的 median read/write issue 为 `28/29`。
  - 由于该配置每个 worker 每行只有 `1` 个 chunk，read issue 包括
    `REDUCE_MAX` 读和 `EXP` 读，故 `28` 次 read 约等于推进到 `14/16` 行。
  - write issue 包括仿真开始时的 `16` 行输入初始化写和每行 output write；
    `29` 次 write 约等于完成 `13/16` 行输出。
  - 可见 core 的 DMA timeout retry/exhausted 为 `0/0`，因此不是 HBM retry
    或 writeback 卡死。
- 关键对照：`rows=8,dim=2048` 与 `rows=16,dim=1024` 总元素数均为 `16384`。
  前者 PASS，后者 timeout，因此本轮瓶颈不能只用总元素数或 HBM byte 解释；
  row 数增加会放大 per-row cross-core reduction、mailbox/guest loop、batch
  issue/wait 和 Vanadis/local-GM 指令推进成本。
- 后续不应继续盲目扩大 rows sweep；下一步应先定位 `rows=16` 的 per-row 控制
  与同步开销，并决定是否优化 row scheduling、mailbox 复用或 batch/issue window。
- 已新增诊断文件：
  `artifacts/sweeps/sfu_softmax_primitive_sweep_20260708_multicore_multirow_v1/figures/softmax_primitive_timeout_diagnosis.md`。

## 2026-07-08 Coordinator-only Reciprocal Result

- 已实现 coordinator-only reciprocal broadcast：
  - 旧路径：每个 worker 每行都执行一次 `RECIPROCAL(row_sum)`。
  - 新路径：coordinator 每行只执行一次 `RECIPROCAL(global_sum)`，然后广播
    `inv_sum` 给所有 worker。
- 统计口径已更新：planned batches 从
  `rows * (planned_groups_per_row * 3 + worker_cores)` 改为
  `rows * (planned_groups_per_row * 3 + 1)`。
- 小规模正确性点 PASS：
  `rows=1,dim=512,chunk=256,worker_cores=16,verify=1` 输出 `batches=49`，
  `max_abs_diff=1.94327e-10`，`max_row_sum_error=7.84031e-10`。
- 关键 `rows=16` probe 仍 timeout：
  `rows=16,dim=1024,chunk=256,worker_cores=16,perf_profile=1` 在 `360s`
  timeout，emergency simulated time 为 `1.08284 ms`，可见未完成 core median
  read/write issue 为 `27/29`，DMA retry/exhausted 为 `0/0`。
- 与旧 timeout 诊断相比，simulated time 从 `1.04688 ms` 增至 `1.08284 ms`，
  但 wall-time 仍未过线，且可见未完成 core 的 read/write 进度没有明显改善。
  因此重复 scalar reciprocal 不是主导瓶颈；下一轮应转向更大的 per-row 固定开销，
  例如多 row 合并 global max/sum reduction、减少 mailbox polling/broadcast，或
  row-level pipeline。

## 2026-07-08 Row-block Reduction Result

- 已实现 `GOLEM_SFU_PRIMITIVE_SOFTMAX_ROW_BLOCK`：
  - `rows > 1` 时默认可使用 block 路径，`row_block=1` 保留为 legacy/debug fallback。
  - `row_block=4` 将 max/sum 的 worker-ready polling 从逐行发布改为每 4 行发布一次。
  - PASS 输出新增 `row_block`、`row_blocks` 和 `block_syncs`。
- 小规模正确性点 PASS：
  `rows=4,dim=512,chunk=256,worker_cores=16,row_block=4,verify=1` 输出
  `row_block=4,row_blocks=1,block_syncs=2`，`max_abs_diff=1.94327e-10`，
  `max_row_sum_error=7.84031e-10`。
- 关键 `rows=16` probe 仍 timeout：
  `rows=16,dim=1024,chunk=256,worker_cores=16,row_block=4,perf_profile=1` 在
  `360s` timeout，emergency simulated time 为 `1.03686 ms`，可见未完成 core
  median read/write issue 为 `24/26`，DMA retry/exhausted 为 `0/0`。
- 与旧逐行 timeout 诊断相比：
  - 原始逐行路径：median read/write issue `28/29`，约推进到 `14/16` 行。
  - coordinator-only reciprocal 后：median read/write issue `27/29`。
  - row-block=4 后：median read/write issue `24/26`，约推进到 `12/16` 行。
- 结论：简单 row-block 化功能正确，但不应作为下一步主优化方向。它减少 ready
  polling 次数的同时，也让 worker 必须先完成一个 block 内多行的 max/sum 阶段，
  推迟 broadcast、normalize 和 output writeback，导致 360s 内可见推进度下降。
  下一步应优先考虑 row-level pipeline 或更细粒度 overlap，而不是继续增大
  `row_block`。

## 2026-07-08 Unified Job Streaming Follow-up

- unified `SFUJobDesc` softmax executor 已从整行 functional buffer 改为
  row-band/chunk streaming functional model。
- 当前实现把 cross-core cooperative execution 表达为 SFU 内部 worker column slices：
  每个 worker slice 先产生 `localMax`，SFU executor 合并为 `globalMax`；随后每个
  worker slice 产生 `localSum`，executor 合并为 `globalSum`；normalize 再按 chunk
  写回。
- `chunk_elems` 现在实际控制 SFU job 内部读写 chunk，而不是只作为 guest descriptor
  字段存在。
- 该实现仍是 host C++ functional/timing model，不是 RTL；但控制结构已经从 guest
  primitive/mailbox stage-machine 迁回 SFU job executor 内部。
- 用户提醒的“直接 HBM 生成 logits、再读取+计算、脱离 GEMM”路径就是
  `GOLEM_SFU_STANDALONE_SOFTMAX=1`，后续 unified job smoke 应复用这条路线，而不是
  发明新的 GEMM-free 数据通路。
- standalone + unified job 的组合语义：
  - `GOLEM_SFU_STANDALONE_SOFTMAX=1` 负责 HBM logits preload 和 logits golden。
  - `GOLEM_SFU_JOB_SOFTMAX=1` 负责把计算入口切到 unified `SFUJobDesc`。
  - guest 只搬运 HBM C tile 区中的 logits 到 local GM row-major buffer，调用
    `golemRunStandaloneSoftmaxSfuJobForCore`，再写回 HBM C tile 区；GEMM 不参与计算。
- wrapper 默认 logits 文件必须带 shape：
  `data/softmax_logits_${GOLEM_GEMM_M}x${GOLEM_GEMM_N}.bin`。旧
  `data/softmax_logits.bin` 容易在不同 shape 间留下 stale 文件，导致 HBM init 或
  verifier 使用错误大小的 logits。
- unified standalone job 真实 SST `64x64` 已 PASS：
  - run id `run_20260708_191142_2928451`。
  - guest stdout 显示
    `dispatch=sfu-standalone-unified-job-softmax rows=64 dim=64 chunk=16 workers=4`
    和 `mode=sfu-standalone-job-softmax ... PASS`。
  - offline verifier 显示
    `[VERIFY-SFU-SOFTMAX] PASS reference=logits dtype=fp32 checked=4096
    mismatches=0 max_abs_diff=3.77493008e-09`。
  - DMA 统计为 `read_issue=1`、`write_issue=1`、`read/write_bytes=16384/16384`、
    `timeout_retry=0`、`timeout_exhausted=0`、`completion/write_completion=1/1`。
- sandbox 内运行真实 SST 仍可能因 OpenMPI socket/OOB 权限失败；真实 SST 需要按权限
  流程在 sandbox 外运行或使用已有可用环境。

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

- 2026-07-04：新增 HBM streaming primitive chunk/batch 科研图，输出在
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_chunk_batch_summary_20260704/figures/`。
  该图使用六个 op、`<=4096` elements/op 的 PASS 数据和一个 timeout 诊断点。
  主要发现是：`chunk_elems` 增大/减小直接改变 chunk 与 DMA request 数；batch
  只压缩 SFU/RoCC control issue，不改变 HBM bytes 或 DMA read/write 数。
  最清晰的对照是 `1024` elems/op、`chunk=256`、all-op：non-batch
  `sfu_ops_issued=24`，batch `sfu_ops_issued=6`，而 DMA read/write 均为 `24/24`。
  `4096` elems/op、`chunk=1024`、batch 在 `1200s` timeout，说明大维度小 chunk
  仍不适合作为常规全统计 sweep 点。
- 2026-07-07：新增 batch-default SFU primitive softmax sweep 图和中文 notes，输出在
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_softmax_primitive_report/figures/`。
  PASS 点共 `5` 个：`128/256/512` 使用 `chunk=dim`，`1024` 使用
  `chunk=512` 和 `chunk=1024`。最大已验证 dim 为 `1024`，最大
  `max_abs_diff=1.44813e-09`。
  `1024/chunk=1024` 相比 `1024/chunk=512` 将 chunks 和 DMA issue 数减半，但
  simulated time 只从 `879.43 us` 变为 `875.113 us`，说明当前功能模型下 chunk
  优化主要减少 descriptor/DMA event，尚未带来明显模拟硬件时间收益。
  `2048/chunk=512` 与 `2048/chunk=2048` 均在 `240s` timeout；日志分别显示被中断
  时已推进到约 `962.57 us` 和 `929.377 us` 模拟时间。判断为当前
  guest/SST event 推进与功能模型开销限制，不是 PASS 点数值错误。
- 2026-07-07：softmax primitive 仿真耗时诊断：
  - 根因调查发现 sweep 已设置 `GOLEM_SKIP_BUILD=1`，但
    `run_noc_dma_softmax_sfu_pipeline.sh` 内部强制 `export GOLEM_SKIP_BUILD=0`，
    导致底层 `run_noc_dma_pipeline.sh` 仍重复构建默认 `test_noc_dma`。该 binary
    不参与 SFU primitive softmax 运行，因为实际使用的是
    `VANADIS_EXE=test_noc_dma_softmax_sfu`。
  - 修复后，`1024/chunk=1024/verify=1` 诊断点跳过 HBM 和底层 build，wall time
    从旧 `234s` 降至 `214s`；simulated time 仍为 `875.113 us`，说明准备阶段
    不是主瓶颈。
  - `verify=0` 诊断点 wall time 为 `166s`、simulated time 为 `672.335 us`。
    这说明 guest 端 per-element golden `std::exp` 有明显开销，但关闭 verify 后
    仍无法解决 200 秒量级问题。
  - 当前最强证据：`1024/chunk=1024` 的业务 DMA 只有 read/write `2/2`，但日志中
    仍有约 `19798` 次 DRAMSim3 backend read 和 `82328` 个 Merlin packet。因此
    剩余主瓶颈不是 softmax HBM bytes，而是 Vanadis guest 指令执行、local GM
    读写、primitive issue/wait 和系统事件推进。
- 2026-07-07：用户明确拒绝将 summary/timing-only 作为当前主实验方案，后续仍以
  真实 SST 仿真为准。新增 `GOLEM_SFU_PERF_PROFILE=1` 后完成 focused real-SST
  softmax primitive sweep：
  - sweep root:
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_softmax_perf_profile_v1`。
  - perf profile 仍运行真实 `sst --num-threads=1`，但关闭逐元素 verification、
    HBM dump 和 SST 全量统计，并在已有 `hbm_config.env` 后复用 HBM/tensor 准备。
  - `1024/chunk=1024`：PASS，simulated time `672.335 us`，wall `171s`，
    HBM stream `12288 B`，DMA read/write issue `2/2`。
  - `2048/chunk=2048`：PASS，simulated time `1.07095 ms`，wall `276s`，
    HBM stream `24576 B`，DMA read/write issue `2/2`。
  - 本轮没有 timeout，但 `2048` 已接近 5 分钟阈值；后续 `4096` 应先做单点
    timeout 诊断，不应直接做全矩阵 sweep。
  - `plot_sfu_softmax_primitive_sweep.py` 的中文 notes 已改为基于 manifest/summary
    动态生成，避免继续带入旧的 `2048 timeout` 硬编码结论。
- 2026-07-07：`4096/chunk=4096` focused real-SST probe：
  - 第一次运行在沙箱内失败，日志为 OpenMPI 无可用网络接口，属于运行环境问题。
  - 第二次沙箱外复跑进入 SST 主仿真，并在 `300s` timeout。
  - emergency shutdown 记录 `# Simulated time: 1.17671 ms`。
  - timeout 前没有 `[SOFTMAX] ... PASS` stdout，也没有非零业务 DMA/SFU issue 统计；
    系统级统计包括 DRAMSim3 backend reads `20379`、Merlin packets `85790`、
    memory queue delay samples `36679`。
  - 结论：`4096` 目前只能作为 timeout 诊断点，不能作为组会趋势图中的 PASS
    性能数据；瓶颈继续指向 Vanadis guest 指令、local GM、primitive issue/wait
    和 SST 系统事件推进。
- 2026-07-07：技术路线纠偏：
  - 用户确认核心路线应为多个 core 联合做 softmax primitive，用于多核架构仿真。
  - 当前 `GOLEM_SFU_PRIMITIVE_SOFTMAX` 只让 `requested_core_id==0` 处理整行，
    与目标架构不符。
  - 因此现有 softmax primitive sweep 和 `4096` probe 必须标记为单核原型诊断；
    它们不能支撑最终多核 SFU primitive softmax 的性能结论。
  - 正确实现方向是：同一行按 column slice 分配到多个 worker core；每个 core
    本地执行 partial max、EXP、partial sum 和 normalize/writeback；架构层面合并
    global row max 和 global row sum。
- 2026-07-07：multi-core primitive softmax 第一版实现/验证发现：
  - 多核不是 `4096` 专用策略；已加入 `GOLEM_SFU_PRIMITIVE_SOFTMAX_MULTICORE_MIN_DIM`
    表示达到较大维度后默认尽量使用多 worker core。
  - 第一版 guest workload 已按 worker slot 切 column slice，并新增
    `worker_cores`、`dim_per_core`、`cross_core_reduce_stages=2` 输出字段。
  - 真实 SST `dim=512, worker_cores=4, chunk=64` probe 未 timeout，但失败于
    GlobalMemory 本地地址断言：
    - 先前 `rd_from_globalmem` 越界来自 coordinator 直接读取 worker GM；已改为
      worker remote 写 coordinator 本地 scratch。
    - 当前剩余失败为 `wr_to_globalmem: wr_addr >= baseAddr`，发生在 DMA read
      response 写回路径，说明 executor/local GM base 与 logical worker id 的映射
      仍需修正。
  - 因此当前不可开展多核性能 sweep；必须先增加/修复 core-id 到 local-GM-base 的
    显式映射或复用现有 runtime 的正确绑定策略。
- 2026-07-07：multi-core primitive softmax local-GM 映射已修复并通过真实 SST
  smoke：
  - 新增 worker/multicore softmax env knob 到 `ncores_selfcom_dma_ctrl.py`，否则 guest
    端不会收到命令行指定的 worker 数和多核阈值。
  - 修复核心是区分 `requested_core_id` 和 `executor_core_id`：local DMA/SFU/GM
    操作必须使用实际 `sched_getcpu()` 对应的 executor/local-GM core；primitive
    softmax 的 worker slot 和本地 mailbox wait 已切到 `executor_core_id`。
  - 真实 SST 点 `dim=512, worker_cores=4, chunk=64, rows=1, verify=1,
    timeout=180s` PASS；输出显示 `dim_per_core=128`、`chunks=8`、`batches=16`、
    `cross_core_reduce_stages=2`，最大绝对误差 `1.90714e-10`。
  - 结论：此前 `wr_to_globalmem: wr_addr >= baseAddr` 不是仿真时间过长，也不是
    SFU 数值错误，而是 logical worker id 与实际 RoCC/GlobalMemory 本地窗口混用。
- 2026-07-07：新增 multi-core cooperative primitive softmax 小矩阵真实 SST sweep
  图和中文 notes，输出在
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_multicore_sweep_d512_c64_128_w4_8_16_v4/figures/`。
  PASS 点共 `6` 个，覆盖 `dim=512`、`chunk=64/128`、`worker_cores=4/8/16`。
  主要发现是：`worker_cores` 控制每个 core 的 column slice 大小，
  `dim_per_core` 从 `128` 降到 `32` 后 simulated time 从约 `348 us` 降到
  `311 us`；`chunk` 会改变 chunks、batches 和 DMA issue 数，例如
  `w=4` 时 `chunk=64` 的 DMA read/write issue 为 `16/16`，`chunk=128`
  降为 `8/8`，但当前功能模型下 simulated time 几乎不变。该结果应作为多核
  primitive softmax 后续 `1024/2048/4096` probe 的基线。
- 2026-07-07：新增 multi-core cooperative primitive softmax 4096 probe 图和中文
  notes，输出在
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_multicore_probe_d1024_2048_4096_c256_w16/figures/`。
  本轮固定 `worker_cores=16`、`chunk=256`，覆盖 `dim=1024/2048/4096`，
  三点全部 PASS。`4096` 的 `dim_per_core=256`，simulated time `415.888 us`，
  wall time `123s`，HBM stream `49152 B`，DMA read/write issue `32/32`。
  这说明单核 `4096` timeout 是错误技术路线下的诊断结果；多核 cooperative
  路线下 `4096` 已可作为真实 SST PASS 数据点。下一步应做同一维度下的
  `chunk/worker_cores` DSE，而不是继续优化单核整行 path。
- 2026-07-08：新增 row-level pipeline depth 控制的第一版保守实现：
  - 新增 `GOLEM_SFU_PRIMITIVE_SOFTMAX_PIPELINE_DEPTH`，已在 wrapper、architecture
    env passthrough 和 RISC-V workload 中打通；`pipeline_depth > 1` 与
    `row_block > 1` 互斥，row-block 继续保留为 debug/负结果路径。
  - 新增 `SoftmaxRowPipelineStage` 和 `SoftmaxRowPipelineState`，并新增
    `run_sfu_primitive_softmax_row_pipeline_for_core`。当前实现是
    `dispatch=conservative-row`：进入 pipeline path、保留 pipeline 元数据和真实
    SST 多核执行，但内部先复用 `row_block=1` 的逐行执行语义，还没有真正做到
    多行 overlap。
  - 验证：
    - `python3 -m unittest discover -s . -p 'test_*.py'`：`93` tests OK。
    - `make clean ARCH=riscv64 && make ARCH=riscv64`：通过。
    - 真实 SST correctness：
      `rows=2, dim=512, chunk=256, worker_cores=16, row_block=1,
      pipeline_depth=2, verify=1` PASS；输出包含
      `pipeline_depth=2 pipeline_mode=row`，`max_abs_diff=1.94327e-10`、
      `max_rel_diff=8.101e-08`、`max_row_sum_error=7.84031e-10`，
      simulated time `478.112 us`，wall time `128s`。
    - 真实 SST 长点 probe：
      `rows=16, dim=1024, chunk=256, worker_cores=16, row_block=1,
      pipeline_depth=2, verify=0` 在 `360s` timeout；emergency shutdown
      simulated time `1.18338 ms`，未产生 PASS。
  - timeout 诊断：
    - 业务 DMA 统计显示 `timeout_retry=0`、`timeout_exhausted=0`、
      `write_timeout_retry=0`，代表不是 DMA retry/deadlock。
    - 多个 active worker core timeout 前约 `read_issue_count=28`、
      `write_issue_count=29`，说明仿真仍在推进，但逐行同步/primitive
      issue-wait/guest 指令开销仍然过大。
    - 结论：当前 conservative-row 只完成了 architecture knob 和真实 SST
      可验证入口，不应作为性能优化结果汇报；下一步必须实现真正 row pipeline
      overlap，把 row N 的等待窗口与 row N+1 的 local max/exp 准备重叠。
- 2026-07-08：将 `pipeline_depth=2` 从 conservative-row 改为 windowed-row：
  - 新增静态测试要求 pipeline path 出现 `dispatch=windowed-row`，且不再保留
    `dispatch=conservative-row`。
  - 实现方式：`pipeline_window_rows=pipeline_depth`，depth=2 时两行组成一个
    row pipeline window。窗口内先完成两行 local max，再广播两行 global max，
    再做两行 EXP/local sum，最后广播 reciprocal 并 normalize/writeback。
    该实现复用 row-block 执行引擎作为 pipeline window，不再是逐行
    conservative scaffold。
  - 验证：
    - `python3 -m unittest discover -s . -p 'test_*.py'`：`94` tests OK。
    - `make clean ARCH=riscv64 && make ARCH=riscv64`：通过。
    - 真实 SST correctness：
      `rows=2, dim=512, chunk=256, worker_cores=16, row_block=1,
      pipeline_depth=2, verify=1` PASS；stdout 显示
      `pipeline_window_rows=2 dispatch=windowed-row`，
      `max_abs_diff=1.94327e-10`、`max_rel_diff=8.101e-08`、
      `max_row_sum_error=7.84031e-10`，simulated time `440.114 us`，
      wall time `129s`。
    - 与 conservative-row 小点相比：simulated time 从 `478.112 us`
      降到 `440.114 us`，说明两行窗口减少了部分同步开销。
    - 真实 SST rows=16 probe：
      `rows=16, dim=1024, chunk=256, worker_cores=16, pipeline_depth=2,
      verify=0` 在 `360s` timeout；emergency simulated time `1.163 ms`。
  - timeout 诊断：
    - active worker core timeout 前约 `read_issue_count=24`、
      `write_issue_count=27`，比 conservative-row 的 `28/29` 少，说明窗口化减少了
      一部分 row-level 同步/issue 数。
    - `timeout_retry=0`、`timeout_exhausted=0`、`write_timeout_retry=0`，仍不是
      DMA retry/deadlock。
    - 结论：windowed-row 是正确方向上的小幅改进，但不足以让 `rows=16,
      dim=1024` 在 360 秒内完成。继续加大窗口风险会回到 row-block 负结果；
      下一步应拆出更细粒度的阶段函数，做非阻塞式 coordinator polling 或更少
      mailbox/remote-write 的 reduction 协议。
- 2026-07-08：新增 windowed-row two-row packed sync：
  - 目标：`pipeline_depth=2` 时每个窗口只有两行，将两行 FP32 标量打包进一个
    64-bit GM word，减少 worker->coordinator 和 coordinator->worker 的标量 value
    写入次数。该优化只在 `row_block <= 2` 启用，`row_block > 2` debug path
    保持原逐行 value 布局。
  - 实现：
    - 新增 `pack_two_fp32_to_reg`、`low_fp32_from_packed_reg`、
      `high_fp32_from_packed_reg`。
    - 对 local max、global max、local sum、global reciprocal/inv_sum 的两行
      标量值使用 packed 64-bit word；ready seq 仍单独保留。
    - pipeline dispatch 日志改为 `dispatch=windowed-row-packed-sync`。
  - 验证：
    - 新增静态测试覆盖 packed helper、`packed_two_row_sync` 分支和 dispatch 字段。
    - `python3 -m unittest discover -s . -p 'test_*.py'`：`95` tests OK。
    - `make clean ARCH=riscv64 && make ARCH=riscv64`：通过。
    - 真实 SST correctness：
      `rows=2, dim=512, chunk=256, worker_cores=16, row_block=1,
      pipeline_depth=2, verify=1` PASS；stdout 显示
      `pipeline_window_rows=2 dispatch=windowed-row-packed-sync`，
      `max_abs_diff=1.94327e-10`、`max_rel_diff=8.101e-08`、
      `max_row_sum_error=7.84031e-10`，simulated time `438.836 us`，
      wall time `124s`。
    - 对比：小点 simulated time 从 windowed-row 的 `440.114 us` 降到
      `438.836 us`，改善很小但方向正确。
    - 真实 SST rows=16 probe：
      `rows=16, dim=1024, chunk=256, worker_cores=16, pipeline_depth=2,
      verify=0` 仍在 `360s` timeout；emergency simulated time `1.11927 ms`。
  - timeout 诊断：
    - active worker core timeout 前约 `read_issue_count=24`、
      `write_issue_count=26`，比 windowed-row 的 `24/27` 少一个 write issue。
    - `timeout_retry=0`、`timeout_exhausted=0`、`write_timeout_retry=0`，仍不是
      DMA deadlock。
    - packed sync 只能减少少量标量同步写入，不能解决主要的 guest
      issue/wait 和逐阶段阻塞问题；下一步需要真正改 protocol，例如非阻塞
      coordinator polling、减少 reciprocal/EXP/SUM 的 wait 次数，或把多个 row
      的 primitive child desc 进一步融合到更大的 batch 提交。
- 2026-07-08：新增 windowed-row packed crossbatch：
  - 目标：在 `pipeline_depth=2` 的 two-row window 内，不再对两个 row 分别 issue
    EXP/SUM batch，而是把同一 column slice 上两个 row 的 child desc 聚合进一个
    `cross_row_exp_descs` 和一个 `cross_row_sum_descs`，减少 primitive batch
    issue/wait 次数。每个 row 的 GM slot 使用 `combined_slot` 分配，保证两个 row
    的 input/output GM scratch 不重叠；当 batch slot 不足或窗口只剩 1 行时仍回退
    到原逐 row 路径。
  - 同步修复：`run_sfu_softmax_primitive_sweep.sh` 会把 `GOLEM_SWEEP_ROOT`
    规范成绝对路径，避免 wrapper/architecture 在不同 cwd 下解释相对 artifact root，
    导致 guest stdout 写到另一处而 manifest 误记 `FAIL`。
  - 验证：
    - 新增静态测试覆盖 `cross_row_batch_rows`、`combined_slot`、
      `cross_row_exp_descs/cross_row_sum_descs` 和
      `dispatch=windowed-row-packed-crossbatch`。
    - `python3 -m unittest discover -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu -p 'test_*.py'`：
      `97` tests OK。
    - `make clean ARCH=riscv64`、`make ARCH=riscv64`：通过。
    - 真实 SST correctness：
      `rows=2, dim=512, chunk=256, worker_cores=16, pipeline_depth=2,
      verify=1` PASS；manifest 正确记录 `PASS,0`；stdout 显示
      `dispatch=windowed-row-packed-crossbatch`；simulated time `429.63 us`，
      `max_abs_diff=1.94327e-10`、`max_rel_diff=8.101e-08`、
      `max_row_sum_error=7.84031e-10`。
    - 相比 packed sync 小点 `438.836 us`，crossbatch 降到 `429.63 us`，
      改善约 `2.1%`，说明减少 EXP/SUM batch issue 有效但收益有限。
    - 真实 SST rows=16 probe：
      `rows=16, dim=1024, chunk=256, worker_cores=16, pipeline_depth=2,
      verify=0` 在 `360s` timeout；emergency simulated time `1.21492 ms`。
      active worker timeout 前约 `read_issue_count=28`、`write_issue_count=29-30`；
      `timeout_retry=0`、`timeout_exhausted=0`、`write_timeout_retry=0`。
  - 结论：
    - crossbatch 是正确的小幅优化，但仍不能让 `rows=16, dim=1024` 在 360 秒内
      完成。
    - 当前主要限制不是 HBM/DMA retry，也不是 correctness；更像 Vanadis guest
      指令、mailbox polling、primitive issue/wait 和阶段阻塞导致 wall time
      过高。
    - 下一步应进入 stage-level protocol 重构：把 local max、global max
      reduce/broadcast、EXP+SUM、reciprocal broadcast、normalize/writeback 拆成
      可推进状态机，并优先减少阻塞 wait 或让 coordinator 非阻塞服务多个 row。
- 2026-07-08：coordinator nonblocking polling 实验已转为可选负结果路径：
  - 新增 `GOLEM_SFU_PRIMITIVE_SOFTMAX_NBPOLL`，默认值为 `0`。默认架构仍是
    `dispatch=windowed-row-packed-crossbatch`；只有显式设置
    `GOLEM_SFU_PRIMITIVE_SOFTMAX_NBPOLL=1` 时才进入
    `dispatch=windowed-row-packed-crossbatch-nbpoll`。
  - nbpoll 思路：coordinator 不再按 worker 顺序阻塞等待每个 ready word，而是扫描
    `observed_max_workers/observed_sum_workers`，用
    `softmax_primitive_poll_ready` 判断 max/sum ready 状态，希望减少线性等待。
  - 验证：
    - 静态回归测试覆盖 `GOLEM_SFU_PRIMITIVE_SOFTMAX_NBPOLL`、wrapper export、
      architecture env passthrough、`coordinator_nbpoll`、ready polling helper 和
      两种 dispatch 字段。
    - `python3 -m unittest discover -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu -p 'test_*.py'`：
      `98` tests OK。
    - `make clean ARCH=riscv64`、`make ARCH=riscv64`：通过。
    - nbpoll correctness：`rows=2, dim=512, chunk=256, worker_cores=16,
      verify=1` PASS；`dispatch=windowed-row-packed-crossbatch-nbpoll`；
      simulated time `430.941 us`，`max_abs_diff=1.94327e-10`，
      `max_rel_diff=8.101e-08`，`max_row_sum_error=7.84031e-10`。
    - nbpoll rows=16 probe：`rows=16, dim=1024, chunk=256, worker_cores=16,
      verify=0` 在 `360s` timeout；emergency simulated time `1.14609 ms`；
      DMA retry/exhausted/write retry 均为 `0`。
    - 默认 crossbatch 回归：显式 `GOLEM_SFU_PRIMITIVE_SOFTMAX_NBPOLL=0` 的
      `rows=2, dim=512, chunk=256, worker_cores=16, verify=1` 真实 SST PASS；
      stdout 显示 `dispatch=windowed-row-packed-crossbatch`；simulated time
      `449.977 us`；`max_abs_diff=1.94327e-10`，
      `max_rel_diff=8.101e-08`，`max_row_sum_error=7.84031e-10`。
  - 结论：
    - 简单 GM ready-word 扫描不是当前主优化方向。它在小点上没有超过原
      crossbatch 基线，长点仍然 timeout。
    - nbpoll 应保留为负实验/诊断开关，不作为默认架构。
    - 下一步应做真正的 stage-level row state machine：把每行/每阶段的状态拆开，
      让 max、sum、EXP、normalize 的推进可以跨 row 交错，而不是只在 coordinator
      内部把阻塞等待改成扫描。
- 2026-07-08：实现第一版 stage-level row state machine：
  - pipeline 主路径不再直接 `return run_sfu_primitive_softmax_row_block_for_core`，
    而是新增 stage mailbox：
    `kSoftmaxPrimitiveStageWorkerStride`、
    `softmax_primitive_stage_worker_addr` 和
    `softmax_primitive_stage_global_addr`。
  - 新增三个阶段推进函数：
    `advance_softmax_stage_local_max`、
    `advance_softmax_stage_exp_sum`、
    `advance_softmax_stage_normalize`。
  - 调度顺序变为：window 内每行先发布自己的 `LOCAL_MAX` ready，coordinator
    可以按行收集并广播 `GLOBAL_MAX`；随后每行独立执行 EXP/SUM、发布
    `LOCAL_SUM`，coordinator 按行广播 reciprocal/inv_sum，最后 normalize。
  - dispatch 字段更新为 `dispatch=stage-row-state-machine`，PASS 输出新增
    `pipeline_stage_cycles`。
  - 验证：
    - 新增静态测试覆盖 stage mailbox、stage advance 函数、状态枚举和
      `dispatch=stage-row-state-machine`。
    - `python3 -m unittest discover -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu -p 'test_*.py'`：
      `99` tests OK。
    - `make clean ARCH=riscv64`、`make ARCH=riscv64`：通过。
    - 小点 correctness：`rows=2, dim=512, chunk=256, worker_cores=16,
      pipeline_depth=2, verify=1` 真实 SST PASS；stdout 显示
      `dispatch=stage-row-state-machine`；simulated time `432.92 us`；
      `pipeline_stage_cycles=6`；`max_abs_diff=1.94327e-10`，
      `max_rel_diff=8.101e-08`，`max_row_sum_error=7.84031e-10`。
    - `rows=16, dim=1024, chunk=256, worker_cores=16, verify=0` 在 `360s`
      timeout；emergency simulated time `971.749 us`。
      emergency log 可见 active worker 的 read/write issue 约 `29-30/30`，
      DMA retry/exhausted/write retry 均为 `0`。
  - 结论：
    - 第一版 stage-machine 数学正确，但没有解决 `rows=16` wall-time timeout。
    - 与 crossbatch rows=16 probe 的 emergency simulated time `1.21492 ms`
      相比，stage-machine 在同样 360 秒内只推进到 `971.749 us`，说明新增
      stage mailbox 和逐行 stage 调度增加了 guest/SST 事件开销。
    - 继续方向不应是“更细粒度但更多 guest 指令”的软件状态机；需要把 stage
      状态推进进一步下沉为 SFU/RoCC 侧队列或更粗粒度的硬件 descriptor，
      减少 guest 轮询、remote write 和 primitive issue/wait 次数。

- 2026-07-09：direct row-major unified SFU job 的大维度问题定位：
  - 问题：
    GEMM 能跑 4096 维，但 unified SFU job softmax 在 1024 direct HBM 点会出
    correctness 问题。
  - 根因差异：
    GEMM 的 4096 不是“一次把整行/整 band 搬进 GM 再算”，而是长期沿用
    block/chunk DMA：A/B panel 按 tile、slot、credit 和 K window 流动。
    早期 direct softmax 则让每个 active band core 一次搬一个完整 row-band
    (`staging_rows * dim * sizeof(float)`)，512 时就是 128KB/core，并发后会
    触发 GlobalMemory DMA read retry/exhausted；exhausted 后 guest 仍继续
    计算，local GM 中留下上一波数据，因此 verifier 出现整 band 或半矩阵
    mismatch。
  - 已修复的代码路径：
    direct row-major path 改为按 `GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS` 子块 streaming：
    每个 sub-job 执行 `direct-load -> SFU_JOB SOFTMAX_ROW -> direct-store`。
    同时 local GM required bytes 从整 band 改成 sub-job input/output buffer。
  - 512 维验证：
    `band_cores=8, staging_rows=64, job_rows=16, chunk=256, workers=16` 真实 SST
    PASS，`checked=262144 mismatches=0 max_abs_diff=3.90064624e-10`；
    DMA summary `timeout_retry=0 timeout_exhausted=0`。
  - 1024 维验证：
    高并发档 `band_cores=8, job_rows=8, retry_ticks=256` 仍失败：
    verifier 从 row 512 开始 mismatch，`mismatches=524288`；
    DMA summary `timeout_retry=1407 timeout_exhausted=128`。
    对比输出发现 row 512 后复用了前一波 local GM 数据，说明不是地址 ABI 或
    softmax 数学错误。
    保守正确性档 `band_cores=4, job_rows=8, retry_ticks=4096,
    max_retries=16` 真实 SST PASS，`checked=1048576 mismatches=0
    max_abs_diff=2.23162767e-10`，DMA timeout/retry 均为 0。
- 结论：
    对 unified SFU job direct softmax，1024 问题的第一根因是 DMA 粒度与并发
    压力，而不是 GEMM 能力或 SFU softmax 数学。下一步应围绕 direct job 的
    band-core 并发、retry window、DMA credits 做 sweep，形成默认稳定档和
    性能档；不要回到 primitive/batch softmax 主线。

## 2026-07-09 Unified Job Direct Sweep Decision

- `run_sfu_unified_job_direct_sweep.sh` 是 unified SFU job direct row-major HBM
  路线的 sweep 入口；它固定使用 standalone logits、`SFU_JOB` softmax 和 direct
  row-major HBM 输入/输出，不启用 `GOLEM_SFU_PRIMITIVE_SOFTMAX`。
- 默认 profile 是 correctness-stable profile，而不是压力 profile：
  - `stable512` 复现实测 512 PASS 配置。
  - `stable1024` 复现实测 1024 PASS 配置：降低 `band_cores` 到 4，并把
    DMA read retry window 放宽到 `retry_ticks=4096,max_retries=16`。
- `pressure1024` 是显式负实验档，用于复现
  `band_cores=8,retry_ticks=256` 下的 DMA retry/exhausted 边界。该点的
  `expect=fail` 应记录为 `EXPECTED_FAIL`，不应作为默认正确性回归失败处理。
- 新增 `GOLEM_SFU_JOB_DIRECT_POINT_LIST` 支持手工覆盖 profile，格式为
  `dim:band_cores:job_rows:retry_ticks:max_retries:expect`。这让后续评估
  band-core 并发、retry window 和 DMA credit 时可以复用同一个 manifest 口径。
- 真实 SST profile 已验证该脚本口径：
  - `stable1024` artifact root:
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable1024_real`。
    manifest 为 `PASS`，verifier 为 `checked=1048576,mismatches=0,
    max_abs_diff=2.23162767e-10`，DMA retry/exhausted 为 `0/0`，read/write
    completion 为 `256/256`。
  - `pressure1024` artifact root:
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_pressure1024_real`。
    manifest 为 `EXPECTED_FAIL`，verifier 从 row 512 开始失败，
    `mismatches=524288`；DMA retry/exhausted 为 `1407/128`，read completion
    只有 `128/256`，write completion 为 `208/256`。
- 因此，当前 1024 维边界已经可以稳定复现为：
  `band_cores=4,retry_ticks=4096,max_retries=16` 正确；
  `band_cores=8,retry_ticks=256,max_retries=8` 因 DMA read exhausted 失败。
  下一步的 sweep 应优先找出最小 retry window 或 band-core 并发阈值，而不是修改
  SFU softmax 数学路径。

## 2026-07-09 DMA Load Guard and Failure Classification

- direct row-major HBM pressure failure 已从“继续计算后产生半矩阵 mismatch”收敛为
  “guest 明确报告 DMA load failure”：
  - 每个 sub-job 在 input GM buffer 中预置 sentinel。
  - DMA read 返回后若 sentinel 仍在，说明 local GM 没有被本次 HBM 数据覆盖，
    guest 直接返回失败。
  - 这避免了 row 512 后复用上一波 GM 数据继续发 `SFU_JOB` 的静默错误。
- wrapper 现在会在 offline verifier 前扫描 guest failure：
  - 匹配 `DMA_LOAD_FAILED`、`standalone unified job failed`、
    `[SOFTMAX-SFU-JOB].*failed`。
  - 命中后输出 `[SFU][ERROR] guest reported failure; skip softmax verifier`，
    退出码为 1。
  - 这使 pressure profile 的 `EXPECTED_FAIL` 表示真实的 DMA read exhaustion，
    而不是后续 verifier 对 partial HBM output 的二次噪声。
- 实测结论：
  - guarded `stable1024` 保持 PASS：
    `checked=1048576,mismatches=0,max_abs_diff=2.23162767e-10`。
  - guarded `pressure1024` 在
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_guard_pressure1024_wrapperfail_real`
    中变为干净的 `EXPECTED_FAIL`：
    8 个 active executor core 报 `DMA_LOAD_FAILED`，manifest exit_code 为 1，
    wrapper 没有继续 unpack/verifier。
- 下一步判断：
  direct softmax 的正确性边界已经可以用 `DMA_LOAD_FAILED` 作为明确分类信号。
  后续 DSE 应继续沿 unified SFU job direct path 扫：
  `band_cores`、`GOLEM_DMA_READ_RETRY_TICKS`、`GOLEM_DMA_READ_MAX_RETRIES`、
  `GOLEM_DMA_SLOT_COUNT`，目标是找出比 `band_cores=4,retry=4096/16` 更接近
  高并发的稳定点，而不是回退到 primitive/batch softmax。

## 2026-07-09 1024 Direct Job Band/Retry DSE

- DSE 矩阵：
  `dim=1024, job_rows=8, chunk=256, workers=16` 固定；
  `band_cores={4,6,8}`、
  `retry_ticks={512,1024,2048,4096}`、
  `max_retries={8,16}`。
- 真实 SST sweep artifact：
  `artifacts/sweeps/sfu_unified_job_direct_dse_1024_bc_retry_20260709_real`。
- 结果：
  - 24 个点全部 PASS。
  - 所有点 `DMA timeout_retry=0`、`timeout_exhausted=0`。
  - 所有点 read/write completion 均为 `256/256`。
  - `retry_ticks` 和 `max_retries` 在该网格中没有影响 simulated time，因为没有
    实际 retry。
  - `band_cores=8` 的 simulated time 最好：`482.485 us`；
    `band_cores=6` 为 `498.133 us`；
    `band_cores=4` 为 `501.695 us`。
- 关键判断：
  - 当前可把 1024 direct row-major unified SFU job 的默认高并发稳定候选从
    `band_cores=4,retry_ticks=4096,max_retries=16` 推进到
    `band_cores=8,retry_ticks=512,max_retries=8`。
  - 已知 `band_cores=8,retry_ticks=256,max_retries=8` 会触发
    `DMA_LOAD_FAILED`，而 `512/8` 稳定 PASS，因此 retry threshold 位于
    `256 < retry_ticks <= 512`。
  - 下一轮若要进一步压低等待窗口，应固定 `band_cores=8,max_retries=8`，
    细扫 `retry_ticks={288,320,384,448,512}`，或先做
    `{320,384,448}` 的三点 probe。

## 2026-07-09 1024 Direct Job Retry Fine Sweep

- 细扫矩阵：
  `dim=1024, band_cores=8, job_rows=8, chunk=256, workers=16,
  max_retries=8` 固定；
  `retry_ticks={288,320,384,448,512}`。
- 真实 SST sweep artifact：
  `artifacts/sweeps/sfu_unified_job_direct_dse_1024_retry_fine_20260709_real`。
- 结果：
  - 5 个点全部 PASS。
  - `retry_ticks=288` 没有 exhausted，但有 `dma_timeout_retry_sum=159`，
    simulated time 上升到 `587.451 us`。
  - `retry_ticks=320/384/448/512` 全部 clean：
    `dma_timeout_retry_sum=0`、`timeout_exhausted_sum=0`、
    read/write completion 都是 `256/256`，simulated time 都是 `482.485 us`。
- 关键判断：
  - 最小“能正确跑完”的实测点是
    `band_cores=8,retry_ticks=288,max_retries=8`。
  - 最小“无 DMA retry 的 clean PASS”实测点是
    `band_cores=8,retry_ticks=320,max_retries=8`。
  - 由于 `288` 已出现 retry 并显著拉长 simulated time，不建议作为默认。
    默认稳定高并发配置应选：
    `band_cores=8,retry_ticks=320,max_retries=8`。
  - 若还要更精细地找临界点，可在 `retry_ticks=256..320` 之间扫
    `{272,280,288,296,304,312,320}`，但工程默认没有必要压到 retry 边缘。

## 2026-07-09 Stable Profile Update

- `run_sfu_unified_job_direct_sweep.sh` 的 `stable1024` 已更新为：
  `band_cores=8, job_rows=8, retry_ticks=320, max_retries=8, expect=pass`。
- 这是当前实测的最小 clean PASS 高并发点：
  `dma_timeout_retry_sum=0`、`timeout_exhausted_sum=0`、
  read/write completion 为 `256/256`，simulated time 为 `482.485 us`。
- `pressure1024` 保持：
  `band_cores=8, job_rows=8, retry_ticks=256, max_retries=8, expect=fail`。
  它继续作为 direct DMA load guard 的负实验边界。
- 解释：
  - `288/8` 虽 PASS，但已有 `dma_timeout_retry_sum=159`，不适合作为默认。
  - `320/8` 是 clean PASS，且不会像旧的 `4096/16` 那样过度保守。

## 2026-07-09 Stable Profile Canonical Real-SST Regression

- 更新后的 `GOLEM_SFU_JOB_DIRECT_PROFILE=stable` 已完成真实 SST 合并回归：
  `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable_profile_real`。
- manifest 结果：
  - `stable512`: `dim=512, band_cores=8, job_rows=16,
    retry_ticks=256, max_retries=8, status=PASS, exit_code=0`。
  - `stable1024`: `dim=1024, band_cores=8, job_rows=8,
    retry_ticks=320, max_retries=8, status=PASS, exit_code=0`。
- `stable1024` 的 DMA 行为是 clean PASS：
  `dma_timeout_retry_sum=0`、`timeout_exhausted_sum=0`、
  read/write issue 和 completion 都是 `256/256`，simulated time 为
  `482.485 us`。
- `stable512` 在 `retry_ticks=256` 下有少量 read timeout retry：
  `dma_timeout_retry_sum=42`，但 `timeout_exhausted_sum=0`，read/write
  completion 都是 `64/64`，manifest 仍为 PASS。
- 控制台 verifier 证据显示 `stable1024`：
  `checked=1048576,mismatches=0,max_abs_diff=2.23162767e-10`。
- sweep 结束时的 `no router stats found` 来自关闭完整 router stats 后的附加
  NoC 统计提取器，不是 SFU guest、DMA guard 或 verifier 失败。
- 决策：
  `band_cores=8,job_rows=8,retry_ticks=320,max_retries=8` 从“fine sweep 推荐”
  升级为当前 1024 direct row-major unified SFU job softmax 的默认稳定配置。
  下一阶段可以基于该基线向 `dim=2048` 扩展，而不是继续在 1024 上压低 retry
  临界点。

## 2026-07-09 2048 Direct Job Scaling Probe

- 首个 2048 direct row-major unified SFU job 单点已真实 SST PASS：
  `artifacts/sweeps/sfu_unified_job_direct_2048_probe_20260709_rt320_jr4_real`。
- 配置：
  `dim=2048, band_cores=8, job_rows=4, chunk=256, workers=16,
  retry_ticks=320, max_retries=8`。
- manifest：
  `status=PASS, exit_code=0`。
- verifier：
  `checked=4194304,mismatches=0,max_abs_diff=1.13844124e-10`。
- DMA 行为：
  - read/write issue: `1024/1024`。
  - read/write completion: `1024/1024`。
  - read/write bytes: `16777216/16777216`。
  - `dma_timeout_retry_sum=83`，`timeout_exhausted_sum=0`。
- timing：
  `simulated_time=1.06516 ms`，`wall_time_sec=244`。
- 判断：
  2048 correctness 和 full HBM read/write completion 已跑通，说明 direct
  row-band/chunk streaming 可以继续扩展维度。但 `retry_ticks=320` 已出现少量
  read retry，因此它是 2048 的 correctness anchor，不一定是 clean default。
  下一轮应固定 `dim=2048,band_cores=8,job_rows=4,max_retries=8` 比较
  `retry_ticks={384,512}`，并单独评估 `job_rows=8` 的 sub-job 数量/稳定性权衡。

## 2026-07-09 2048 Direct Job Clean Retry Sweep

- 2048 clean retry sweep 已真实 SST PASS：
  `artifacts/sweeps/sfu_unified_job_direct_2048_retry_clean_20260709_real`。
- 固定配置：
  `dim=2048, band_cores=8, job_rows=4, chunk=256, workers=16,
  max_retries=8`。
- 扫描点：
  `retry_ticks={384,512}`。
- manifest：
  两点均 `status=PASS, exit_code=0`。
- DMA/timing：
  - `retry_ticks=384`：
    `dma_timeout_retry_sum=0`、`timeout_exhausted_sum=0`、
    read/write completion `1024/1024`、simulated time `1.01116 ms`、
    wall `218s`。
  - `retry_ticks=512`：
    `dma_timeout_retry_sum=0`、`timeout_exhausted_sum=0`、
    read/write completion `1024/1024`、simulated time `1.01116 ms`、
    wall `222s`。
- verifier：
  两点控制台 verifier 均为
  `checked=4194304,mismatches=0,max_abs_diff=1.13844124e-10`。
- 对比 2048 `retry_ticks=320`：
  `320/8` 已能 PASS，但有 `dma_timeout_retry_sum=83`，simulated time 为
  `1.06516 ms`；`384/8` 是当前最小实测 clean PASS 点。
- 决策：
  2048 direct row-major unified SFU job 的 clean profile 候选应设为
  `band_cores=8,job_rows=4,retry_ticks=384,max_retries=8`。下一步再测
  `job_rows=8,retry_ticks=384,max_retries=8`，判断能否用更少 sub-job 保持
  clean PASS。

## 2026-07-09 2048 Direct Job `job_rows=8` Boundary

- `job_rows=8` throughput-oriented probe 当前未通过，失败分类清晰：
  guest-side direct DMA load guard 触发，wrapper 跳过 verifier。
- `retry_ticks=384,max_retries=8`：
  - artifact：
    `artifacts/sweeps/sfu_unified_job_direct_2048_jr8_probe_20260709_real`。
  - manifest：`FAIL, exit_code=1`。
  - 8 个 active executor core 报 `DMA_LOAD_FAILED`。
  - 失败 sub-job 粒度：`sub_job_rows=8, bytes=65536`。
  - DMA summary：
    `timeout_retry_sum=541, timeout_exhausted_sum=32,
    read_issue_count_sum=144, write_issue_count_sum=112,
    completion_sum=112, write_completion_sum=108`。
- `retry_ticks=512,max_retries=8`：
  - artifact：
    `artifacts/sweeps/sfu_unified_job_direct_2048_jr8_rt512_probe_20260709_real`。
  - manifest：`FAIL, exit_code=1`。
  - 仍为 `DMA_LOAD_FAILED`，不是 verifier mismatch。
  - DMA summary：
    `timeout_retry_sum=861, timeout_exhausted_sum=32,
    read_issue_count_sum=236, write_issue_count_sum=204,
    completion_sum=204, write_completion_sum=200`。
- 对比：
  `job_rows=4,retry_ticks=384,max_retries=8` 已 clean PASS：
  read/write completion `1024/1024`、`timeout_retry_sum=0`、
  `timeout_exhausted_sum=0`。
- 决策：
  `job_rows=8` 不是当前 2048 clean default。它把单次 sub-job load 提高到 64KB，
  在 8 band cores 并发下触发 direct DMA exhausted；`512/8` 仍不足以稳定。
  当前 2048 默认应保持
  `band_cores=8,job_rows=4,retry_ticks=384,max_retries=8`。
  后续若要继续探索 `job_rows=8`，应将其作为压力/吞吐边界实验，显式扫
  `retry_ticks=1024` 或 `max_retries=16`。

## 2026-07-09 Stable2048 Profile Solidified

- `run_sfu_unified_job_direct_sweep.sh` 已新增
  `GOLEM_SFU_JOB_DIRECT_PROFILE=stable2048`：
  `dim=2048, band_cores=8, job_rows=4, retry_ticks=384,
  max_retries=8, expect=pass`。
- 默认 `GOLEM_SFU_JOB_DIRECT_PROFILE=stable` 现在包含三点：
  `stable512`、`stable1024`、`stable2048`。
- canonical stable2048 artifact：
  `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable2048_profile_real`。
- manifest：
  `sfu_job_direct_stable2048,2048,8,4,384,8,pass,PASS,0,1800`。
- verifier：
  `checked=4194304,mismatches=0,max_abs_diff=1.13844124e-10`。
- clean DMA evidence：
  `dma_timeout_retry_sum=0`、`timeout_exhausted_sum=0`、
  read/write issue `1024/1024`、read/write completion `1024/1024`。
- timing：
  `simulated_time=1.01116 ms`、`wall_time_sec=226`。
- 决策：
  2048 direct row-major unified SFU job softmax 的默认 clean profile 已从候选
  升级为稳定 profile。下一步可以从该 profile 出发探 `dim=4096`，或者把
  `job_rows=8` 作为单独 pressure profile 继续扫更宽 retry policy；两者都不应
  回到 primitive/batch softmax 主线。

## 2026-07-09 4096 Direct Job First Probe

- 4096 direct row-major unified SFU job 已真实 SST clean PASS。
- 4096 的 direct row-major HBM footprint 需要显式放大 per-node backing：
  - input matrix: `4096 * 4096 * 4 = 64MiB`。
  - output matrix: `64MiB`。
  - input+output 共 `128MiB`，再加 GEMM C/direct metadata 区域后默认
    `128MiB` per-node backing 不够。
  - 成功 probe 使用 `--mem-node-size 268435456`。
- 首次启动被用户中断后只留下 partial artifact：
  `artifacts/sweeps/sfu_unified_job_direct_4096_probe_20260709_rt384_jr2_mem256_real`；
  manifest 只有表头，未作为结果。
- canonical 4096 probe artifact：
  `artifacts/sweeps/sfu_unified_job_direct_4096_probe_20260709_rt384_jr2_mem256_retry1_real`。
- 配置：
  `dim=4096, band_cores=8, job_rows=2, chunk=256, workers=16,
  retry_ticks=384, max_retries=8, mem_node_size=256MiB`。
- 配置理由：
  `job_rows=2` 让单次 direct DMA load 粒度保持为 `32768B`，与 2048 clean
  profile 的 `job_rows=4` 相同。
- manifest：
  `sfu_job_direct_d4096_bc8_jr2_rt384_mr8_pass,4096,8,2,384,8,pass,PASS,0,2400`。
- verifier：
  `checked=16777216,mismatches=0,max_abs_diff=5.72476014e-11`。
- clean DMA evidence：
  `dma_timeout_retry_sum=0`、`timeout_exhausted_sum=0`、
  read/write issue `4096/4096`、read/write completion `4096/4096`。
- timing：
  `simulated_time=3.16213 ms`、`wall_time_sec=745`。
- 决策：
  4096 已成为 unified-job direct row-major path 的 correctness anchor。下一步
  可以新增 `stable4096` profile，但它必须携带或自动派生 256MiB mem-node size；
  否则 HBM init/layout 失败会掩盖 SFU/DMA 行为。

## 2026-07-09 Stable4096 Profile Solidified

- `run_sfu_unified_job_direct_sweep.sh` 已新增
  `GOLEM_SFU_JOB_DIRECT_PROFILE=stable4096`：
  `dim=4096, band_cores=8, job_rows=2, retry_ticks=384,
  max_retries=8, mem_node_size=268435456, expect=pass`。
- `run_point` 现在支持第 9 个可选参数 `mem_node_size`，默认仍为
  `134217728`；只有 `stable4096` 显式使用 `268435456`。
- 默认 `stable` profile 仍只包含 `stable512`、`stable1024`、`stable2048`。
  4096 需要单独通过 `GOLEM_SFU_JOB_DIRECT_PROFILE=stable4096` 调用。
- canonical stable4096 artifact：
  `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable4096_profile_real`。
- manifest：
  `sfu_job_direct_stable4096,4096,8,2,384,8,pass,PASS,0,2400`。
- verifier：
  `checked=16777216,mismatches=0,max_abs_diff=5.72476014e-11`。
- clean DMA evidence：
  `dma_timeout_retry_sum=0`、`timeout_exhausted_sum=0`、
  read/write issue `4096/4096`、read/write completion `4096/4096`。
- timing：
  `simulated_time=3.16213 ms`、`wall_time_sec=721`。
- 决策：
  4096 direct row-major unified SFU job softmax 的 clean profile 已从
  correctness anchor 升级为稳定 profile。下一步可做 `stable4096` 的压力变体，
  例如 `job_rows=4` 或更低 retry window，但应保持与默认 stable 回归分离。

## 2026-07-09 4096 Job-Rows=4 Pressure Window

- `job_rows=4` 会把 direct row-major sub-job 从 stable4096 的 32KiB
  扩大到 64KiB；在同样 `band_cores=8` 下，active band cores 同时发更大的
  DMA read burst。
- `retry_ticks=384` 的初始 pressure run 失败：
  `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_pressure4096_jr4_real`。
  失败不是 softmax mismatch，而是 guest DMA load guard 报
  `DMA_LOAD_FAILED direct row-major sub-job`。DMA 汇总：
  `dma_timeout_retry_sum=520`、`timeout_exhausted_sum=32`。
- `retry_ticks=512` 仍失败：
  `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_pressure4096_jr4_rt512_real`。
  它能推进到更靠后的 sub-job，但仍有
  `dma_timeout_retry_sum=839`、`timeout_exhausted_sum=32`，最终同样触发
  guard failure。
- `retry_ticks=1024` clean PASS：
  `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_pressure4096_jr4_rt1024_real`。
  manifest：
  `sfu_job_direct_pressure4096_jr4_rt1024,4096,8,4,1024,8,pass,PASS,0,2400`。
  verifier：
  `checked=16777216,mismatches=0,max_abs_diff=5.72476014e-11`。
  DMA：
  `timeout_retry_sum=0`、`timeout_exhausted_sum=0`、read/write issue
  `4096/4096`、read/write completion `4096/4096`。
  timing：
  `simulated_time=3.05433 ms`、`wall_time_sec=700`。
- 决策：
  4096 `job_rows=4` throughput pressure path 可行，但 clean profile 需要
  `retry_ticks=1024`。`rt384` 和 `rt512` 应保留为 expected-fail pressure
  points；`rt1024` 是下一轮 4096 throughput/profile 对比的基线。

## 2026-07-10 Retry Window 768 Is Clean

- `pressure4096_jr4_rt768` real SST artifact：
  `artifacts/sweeps/sfu_unified_job_direct_sweep_20260710_pressure4096_jr4_rt768_real`。
- Full verifier PASS：
  `checked=16777216,mismatches=0,max_abs_diff=5.72476014e-11`。
- DMA is fully clean：
  `timeout_retry_sum=0`、`timeout_exhausted_sum=0`、read/write issue
  `4096/4096`、read/write completion `4096/4096`。
- Timing is effectively identical to rt1024：
  `simulated_time=3.05433 ms`、`wall_time_sec=692`；active cores observed
  `max_rtt_ticks=688`。
- Decision：
  `retry_ticks=1024` is no longer the minimum known clean point；768 is the
  current clean pressure baseline. Since 512 fails and 768 has 80 ticks of
  headroom above the observed maximum RTT, test 704 next to tighten the
  zero-retry boundary without entering the obviously-too-small 640 window.

## 2026-07-13 Retry Window 704 Is Clean

- `pressure4096_jr4_rt704` real SST artifact：
  `artifacts/sweeps/sfu_unified_job_direct_sweep_20260713_pressure4096_jr4_rt704_real`。
- Full verifier PASS：
  `checked=16777216,mismatches=0,max_abs_diff=5.72476014e-11`。
- DMA remains fully clean and identical to rt768/rt1024：
  `timeout_retry_sum=0`、`timeout_exhausted_sum=0`、read/write completion
  `4096/4096`、observed `max_rtt_ticks=688`。
- Timing：`simulated_time=3.05433 ms`、`wall_time_sec=738`。
- Decision：704 replaces 768 as the minimum known zero-retry clean point.
  The next boundary probe should test 688 directly to determine equal-tick
  timeout ordering, or 696 if the goal is a small explicit safety margin.

## 2026-07-13 Unified Job Physical Multi-SFU Cooperation

- Phase 2 的 `worker_cores` 只在单个 SFU 内顺序遍历 logical slices；它能验证
  数学分块，但不是多个物理 core/SFU 协作。Phase 3A 新增 distributed-columns
  descriptor 语义后，每个物理 SFU 只持有并计算一个 compact column slice。
- reducer 由 SFU 管理，最终按 `(job_id, tag, owner_core, row)` 保存各
  `worker_slot` 的 local max/sum，
  `issueJob()` 提交 local max 后保持 Pending，`wait()` 分阶段推进 sum 和
  normalize。所有 worker 完成后清理 row reducer entry。
- 第一次 `run_20260713_145403_2376829` 虽然 golden PASS，但 guest 日志显示
  `distributed_columns=0, band_cores=1`，只有一个 SFU 有计数。根因是旧 archive
  shim 没有转发新 env；该结果不能作为 distributed 证据。
- env 转发修复后，`run_20260713_145840_2416014` 四个 guest 都进入 distributed
  path，但 descriptor 返回 `InvalidShape(5)`、输出全零。verbose 诊断进一步在
  `run_20260713_150559_2473003` 发现 descriptor 的 `worker_cores=4` 正确，而
  SFU 内 `active_workers=1`。
- `active_workers=1` 的根因是 RoCC parent 没收到 `active_worker_cores` 参数，
  `setCoreInfo(... params.find("active_worker_cores", 1))` 用默认 1 覆盖了 SFU
  child 的正确配置。`cpu_builder.py` 同时向 RoCC parent 和 SFU child 传参后修复。
- canonical real-SST run：`run_20260713_150940_2502347`，配置
  `rows=4, dim=64, chunk=16, worker_cores=4, band_cores=4`。四个 participating
  SFU 各自记录 `ops=1, rows=4, max/sum/norm chunks=4/4/4,
  partial_submits=8, partial_done=4`，其余 12 个 SFU 均为 0。
- DMA 全局 `read/write issue=16/16`、`completion=16/16`、读写各 1024 bytes，
  `timeout_retry=0, timeout_exhausted=0`。golden 检查 256 个 FP32 值，
  0 mismatch，`max_abs_diff=2.71942902e-09`。
- 这证明 unified job 已有真实 physical multi-SFU functional cooperation；当前
  reducer 仍是 SST 内 shared state，没有通过 SimpleNetwork 发送 partial，故
  不能用本次 simulated time 推断真实 NoC reduction 成本。

## 2026-07-13 Distributed Reducer Reliability Review

- 初版 `(job_id, local_row)` key 无法为未来的并发同 shape job 和多个
  cooperative group 提供充分隔离。最终 key 改为
  `(job_id, tag, owner_core, local_row)`；tag 隔离 sub-job，owner 隔离 group。
- descriptor validator 现在验证 physical membership：当前 SFU core 必须位于
  `[owner_core, owner_core + worker_cores)`，且
  `coreId - owner_core == worker_slot`。这让 `owner_core` 从日志字段变为协议约束。
- 新增 `SFU_JOB_FLAG_DISTRIBUTED_ABORT`。DMA guard 失败的 worker 仍通过相同
  unified job API 发 abort descriptor；reducer tombstone 使已提交或等待中的
  peers 返回错误，而不是永久 Pending。`abortSeen` 在所有 physical workers
  观察后清理 tombstone；`distributedAbortObserved` 防止最后观察者擦除后又由
  通用错误路径二次创建。
- duplicate in-flight tag 不再覆盖 `pendingJobOps_[tag]` 或重复增加 inflight；
  它 poison 该冲突 tag identity，并将已有 operation 标记为
  `InvalidDescriptor`，让同 hart 后续 `wait` 可以有限退出。
- abort 和 metadata mismatch 会扫描同一 `(job_id, tag, owner_core)` cohort
  已存在的全部 row key；最后一个 observer 清理后不会因为 abort creation loop
  再创建 tombstone，避免 shape 不一致留下永久 Pending 或 reducer 泄漏。
- review 随后发现仅扫描现有 key 仍不足：若两个 worker 的 `rows` 分别为 1 和
  2，多出的 row 在没有任何 error 的情况下会永远等缺失 worker。最终每个 row
  state 记录 canonical `expectedWorkers/expectedRows/expectedCols`，第一个 local
  max submission 即检查完整 cohort shape；不匹配时进入现有 abort/cleanup 路径。
- distributed direct path 的 local-GM staging 现在按当前 worker 的 compact
  slice 宽度计算，而不是仍按完整 `dim` 计算。这消除了大维度 physical
  column split 被假容量检查提前拒绝的问题。
- final positive run：`run_20260713_162719_3103304`，golden 256 values、
  0 mismatch、`max_abs_diff=2.71942902e-09`。带完整计数的 post-review run
  `run_20260713_155053_2817319` 中四核各
  `ops=1, rows=4, max/sum/norm=4/4/4, partial_submits=8,
  partial_done=4, retry_events=0`；DMA 16/16 read/write、0 retry/exhausted；
  golden 结果相同。
- final negative run：`run_20260713_162857_3116951`，`retry_ticks=1,
  max_retries=0`。一个 worker 报 `DMA_LOAD_FAILED`，另外三个 worker 的
  `sfu_job_wait` 返回 status 2；simulation 有限完成，wrapper 返回 1。
- negative run 同时确认旧 failure detector 的假成功根因：base pipeline 子
  shell 内的 `GOLEM_RUN_ID` 不会回传 wrapper。wrapper 现在从
  `stats/run_summary.csv` 最后一行恢复 run id，再只扫描本次 stdout；显式设置
  `GOLEM_RUN_SUMMARY_CSV` 时使用该 override。

## 2026-07-13 Unified Job Distributed Scaling Matrix

- 新增独立 `run_sfu_unified_job_distributed_scaling.sh`。没有修改旧 direct
  sweep，因为旧脚本属于单 SFU logical-worker profile，且 rows 与 dim 绑定。
- 固定架构参数：`rows=16, staging_rows=4, job_rows=4, chunk=256,
  retry_ticks=1024, max_retries=8`。四个 4-row bands 让
  `band_cores/worker_cores` 真正形成 1、2 或 4 个 cooperative groups。
- real artifact：
  `artifacts/sweeps/sfu_unified_job_distributed_scaling_20260713_real`。
- 512 四点全部 PASS：`W4/BC4/G1`、`W4/BC16/G4`、`W8/BC16/G2`、
  `W16/BC16/G1`。每点 golden checked 8192、0 mismatch，
  `max_abs_diff=3.90064624e-10`。
- 1024 四点采用同一 worker/group 组合，全部 PASS；每点 golden checked
  16384、0 mismatch，`max_abs_diff=2.05596251e-10`。
- simulated time（us）：512 为 `407.503, 345.521, 366.589, 410.959`；
  1024 为 `407.366, 343.915, 365.365, 411.846`，顺序均为上述四个组合。
- DMA read/write issue 与 completion 均严格等于 `rows * worker_cores`：W4/W8/W16
  分别为 64/128/256；读写 bytes 分别为 512 的 32768 和 1024 的 65536；
  timeout retry、exhausted 和 write retry 全部为 0。
- runner 同时校验 active SFU 数等于 band cores、max/sum/norm stage counters、
  partial submit/done totals，并仅在全部通过后写 `.pass` marker。
- 首次校准在 SST 前因 `GOLEM_SKIP_BUILD=1` 与旧 4x64 base workload metadata
  不匹配而停止。修复为 shape-specific rebuild，并把 tensor/HBM/output 全部限定
  在 sweep root；这不是架构或 softmax failure。
- 这些 timing 只证明当前 shared-SST reducer 下的调度扩展性，不能解释为显式
  NoC reduction 性能。正式性能比较仍需 Phase 4 NoC reducer traffic。

## 2026-07-13 Distributed Scaling Runner Reliability

- Bash 的 `set -e` 在函数作为 `if` 条件执行时不会保证函数体内命令失败就退出。
  因此 artifact validator 必须对每个 metric extraction 和 equality check 显式
  `return 1`，不能把 errexit 当作 correctness contract。
- completion marker 不能只用 run id。当前 run id 不包含 chunk、staging rows、
  job rows、retry policy 和可覆盖固定 CLI 的 pipeline args；marker 改为存储完整
  配置签名，并在缓存命中时重新跑 artifact validator，才能防止参数变化、stats
  损坏或目录被删后的 stale PASS。
- wrapper 的 golden PASS 只发生在 real run 当时。缓存要继续代表该 golden 结果，
  marker 还必须记录输出 tensor SHA-256；重入时同时检查文件大小和 hash，才能拒绝
  被删除、截断或同尺寸篡改的输出。
- `pipeline_args` 允许重复 wrapper CLI option，而 parser 采用后出现者覆盖前值。
  因此 artifact contract 依赖的 `--softmax-c-file` 必须由 runner 在附加参数之后
  最终固定；否则真实输出和校验输出可能指向两个文件。
- sandbox 内 Open MPI 报 `No network interfaces were found` 属于执行环境限制，
  发生在 SST workload 启动之前；同一命令在 sandbox 外完成 8/8 PASS，所以该行
  应归类为 host launch failure，而不是 SFU architecture failure。

## 2026-07-13 Phase 4A Modeled Reduction Transport

- `distributed_reduction_transport=modeled_noc` 目前是 message-equivalent
  observability contract，不是 `SimpleNetwork` implementation。它在 shared
  reducer 的 max/sum request 和 ready response 边界计数，用来约束后续显式
  NoC reducer 的消息数和生命周期。
- 对 `rows=16, dim=512, worker_cores=4, band_cores=4`，每个 active SFU 处理
  16 个 worker-row，因此四个新统计项在 core0-3 上各为 16，total 都应为
  `rows * worker_cores = 64`。这与 `partial_submits=128`、`partial_done=64`
  的关系一致：max 和 sum 各一次 submit，所以 partial submit 是 response 行数的
  两倍。
- real smoke `sfu_unified_job_modeled_noc_smoke_20260713_rerun` 证明：
  golden checked 8192、0 mismatch，reduction request/response stats 出现在
  `stats_selfcom.txt` 且 artifact validator 接受，DMA retry/exhausted/write
  retry 全部为 0。
- reviewfix 后的 canonical smoke 为
  `sfu_unified_job_modeled_noc_smoke_20260713_reviewfix`，同样 manifest
  `PASS,0,...,artifact_validation=PASS`、golden checked 8192、0 mismatch、
  四个 active SFU 的 reduction request/response stats 均为 16。
- distributed scaling runner 是 Phase 4A modeled-NoC observability runner，
  不是 shared reducer functional runner。若
  `GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=shared`，入口会 exit 2，因为 shared
  模式按设计不记录 message-equivalent counters，继续运行只会产生误导性的
  artifact validation failure。
- SFU runtime 对未知 `distributed_reduction_transport` 字符串 fatal；这能把
  `modelled_noc` 这类拼写错误变成配置错误，而不是静默变成 `shared` 后得到
  zero-counter 产物。
- 如果修改了 `src/sst/elements/golem/sfu/*` 后只重跑 workload 而没有重新链接
  `build/sst-elements/src/sst/elements/golem/.libs/libgolem.so`，可能出现
  softmax golden PASS 但新增 SFU stats 完全缺失的假失败。遇到这种情况应先执行
  `make -C build/sst-elements/src/sst/elements/golem -j4`，并确认 `.libs/libgolem.so`
  中包含新 statistic 字符串，再重跑真实 SST。
- 下一阶段不能再只比较 simulated time；需要把当前四个 message-equivalent
  counters 映射到真实 NoC request/response event，开始观察 explicit reduction
  latency、contention 和队列压力。

## 2026-07-13 Phase 4B Queue Boundary Decision

- The Phase 4B first cut is deliberately not a `SimpleNetwork` implementation:
  `explicit_noc` introduces typed max/sum request and response FIFO messages,
  but drains them to the existing shared reducer so golden behavior remains the
  reference. This establishes a replaceable transport boundary before later
  latency, contention, and queue-pressure modeling.

## 2026-07-13 Phase 4B Explicit-NoC Queue Smoke

- `explicit_noc` is now a working queue-backed transport boundary, not just a
  parser value. It enqueues/drains typed max/sum request messages into the
  existing shared reducer and queues reducer-ready values as responses before
  row state consumes them. It does not yet emit SST `SimpleNetwork` reduction
  events, so it must not be interpreted as a latency, contention, or queue
  pressure model.
- Real artifact `tests/artifacts/sweeps/sfu_unified_job_explicit_noc_smoke_20260713`
  passed the `16:512:4:4` smoke with artifact validation, logits golden
  `checked=8192, mismatches=0, max_abs_diff=3.90064624e-10`, and each active
  core0-core3 reporting 16 for all four reduction request/response counters.
  Per-kind total is 64, exactly `rows * worker_cores`; DMA read/write issue and
  completion are 64/64 with zero retry or exhaustion.
- The next Phase 4B cut should replace `drainDistributedReductionMessages()`'
  immediate shared-reducer handoff with a real SST NoC event path while keeping
  this artifact contract unchanged.
- Final review found no Phase 4B defect. Residual risk is intentional: immediate
  FIFO drain cannot model NoC latency, contention, backpressure, or delayed
  multi-message delivery; those must be validated only after SimpleNetwork
  integration.

## 2026-07-14 SimpleNetwork and Runner Findings

| Observation | Evidence and operational rule |
|---|---|
| Real explicit reduction transport works | The `16:512:4:4` anchor passes golden `8192/0`; four SFU max/sum request/response totals are 64, exported SFU transport receive total is 256, and GlobalMemory runtime diagnostics are immediate/queued/received `256/0/256`. |
| Rebuild scope includes memHierarchy | `memHierarchy` exports a GlobalMemory ELI builder through `memNICBase.h`; after changing GlobalMemory ELI metadata, rebuild/install `memHierarchy` as well as `golem`, otherwise old metadata can override the current `libgolem.so` at runtime. |
| ELI aggregation is explicit | `golem.cc` must include `globalmemory/globalmemory.h`; compiling only `globalmemory.cc` cannot refresh the library's registered metadata. |
| Exported SFU stats are the artifact contract | Nested GlobalMemory statistics are not present in `stats_selfcom.txt`; validate explicit transport using `sfu_reduction_transport_received == 4 * rows * worker_cores`. Keep GlobalMemory counters as diagnostics, not required CSV rows. |
| Original GEMM baseline is default-path only | Use `run_noc_dma_pipeline.sh` with SFU/softmax variables unset and no architecture/group/control/WCP overrides. `architecture/ncores_selfcom_dma_ctrl.py` is the SST topology script; the guest remains `small/mvm_noc_int_array/riscv64/test_noc_dma`. |
| Generic RISC-V/SST runner must be self-contained | Non-interactive shells do not guarantee toolchain, `sst`, or `libpython3.13` discovery. The generic runner now configures the RISC-V musl toolchain, absolute SST executable, and documented Conda/SST library path while preserving caller overrides. |
