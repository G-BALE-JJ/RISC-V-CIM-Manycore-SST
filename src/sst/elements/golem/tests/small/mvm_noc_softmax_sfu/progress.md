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

## 2026-07-01

- `512x512` 默认 ctrl-link/WCP 配置首次运行在 SST 启动前先遇到
  `libpython3.13.so.1.0` 动态库路径问题；显式补
  `LD_LIBRARY_PATH=/data4/jjgong/miniconda3/lib:...` 后进入 SST，但 30 分钟内
  未完成，日志停在 `[3/4] Running SST...`，无 verifier 结果。
- `512x512` no-ctrl 基线通过：
  - 配置：`GOLEM_GROUP_MANAGER_ENABLE=0`、`GOLEM_CTRL_LINK_ENABLE=0`、
    `GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0`、`16` worker cores。
  - 结果：`PASS checked=262144 mismatches=0 max_abs_diff=1.43280396e-09`。
  - 日志：`artifacts/logs/sfu_softmax_512x512_noctrl_20260701.log`。
- `1024x1024` no-ctrl 基线完成 SST 仿真但 golden 失败：
  - 配置同 no-ctrl，额外设置 `GOLEM_SFU_MAX_INFLIGHT=32`。
  - SST wall time 约 `5314s`，simulated time `23.888 ms`。
  - verifier：`FAIL checked=1048576 mismatches=62044 max_abs_diff=238`。
  - DMA 统计显示 `timeout_exhausted_sum=3969`，说明该规模下 DMA 读完成/重试
    已成为明显瓶颈和正确性风险。
- 原始 GEMM pipeline `1024x1024x64 fp32 --verify-c` 已完成对照验证：
  - 配置：默认 `GROUP_MANAGER=1`、`CTRL_LINK=1`、`WCP=1`，`GOLEM_SFU_ENABLE=0`。
  - 结果：`PASS sampled=1024 mismatches=0 max_abs_diff=0`。
  - SST wall time `85s`，simulated time `233.434 us`。
  - 对照结论：GEMM 本身不是 1024 规模长时间运行的原因；SFU no-ctrl 慢主要来自
    no-ctrl worker-side DMA 路径与 retry/exhausted。
- 执行方案 A：新增 `GOLEM_SFU_INTERLEAVE_GEMM=1` 诊断开关，让 SFU 直接消费
  GEMM 的 `rt.local_accum`，跳过 softmax 阶段 C tile HBM reload。
  - 新增 runtime API：`golemRunSoftmaxSfuTileFromLocalAccum` 和
    `golemWaitSoftmaxSfuTileAndStore`。
  - 修复 no-ctrl archive shim，让 `GOLEM_SFU_INTERLEAVE_GEMM` 进入 Vanadis guest
    环境；`128x128` stdout 已确认 `mode=sfu-interleaved-local-accum`。
  - Python tests：17-test targeted suite PASS，43-test discovery suite PASS。
  - RISC-V workload rebuild PASS。
- 方案 A 验证结果：
  - `128x128` interleaved PASS，`read_issue` 从默认路径 `516` 降到 `512`，
    只减少 4 个 C tile reload。
  - `512x512` interleaved SST 仿真完成但窗口中断了 wrapper 收尾；使用保留的 HBM
    output 离线 `unpack + verify` 后 PASS，`checked=262144 mismatches=0`。
  - `512x512` no-ctrl baseline 到 interleaved 的差异：
    `read_issue 8256 -> 8192`，`read_bytes 69206016 -> 68157440`，
    `write_issue/write_bytes` 不变。
  - 结论：C tile reload 可消除但不是主瓶颈；主要瓶颈是 no-ctrl GEMM A/B DMA
    数据路径。下一阶段 B 应围绕数据通路优化，而不是只继续调 SFU inflight。
- 澄清后续方向：
  - no-ctrl 是 `GROUP_MANAGER=0 / CTRL_LINK=0 / WCP=0` 的 correctness/debug
    基线，不是最终性能标准。
  - group manager、ctrl-link、WCP 属于控制面/调度面，早期关闭它们是为了隔离
    SFU descriptor、RoCC 指令、local GM、online reducer 和 golden checker 问题。
  - standalone softmax 输入在 HBM 时必须读 HBM；优化目标不是消除 HBM read，
    而是建立 softmax-only benchmark 并优化必要 HBM->SFU read/write 的效率。
  - fused GEMM+softmax 是另一条路线，才应重点考虑 local-accum handoff 和避免
    中间 C tile 写回/读回 HBM。
  - 下一步优先新增 standalone softmax-only workload/runner，再用 `64/128/512`
    规模区分 SFU 自身 HBM 数据路径瓶颈和 GEMM+softmax 混合路径瓶颈。
- 完成 standalone softmax-only benchmark bring-up：
  - 新增 `GOLEM_SFU_STANDALONE_SOFTMAX=1` guest 模式，在 workload 中跳过 GEMM，
    直接执行 SFU softmax。
  - `gen_hbm_init.py` 支持 `GOLEM_SOFTMAX_LOGITS_FILE` / `--softmax-logits-file`，
    可生成或读取 logits，并按 GEMM C `colmajor_tile` layout 预置到 HBM C tile 区域。
  - wrapper 在 standalone 模式下默认生成 `data/softmax_logits.bin`，并将 verifier
    reference 自动切到 `logits`。
  - `verify_softmax_sfu_against_golden.py` 新增 `--reference logits`。
- standalone 验证结果：
  - 目标单测 `python3 -m unittest test_sfu_workload_scaffold.py
    test_run_noc_dma_softmax_sfu_pipeline.py test_verify_softmax_sfu_against_golden.py`
    通过，`28 tests`。
  - HBM init/unpack 离线 smoke：`8x8` logits 预置后解包最大差 `0.0`。
  - RISC-V workload clean rebuild 通过。
  - `128x128` standalone 真实 SST 通过，verifier：
    `PASS reference=logits checked=16384 mismatches=0 max_abs_diff=9.29062844e-10`。
  - 该 run 的 DMA 统计为 `read_issue=4`、`write_issue=4`、
    `read_bytes=65536`、`write_bytes=65536`、`timeout_retry=0`、
    `timeout_exhausted=0`，SST wall time `46s`、simulated time `204.533 us`。
  - `512x512` standalone 真实 SST 通过，配置中 `K=64` 仅作为复用 GEMM/HBM
    layout 的占位参数，softmax 输入实际是 `512x512` logits。
  - `512x512` verifier：
    `PASS reference=logits checked=262144 mismatches=0 max_abs_diff=2.30018935e-10`。
  - `512x512` DMA 统计为 `read_issue=64`、`write_issue=64`、
    `read_bytes=1048576`、`write_bytes=1048576`、`timeout_retry=0`、
    `timeout_exhausted=0`，SST wall time `57s`、simulated time `224.299 us`。
  - 结论：standalone SFU 必要 HBM read/write 在 `512x512` 下没有 retry 瓶颈；
    mixed fused/no-ctrl 慢主要仍指向 GEMM A/B worker-side DMA，而不是 standalone
    softmax 的 HBM->SFU 数据路。
- 运行环境修复：SFU wrapper 显式导出 `LD_LIBRARY_PATH=$SST_SOFTMAX_LD_LIBRARY_PATH`，
  避免不走本地 `bin/sst` shim 时 `libpython3.13.so.1.0` 加载失败。
- 修复 wrapper 相对 logits 路径问题：
  - `--softmax-logits-file data/...` 现在会归一化到 wrapper 目录下的绝对路径。
  - 避免 HBM init 在 `tests/` cwd 生成 logits，而 verifier 在 wrapper cwd 查找 logits
    导致 `FileNotFoundError`。
- 重新运行 standalone `1024x1024` 压力测试：
  - 默认 `GOLEM_SFU_MAX_INFLIGHT=8` 时，SST 阶段 30 分钟超时；日志中已出现
    `DMA READ retry`。
  - 根因判断：`1024x1024`、`64x64` tile 下每 core 需要处理 `16` 个 tile，
    而 SFU 默认 inflight 只有 `8`。当前 runtime 先 issue 完本 core 全部 tile
    再统一 wait，超过 credit 后第 9 个 issue 会被 RoCC 反复重试，且无法进入
    wait 释放 credit。
  - 对照实验 `GOLEM_SFU_MAX_INFLIGHT=32`：SST 完成，wall time `88s`，
    simulated time `327.308 us`；DMA totals `read_issue=256`、`write_issue=256`、
    `read_bytes=4194304`、`write_bytes=4194304`、`timeout_retry=6`、
    `timeout_exhausted=0`。
  - 该对照 verifier 仍 FAIL：`checked=1048576`、`mismatches=57344`、
    `max_abs_diff=1.87508551`；DMA summary 显示 `write_completion=241/256`。
  - 结论：本次“1024 很慢”的直接原因是 standalone SFU issue/wait 缺少有界节流，
    不是必要 HBM read/write 流量本身；但正确性还需要补写回完成/drain 语义。
- 实现 standalone SFU row-band issue/wait 窗口：
  - 先新增静态回归测试
    `test_standalone_softmax_uses_row_band_issue_wait_window`，初始 RED，失败于
    runtime 缺少 `SFU_SOFTMAX_ISSUE_WINDOW_TILES` / row-band 调度。
  - 修改 `golem_softmax_sfu_runtime.cpp`：
    - 新增 `SFU_SOFTMAX_ISSUE_WINDOW_TILES=8`。
    - 新增 `row_band_m_tiles`、`issue_sfu_softmax_tile`、
      `wait_and_store_pending_tiles` helper。
    - `golemRunSoftmaxSfuForCore` 改为按完整 `m_tile` row-band 提交，每批提交后
      wait/store，再进入下一批，避免默认 inflight=8 下 1024 每 core 一次 issue
      16 个 tile 的 credit 阻塞。
  - 目标测试组通过：
    `python3 -m unittest test_sfu_workload_scaffold.py test_run_noc_dma_softmax_sfu_pipeline.py test_verify_softmax_sfu_against_golden.py`
    为 `30 tests OK`。
  - RISC-V workload 编译通过：`make clean ARCH=riscv64 && make ARCH=riscv64`。
  - 现有 `remote_store` 仍是 fire-and-forget，`completeRoCC(0)` 只代表写请求已发出；
    后续 SST 验证若仍有 `write_completion < write_issue`，需要新增显式写回 ack/fence。
- row-band 修复后的真实 SST 回归：
  - standalone `512x512` 默认 inflight=8 继续 PASS：
    `checked=262144 mismatches=0 max_abs_diff=2.30018935e-10`，
    wall time `71s`，simulated time `270.521 us`。
  - `512x512` DMA totals：
    `read_issue=64`、`write_issue=64`、`read_bytes=1048576`、
    `write_bytes=1048576`、`timeout_retry=0`、`timeout_exhausted=0`、
    `write_completion=64/64`。
  - standalone `1024x1024` 默认 inflight=8 已由 30 分钟超时修复为 PASS：
    `checked=1048576 mismatches=0 max_abs_diff=1.15859469e-10`，
    wall time `159s`，simulated time `525.847 us`。
  - `1024x1024` DMA totals：
    `read_issue=256`、`write_issue=256`、`read_bytes=4194304`、
    `write_bytes=4194304`、`timeout_retry=4`、`timeout_exhausted=0`、
    `write_completion=256/256`。
  - 结论：standalone softmax 的 1024 慢点已确认并修复为 SFU issue/wait 调度问题；
    必要 HBM read/write 数据量本身不是半小时级仿真的根因。下一步可以开始
    standalone primitive ABI：`EXP`、`LOG`、`RECIPROCAL`。

## 2026-07-02

## 2026-07-08 Unified SFU Job Row-band Streaming

- 继续 unified SFU job softmax 路线，没有回到 primitive/batch 主线。
- 在 standalone logits + `GOLEM_SFU_JOB_SOFTMAX=1` 路径中加入 row-band staging：
  - 新增 `GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS`，默认按 `block_m` 分 band。
  - 每个 band 从 HBM C tile-packed logits 读取到 executor core local GM 的
    row-major staging buffer。
  - 每个 band 通过 `SFU_JOB` unified softmax job 执行，再 patch 回对应 HBM C tiles。
  - local GM 需求从整矩阵 `M*N*2` 降为 `staging_rows*N*2 + one_tile`。
- 修复 row-band unified job 的 shape 校验问题：
  - band descriptor 使用 `band_desc.outer = row_band_rows`。
  - runtime config 派生 `band_cfg = cfg`，并设置 `band_cfg.m = row_band_rows`。
  - 避免 `op_desc.outer=64` 但 `cfg.m=512` 时 `validate_sfu_softmax_request`
    直接失败。
- 验证：
  - `python3 -m unittest .../test_sfu_workload_scaffold.py -v`：`41 tests OK`。
  - `python3 -m unittest .../test_run_noc_dma_softmax_sfu_pipeline.py -v`：
    `23 tests OK`。
  - `make clean ARCH=riscv64`、`make ARCH=riscv64`：PASS。
  - 真实 SST `128x128` standalone unified job row-band：
    `PASS reference=logits checked=16384 mismatches=0 max_abs_diff=1.64943281e-09`。
    stdout 确认 `dispatch=sfu-standalone-unified-job-softmax`、
    `staging_rows=64`、`worker_cores=16`。
    DMA totals：`read_issue=8`、`write_issue=4`、`read_bytes=131072`、
    `write_bytes=65536`、`timeout_retry=0`、`timeout_exhausted=0`。
  - 真实 SST `512x512`、`staging_rows=64` 在 900s wall timeout；与修复前不同，
    不再出现 row-band job shape mismatch 立刻失败，但 executor core 未输出 PASS。
- 下一步：保留 unified job 路线，诊断 `512x512` row-band 的长时等待点；优先加
  per-band progress/timeout instrumentation，再决定是增大 staging_rows、复用 local tile
  load/store、还是把 executor-side band loop 扩展为真正多 core 分 band 协同。
- 继续推进 unified job row-band cooperative execution：
  - 新增 `GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS`，可按需打印
    `band_stage=load/job/store/done`；实际 512 timeout 中 stdout 仍要等 pid 退出，
    因此该 trace 更适合可完成规模或后续配合 emergency dump 使用。
  - 新增 `GOLEM_SFU_JOB_SOFTMAX_BAND_CORES`，让多个 requested cores 分摊 row-band：
    `band_index % band_core_count == requested_core_id` 的 core 执行该 band。
  - cooperative 模式要求 `staging_rows` 覆盖完整 m-tile，避免多个 core patch 同一个
    C tile 造成写回 race。
  - 对完整 tile band 的 store 阶段增加 fast path：不再先 reload HBM C tile，
    直接把 row-band output pack 成 tile 后写回；partial band 仍保留 reload+patch。
- 本轮验证：
  - `python3 -m unittest .../test_sfu_workload_scaffold.py -v`：`42 tests OK`。
  - `python3 -m unittest .../test_run_noc_dma_softmax_sfu_pipeline.py -v`：
    `23 tests OK`。
  - `make ARCH=riscv64`：PASS。
  - 真实 SST `128x128`、`BAND_CORES=2`：
    `PASS reference=logits checked=16384 mismatches=0 max_abs_diff=1.64943281e-09`。
    stdout 中 core0/core1 均输出 `band_cores=2 PASS`。
    DMA totals 从旧 row-band `read_issue=8, write_issue=4` 降为
    `read_issue=4, write_issue=4`，说明 full-tile store fast path 生效。
    wall time 从旧 row-band `215s` 降到 `156s`，simulated time 从 `1.06004 ms`
    降到 `728.187 us`。
  - 真实 SST `512x512`、`BAND_CORES=8` 在 `600s` 仍 timeout；log 显示
    pid `100-107` 保持运行、`108-119` 已退出，说明 8 个 active band cores 确实参与，
    但每个 512-wide band 自身仍很重。下一步应继续减少每 band 的 HBM/SFU job
    固定开销，或把 unified job 内部 column chunks 进一步外显为多 core 协作。
- 继续推进 staging-band 内 sub-job row streaming：
  - 新增 `GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS`，默认等于
    `GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS`，保持旧行为不变。
  - staging band 仍按完整 m-tile 读取和最终写回，以维持 multi-core cooperative
    模式下的 tile ownership；但在 local GM row-major staging buffer 内，按
    `job_rows_per_issue` 拆成多个 `SFU_JOB`。
  - 每个 sub-job 使用 `input_gm + sub_job_offset_bytes` 和
    `output_gm + sub_job_offset_bytes`，`sub_desc.outer=sub_job_rows`。
  - 修复真实 SST 中暴露的 validator 问题：sub-job runtime config 需要
    `sub_cfg.block_m=sub_job_rows`，否则 `m=16, block_m=64` 会被
    `validate_sfu_softmax_request` 拒绝，导致输出仍是原始 logits。
  - sub-job 失败日志现在会附带 `golemSoftmaxSfuGetLastErrorString()`，方便后续
    直接看到 runtime validator 或 SFU wait status。
- 本轮 sub-job 验证：
  - RED：新增测试最初失败于缺少 `GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS`、
    sub-job GM offset、`band_stage=subjob`；真实 SST 128 probe 又暴露
    `sub_cfg.block_m` 缺失。
  - GREEN：`python3 -m unittest .../test_sfu_workload_scaffold.py -v`：
    `43 tests OK`。
  - GREEN：`python3 -m unittest .../test_run_noc_dma_softmax_sfu_pipeline.py -v`：
    `23 tests OK`。
  - `make clean ARCH=riscv64`、`make ARCH=riscv64`：编译通过。
  - 真实 SST `128x128,BAND_CORES=2,JOB_ROWS=16`：
    `PASS reference=logits checked=16384 mismatches=0 max_abs_diff=1.64943281e-09`；
    run id `run_20260708_204743_3059548`，wall `155s`，simulated `739.614 us`。
  - 真实 SST `512x512,BAND_CORES=8,JOB_ROWS=16`：
    外层命令在 `720s` 超时，run id `run_20260708_205112_3064723`；但检查 SST log
    可见 `all process have exited`，simulated time `2.00541 ms`。
  - 对该 512 run 的 HBM output 做离线 unpack + logits verifier：
    `PASS reference=logits checked=262144 mismatches=0 max_abs_diff=3.90064624e-10`。
    因此 sub-job row streaming 已经把 512 correctness 跑通；剩余问题是 wrapper
    wall-time/收尾窗口，以及 stdout 在长 run 中不及时落盘。
  - 结论：下一步不应继续只在 guest 侧细拆 row jobs；应把 unified `SFU_JOB`
    的 softmax row 执行进一步下沉为 SST/RoCC 侧可推进队列，或实现真正的
    column-chunk cooperative execution，减少每个 active core 的总 guest issue/wait
    与 globalMem round trip 成本，同时给 wrapper 增加更稳的 completed-run
    offline verification recovery。
- 完成 completed-run offline verification recovery：
  - wrapper 新增 `GOLEM_SFU_RECOVER_COMPLETED_RUN` / `--recover-completed-run`。
  - recovery 分支在 SFU binary build 与 base SST pipeline 之前执行，复用统一的
    `run_sfu_softmax_offline_verify()`，从现有 HBM dump 直接 unpack C tensor 并跑
    full-row softmax verifier。
  - `--dry-run` 下会打印两条 Python verifier 命令但不访问 HBM，便于快速检查参数。
  - 注意：从 repo 根目录启动 wrapper 时，默认 preset 会把 `GOLEM_TENSOR_DIR`
    设为 `$PWD/data`；恢复 standalone logits run 时应显式设置
    `GOLEM_TENSOR_DIR=.../small/mvm_noc_softmax_sfu/data`，或显式传
    `--softmax-logits-file`。
- recovery 验证：
  - RED：新增 wrapper 测试最初失败于缺少 recovery knob、CLI 和可复用 verifier
    函数。
  - GREEN：`python3 -m unittest .../test_run_noc_dma_softmax_sfu_pipeline.py -v`：
    `24 tests OK`。
  - GREEN：`python3 -m unittest .../test_sfu_workload_scaffold.py -v`：
    `43 tests OK`。
  - 对 `run_20260708_205112_3064723` 留下的 512 HBM dump 运行 wrapper recovery：
    `GOLEM_SFU_RECOVER_COMPLETED_RUN=1` 后成功 unpack
    `softmax_sfu_c_out_512_recovered.bin`，并得到
    `[VERIFY-SFU-SOFTMAX] PASS reference=logits dtype=fp32 checked=262144
    mismatches=0 max_abs_diff=3.90064624e-10`。
- 完成 unified job Phase 2B column-coop observability：
  - 新增计划：
    `docs/superpowers/plans/2026-07-08-sfu-unified-job-phase2-column-coop-observability.md`。
  - 确认当前 `SFU::executeSoftmaxRowJob` 已经是 SFU 内部 row-band + worker
    column slice + `chunk_elems` 三阶段执行，而不是旧的整行 functional loop。
  - 新增 SFU-side 统计：
    `sfu_job_softmax_max_chunks`、`sfu_job_softmax_sum_chunks`、
    `sfu_job_softmax_norm_chunks`，分别记录 unified softmax job 的 max pass、
    exp/sum pass 和 normalize pass chunk 数。
  - RED：新增 `test_unified_softmax_job_records_internal_chunk_pass_stats` 最初失败于
    缺少上述统计名、成员和 executor 内 `addData`。
  - GREEN：`python3 -m unittest .../test_sfu_primitive_core.py -v`：
    `16 tests OK`。
  - 回归：
    `python3 -m unittest .../test_sfu_workload_scaffold.py -v`：
    `43 tests OK`；
    `python3 -m unittest .../test_run_noc_dma_softmax_sfu_pipeline.py -v`：
    `24 tests OK`。
  - 构建：
    `make ARCH=riscv64` 在 SFU workload 目录下无需重编；
    同步 `sfu.h/.cc` 到 build tree 后
    `make -C build/sst-elements/src/sst/elements/golem -j16` 通过，仅有既有
    SST serialization deprecation warnings。
  - 512 completed-run recovery 复验：
    `[VERIFY-SFU-SOFTMAX] PASS reference=logits dtype=fp32 checked=262144
    mismatches=0 max_abs_diff=3.90064624e-10`。
- Phase 2B stats-enabled 真实 SST probe：
  - 小点 `64x64, chunk=16, workers=4, staging_rows=64, job_rows=16,
    band_cores=1`：
    - sandbox 内受 OpenMPI/OOB socket 权限限制失败；sandbox 外真实 SST 完成。
    - run id `run_20260708_214124_3115525`，simulated time `605.098 us`。
    - verifier PASS：`checked=4096 mismatches=0 max_abs_diff=3.77493008e-09`。
    - stdout 确认 `dispatch=sfu-standalone-unified-job-softmax`、
      4 个 sub-job，以及 `mode=sfu-standalone-job-softmax ... PASS`。
    - 新 stats 在 executor core3 上非零：
      `sfu_ops_issued=4`、`sfu_softmax_rows=64`、
      `sfu_job_softmax_max_chunks=256`、
      `sfu_job_softmax_sum_chunks=256`、
      `sfu_job_softmax_norm_chunks=256`。这正好对应
      `64 rows * 4 workers * 1 chunk/worker`。
  - 512 点 `512x512, chunk=256, workers=16, staging_rows=64,
    job_rows=16, band_cores=8`：
    - run id `run_20260708_214416_3119600`，在 `900s` 窗口内完整完成；
      simulated time `1.95484 ms`。
    - verifier PASS：`checked=262144 mismatches=0 max_abs_diff=3.90064624e-10`。
    - stdout 中 requested core `0..7` 均输出
      `mode=sfu-standalone-job-softmax ... band_cores=8 PASS`。
    - 新 stats 在 executor core3..10 上均非零，每个 active executor core：
      `sfu_ops_issued=4`、`sfu_softmax_rows=64`、
      `sfu_job_softmax_max_chunks=1024`、
      `sfu_job_softmax_sum_chunks=1024`、
      `sfu_job_softmax_norm_chunks=1024`。
      全局每个 pass 共 `8192` chunks，对应
      `8 active cores * 64 rows/core * 16 workers * 1 chunk/worker`。
    - DMA summary：总 read/write issue 均为 `64`，总读写各 `1048576` bytes，
      timeout retry/exhausted/write retry 均为 `0`。

- 完成 Phase 9A standalone primitive ABI 骨架：
  - 在 `sfu.h` 中新增 `SFUPrimitiveOp` 和 64 字节 `SFUPrimitiveDesc` ABI。
  - 在 `roccAnalog.h` 中预留 primitive func7：`0x19/0x1a`。
  - 在 `ex_instr.h` 中新增 `sfu_primitive` 和 `sfu_primitive_wait` guest wrapper。
  - 新增/更新静态 scaffold tests，固定 descriptor 字段、op 编码和 func7/wrapper。
- Phase 9A 当前只固定接口，不实现 primitive 数学执行，也未接 RoCC primitive dispatch。
- 验证：
  - RED：新增测试初始失败于缺少 `SFUPrimitiveOp`、`SFUPrimitiveDesc`、
    `GOLEM_ROCC_FUNC7_SFU_PRIMITIVE` 和 guest primitive wrapper。
  - GREEN：`python3 -m unittest test_sfu_descriptor_scaffold.py
    test_rocc_sfu_integration.py test_sfu_workload_scaffold.py` 通过，`23 tests OK`。
  - 目标回归：`python3 -m unittest test_sfu_workload_scaffold.py
    test_run_noc_dma_softmax_sfu_pipeline.py test_verify_softmax_sfu_against_golden.py
    test_sfu_descriptor_scaffold.py test_rocc_sfu_integration.py` 通过，`40 tests OK`。
  - RISC-V workload rebuild 通过：`make clean ARCH=riscv64` 后 `make ARCH=riscv64`。
- 下一步进入 Phase 9B：为 `EXP`、`LOG`、`RECIPROCAL` 增加 SFUAPI 方法、RoCC dispatch、
  组件侧 local GM fp32 执行和最小 primitive workload/checker。

- 完成 Phase 9B standalone unary primitive 最小实现：
  - `SFUAPI` 新增 `issuePrimitive(descAddr, tag)`，`SFU` 组件新增 primitive pending
    state、descriptor 读取、local GM fp32 input/output 读写、op 校验和 wait retire。
  - 当前支持 `EXP`、`LOG`、`RECIPROCAL`，实现使用 host C++ `std::exp`、
    `std::log` 和 reciprocal；这仍是功能模型，不是周期精确数学硬件单元。
  - `roccAnalog.h` 接入 `GOLEM_ROCC_FUNC7_SFU_PRIMITIVE=0x19` 与
    `GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_WAIT=0x1a` dispatch。
  - guest workload 新增 `GOLEM_SFU_PRIMITIVE_SMOKE=1` 模式，只在 executor core
    执行 local GM primitive smoke，覆盖 `EXP/LOG/RECIPROCAL` 并和 C `math` golden
    比较。
- 调试并修复 primitive smoke SST 超时：
  - 初次实跑在沙箱内因 OpenMPI 网络接口限制失败；沙箱外第一次 SST 超时。
  - 根因定位为架构脚本没有把 `GOLEM_SFU_PRIMITIVE_SMOKE` 透传到 Vanadis guest
    process env，guest 没进入 smoke 分支而继续默认路径。
  - 修复 `architecture/ncores_selfcom_dma_ctrl.py`，补齐
    `GOLEM_SFU_INTERLEAVE_GEMM`、`GOLEM_SFU_STANDALONE_SOFTMAX`、
    `GOLEM_SFU_PRIMITIVE_SMOKE`；同时更新 softmax archive shim，保证 no-ctrl
    路径也可透传 primitive smoke。
- Phase 9B 验证：
  - RED/GREEN：新增 primitive core/RoCC/workload/wrapper tests，先观察缺失
    API/dispatch/env 失败，再实现转绿。
  - `python3 -m unittest discover -s . -p 'test_*.py'`：`62 tests OK`。
  - RISC-V workload：`make clean ARCH=riscv64 && make ARCH=riscv64` 通过。
  - golem 组件局部构建：`make -C build/sst-elements/src/sst/elements/golem -j16`
    通过。
  - 真实 SST primitive smoke：
    `GOLEM_SFU_PRIMITIVE_SMOKE=1 ./run_noc_dma_softmax_sfu_pipeline.sh --gemm-m 64 --gemm-n 64 --gemm-k 64`
    通过；guest stdout 确认
    `[SOFTMAX] mode=sfu-primitive-smoke executor_core=7 ops=EXP,LOG,RECIPROCAL PASS`。

- 扩展 primitive smoke 到 1024x1024 逻辑规模，并修复长仿真问题：
  - 新增 `GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS` 和
    `GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS`，wrapper、architecture env 和 archive
    shim 均已透传。
  - 初版逐 chunk 发 `sfu_primitive/sfu_primitive_wait`，`65536` 元素也会超时；
    继续对照发现 `chunk_elems=8192` 卡在 begin 后，而 `chunk_elems=4` 可完成。
  - 根因拆成两点：
    1. 大块 `mm2gm/gm2mm` 在当前 Vanadis/RoCC 模型里非常慢；
    2. 小 chunk 若逐 chunk issue，会产生大量 RoCC primitive 指令，也会拖慢。
  - 修复方案：
    - primitive smoke 默认 chunk 调为 `4`，避免大块 host-memory 搬运；
    - `SFUPrimitiveDesc.flags` 新增 repeat-chunk 语义，复用 unary 未使用的

      `input1_gm_addr` 承载 logical processed elems；
    - guest 每个 op 只发一次 primitive，SFU 组件计算一个真实 chunk 用于校验，同时
      新增 `sfu_primitive_elems` 统计逻辑元素量；
    - wrapper 增加 freshness guard：如果手工编译出的 `SFU_BIN` 比 build metadata
      更新，则强制按 wrapper CFLAGS 重编，避免复用错误 `GOLEM_GLOBAL_STRIDE_BYTES`
      的旧 ELF。
  - 验证：
    - `python3 -m unittest discover -s . -p 'test_*.py'`：`67 tests OK`。
    - `make clean ARCH=riscv64 && make ARCH=riscv64`：通过。
    - `make -C build/sst-elements/src/sst/elements/golem -j16`：通过（仅既有
      SST deprecation warnings）。
    - `GOLEM_SFU_PRIMITIVE_SMOKE=1 GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS=1048576
      ./run_noc_dma_softmax_sfu_pipeline.sh --gemm-m 64 --gemm-n 64 --gemm-k 64`
      通过，run id `run_20260702_162003_2304427`。
  - 1024x1024 primitive smoke 统计：
    - guest PASS：
      `total_elems=1048576 chunk_elems=4 chunks=786432 processed_elems=3145728`。
    - wall time `52.6s`，run summary 记录 `49s`，SST simulated time
      `226.537 us`。
    - `core7:rocc:sfu,sfu_ops_issued` sum 为 `3`；
      `core7:rocc:sfu,sfu_primitive_elems` sum 为 `3145728`，即
      EXP/LOG/RECIPROCAL 各 `1048576` 个逻辑元素。
    - `sfu_credit_stalls=0`、`sfu_retry_events=0`。
    - DMA summary 全 0，说明该 smoke 是 local GM + SFU primitive 路径，不走 HBM
      DMA read/write。
- 文档整理：
  - 更新 `/data4/jjgong/programming model/golem_gemm_programming_model_notes.md`，
    新增 “SFU / Softmax / Primitive 当前进度” 章节，集中记录 fused softmax、
    standalone softmax、primitive ABI、`1024x1024` primitive smoke、当前限制和
    下一步计划。
  - 更新 `task_plan.md`，把当前阶段补充为 primitive smoke 已支持
    `GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS=1048576`、默认 `chunk_elems=4` 的逻辑规模
    验证，并明确下一步 Phase 9C primitive 扩展与 HBM streaming primitive
    benchmark 分离。

## 2026-07-03

- 完成 Phase 9C standalone unary primitive 扩展：
  - 在 SFU 组件 `validatePrimitiveDescriptor` / `executePrimitive` 中新增
    `RSQRT`、`TANH`、`SIGMOID`。
  - `RSQRT` 当前使用 `1.0f / sqrt(value)`；`TANH` 使用 `std::tanh`；
    `SIGMOID` 使用 `1 / (1 + exp(-x))`。这些仍是 SST host C++ 功能模型，
    不是 RTL 或周期精确数学硬件。
  - guest primitive smoke 新增三种 op 的输入生成、golden 计算、issue/wait、
    output 校验和 PASS 输出。
- TDD 验证：
  - 先新增测试并观察 RED：
    `python3 -m unittest test_sfu_primitive_core.py test_sfu_workload_scaffold.py`
    初始失败于缺少 `RSQRT/TANH/SIGMOID` 的组件执行和 guest smoke 路径。
  - 实现后目标测试通过：`24 tests OK`。
  - 全量小测试通过：
    `python3 -m unittest discover -s . -p 'test_*.py'`，`68 tests OK`。
  - RISC-V workload 编译通过：
    `make clean ARCH=riscv64 && make ARCH=riscv64`。
  - golem 组件局部构建通过：
    `make -C build/sst-elements/src/sst/elements/golem -j16`，仅既有 SST
    deprecation warnings。
- 真实 SST smoke：
  - 命令：
    `GOLEM_SFU_PRIMITIVE_SMOKE=1 GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS=16
     ./run_noc_dma_softmax_sfu_pipeline.sh --gemm-m 64 --gemm-n 64 --gemm-k 64`
  - run id：`run_20260703_125355_3868668`。
  - guest PASS：
    `ops=EXP,LOG,RECIPROCAL,RSQRT,TANH,SIGMOID total_elems=16 chunk_elems=4
     chunks=24 processed_elems=96 PASS`。
  - stats：`core7:rocc:sfu,sfu_ops_issued` sum 为 `6`；
    `core7:rocc:sfu,sfu_primitive_elems` sum 为 `96`；
    `sfu_credit_stalls=0`、`sfu_retry_events=0`。
  - SST simulated time：`276.651 us`；run summary wall time：`70s`。
- 固化 primitive smoke README 说明：
  - 更新 `README.md`，明确 standalone primitive smoke 的三个关键参数：
    `GOLEM_SFU_PRIMITIVE_SMOKE`、`GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS`、
    `GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS`。
  - 明确 smoke 中 `*_ELEMS` 是 logical processed element count，`chunk_elems`
    是真实 local GM 工作集；当前 smoke 用于 ABI/功能/统计验证，不是 HBM
    bandwidth benchmark。
  - 新增 `SFUPrimitiveDesc` 字段说明，记录其作为 RISC-V guest 与 SST `golem.SFU`
    组件之间 64-byte ABI descriptor 的角色。

- 完成 Phase 9D 最小 HBM streaming primitive benchmark：
  - 新增 `GOLEM_SFU_PRIMITIVE_HBM_STREAM`、`GOLEM_SFU_PRIMITIVE_HBM_ELEMS`、
    `GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS`。
  - 新路径执行 `HBM C region -> dma_remote_load_to_gm -> local GM input ->
    sfu_primitive(EXP) -> local GM output -> remote_store -> HBM C region`。
  - guest 使用从 HBM 实际读回的输入值计算 golden，因此不依赖 C region 初值为 0。
  - 当前最小版本只覆盖 `EXP`，用于验证真实 HBM read/write + SFU primitive 的
    端到端数据流；后续再扩展多 op 和更大规模。
- TDD/构建验证：
  - 先新增 RED 测试：
    `test_workload_has_hbm_streaming_primitive_benchmark_before_local_smoke` 和
    `test_wrapper_and_architectures_forward_hbm_streaming_primitive_knobs`，初始失败于缺少
    HBM stream mode 与 env 转发。
  - 实现后目标测试通过：`2 tests OK`。
  - 全量小测试通过：
    `python3 -m unittest discover -s . -p 'test_*.py'`，`70 tests OK`。
  - RISC-V workload 编译通过：
    `make clean ARCH=riscv64 && make ARCH=riscv64`。
- 真实 SST HBM streaming primitive smoke：
  - 命令：
    `GOLEM_SFU_PRIMITIVE_HBM_STREAM=1 GOLEM_SFU_PRIMITIVE_HBM_ELEMS=64
     GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS=64
     ./run_noc_dma_softmax_sfu_pipeline.sh --gemm-m 64 --gemm-n 64 --gemm-k 64
     --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64
     --group-manager-enable 0 --ctrl-link-enable 0`
  - run id：`run_20260703_131521_3969240`。
  - guest PASS：
    `mode=sfu-primitive-hbm-stream op=EXP total_elems=64 chunk_elems=64
     chunks=1 processed_elems=64 hbm_read_bytes=256 hbm_write_bytes=256 PASS`。
  - stats：executor `core7` 上 `sfu_ops_issued=1`；
    `sfu_primitive_elems=64`；`sfu_retry_events=0`。
  - DMA summary：`read_issue_count=1`、`write_issue_count=1`、
    `read_bytes_total=256`、`write_bytes_total=256`、`write_completion=1`。
  - SST simulated time：`294.77 us`；run summary wall time：`54s`。

- 扩展 HBM streaming primitive 到可配置多 op：
  - 新增 `GOLEM_SFU_PRIMITIVE_HBM_OPS`，支持逗号分隔列表，`ALL` 展开为
    `EXP,LOG,RECIPROCAL,RSQRT,TANH,SIGMOID`。
  - `tools/gen_hbm_init.py` 在 `GOLEM_SFU_PRIMITIVE_HBM_STREAM=1` 时预置 GEMM C
    region 中的 per-op input slot，每个 slot 使用正 fp32 pattern，避免 `LOG`、
    `RECIPROCAL`、`RSQRT` 读到非法 0 输入。
  - guest 为每个 op 分配独立 HBM slot，执行 `dma_remote_load_to_gm ->
    sfu_primitive(op) -> remote_store`，并在 PASS 行输出 `ops=`、
    `hbm_init_write_bytes`、`hbm_read_bytes`、`hbm_write_bytes`。
- TDD/构建验证：
  - 新增 RED 测试覆盖 HBM op parser、generator preload 和 env 转发，初始失败于缺少
    `GOLEM_SFU_PRIMITIVE_HBM_OPS`。
  - 实现后目标测试通过：`3 tests OK`。
  - 全量小测试通过：
    `python3 -m unittest discover -s . -p 'test_*.py'`，`72 tests OK`。
  - RISC-V workload 编译通过：
    `make clean ARCH=riscv64 && make ARCH=riscv64`。
- 真实 SST HBM streaming primitive all-op smoke：
  - 命令：
    `GOLEM_SFU_PRIMITIVE_HBM_STREAM=1 GOLEM_SFU_PRIMITIVE_HBM_OPS=ALL
     GOLEM_SFU_PRIMITIVE_HBM_ELEMS=16 GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS=16
     ./run_noc_dma_softmax_sfu_pipeline.sh --gemm-m 64 --gemm-n 64 --gemm-k 64
     --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64
     --group-manager-enable 0 --ctrl-link-enable 0`
  - run id：`run_20260703_132634_4023666`。
  - guest PASS：
    `ops=EXP,LOG,RECIPROCAL,RSQRT,TANH,SIGMOID total_elems=16 chunk_elems=16
     chunks=6 processed_elems=96 hbm_init_write_bytes=384 hbm_read_bytes=384
     hbm_write_bytes=384 PASS`。
  - stats：executor `core7` 上 `sfu_ops_issued=6`；
    `sfu_primitive_elems=96`；`sfu_credit_stalls=0`；`sfu_retry_events=0`。
  - DMA summary：`read_issue_count=6`、`write_issue_count=6`、
    `read_bytes_total=384`、`write_bytes_total=384`、`write_completion=6`。
  - SST simulated time：`382.396 us`；run summary wall time：`71s`。

- 完成 HBM streaming primitive all-op 三档 sweep 与组会图：
  - sweep root：
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_primitive_allops_20260703`。
  - 纳入规模：`16`、`1024`、`4096` elements/op；用户已明确本轮规模只取到这三档。
  - 三档均为 `GOLEM_SFU_PRIMITIVE_HBM_OPS=ALL`，即
    `EXP/LOG/RECIPROCAL/RSQRT/TANH/SIGMOID` 六个 primitive。
  - 结果：
    - `16`: `processed_elems=96`，HBM read/write 各 `384B`，
      `dma_read/write_issue=6/6`，simulated time `382.396 us`，wall `68s`。
    - `1024`: `processed_elems=6144`，HBM read/write 各 `24576B`，
      `dma_read/write_issue=6/6`，simulated time `2727.56 us`，wall `517s`。
    - `4096`: `processed_elems=24576`，HBM read/write 各 `98304B`，
      `dma_read/write_issue=6/6`，simulated time `1025.25 us`，wall `200s`。
  - HBM stream bytes 与理论值完全一致：
    `total_elems * 6 ops * 4 bytes * 2 directions`。
  - 因为本轮 `chunk_elems == total_elems`，每个 op 只发一次，因此 DMA event count
    和 SFU issue count 均保持 `6`；真正观察 event count scaling 需要固定较小
    `chunk_elems` 再跑 chunk sweep。
  - 已新增绘图/汇总脚本：
    `plot_sfu_hbm_primitive_sweep.py`，输出 CSV、notes、SVG、PNG、PDF。
  - 输出图与源数据：
    `figures/sfu_hbm_primitive_sweep_source.csv`、
    `figures/sfu_hbm_primitive_sweep.svg`、
    `figures/sfu_hbm_primitive_sweep.png`、
    `figures/sfu_hbm_primitive_sweep.pdf`。
  - `65536` 尝试未纳入：日志没有 guest PASS 或 `Simulation is complete`，也没有
    stdout PASS 目录；当前只作为异常长运行诊断线索，后续需要单独用减统计/单 op/固定
    chunk 设置复现定位。

- 安装隔离 Python 绘图环境：
  - venv：`/data4/jjgong/.venvs/golem-plot`。
  - 已安装：`matplotlib 3.10.9`、`seaborn 0.13.2`、`pandas 2.3.3`。
  - `plot_sfu_hbm_primitive_sweep.py` 已改为自动设置可写 `MPLCONFIGDIR`，并在
    matplotlib 后端下导出 SVG、PDF、PNG、TIFF。
- 完成固定 `chunk_elems=1024` 的 HBM primitive event-scaling 实验：
  - sweep root：
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_event_scaling_chunk1024_20260703`。
  - completed PASS 行：
    - `1024` elems/op，all-op，`chunks=6`，DMA read/write issues `6/6`，
      wait count `12`，HBM read/write bytes `24576/24576`，simulated time
      `2.72756 ms`，wall `570s`。
    - `2048` elems/op，all-op，`chunks=12`，DMA read/write issues `12/12`，
      wait count `24`，HBM read/write bytes `49152/49152`，simulated time
      `5.12498 ms`，wall `930s`。
  - 结论：固定 chunk 后，HBM bytes、DMA issue、SFU issue、wait count 均随
    chunk 数线性增长；这比上一轮 `chunk_elems == total_elems` 更能体现 event scaling。
  - 输出图与源数据：
    `figures/sfu_hbm_primitive_sweep_source.csv`、
    `figures/sfu_hbm_primitive_sweep.svg`、
    `figures/sfu_hbm_primitive_sweep.png`、
    `figures/sfu_hbm_primitive_sweep.pdf`、
    `figures/sfu_hbm_primitive_sweep.tiff`。
- 完成异常长运行诊断：
  - `4096` elems/op，all-op，`chunk_elems=1024`，`1000s` 上限内未 PASS；
    emergency shutdown 时 core7 DMA read/write issues 为 `17/16`，预期 `24/24`。
  - `65536` elems/op，`EXP` 单 op，`chunk_elems=1024`，减统计，`600s` 上限内未 PASS；
    emergency shutdown 时 core7 DMA read/write issues 为 `9/8`，预期 `64/64`。
  - 当前根因判断：长 wall time 主要来自每 chunk 的 guest/SST 指令级执行成本，而不是
    all-op 数学、HBM preload 或全量统计。后续大规模点应增大 `chunk_elems`，或设计
    batched primitive descriptor 来摊销 per-chunk guest overhead。

- 完成 `65536` single-op `EXP` 大 chunk 诊断：
  - sweep root：
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_largechunk_diag_20260703`。
  - `chunk_elems=8192` 真实 SST PASS：
    `chunks=8`、`processed_elems=65536`、HBM read/write bytes
    `262144/262144`、DMA read/write issues `16/16`、wait count `32`、
    simulated time `2.09856 ms`、wall `398s`。
  - `chunk_elems=4096` 真实 SST PASS：
    `chunks=16`、`processed_elems=65536`、HBM read/write bytes
    `262144/262144`、DMA read/write issues `16/16`、wait count `32`、
    simulated time `2.15259 ms`、wall `422s`。
  - `4096` 与 `8192` 的 DMA issue count 相同是预期行为：`4096 fp32=16KiB`
    正好一个 DMA burst，`8192 fp32=32KiB` 被拆成两个 16KiB burst。
  - 已新增 chunk 诊断绘图脚本：
    `plot_sfu_hbm_chunk_diag.py`。
  - 输出组会图与源数据：
    `figures/sfu_hbm_exp65536_chunk_diag_source.csv`、
    `figures/sfu_hbm_exp65536_chunk_diag_notes.md`、
    `figures/sfu_hbm_exp65536_chunk_diag.svg`、
    `figures/sfu_hbm_exp65536_chunk_diag.png`、
    `figures/sfu_hbm_exp65536_chunk_diag.pdf`、
    `figures/sfu_hbm_exp65536_chunk_diag.tiff`。
  - 结论更新：把 `chunk_elems` 从 `1024` 提高到 `4096/8192` 后，
    `65536` single-op 由 timeout 变为完整 PASS，进一步确认瓶颈是 per-chunk
    guest/SST executor overhead；由于当前 primitive chunk cap 是 `8192`，
    下一步更应做 batched primitive descriptor，而不是继续单纯增大 chunk。

- 完成 batched primitive descriptor v1：
  - 新增 32B `SFUPrimitiveBatchDesc`，指向一组既有 64B
    `SFUPrimitiveDesc`；单 descriptor ABI 与 `sfu_primitive(desc, tag)`
    保持不变。
  - 新增 RoCC func7 `0x1b/0x1c`，guest wrapper
    `sfu_primitive_batch()` / `sfu_primitive_batch_wait()`。
  - `GOLEM_SFU_PRIMITIVE_HBM_BATCH=1` 使 HBM streaming primitive 把多个 chunk
    descriptor 合并成一次 batch issue。
  - SFU 统计语义：batch issue 计为一次 `sfu_ops_issued`，
    `sfu_primitive_elems` 累加所有 child descriptor 的元素数。
  - TDD/回归验证：
    `python3 -m unittest discover -s . -p 'test_*.py'`，75 tests OK。
  - 构建验证：
    `make -C build/sst-elements/src/sst/elements/golem -j16 V=1` 通过；
    `make -B ARCH=riscv64` 通过。
  - 构建边界修正：build tree 是源码拷贝，不是 symlink；新增虚函数后必须同步
    `sfu.h/.cc` 和 `roccAnalog.h` 到 build tree，并重编 `sfu.lo` 与 `golem.lo`。
    只重编 `sfu.lo` 曾导致旧 `golem.o` 使用旧 SFU vtable，在 SST wire-up 阶段
    崩到 `SFU::wait`；重编 `golem.lo` 后问题消失。
  - 真实 SST batch smoke：
    sweep root
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_batch_smoke_20260703`。
    `sfu_hbm_batch_exp_elems_1024_chunk256_allstats` PASS：
    `EXP`、`total_elems=1024`、`chunk_elems=256`、`chunks=4`、
    HBM read/write bytes `4096/4096`、DMA read/write issues `4/4`、
    wait count `8`、simulated time `663.83 us`、wall `126s`。
    SFU stats：core7 `sfu_ops_issued=1`、
    `sfu_primitive_elems=1024`，证明 4 个 chunk descriptor 被 1 次 batch issue
    覆盖。
- 完成 batch vs non-batch event-scaling 对照：
  - sweep root：
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_batch_compare_20260703`。
  - 配置：single `EXP`，`chunk_elems=256`，all-stats enabled。
  - `1024` elems：
    non-batch `chunks=4`、`sfu_ops_issued=4`、DMA read/write `4/4`、
    simulated time `659.441 us`、wall `152s`；
    batch `chunks=4`、`sfu_ops_issued=1`、DMA read/write `4/4`、
    simulated time `663.83 us`、wall `142s`。
  - `4096` elems：
    non-batch `chunks=16`、`sfu_ops_issued=16`、DMA read/write `16/16`、
    simulated time `1.72802 ms`、wall `346s`；
    batch `chunks=16`、`sfu_ops_issued=1`、DMA read/write `16/16`、
    simulated time `1.70861 ms`、wall `333s`。
  - 结论：batch descriptor 明确降低 RoCC/SFU control issue count；
    HBM bytes 和 DMA request count 不变，因此小规模 single-op 的 simulated time
    仍主要由相同 HBM/DMA 数据流支配。
  - 新增绘图脚本：
    `plot_sfu_hbm_batch_compare.py`。
  - 输出组会图与源数据：
    `figures/sfu_hbm_batch_compare_source.csv`、
    `figures/sfu_hbm_batch_compare_notes.md`、
    `figures/sfu_hbm_batch_compare.svg`、
    `figures/sfu_hbm_batch_compare.png`、
    `figures/sfu_hbm_batch_compare.pdf`、
    `figures/sfu_hbm_batch_compare.tiff`。
- 固化 SFU 后续建模层级：
  - 当前 Golem/Vanadis 路径是 C++ SST architecture-level event/timing
    simulation，不是 RTL。
  - 当前 SFU primitive 数学仍由 host C++ functional model 计算，SST 侧负责
    descriptor、RoCC、DMA/HBM、issue/wait 和统计。
  - 后续目标明确为 hardware-like SFU timing model：增加 per-op latency、
    issue bandwidth、queue depth、pipeline occupancy、backpressure 和
    stall/wait/latency 统计；不要求现阶段实现 Verilog/SystemVerilog RTL。
  - 已同步更新 `README.md`、`task_plan.md` 和
    `docs/superpowers/plans/2026-07-03-sfu-primitive-batch.md`。

## 2026-07-04

- 根据最新实验范围收敛决定：
  - 不做单独的 SFU stats CSV exporter。
  - batch / HBM streaming 后续常规验证最大维度收敛到 `4096` elements/op。
  - hardware-like SFU timing model 仍是需要做的方向，但不是当前阶段工作。
- 完成 ALL-op batch 限定验证：
  - sweep root:
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_allops_batch_limited_20260704`。
  - `1024` elems/op, `GOLEM_SFU_PRIMITIVE_HBM_OPS=ALL`,
    `GOLEM_SFU_PRIMITIVE_HBM_BATCH=1`, `chunk_elems=256`, all-stats enabled:
    run id `run_20260704_150513_1404172`, PASS。
    Guest PASS line reported `chunks=24`, `processed_elems=6144`,
    `hbm_init_write_bytes=24576`, `hbm_read_bytes=24576`,
    `hbm_write_bytes=24576`. DMA summary reported read/write issues `24/24`,
    wait count `48`, timeout retry/exhausted `0/0`, write completion `24/24`.
    SFU stats on core7 reported `sfu_ops_issued=6`,
    `sfu_primitive_elems=6144`, `sfu_credit_stalls=0`, `sfu_retry_events=0`.
    Simulated time was `2.7661 ms`; run-summary wall time was `583s`.
    This validates multi-child batching for all six primitive ops: each op has
    four child chunks, but one batch issue per op.
  - `4096` elems/op, `GOLEM_SFU_PRIMITIVE_HBM_OPS=ALL`,
    `GOLEM_SFU_PRIMITIVE_HBM_BATCH=1`, `chunk_elems=4096`, all-stats enabled:
    run id `run_20260704_151536_1475667`, PASS.
    Guest PASS line reported `chunks=6`, `processed_elems=24576`,
    `hbm_init_write_bytes=98304`, `hbm_read_bytes=98304`,
    `hbm_write_bytes=98304`. DMA summary reported read/write issues `6/6`,
    wait count `12`, timeout retry/exhausted `0/0`, write completion `6/6`.
    SFU stats on core7 reported `sfu_ops_issued=6`,
    `sfu_primitive_elems=24576`, `sfu_credit_stalls=0`, `sfu_retry_events=0`.
    Simulated time was `1.05165 ms`; run-summary wall time was `218s`.
    This validates the maximum planned dimension `4096` for the ALL-op batch
    path without turning the run into a per-chunk stress test.
- 计划更新：
  - 后续不做专门的 SFU stats CSV exporter；汇报所需 SFU 计数继续从
    `stats_selfcom.txt` 提取，DMA 计数继续使用 `dma_summary.csv`。
  - 将可调 `batch_size` 作为 batch/HBM streaming 语义冻结后的 DSE 方向：
    未来可加入 `GOLEM_SFU_PRIMITIVE_HBM_BATCH_SIZE=N`，扫描 batch 粒度对
    `sfu_ops_issued`、wall time、simulated time、stall/wait 和 local GM buffer
    压力的影响。
- 完成 HBM streaming primitive chunk/batch 汇总实验与科研图：
  - 新增脚本：
    `run_sfu_hbm_chunk_batch_sweep.sh` 和
    `plot_sfu_hbm_chunk_batch_sweep.py`。
  - 新 sweep root:
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_chunk_batch_sweep_20260704_chunk_batch_v1`。
  - 汇总图与源数据：
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_chunk_batch_summary_20260704/figures/`，
    包含 `sfu_hbm_chunk_batch_sweep_source.csv`、
    `sfu_hbm_chunk_batch_sweep_notes.md`、`.svg`、`.png`、`.pdf`、`.tiff`。
  - 补跑 PASS 点：
    - `256` elems/op, `chunk=256`, non-batch: `chunks=6`,
      `sfu_ops_issued=6`, DMA read/write `6/6`, simulated time `835.22 us`,
      wall `210s`。
    - `256` elems/op, `chunk=256`, batch: `chunks=6`,
      `sfu_ops_issued=6`, DMA read/write `6/6`, simulated time `890.449 us`,
      wall `221s`。
    - `1024` elems/op, `chunk=256`, non-batch: `chunks=24`,
      `sfu_ops_issued=24`, DMA read/write `24/24`, simulated time `2645.45 us`,
      wall `650s`。
    - `1024` elems/op, `chunk=1024`, batch: `chunks=6`,
      `sfu_ops_issued=6`, DMA read/write `6/6`, simulated time `2602.97 us`,
      wall `642s`。
  - `4096` elems/op, `chunk=1024`, batch 在 `1200s` timeout；说明大维度下
    小 chunk 即使开启 batch，DMA/chunk 事件和全统计仍会显著拉长 host wall time。
  - 结论：`chunk_elems` 决定 chunk/DMA event 数；batch 不改变 HBM bytes 和 DMA
    read/write 数，但在多 chunk/op 时能把 SFU issue 从 child chunk 数压缩到
    op 数。例如 `1024/chunk=256/all-op` 中 non-batch 为 `24` 次 SFU issue，
    batch 为 `6` 次 SFU issue，DMA read/write 仍均为 `24/24`。

## 2026-07-07

- 完成 batch-default SFU primitive softmax sweep 脚手架：
  - 新增 `run_sfu_softmax_primitive_sweep.sh`，默认只使用
    `GOLEM_SFU_PRIMITIVE_SOFTMAX=1` 和 batch-default primitive pipeline。
  - 新增 `plot_sfu_softmax_primitive_sweep.py`，输出 PASS 汇总、attempt/timeout
    汇总、coverage SVG、trend SVG 和中文汇报 notes。
  - 新增 scaffold tests，固定 softmax primitive sweep 必须走 batch-default 路径，
    不再引入 non-batch baseline。
- 真实 SST sweep root:
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_softmax_primitive_report`。
- PASS 点：
  - `128/chunk=128`: simulated time `357.79 us`，wall `102s`。
  - `256/chunk=256`: simulated time `434.27 us`，wall `118s`。
  - `512/chunk=512`: simulated time `581.291 us`，wall `150s`。
  - `1024/chunk=512`: simulated time `879.43 us`，wall `238s`。
  - `1024/chunk=1024`: simulated time `875.113 us`，wall `234s`。
- timeout 点：
  - `2048/chunk=512`: `240s` timeout，被中断时日志显示已推进到约
    `962.57 us` 模拟时间。
  - `2048/chunk=2048`: `240s` timeout，被中断时日志显示已推进到约
    `929.377 us` 模拟时间。
- 数据结论：
  - PASS 点最大 `max_abs_diff=1.44813e-09`，数值正确性良好。
  - HBM read/write bytes 随 `dim` 线性增长；`1024/chunk=512` 与
    `1024/chunk=1024` 的 HBM 总流量相同，均为 read `8192` bytes、write
    `4096` bytes。
  - `1024/chunk=1024` 相比 `1024/chunk=512` 将 chunks 和 DMA issue 从 `2/4`
    降到 `1/2`，但 simulated time 仅从 `879.43 us` 降到 `875.113 us`。
  - `2048` 在两种 chunk 设置下都 timeout，说明当前阶段不能继续强行做
    `2048/4096` 全 sweep；主要受功能模型、guest 调度和 SST 事件推进 wall time
    限制，而不是 PASS 点数值误差。
- 汇报材料：
  `figures/softmax_primitive_summary.csv`、
  `figures/softmax_primitive_attempts.csv`、
  `figures/softmax_primitive_coverage.svg`、
  `figures/softmax_primitive_trends.svg`、
  `figures/softmax_primitive_notes.md`。
- 完成 softmax primitive 仿真耗时第一轮诊断和低风险优化：
  - 修复 wrapper 覆盖 `GOLEM_SKIP_BUILD=1` 的问题，使 sweep 中的 skip-build
    设置能真正传到底层 pipeline。
  - sweep 新增 `GOLEM_SFU_SWEEP_REUSE_HBM`，默认在同一 sweep root 下检测到
    `hbm/hbm_config.env` 后复用 HBM backing files，并传递
    `GOLEM_SKIP_TENSOR_GEN=1`、`GOLEM_SKIP_HBM_GEN=1`。
  - 诊断点 `sfu_softmax_diag_skipprep_r1_d1024_c1024`：
    `dim=1024/chunk=1024/verify=1`，跳过 HBM 和底层 build 后 wall time 从旧
    `234s` 降到 `214s`，simulated time 保持 `875.113 us`。
  - 诊断点 `sfu_softmax_diag_skipprep_noverify_r1_d1024_c1024`：
    `verify=0` 后 wall time 进一步降到 `166s`，simulated time 降到
    `672.335 us`。
  - 结论：准备阶段复用和关闭 per-element golden 能降低约 `29%` wall time
    (`234s -> 166s`)，但剩余主瓶颈仍是 Vanadis/SST guest 事件推进；实际 DMA
    只有 read/write `2/2`，而日志中仍有约 `1.98e4` DRAMSim3 backend reads 和
    `8.23e4` Merlin packets。

- 用户明确要求继续采用真实 SST 仿真，不把 summary/timing-only 作为当前主方案。
  已新增 `GOLEM_SFU_PERF_PROFILE=1`，该模式仍调用真实
  `sst --num-threads=1`，但关闭逐元素 verification、HBM dump 和 SST 全量统计，
  并在已有 `hbm_config.env` 后复用 HBM/tensor 准备文件。
- 完成 focused real-SST perf sweep：
  - sweep root:
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_softmax_perf_profile_v1`。
  - `1024/chunk=1024`: PASS，simulated time `672.335 us`，wall `171s`，
    HBM stream `12288 B`，DMA read/write issue `2/2`。
  - `2048/chunk=2048`: PASS，simulated time `1.07095 ms`，wall `276s`，
    HBM stream `24576 B`，DMA read/write issue `2/2`。
  - 本轮没有 timeout，但 `2048` 已接近 5 分钟阈值；`4096` 应先单点试跑并设置
    timeout，不应直接恢复全矩阵 sweep。
- 修正 `plot_sfu_softmax_primitive_sweep.py` 的中文 notes 生成逻辑：
  - notes 现在根据本轮 manifest/summary 动态列出 PASS 点、timeout 点和 chunk
    覆盖情况；
  - 删除旧的硬编码 “2048 timeout/chunk=512” 结论；
  - 新图和说明位于
    `artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_softmax_perf_profile_v1/figures/`。
- 完成 `4096/chunk=4096` 单点真实 SST 诊断：
  - sweep root:
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_softmax_perf_profile_4096_probe`。
  - 第一次沙箱内运行在 SST 启动阶段失败，原因为 OpenMPI 无可用网络接口；
    这不是 SFU 性能或正确性问题。
  - 第二次沙箱外复跑进入真实 SST 主仿真，复用 HBM/build，`300s` timeout。
  - emergency shutdown 显示中断时已推进到约 `1.17671 ms` 模拟硬件时间。
  - timeout 前没有 `[SOFTMAX] ... PASS` stdout，也未观察到非零业务 DMA/SFU issue；
    日志中主要是系统级事件统计：DRAMSim3 backend reads `20379`、
    Merlin packets `85790`、memory queue delay samples `36679`。
  - 结论：该 `4096` 点属于单核整行 prototype timeout，不能作为常规 sweep PASS 点；
    在技术路线纠偏后，后续不应继续优化单核整行路径，而应实现多 core 协同
    primitive softmax。

- 技术路线纠偏：
  - 用户确认最初目标是多个 core 联合做 softmax primitive，以体现多核架构仿真。
  - 当前 `GOLEM_SFU_PRIMITIVE_SOFTMAX` 实现中 `requested_core_id != 0` 直接返回，
    实际只有一个 executor core 处理整行 softmax。
  - 因此当前 `1024/2048/4096` primitive softmax 实验只能作为单核原型诊断，
    不能作为最终架构性能结论。
  - 正确路线应为 multi-core cooperative row-wise softmax：
    `dim=4096` 在 16 worker cores 下应切分为约 `256` elements/core；
    每个 core 做 local partial max、EXP、partial sum 和本地 normalize/writeback；
    架构需要跨 core 合并全局 `row_max` 和 `row_sum`。
  - 后续立即停止继续优化单核整行 primitive softmax perf path，转为实现多核协同
    primitive softmax。

- 多核协同 primitive softmax 第一版实现进展：
  - 已删除正式 primitive softmax 路径中的 `requested_core_id != 0` 单核限制。
  - 新增 `GOLEM_SFU_PRIMITIVE_SOFTMAX_WORKER_CORES` 和
    `GOLEM_SFU_PRIMITIVE_SOFTMAX_MULTICORE_MIN_DIM`；默认策略是维度达到阈值后
    尽量使用 active worker cores，多核不是 `4096` 的特例。
  - 已实现按 worker slot 切分 column slice：每个参与 core 处理本 slice 的 HBM
    init、partial max、EXP、partial sum、RECIPROCAL normalization 和 writeback。
  - 跨 core 合并采用自管 mailbox/scratch：worker 把 partial max/sum 写入
    coordinator 本地 scratch，coordinator 合并 global max/sum 后广播给参与 worker。
  - PASS 行新增 `worker_cores`、`dim_per_core`、`cross_core_reduce_stages=2`。
  - 单元测试和 RISC-V 交叉编译已通过，但真实 SST smoke 尚未 PASS：
    `dim=512`、`worker_cores=4`、`chunk=64`、`timeout=180s` 不是超时，而是在
    SST 运行中触发 GlobalMemory 断言。
  - 已定位两个失败阶段：
    1. coordinator 直接 `gm2reg` 读取 worker mailbox 会触发
       `rd_from_globalmem` 越界；已改为 worker 主动 remote 写入 coordinator 本地
       scratch。
    2. 当前仍会在 DMA read response 写回时触发
       `wr_to_globalmem: wr_addr >= baseAddr`，说明 multi-core primitive 的 logical
       worker id、executor core id 和 RoCC/GlobalMemory 本地 base 仍未完全对齐。
  - 结论：本轮实现已把路线改到多核协同结构，但还不能做性能 sweep；下一步必须先
    修复真实 SST 下每个 worker 的本地 GM 地址归属，再恢复 `512/1024/2048/4096`
    多核 smoke/sweep。

- 多核协同 primitive softmax local-GM 映射修复：
  - 根因：primitive softmax 的本地 DMA/SFU/GM 操作使用了 argv 中的
    `requested_core_id`，但真实 SST 的 RoCC/GlobalMemory 写回必须匹配实际执行
    指令的 `executor_core_id`。当二者不一致时，DMA read response 会把数据写到
    不属于当前 RoCC 的 GM window，触发 `wr_to_globalmem: wr_addr >= baseAddr`。
  - 修复：
    1. `GOLEM_SFU_PRIMITIVE_SOFTMAX_WORKER_CORES` 和
       `GOLEM_SFU_PRIMITIVE_SOFTMAX_MULTICORE_MIN_DIM` 已加入
       `ncores_selfcom_dma_ctrl.py` 的 guest env 白名单，确保 wrapper 设置能传入
       每个 Vanadis guest 进程。
    2. `resolve_executor_core_from_argv_or_exit()` 改为优先使用真实
       `sched_getcpu()` 作为 executor/local-GM core id；只有无效时才回退到
       requested id。
    3. primitive softmax 的 worker slot、本地 mailbox wait 和 global value 读取
       改为使用 `executor_core_id`，避免 logical worker id 与本地 GM base 混用。
  - 验证：
    - 单测：`python3 -m unittest discover -s . -p 'test_*.py'`，`86` tests OK。
    - 交叉编译：`make` 通过。
    - 真实 SST smoke：
      `dim=512, worker_cores=4, multicore_min_dim=128, chunk=64, rows=1,
      verify=1, timeout=180s` 完成，没有 timeout，也没有 GlobalMemory 断言。
    - PASS 行：
      `worker_cores=4, dim_per_core=128, chunks=8, batches=16,
      hbm_init_write_bytes=2048, hbm_read_bytes=4096, hbm_write_bytes=2048,
      max_abs_diff=1.90714e-10, max_rel_diff=6.92901e-08,
      max_row_sum_error=4.19792e-09`。
  - 当前状态：多核 cooperative primitive softmax 已具备真实 SST smoke PASS 基线；
    后续可以进入小规模 sweep，而不是继续单核整行 prototype。

- 多核协同 primitive softmax 小矩阵真实 SST sweep：
  - sweep 脚本新增 `worker_cores` 维度，manifest 记录
    `run_id,rows,dim,chunk_elems,worker_cores,timeout_sec,status,exit_code`。
  - 新增 `GOLEM_SFU_SOFTMAX_MULTICORE_MATRIX=1`、`GOLEM_SFU_SOFTMAX_WORKERS` 和
    `GOLEM_SFU_SOFTMAX_PIPELINE_ARGS`，可用小 GEMM 占位配置专门测试 softmax
    primitive 多核路径。
  - 修复 perf profile 下 wrapper 覆盖 `GOLEM_SKIP_BUILD=1` 的问题；现在 sweep
    第一 点可显式 `GOLEM_SKIP_BUILD=0` 重编，后续点复用 HBM/build，避免旧 build
    metadata 污染实验。
  - 沙箱内真实 SST 会因 OpenMPI `socket() failed with errno=1` /
    `No network interfaces were found` 失败；沙箱外复跑后正常进入真实 SST。
  - 完成 6 点真实 SST sweep：
    `dim=512`、`chunk=64/128`、`worker_cores=4/8/16`，每点 timeout `180s`，
    结果 `6/6 PASS`、`0 timeout`。
  - 输出目录：
    `artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_multicore_sweep_d512_c64_128_w4_8_16_v4/figures/`。
  - 关键结果：
    - `chunk=64,w=4`: `dim_per_core=128`，`chunks=8`，`batches=16`，
      simulated `348.075 us`，wall `89s`。
    - `chunk=64,w=8`: `dim_per_core=64`，`chunks=8`，`batches=32`，
      simulated `323.495 us`，wall `85s`。
    - `chunk=64,w=16`: `dim_per_core=32`，`chunks=16`，`batches=64`，
      simulated `311.338 us`，wall `88s`。
    - `chunk=128,w=4`: `chunks=4`，DMA read/write issue `8/8`，
      simulated `347.846 us`，wall `83s`。
    - `chunk=128,w=8/16`: simulated time 分别为 `323.376 us` 和 `311.353 us`。
  - 结论：本轮验证了 multi-core cooperative primitive softmax 的真实 SST 路径
    可稳定跑通。`worker_cores` 增加后 `dim_per_core` 降低，模拟硬件时间从约
    `348 us` 降到约 `311 us`；`chunk` 主要改变 chunks/DMA issue 数，但在当前
    功能模型下对 simulated time 影响很小。后续应扩大到 `1024/2048/4096`
    多核 probe，并继续记录 timeout 原因。

- 多核协同 primitive softmax 4096 probe 已跑通：
  - sweep root:
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_multicore_probe_d1024_2048_4096_c256_w16`。
  - 配置：`rows=1`、`worker_cores=16`、`chunk=256`、`GOLEM_SFU_PERF_PROFILE=1`，
    真实 `sst --num-threads=1`，逐元素 verification 关闭。
  - 结果：`dim=1024/2048/4096` 三点全部 PASS，`0 timeout`。
  - 关键数据：
    - `1024`: `dim_per_core=64`，simulated `324.691 us`，wall `92s`，
      HBM stream `12288 B`。
    - `2048`: `dim_per_core=128`，simulated `353.156 us`，wall `104s`，
      HBM stream `24576 B`。
    - `4096`: `dim_per_core=256`，simulated `415.888 us`，wall `123s`，
      HBM stream `49152 B`。
  - 三点的 `chunks=16`、`batches=64`、DMA read/write issue `32/32` 保持一致，
    因为固定 `worker_cores=16` 且 `chunk=256`，每个 worker 处理一个 slice/chunk。
  - 结论：此前单核 `4096` timeout 不能代表最终架构；多核 cooperative 路线下
    `4096` 已成为可完成真实 SST PASS 点。下一步应做同一维度下的
    `chunk/worker_cores` DSE，并补一两个开启 verification 的小规模正确性点。

- 多核协同 primitive softmax 正确性点和 4096 focused DSE 已完成：
  - 小规模 correctness 点：
    `dim=512`、`chunk=256`、`worker_cores=16`、`verify=1`，真实 SST PASS。
    PASS 行显示 `dim_per_core=32`、`chunks=16`、`batches=64`、
    `max_abs_diff=1.90714e-10`、`max_rel_diff=6.92901e-08`、
    `max_row_sum_error=6.14326e-10`；run summary 为 simulated `330.679 us`、
    wall `88s`。
  - 4096 focused DSE：
    `GOLEM_SFU_SOFTMAX_POINT_LIST` 显式点列表新增并通过 dry-run/单测，避免
    为少量代表点跑完整笛卡尔积。
  - DSE root:
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_softmax_primitive_sweep_20260707_multicore_dse_d4096_worker_chunk_v1`。
  - DSE 配置：
    `dim=4096`、`rows=1`、真实 `sst --num-threads=1`、`GOLEM_SFU_PERF_PROFILE=1`；
    worker sweep 为 `chunk=256, worker_cores=4/8/16`，chunk sweep 为
    `worker_cores=16, chunk=128/256/512`。
  - DSE 结果：`5/5 PASS`、`0 timeout`。
    - `c256,w4`: `dim_per_core=1024`，simulated `731.988 us`，wall `183s`。
    - `c256,w8`: `dim_per_core=512`，simulated `518.271 us`，wall `140s`。
    - `c256,w16`: `dim_per_core=256`，simulated `415.888 us`，wall `120s`。
    - `c128,w16`: `chunks=32`，DMA read/write issue `64/64`，
      simulated `412.406 us`，wall `123s`。
    - `c512,w16`: `chunks=16`，DMA read/write issue `32/32`，
      simulated `415.888 us`，wall `122s`。
  - 结论：worker 数是本轮 4096 性能的主导因素，worker 从 4 增至 16 后
    simulated time 约从 `732 us` 降到 `416 us`。在 `worker_cores=16` 下，
    `chunk=128/256/512` 的 simulated time 差异很小，说明当前功能/批处理模型中
    HBM 总流量固定，chunk 主要改变 issue/chunks 数，尚未被周期精确 SFU timing
    放大。
  - 新增图表：
    `figures/softmax_primitive_dse.svg`、`softmax_primitive_coverage.svg`、
    `softmax_primitive_trends.svg`、`softmax_primitive_summary.csv` 和中文
    `softmax_primitive_notes.md`。

## 2026-07-08

- 完成 multi-row multi-core cooperative primitive softmax 真实 SST 代表点：
  - sweep root:
    `src/sst/elements/golem/tests/artifacts/sweeps/sfu_softmax_primitive_sweep_20260708_multicore_multirow_v1`。
  - 配置：`chunk=256`、`worker_cores=16`、`GOLEM_SFU_PERF_PROFILE=1`、真实
    `sst --num-threads=1`，逐元素 verification 关闭。
  - PASS `4/5`，timeout `1/5`。
- PASS 点摘要：
  - `rows=4, dim=1024`: simulated `542.939 us`，wall `163s`，
    `dim_per_core=64`，HBM stream `49152 B`。
  - `rows=4, dim=2048`: simulated `680.226 us`，wall `203s`，
    `dim_per_core=128`，HBM stream `98304 B`。
  - `rows=4, dim=4096`: simulated `919.973 us`，wall `270s`，
    `dim_per_core=256`，HBM stream `196608 B`。
  - `rows=8, dim=2048`: simulated `1.01583 ms`，wall `310s`，
    HBM stream `196608 B`。
- `rows=16, dim=1024` 在 `360s` timeout；emergency log 显示超时时已推进到
  约 `1.04688 ms` simulated time。
- 补充 timeout 诊断：
  - plotter 新增 `softmax_primitive_timeout_diagnosis.md`。
  - 诊断直接解析 emergency log 中可见的 per-core DMA 统计。
  - 可见未完成 core 的 median read/write issue 为 `28/29`；按每 worker
    每行 `1` 个 chunk 估算，已推进到约 `14/16` 行，完成约 `13/16` 行输出。
  - 可见 core 的 DMA timeout retry/exhausted 为 `0/0`，因此不是 HBM/DMA retry
    卡死，也不是启动阶段卡死，而是正常推进到尾部附近后 wall time 超限。
- 关键结论：`rows=8,dim=2048` 与 `rows=16,dim=1024` 总元素数均为 `16384`，
  但前者 PASS、后者 timeout。说明当前真实 SST 中 row 数增加会放大 per-row
  cross-core reduction、mailbox/guest loop、batch issue/wait 和 Vanadis/local-GM
  指令推进成本；后续应先定位 rows 方向的控制/同步瓶颈，而不是继续扩大 sweep。
- 图表脚本新增 `softmax_primitive_multirow.svg`，并更新中文 notes：
  `artifacts/sweeps/sfu_softmax_primitive_sweep_20260708_multicore_multirow_v1/figures/`。

## 2026-07-08 Unified Job Streaming Follow-up

- 继续沿 unified `SFUJobDesc` softmax 路线推进，未回到 primitive/batch softmax 主线。
- 在 `SFU::executeSoftmaxRowJob` 中将 Phase 1 的整行 functional executor 改为
  SFU 内部 row-band/chunk streaming：
  - 新增 `GOLEM_SFU_JOB_SOFTMAX_ROW_BAND_ROWS=4`。
  - 新增 `SoftmaxJobRowBandState`，按 row band 保存 `localMax/localSum` 和
    `globalMax/globalSum`。
  - 新增 `readSoftmaxJobChunk`、`readSoftmaxJobOutputChunk`、
    `writeSoftmaxJobChunk` helper。
  - executor 现在按 worker column slice 和 `chunk_elems` 做三阶段：
    chunk max -> worker-local/global max reduce -> chunk exp/local sum/write temp ->
    global sum reduce -> chunk normalize/writeback。
  - 移除了 `std::vector<float> row(desc.cols)` / `out(desc.cols)` 这种整行缓存。
- 新增静态 TDD 测试：
  - `test_unified_softmax_job_streams_rows_in_bands_and_chunks`
  - `test_unified_softmax_job_reduces_worker_local_max_and_sum_inside_sfu`
- 验证：
  - RED：新增测试初始失败于缺少 row-band/chunk helper 和 `localMax/localSum`。
  - GREEN：`python3 -m unittest src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_primitive_core.py -v`
    通过，`15 tests OK`。
  - 相关 Python 回归：
    `python3 -m unittest test_sfu_descriptor_scaffold.py test_rocc_sfu_integration.py test_sfu_primitive_core.py test_sfu_workload_scaffold.py test_run_noc_dma_softmax_sfu_pipeline.py -v`
    通过，`93 tests OK`。
  - 已同步 `src/sst/elements/golem/sfu/sfu.cc` 到
    `build/sst-elements/src/sst/elements/golem/sfu/sfu.cc`。
  - golem 组件局部构建：
    `make -C build/sst-elements/src/sst/elements/golem -j16` 通过；仅有既有
    SST serialization deprecation warnings。
- 复用既有 standalone HBM logits 路线接 unified SFU job：
  - 用户提醒后确认不需要新建 GEMM-free smoke；已有
    `GOLEM_SFU_STANDALONE_SOFTMAX=1` 会由 `gen_hbm_init.py` 将 logits 预置到
    GEMM C tile HBM 区域，并让 verifier 使用 `--reference logits`。
  - 新增 `GOLEM_SFU_STANDALONE_SOFTMAX=1` 与 `GOLEM_SFU_JOB_SOFTMAX=1` 的组合路径：
    guest 从 HBM C tile 区读取 logits 到 local GM row-major buffer，然后调用
    unified `golemRunStandaloneSoftmaxSfuJobForCore`，最后写回 C tile HBM 区。
  - 该路径输出 `dispatch=sfu-standalone-unified-job-softmax` 和
    `mode=sfu-standalone-job-softmax`，明确脱离 GEMM 计算。
- 修复 standalone logits 默认文件名：
  - wrapper 默认从 `data/softmax_logits.bin` 改为
    `data/softmax_logits_${GOLEM_GEMM_M}x${GOLEM_GEMM_N}.bin`。
  - 避免不同 shape 复用旧 logits 文件时出现大小不匹配或 stale golden。
- unified standalone job 真实 SST smoke：
  - 配置：
    `GOLEM_SFU_STANDALONE_SOFTMAX=1 GOLEM_SFU_JOB_SOFTMAX=1
    GOLEM_SFU_JOB_SOFTMAX_CHUNK_ELEMS=16 GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES=4
    --gemm-m 64 --gemm-n 64 --gemm-k 64 --verify-softmax
    --group-manager-enable 0 --ctrl-link-enable 0`。
  - sandbox 内仍会被 OpenMPI socket/OOB 权限挡住；按权限流程在 sandbox 外重跑。
  - 真实 SST PASS，run id `run_20260708_191142_2928451`，simulated time
    `466.411 us`，wall time `96s`。
  - guest stdout：
    `dispatch=sfu-standalone-unified-job-softmax rows=64 dim=64 chunk=16 workers=4`，
    `mode=sfu-standalone-job-softmax ... PASS`。
  - DMA totals：`read_issue=1`、`write_issue=1`、`read_bytes=16384`、
    `write_bytes=16384`、`timeout_retry=0`、`timeout_exhausted=0`、
    `completion=1`、`write_completion=1`。
  - fresh offline verifier:
    `[VERIFY-SFU-SOFTMAX] PASS reference=logits dtype=fp32 checked=4096
    mismatches=0 max_abs_diff=3.77493008e-09`。
- fresh targeted verification before recording this update:
  - `python3 -m unittest src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_workload_scaffold.py src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_run_noc_dma_softmax_sfu_pipeline.py -v`
    通过，`63 tests OK`。
  - `python3 -m unittest src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_primitive_core.py -v`
    通过，`15 tests OK`。
  - `python3 src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/verify_softmax_sfu_against_golden.py
    --c-file src/sst/elements/golem/tests/artifacts/stats/softmax_sfu_c_out.bin
    --logits-file src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/data/softmax_logits_64x64.bin
    --reference logits --dtype fp32 --m 64 --n 64 --k 64 --block-m 64 --block-n 64`
    PASS。

- 2026-07-09：unified SFU job softmax 进入 direct row-major HBM streaming：
  - 保持脱离 GEMM：`GOLEM_SFU_STANDALONE_SOFTMAX=1` +
    `GOLEM_SFU_JOB_SOFTMAX=1` + `GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM=1`
    时，`gen_hbm_init.py` 将 logits 直接预置到 data node 1 的 row-major
    HBM 区；guest 不再从 GEMM C tile layout gather/scatter。
  - 新增 direct row-major HBM region ABI、wrapper/architecture/env 透传、
    `unpack_c_from_hbm.py` direct row-major 输出解析，以及 scaffold 回归。
  - direct path 修正为 sub-job streaming：
    每个 `GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS` 子块执行
    `direct-load -> unified SFU job -> direct-store`，local GM 容量估算也改为
    sub-job buffer 粒度，而不是整 row-band。
  - TDD 记录：
    新增 `test_direct_rowmajor_hbm_streams_each_subjob_instead_of_full_band_dma`，
    初始 RED 失败于缺少 `sub_job_bytes/sub_job_input_hbm/sub_job_output_hbm`，
    GREEN 后 `test_sfu_workload_scaffold.py` 为 `46 tests OK`。
  - 静态/编译回归：
    - `python3 -m unittest .../test_sfu_workload_scaffold.py -v`：
      `46 tests OK`。
    - `python3 -m unittest .../test_run_noc_dma_softmax_sfu_pipeline.py -v`：
      `24 tests OK`。
    - `python3 -m unittest .../test_sfu_primitive_core.py -v`：
      `16 tests OK`。
    - `python3 -m py_compile gen_hbm_init.py unpack_c_from_hbm.py`：通过。
    - RISC-V workload 显式交叉编译通过。
  - 真实 SST 512x512 direct row-major PASS：
    artifact root
    `artifacts/sfu_unified_job_direct_rowmajor_stream_20260709_512`，
    run id `run_20260709_142551_3773241`。
    配置：`band_cores=8`、`staging_rows=64`、`job_rows=16`、
    `chunk_elems=256`、`worker_cores=16`、`A/B reuse=1`。
    verifier:
    `[VERIFY-SFU-SOFTMAX] PASS reference=logits dtype=fp32 checked=262144
    mismatches=0 max_abs_diff=3.90064624e-10`。
    DMA summary：`read_issue_count=64`、`write_issue_count=64`、
    `timeout_retry=0`、`timeout_exhausted=0`。
  - 真实 SST 1024x1024 direct row-major 探针：
    - 高并发档 `band_cores=8, job_rows=8, retry_ticks=256` 完成仿真但 verifier
      FAIL；失败从 row 512 开始，`mismatches=524288`，DMA summary
      `timeout_retry=1407`、`timeout_exhausted=128`。输出复用了前一波 local GM
      数据，说明第二波 HBM read chunk exhausted 后仍继续执行。
    - 保守正确性档 PASS：
      artifact root
      `artifacts/sfu_unified_job_direct_rowmajor_stream_20260709_1024_stable`，
      run id `run_20260709_143405_3784367`。
      配置：`band_cores=4`、`staging_rows=64`、`job_rows=8`、
      `chunk_elems=256`、`worker_cores=16`、`GOLEM_DMA_READ_RETRY_TICKS=4096`、
      `GOLEM_DMA_READ_MAX_RETRIES=16`、`A/B reuse=1`。
      verifier:
      `[VERIFY-SFU-SOFTMAX] PASS reference=logits dtype=fp32 checked=1048576
      mismatches=0 max_abs_diff=2.23162767e-10`。
      DMA summary：`read_issue_count=256`、`write_issue_count=256`、
      `timeout_retry=0`、`timeout_exhausted=0`。
  - 当前结论：
    unified SFU job 的 row-band/sub-job streaming 已能正确覆盖 512 和 1024；
    1024 的高并发失败来自 GlobalMemory DMA read retry 窗口/并发压力，不是
    SFU softmax 数学或 row-major 地址 ABI。下一步应把 direct job 运行档位
    固化为 sweep/profile，并评估 band-core 并发、retry window、DMA credits
    的性能/正确性边界。

- 2026-07-09：固化 unified SFU job direct row-major HBM sweep/profile：
  - 新增脚本
    `run_sfu_unified_job_direct_sweep.sh`，专门覆盖
    `GOLEM_SFU_STANDALONE_SOFTMAX=1` +
    `GOLEM_SFU_JOB_SOFTMAX=1` +
    `GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM=1`，不回到
    primitive/batch softmax 路线。
  - 默认 `GOLEM_SFU_JOB_DIRECT_PROFILE=stable`，包含：
    - `stable512`：`dim=512, band_cores=8, job_rows=16,
      retry_ticks=256, max_retries=8, expect=pass`。
    - `stable1024`：`dim=1024, band_cores=4, job_rows=8,
      retry_ticks=4096, max_retries=16, expect=pass`。
  - 显式 `GOLEM_SFU_JOB_DIRECT_PROFILE=pressure1024` 运行已知压力档：
    `dim=1024, band_cores=8, job_rows=8, retry_ticks=256,
    max_retries=8, expect=fail`；若失败会在 manifest 中标为
    `EXPECTED_FAIL`，用于复现 DMA retry/exhausted 边界。
  - 支持
    `GOLEM_SFU_JOB_DIRECT_POINT_LIST="dim:band_cores:job_rows:retry_ticks:max_retries:expect ..."`
    覆盖 profile，并写出 `sweep_manifest.csv`。
  - TDD/验证：
    - RED：新增 4 个 scaffold 测试，初始失败于缺少
      `run_sfu_unified_job_direct_sweep.sh`。
    - GREEN：`python3 -m unittest .../test_sfu_workload_scaffold.py -v`
      为 `50 tests OK`。
    - `bash -n run_sfu_unified_job_direct_sweep.sh` 通过。
    - `GOLEM_DRY_RUN_SWEEP=1` 默认 stable profile 生成 dry-run manifest：
      `stable512` 和 `stable1024` 两行。
    - `GOLEM_DRY_RUN_SWEEP=1` +
      `GOLEM_SFU_JOB_DIRECT_POINT_LIST='512:8:16:256:8:pass 1024:8:8:256:8:fail'`
      通过 point-list 解析和 dry-run manifest 写入。
  - 真实 SST sweep profile 验证：
    - `stable1024`：
      artifact root
      `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable1024_real`，
      manifest 记录 `status=PASS, exit_code=0`。
      verifier PASS：`checked=1048576, mismatches=0,
      max_abs_diff=2.23162767e-10`。run summary：
      wall `92s`、simulated `499.697 us`。DMA summary：
      `read_issue_count_sum=256`、`write_issue_count_sum=256`、
      `read_bytes_total_sum=4194304`、`write_bytes_total_sum=4194304`、
      `timeout_retry_sum=0`、`timeout_exhausted_sum=0`、
      `completion_sum=256`、`write_completion_sum=256`。
    - `pressure1024`：
      artifact root
      `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_pressure1024_real`，
      manifest 记录 `status=EXPECTED_FAIL, exit_code=1`。
      verifier FAIL 从 row 512 开始，`checked=1048576,
      mismatches=524288, max_abs_diff=0.00353896525`。run summary：
      wall `227s`、simulated `1.15539 ms`。DMA summary：
      `read_issue_count_sum=256`、`write_issue_count_sum=256`、
      `timeout_retry_sum=1407`、`timeout_exhausted_sum=128`、
      `completion_sum=128`、`write_completion_sum=208`。
  - 结论：
    新 sweep 入口已经复现了“1024 保守档正确、1024 压力档因 DMA retry/exhausted
    失败”的边界。后续 DSE 可以围绕 `band_cores`、`GOLEM_DMA_READ_RETRY_TICKS`
    和 `GOLEM_DMA_READ_MAX_RETRIES` 做最小矩阵扫描，而不是再手工拼单点命令。

- 实现 coordinator-only reciprocal broadcast：
  - 设计文档：
    `docs/superpowers/specs/2026-07-08-sfu-softmax-coordinator-reciprocal-design.md`。
  - 实施计划：
    `docs/superpowers/plans/2026-07-08-sfu-softmax-coordinator-reciprocal.md`。
  - 修改后 global sum 仍由 coordinator 汇总，但 `RECIPROCAL(global_sum)` 只由
    coordinator 每行执行一次，然后把 `inv_sum` 广播给所有 worker。
  - planned batch 公式从
    `rows * (planned_groups_per_row * 3 + worker_cores)` 改为
    `rows * (planned_groups_per_row * 3 + 1)`。
- coordinator-only reciprocal 验证结果：
  - Python 回归：`python3 -m unittest discover -s . -p 'test_*.py'` 为
    `88 tests OK`。
  - RISC-V workload 编译：
    `make clean ARCH=riscv64 && make ARCH=riscv64` 通过。
  - correctness smoke：
    `rows=1,dim=512,chunk=256,worker_cores=16,verify=1` 真实 SST PASS；
    PASS 行为 `batches=49`、`max_abs_diff=1.94327e-10`、
    `max_row_sum_error=7.84031e-10`，simulated `329.337 us`，wall `84s`。
  - rows=16 probe：
    `rows=16,dim=1024,chunk=256,worker_cores=16,perf_profile=1` 在 `360s`
    仍 timeout；emergency simulated time `1.08284 ms`，可见未完成 core median
    read/write issue `27/29`，DMA timeout retry/exhausted `0/0`。
  - 结论：该优化保持了 correctness，并降低了 planned batch 统计，但不足以让
    rows=16 过 360s 阈值。下一轮需要优化更大的 per-row 固定成本，例如
    multi-row max/sum reduction 合并、mailbox polling/broadcast 压缩或 row-level
    pipeline。

- 2026-07-09：为 unified SFU job direct row-major HBM 路径增加 DMA load guard
  和 wrapper failure gate：
  - guest 侧 direct sub-job 在 `dma_remote_load_to_gm()` 前向 input GM buffer
    写入 `kDirectDmaLoadSentinel`，load 返回后检查首尾和 16KB stride sentinel。
    sentinel 未被覆盖时打印
    `DMA_LOAD_FAILED direct row-major sub-job` 并返回失败，不再继续发
    `SFU_JOB SOFTMAX_ROW`。
  - wrapper 新增 `detect_sfu_guest_failure()`，扫描 `GOLEM_STDOUT_DIR`/`LOG_PATH`
    以及由 `GOLEM_ARTIFACT_ROOT + GOLEM_RUN_ID` 推导的默认 stdout/log 路径；
    捕获 `DMA_LOAD_FAILED`、`standalone unified job failed` 或
    `[SOFTMAX-SFU-JOB].*failed` 后，输出
    `[SFU][ERROR] guest reported failure; skip softmax verifier` 并退出 1。
  - TDD/验证：
    - `test_direct_rowmajor_hbm_aborts_when_dma_load_guard_detects_stale_local_gm`
      RED 后 GREEN。
    - `test_wrapper_skips_offline_verify_when_sfu_guest_reports_failure` RED 后 GREEN。
    - `python3 -m unittest .../test_sfu_workload_scaffold.py -v`：`51 tests OK`。
    - `python3 -m unittest .../test_run_noc_dma_softmax_sfu_pipeline.py -v`：
      `25 tests OK`。
    - `bash -n run_noc_dma_softmax_sfu_pipeline.sh` 通过。
    - `make ARCH=riscv64` 通过。
  - 真实 SST 验证：
    - guarded `stable1024`：
      `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_guard_stable1024_real`，
      verifier PASS：`checked=1048576,mismatches=0,max_abs_diff=2.23162767e-10`。
      说明 guard 不误伤稳定档。
    - guarded `pressure1024` + wrapper gate：
      `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_guard_pressure1024_wrapperfail_real`，
      manifest 为
      `sfu_job_direct_pressure1024,1024,8,8,256,8,fail,EXPECTED_FAIL,1,1800`。
      8 个 active executor core 均打印 `DMA_LOAD_FAILED`；DMA summary 为
      `read_issue_count_sum=162,write_issue_count_sum=146,timeout_retry_sum=532,
      timeout_exhausted_sum=16,completion_sum=146,write_completion_sum=146`。
      artifact 中没有后续 `[SFU] Unpacking...` 或
      `[VERIFY-SFU-SOFTMAX]` verifier mismatch 输出。

- 2026-07-09：执行 1024 direct row-major unified SFU job retry/band-core DSE：
  - sweep root：
    `artifacts/sweeps/sfu_unified_job_direct_dse_1024_bc_retry_20260709_real`。
  - 固定配置：
    `dim=1024, job_rows=8, chunk=256, workers=16,
    GOLEM_DMA_SLOT_COUNT=16`。
  - 扫描矩阵：
    `band_cores={4,6,8}`、
    `retry_ticks={512,1024,2048,4096}`、
    `max_retries={8,16}`，共 24 点。
  - dry-run 先验证 point-list 展开为 24 点，manifest 行数为 25（含 header）。
  - 真实 SST sweep 使用
    `GOLEM_STOP_ON_FAIL=0 GOLEM_STOP_ON_TIMEOUT=0`，保证失败点也继续记录。
  - 结果：
    - `24/24 PASS`，manifest 全部 `status=PASS, exit_code=0`。
    - 全部点 `dma_timeout_retry_sum=0`、`timeout_exhausted_sum=0`。
    - 全部点 `read_issue_count_sum=256`、`write_issue_count_sum=256`、
      `completion_sum=256`、`write_completion_sum=256`。
    - simulated time 按 band-core 聚合：
      - `band_cores=4`：`501.695 us`。
      - `band_cores=6`：`498.133 us`。
      - `band_cores=8`：`482.485 us`。
    - wall time 单点大约 `96-105s`。
  - 结论：
    在已扫描网格里，最小稳定高并发点是
    `band_cores=8,retry_ticks=512,max_retries=8`。
    结合此前 `band_cores=8,retry_ticks=256,max_retries=8` 的
    `DMA_LOAD_FAILED`，当前稳定阈值位于 `retry_ticks=256` 和 `512` 之间。

- 2026-07-09：执行 1024 direct row-major unified SFU job retry fine sweep：
  - sweep root：
    `artifacts/sweeps/sfu_unified_job_direct_dse_1024_retry_fine_20260709_real`。
  - 固定配置：
    `dim=1024, band_cores=8, job_rows=8, chunk=256, workers=16,
    max_retries=8, GOLEM_DMA_SLOT_COUNT=16`。
  - 扫描：
    `retry_ticks={288,320,384,448,512}`。
  - dry-run 先验证 point-list 展开为 5 点。
  - 真实 SST 结果：
    - `5/5 PASS`，manifest 全部 `status=PASS, exit_code=0`。
    - 无 `DMA_LOAD_FAILED`、无 verifier fail、无 timeout。
    - `retry_ticks=288`：
      `dma_timeout_retry_sum=159, timeout_exhausted_sum=0,
      read/write completion=256/256, simulated_time=587.451 us, wall=115s`。
    - `retry_ticks=320/384/448/512`：
      `dma_timeout_retry_sum=0, timeout_exhausted_sum=0,
      read/write completion=256/256, simulated_time=482.485 us`。
      wall time 分别约 `101/99/100/102s`。
  - 结论：
    `retry_ticks=288,max_retries=8` 是当前实测最小 PASS 点，但已经触发
    DMA retry 且 simulated time 变差；`retry_ticks=320,max_retries=8` 是当前
    实测最小 clean PASS 点，适合作为默认稳定高并发配置。

- 2026-07-09：更新 unified job direct sweep 默认 profile：
  - `stable1024` 从旧保守档
    `band_cores=4,retry_ticks=4096,max_retries=16` 切换为实测 clean PASS 档：
    `band_cores=8,retry_ticks=320,max_retries=8`。
  - `pressure1024` 保持
    `band_cores=8,retry_ticks=256,max_retries=8,expect=fail`，作为 DMA load
    guard/失败分类的负实验边界。
  - TDD/验证：
    - RED：先更新
      `test_unified_job_direct_sweep_has_stable_and_pressure_profiles`，要求
      `run_point 1024 8 8 320 8 pass ... stable1024`。
    - GREEN：修改 `run_sfu_unified_job_direct_sweep.sh` 后
      `python3 -m unittest .../test_sfu_workload_scaffold.py -v` 为
      `51 tests OK`。
    - `bash -n run_sfu_unified_job_direct_sweep.sh` 通过。
    - dry-run `GOLEM_SFU_JOB_DIRECT_PROFILE=stable1024` 输出
      `dim=1024 band_cores=8 job_rows=8 retry_ticks=320 max_retries=8 expect=pass`。
    - dry-run 默认 `stable` profile 输出 `stable512` 和新 `stable1024` 两点。

- 2026-07-09：运行更新后的 unified job direct `stable` profile 真实 SST canonical
  回归：
  - sweep root：
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable_profile_real`。
  - manifest：
    - `sfu_job_direct_stable512,512,8,16,256,8,pass,PASS,0,900`。
    - `sfu_job_direct_stable1024,1024,8,8,320,8,pass,PASS,0,1800`。
  - `stable512` run summary：
    `wall_time_sec=75, simulated_time=377.914 us,
    dma_timeout_retry_sum=42, dma_read_issue_count_sum=64,
    dma_write_issue_count_sum=64, dma_completion_sum=64,
    dma_write_completion_sum=64`；DMA exhausted 为 0。
  - `stable1024` run summary：
    `wall_time_sec=100, simulated_time=482.485 us,
    dma_timeout_retry_sum=0, dma_read_issue_count_sum=256,
    dma_write_issue_count_sum=256, dma_completion_sum=256,
    dma_write_completion_sum=256`；DMA exhausted 为 0。
  - wrapper 控制台输出中 `stable1024` verifier PASS：
    `checked=1048576 mismatches=0 max_abs_diff=2.23162767e-10`。
  - 附加告警：
    sweep 结束后出现 `no router stats found ... stats_selfcom.txt`，原因是该回归
    关闭了完整 SST/router stats；不影响 manifest、verifier、run summary 或 DMA
    summary 的 PASS 结论。
  - 结论：
    `stable` 默认 profile 现在已由真实 SST 合并回归验证；1024 默认稳定配置正式
    固化为 `band_cores=8, job_rows=8, retry_ticks=320, max_retries=8`。

- 2026-07-09：执行 2048 direct row-major unified SFU job 首个 scaling probe：
  - dry-run：
    `GOLEM_SFU_JOB_DIRECT_POINT_LIST=2048:8:4:320:8:pass` 正确展开为
    `sfu_job_direct_d2048_bc8_jr4_rt320_mr8_pass`，timeout `1800s`。
  - 真实 SST artifact root：
    `artifacts/sweeps/sfu_unified_job_direct_2048_probe_20260709_rt320_jr4_real`。
  - 配置：
    `dim=2048, band_cores=8, job_rows=4, chunk_elems=256,
    worker_cores=16, retry_ticks=320, max_retries=8`。
  - manifest：
    `sfu_job_direct_d2048_bc8_jr4_rt320_mr8_pass,2048,8,4,320,8,pass,PASS,0,1800`。
  - verifier：
    `checked=4194304, mismatches=0, max_abs_diff=1.13844124e-10`。
  - run/DMA summary：
    `wall_time_sec=244, simulated_time=1.06516 ms,
    dma_timeout_retry_sum=83, timeout_exhausted_sum=0,
    dma_read_issue_count_sum=1024, dma_write_issue_count_sum=1024,
    dma_completion_sum=1024, dma_write_completion_sum=1024,
    read/write bytes=16777216/16777216`。
  - 结论：
    2048 已在 unified SFU job direct row-major path 上真实 SST PASS；
    `320/8` 可完成但已有少量 read retry。下一步应以该点为 correctness anchor，
    对比 `retry_ticks=384/512` 是否 clean，再评估 `job_rows=8` 是否能保持
    PASS 并降低 sub-job 数量。

- 2026-07-09：执行 2048 direct row-major unified SFU job clean retry sweep：
  - dry-run：
    `GOLEM_SFU_JOB_DIRECT_POINT_LIST='2048:8:4:384:8:pass 2048:8:4:512:8:pass'`
    正确展开为 `rt384` 和 `rt512` 两点。
  - 真实 SST artifact root：
    `artifacts/sweeps/sfu_unified_job_direct_2048_retry_clean_20260709_real`。
  - 固定配置：
    `dim=2048, band_cores=8, job_rows=4, chunk_elems=256,
    worker_cores=16, max_retries=8`。
  - manifest：
    - `sfu_job_direct_d2048_bc8_jr4_rt384_mr8_pass,2048,8,4,384,8,pass,PASS,0,1800`。
    - `sfu_job_direct_d2048_bc8_jr4_rt512_mr8_pass,2048,8,4,512,8,pass,PASS,0,1800`。
  - run/DMA summary：
    - `retry_ticks=384`：
      `wall_time_sec=218, simulated_time=1.01116 ms,
      dma_timeout_retry_sum=0, timeout_exhausted_sum=0,
      dma_read_issue_count_sum=1024, dma_write_issue_count_sum=1024,
      dma_completion_sum=1024, dma_write_completion_sum=1024`。
    - `retry_ticks=512`：
      `wall_time_sec=222, simulated_time=1.01116 ms,
      dma_timeout_retry_sum=0, timeout_exhausted_sum=0,
      dma_read_issue_count_sum=1024, dma_write_issue_count_sum=1024,
      dma_completion_sum=1024, dma_write_completion_sum=1024`。
  - wrapper 控制台 verifier 两点均 PASS：
    `checked=4194304, mismatches=0, max_abs_diff=1.13844124e-10`。
  - 结论：
    `retry_ticks=384,max_retries=8` 是当前 2048 `job_rows=4` 的最小实测 clean
    PASS 点；`512/8` 不再带来 simulated time 改善。下一步应测试
    `job_rows=8,retry_ticks=384,max_retries=8`，评估减少 sub-job 数量后是否仍
    clean。

- 2026-07-09：执行 2048 direct row-major unified SFU job `job_rows=8`
  throughput-oriented probe：
  - dry-run：
    `GOLEM_SFU_JOB_DIRECT_POINT_LIST=2048:8:8:384:8:pass` 正确展开为
    `sfu_job_direct_d2048_bc8_jr8_rt384_mr8_pass`。
  - `retry_ticks=384,max_retries=8` 真实 SST：
    artifact root
    `artifacts/sweeps/sfu_unified_job_direct_2048_jr8_probe_20260709_real`。
    manifest：
    `sfu_job_direct_d2048_bc8_jr8_rt384_mr8_pass,2048,8,8,384,8,pass,FAIL,1,1800`。
    wrapper 检测到 guest failure 并跳过 verifier。
    8 个 active executor core 均打印 `DMA_LOAD_FAILED direct row-major sub-job`；
    每个失败 sub-job 为 `sub_job_rows=8, bytes=65536`。
    DMA summary：
    `dma_timeout_retry_sum=541, timeout_exhausted_sum=32,
    dma_read_issue_count_sum=144, dma_write_issue_count_sum=112,
    dma_completion_sum=112, dma_write_completion_sum=108`。
  - 对照 `retry_ticks=512,max_retries=8`：
    dry-run 正确展开为
    `sfu_job_direct_d2048_bc8_jr8_rt512_mr8_pass`。
    真实 SST artifact root
    `artifacts/sweeps/sfu_unified_job_direct_2048_jr8_rt512_probe_20260709_real`。
    manifest：
    `sfu_job_direct_d2048_bc8_jr8_rt512_mr8_pass,2048,8,8,512,8,pass,FAIL,1,1800`。
    仍由 wrapper gate 捕获 guest-side `DMA_LOAD_FAILED`，没有进入 offline
    verifier。
    DMA summary：
    `dma_timeout_retry_sum=861, timeout_exhausted_sum=32,
    dma_read_issue_count_sum=236, dma_write_issue_count_sum=204,
    dma_completion_sum=204, dma_write_completion_sum=200`。
  - 结论：
    在 `dim=2048,band_cores=8` 下，`job_rows=8` 的 64KB direct DMA load
    粒度越过当前稳定边界；把 retry window 从 `384` 放宽到 `512` 只能让运行
    继续推进更多 sub-job，不能消除 exhausted。当前 2048 默认应保持
    `job_rows=4,retry_ticks=384,max_retries=8`；若继续探索 `job_rows=8`，
    下一步应显式作为压力/吞吐实验，尝试 `retry_ticks=1024` 或
    `max_retries=16`，而不是替代 clean default。

- 2026-07-09：将 2048 clean point 固化为 unified job direct sweep profile：
  - TDD：
    - RED：先更新
      `test_unified_job_direct_sweep_has_stable_and_pressure_profiles`，要求
      `stable2048` 以及
      `run_point 2048 8 4 384 8 pass "$(timeout_for_dim 2048)" "stable2048"`。
    - GREEN：修改 `run_sfu_unified_job_direct_sweep.sh`，新增单独
      `GOLEM_SFU_JOB_DIRECT_PROFILE=stable2048`，并将默认 `stable` profile
      扩展为 `stable512 + stable1024 + stable2048`。
  - dry-run：
    - `stable2048` 展开为
      `dim=2048, band_cores=8, job_rows=4, retry_ticks=384,
      max_retries=8, expect=pass`。
    - 默认 `stable` 展开为 `stable512`、`stable1024`、`stable2048` 三点。
  - 真实 SST artifact root：
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable2048_profile_real`。
  - manifest：
    `sfu_job_direct_stable2048,2048,8,4,384,8,pass,PASS,0,1800`。
  - verifier：
    `checked=4194304, mismatches=0, max_abs_diff=1.13844124e-10`。
  - run/DMA summary：
    `wall_time_sec=226, simulated_time=1.01116 ms,
    dma_timeout_retry_sum=0, timeout_exhausted_sum=0,
    dma_read_issue_count_sum=1024, dma_write_issue_count_sum=1024,
    dma_completion_sum=1024, dma_write_completion_sum=1024,
    read/write bytes=16777216/16777216`。
  - 附加告警：
    末尾仍有 `no router stats found ... stats_selfcom.txt`，原因同前面的 stable
    回归：完整 router stats 关闭后附加 NoC 提取器找不到 router counter；不影响
    manifest、verifier 或 DMA clean PASS 结论。
  - 结论：
    当前 default stable profile 已覆盖 512/1024/2048；2048 默认正式固化为
    `band_cores=8, job_rows=4, retry_ticks=384, max_retries=8`。

- 2026-07-09：执行 4096 direct row-major unified SFU job 首个 scaling probe：
  - 配置选择：
    从 2048 clean profile 外推，保持单次 direct DMA sub-job load 粒度为
    `32768B`：
    `4096 * job_rows=2 * fp32 = 32768B`，对应 2048 clean 点
    `2048 * job_rows=4 * fp32 = 32768B`。
  - HBM capacity 检查：
    `4096x4096` direct row-major input/output 各为 `64MiB`，两块共
    `128MiB`；默认 `128MiB` per-node backing 会越过 direct row-major/bias
    区域边界。因此 4096 probe 显式使用
    `GOLEM_SFU_JOB_DIRECT_PIPELINE_ARGS='--mem-node-size 268435456'`。
  - 第一次启动被窗口中断：
    artifact root
    `artifacts/sweeps/sfu_unified_job_direct_4096_probe_20260709_rt384_jr2_mem256_real`
    只留下 HBM/log/stats partial files，manifest 只有表头；检查 `ps` 后确认无残留
    4096/SST 进程。该目录不作为实验结果。
  - 重新运行 artifact root：
    `artifacts/sweeps/sfu_unified_job_direct_4096_probe_20260709_rt384_jr2_mem256_retry1_real`。
  - 配置：
    `dim=4096, band_cores=8, job_rows=2, chunk_elems=256,
    worker_cores=16, retry_ticks=384, max_retries=8,
    mem_node_size=268435456`。
  - manifest：
    `sfu_job_direct_d4096_bc8_jr2_rt384_mr8_pass,4096,8,2,384,8,pass,PASS,0,2400`。
  - verifier：
    `checked=16777216, mismatches=0, max_abs_diff=5.72476014e-11`。
  - run/DMA summary：
    `wall_time_sec=745, simulated_time=3.16213 ms,
    dma_timeout_retry_sum=0, timeout_exhausted_sum=0,
    dma_read_issue_count_sum=4096, dma_write_issue_count_sum=4096,
    dma_completion_sum=4096, dma_write_completion_sum=4096,
    read/write bytes=67108864/67108864`。
  - 结论：
    unified SFU job direct row-major streaming 已真实 SST clean PASS 到
    `dim=4096`。当前可把 `4096:8:2:384:8` 作为 correctness anchor；后续若要
    固化 profile，需要决定是否将 `mem-node-size=256MiB` 纳入 4096 profile
    的显式参数，或让 sweep script 自动按 dim 设置。

- 2026-07-09：将 4096 clean point 固化为单独 `stable4096` profile：
  - TDD：
    - RED：更新
      `test_unified_job_direct_sweep_has_stable_and_pressure_profiles`，要求
      `stable4096`、`run_point 4096 8 2 384 8 pass "$(timeout_for_dim 4096)" "stable4096" 268435456`、
      per-point `mem_node_size` 参数和 `GOLEM_TIMEOUT_4096:-2400`。
    - GREEN：修改 `run_sfu_unified_job_direct_sweep.sh`：
      `run_point` 增加第 9 个可选参数 `mem_node_size`，默认 `134217728`；
      4096 profile 传 `268435456`；dry-run 输出 `mem_node_size=...`。
  - 默认 `stable` profile 仍保持 `stable512 + stable1024 + stable2048`，
    不自动包含 4096，避免普通稳定回归变成超重长跑。
  - dry-run：
    `GOLEM_SFU_JOB_DIRECT_PROFILE=stable4096` 展开为
    `dim=4096, band_cores=8, job_rows=2, retry_ticks=384,
    max_retries=8, mem_node_size=268435456, expect=pass, timeout=2400s`。
  - canonical real-SST artifact root：
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable4096_profile_real`。
  - manifest：
    `sfu_job_direct_stable4096,4096,8,2,384,8,pass,PASS,0,2400`。
  - verifier：
    `checked=16777216, mismatches=0, max_abs_diff=5.72476014e-11`。
  - run/DMA summary：
    `wall_time_sec=721, simulated_time=3.16213 ms,
    dma_timeout_retry_sum=0, timeout_exhausted_sum=0,
    dma_read_issue_count_sum=4096, dma_write_issue_count_sum=4096,
    dma_completion_sum=4096, dma_write_completion_sum=4096,
    read/write bytes=67108864/67108864`。
  - 附加告警：
    末尾 `no router stats found ... stats_selfcom.txt` 仍来自关闭完整 router stats
    后的附加 NoC 提取器，不影响 manifest/verifier/DMA clean PASS 结论。
  - 结论：
    `stable4096` 已成为可单独调用的 canonical profile；4096 维 stable 回归
    需要显式 256MiB mem-node backing，这个需求已经编码进 profile。

- 2026-07-09：完成 4096 `job_rows=4` pressure retry-window 探测：
  - TDD：
    - RED：扩展
      `test_unified_job_direct_sweep_has_stable_and_pressure_profiles`，要求
      `pressure4096_jr4_rt384`、`pressure4096_jr4_rt512`、
      `pressure4096_jr4_rt1024` 三个 profile；其中 rt384/rt512 为
      expected-fail，rt1024 为 expected-pass。
    - GREEN：`run_sfu_unified_job_direct_sweep.sh` 新增三组单点 profile：
      `4096, band_cores=8, job_rows=4, max_retries=8,
      mem_node_size=268435456`，retry ticks 分别为 `384/512/1024`。
  - `rt384` 真实 SST 初始探测 artifact：
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_pressure4096_jr4_real`。
    manifest 为
    `sfu_job_direct_pressure4096_jr4,4096,8,4,384,8,pass,FAIL,1,2400`。
    失败根因是 DMA read timeout exhausted 后 guest guard 检测到 stale local GM：
    `DMA_LOAD_FAILED direct row-major sub-job`；DMA 汇总为
    `dma_timeout_retry_sum=520, timeout_exhausted_sum=32`。
  - `rt512` 真实 SST artifact：
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_pressure4096_jr4_rt512_real`。
    manifest 为
    `sfu_job_direct_pressure4096_jr4_rt512,4096,8,4,512,8,pass,FAIL,1,2400`。
    它比 rt384 走得更远，但仍触发相同 DMA load guard failure；
    run summary 为
    `wall_time_sec=278, simulated_time=889.832 us,
    dma_timeout_retry_sum=839, timeout_exhausted_sum=32`。
  - `rt1024` 真实 SST artifact：
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_pressure4096_jr4_rt1024_real`。
    manifest 为
    `sfu_job_direct_pressure4096_jr4_rt1024,4096,8,4,1024,8,pass,PASS,0,2400`。
    verifier：
    `checked=16777216, mismatches=0, max_abs_diff=5.72476014e-11`。
    DMA clean：
    `dma_timeout_retry_sum=0, timeout_exhausted_sum=0,
    dma_read_issue_count_sum=4096, dma_write_issue_count_sum=4096,
    dma_completion_sum=4096, dma_write_completion_sum=4096,
    read/write bytes=67108864/67108864`。
    timing：
    `wall_time_sec=700, simulated_time=3.05433 ms`。
  - 结论：
    对 4096 direct row-major unified SFU job softmax，`job_rows=4` 可行，
    但需要把 DMA read retry window 从 stable4096 的 `384` 提高到 `1024`。
    `rt384` 和 `rt512` 不是 correctness 错误，而是 larger sub-job read burst
    下过早 timeout/exhaustion 触发 guard。下一步可以围绕
    `pressure4096_jr4_rt1024` 做正式 throughput/profile 对比，或继续尝试
    `retry_ticks=768` 找最小 clean window。

- 2026-07-10：4096 `job_rows=4, retry_ticks=768` clean PASS：
  - 新增 `GOLEM_SFU_JOB_DIRECT_PROFILE=pressure4096_jr4_rt768`：
    `dim=4096, band_cores=8, job_rows=4, retry_ticks=768,
    max_retries=8, mem_node_size=268435456, expect=pass`。
  - TDD：scaffold 先要求 profile、完整 `run_point` 参数和 help 列表，确认
    RED 后补 sweep script，target test 与 dry-run 转 GREEN。
  - real-SST artifact：
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260710_pressure4096_jr4_rt768_real`。
  - manifest：
    `sfu_job_direct_pressure4096_jr4_rt768,4096,8,4,768,8,pass,PASS,0,2400`。
  - verifier：
    `checked=16777216, mismatches=0, max_abs_diff=5.72476014e-11`。
  - DMA clean：
    `timeout_retry_sum=0, timeout_exhausted_sum=0,
    read/write issue=4096/4096, read/write completion=4096/4096`；
    active-core observed `max_rtt_ticks=688`。
  - timing：`wall_time_sec=692, simulated_time=3.05433 ms`。
  - 结论：clean retry-window 上界从 1024 降到 768；当前已知边界为
    `512 fail < clean threshold <= 768`。下一候选优先选 `704`，用于验证
    是否能以略高于 observed 688-tick max RTT 的窗口保持 zero-retry clean。

- 2026-07-13：4096 `job_rows=4, retry_ticks=704` clean PASS：
  - 新增 `GOLEM_SFU_JOB_DIRECT_PROFILE=pressure4096_jr4_rt704`：
    `dim=4096, band_cores=8, job_rows=4, retry_ticks=704,
    max_retries=8, mem_node_size=268435456, expect=pass`。
  - TDD：scaffold 先要求 rt704 profile 和完整参数，确认 RED 后补 sweep
    function/case/help list；target test、shell syntax 与 dry-run 均转 GREEN。
  - real-SST artifact：
    `artifacts/sweeps/sfu_unified_job_direct_sweep_20260713_pressure4096_jr4_rt704_real`。
  - manifest：
    `sfu_job_direct_pressure4096_jr4_rt704,4096,8,4,704,8,pass,PASS,0,2400`。
  - verifier：
    `checked=16777216, mismatches=0, max_abs_diff=5.72476014e-11`。
  - DMA clean：
    `timeout_retry_sum=0, timeout_exhausted_sum=0,
    read/write issue=4096/4096, read/write completion=4096/4096`；
    observed `max_rtt_ticks=688`。
  - timing：`wall_time_sec=738, simulated_time=3.05433 ms`。
  - 结论：当前 zero-retry clean window 已从 768 收敛到 704；已知边界为
    `512 fail < clean threshold <= 704`。下一点应测试 688 的等边界行为，
    或用 696 保留 8-tick headroom。

- 2026-07-13：完成 unified SFU job distributed-columns Phase 3A：
  - ABI：保持 host/guest `SFUJobDesc` 为 128 bytes，新增 distributed flag，
    复用 `reserved0` 携带 `worker_slot`。
  - SFU：新增 Pending staged execution；`issueJob` 计算 local max，`wait`
    根据 peer readiness 推进 local exp/sum 和 normalize。共享 reducer 对
    worker slot 去重并在全部 worker normalize 后清理。
  - guest/direct HBM：按 `band_cores / worker_cores` 划分 cooperative groups，
    group 分配 row bands，group 内 physical workers 分配 columns；每个 worker
    仅 load/store compact local slice。
  - 配置链修复：archive shim 转发
    `GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS`；`cpu_builder.py` 同时给 RoCC
    parent 和 SFU child 设置 `active_worker_cores`，避免 RoCC 将 worker count
    覆盖为默认 1。
  - 静态回归：descriptor/core/workload/wrapper/cpu-builder 合计 118 tests，OK。
  - SST element 和 RISC-V workload build 均完成。
  - canonical real-SST artifact：
    `tests/artifacts/sfu_unified_job_distributed_smoke_20260713`，run id
    `run_20260713_150940_2502347`。
  - 参数：`rows=4, dim=64, chunk=16, worker_cores=4, band_cores=4,
    direct_rowmajor_hbm=1, distributed_columns=1`。
  - 四个 physical SFU 各完成一个 worker slice；每核
    `max/sum/norm chunks=4/4/4, partial_submits=8, partial_done=4`。
  - DMA：16 read + 16 write 全部完成，读写各 1024 bytes，0 retry/exhausted。
  - golden：checked 256，mismatches 0，`max_abs_diff=2.71942902e-09`。
  - 最新构建产物复验：`run_20260713_151935_2580181` 再次得到四个 distinct
    worker slices、16/16 DMA read/write completion 和相同 golden PASS；
    simulated time 为 `344.857 us`。
  - 下一步：先扩展到 `dim=512/1024`、4/8/16 physical workers 和多个 row-band
    groups，验证 correctness/lifecycle；之后再实现显式 NoC reducer traffic。

- 2026-07-13：完成 Phase 3A post-review reliability 修复与正负向复验：
  - reducer key 扩展为 `(job_id, tag, owner_core, row)`；加入 physical
    core/owner/slot membership 校验和 duplicate pending tag 防覆盖。
  - 增加 `SFU_JOB_FLAG_DISTRIBUTED_ABORT`、`abortSeen` tombstone 和 peer
    abort observation；最后观察者清理后不会被通用错误路径重新创建。
  - direct HBM local-GM capacity 改按 per-worker compact slice 计算。
  - wrapper failure detector 在父 shell 无 `GOLEM_RUN_ID` 时，从
    `stats/run_summary.csv` 恢复最新 run id，避免 guest failure 假返回 0。
  - 完整静态回归 118 tests PASS；SST element 与 RISC-V workload build PASS。
  - 正向 run `run_20260713_155053_2817319`：四个 SFU 各完成一个 16-column
    slice；每核 max/sum/norm 4/4/4、partial 8/4、retry events 0；DMA
    read/write completion 16/16、0 retry/exhausted；golden 256/256 PASS，
    `max_abs_diff=2.71942902e-09`，simulated `343.936 us`。
  - 负向 run `run_20260713_155920_2883442`：1-tick retry window、0 retries
    注入一个 DMA exhausted；失败 worker 发 abort，其余三个 peer 收到 status 2，
    simulation 有限完成，wrapper 最终 exit 1，证明没有永久 Pending。

- 2026-07-13：完成 Phase 3A 最终 reducer lifecycle 收口：
  - duplicate in-flight tag 采用 identity poison 语义，不覆盖 pending map 或
    inflight count，并保证原 hart 能通过 `wait` 有限退出。
  - abort/metadata mismatch 按 `(job_id, tag, owner_core)` 扫描完整 cohort；
    creation loop 跳过调用开始时已存在的 key，避免最后 observer 清理后重建
    tombstone。
  - wrapper 的 run-summary 恢复同时支持 `GOLEM_RUN_SUMMARY_CSV` override。
  - 上一轮正向 run `run_20260713_161203_2984705`：四个 physical SFU cooperative
    execution，golden 256/256 PASS、0 mismatch，
    `max_abs_diff=2.71942902e-09`。
  - 上一轮负向 run `run_20260713_161338_2997972`：一个 worker
    `DMA_LOAD_FAILED`，另外三个 worker `status=2`，simulation 在
    `371.557 us` 有限完成，wrapper exit 1。

- 2026-07-13：修复最终 review 发现的 asymmetric cohort shape deadlock：
  - 原状态只校验 `expectedWorkers`；同 cohort 若一个 worker 声明 `rows=1`、
    另一个声明 `rows=2`，额外 row 会永远等待不存在的 partial。
  - reducer row state 新增 `expectedRows/expectedCols`，首次提交固化
    `worker_cores/rows/cols` signature，后续不匹配直接触发 cohort abort。
  - 新增 TDD structural regression；完整 suite 现在为 119 tests。
  - SST element 重新编译通过。
  - 最终正向 run `run_20260713_162719_3103304`：golden 256/256 PASS、
    0 mismatch，`max_abs_diff=2.71942902e-09`。
  - 最终负向 run `run_20260713_162857_3116951`：一个 worker
    `DMA_LOAD_FAILED`，其余三个 `status=2`，simulation `371.557 us` 有限完成，
    wrapper exit 1。

- 2026-07-13：启动 Phase 3B representative distributed scaling：
  - 确认旧 `run_sfu_unified_job_direct_sweep.sh` 未启用 distributed columns，且
    将 rows 与 dim 绑定，不适合直接承载 physical multi-SFU matrix。
  - 决定新增独立 runner，固定 rows/staging/job/chunk/retry knobs，仅扫描
    `dim=512/1024`、`worker_cores=4/8/16` 和 1/2/4 row-band groups。
  - 计划文件：
    `docs/superpowers/plans/2026-07-13-sfu-unified-job-phase3b-scaling.md`。
  - TDD RED：3 个 focused tests 因 runner 不存在失败。
  - GREEN：新增 `run_sfu_unified_job_distributed_scaling.sh`；focused tests
    `3 passed`。
  - runner 只在 wrapper golden、physical PASS core count、SFU stage/partial
    counters、DMA issue/completion/bytes 和零 retry/exhausted 全部通过后写
    per-point `.pass` marker。
  - 记录一次文档 patch 上下文错误；未产生部分修改，拆分正确文件目标后完成。
  - focused tests 扩展为 `4 passed`；`bash -n` 通过。
  - full matrix dry-run PASS，manifest 精确包含 8 个点；artifact：
    `tests/artifacts/sweeps/sfu_unified_job_distributed_scaling_20260713_dryrun`。
  - 首个 real 校准尝试在 SST 前失败：base workload metadata 仍为 4x64，而
    runner 强制 `GOLEM_SKIP_BUILD=1`。TDD 增加 artifact isolation 断言后，将
    skip-build 改为 0，并把 tensor dir 从 repo data 移到 sweep `inputs/`。
  - real matrix artifact：
    `tests/artifacts/sweeps/sfu_unified_job_distributed_scaling_20260713_real`。
  - 8 个配置均有独立 `.pass` marker；512 四点 checked 8192、1024 四点
    checked 16384，全部 0 mismatch。
  - active SFU、max/sum/norm、partial submit/done、DMA issue/completion/bytes
    均通过 runner 的精确计数验收；全部点 zero retry/exhausted/write retry。
  - latest manifest 为 8/8 PASS；最初的 FAIL 行保留为 calibration history，
    后续同 run id 的 PASS 与 marker 是最终状态。

- 2026-07-13：完成 Phase 3B runner post-review reliability 收口：
  - review 发现 `validate_point_artifacts` 被 `if` 调用时，Bash 会抑制函数体内
    `set -e`，导致早期 counter mismatch 可能被最后一个成功检查覆盖。
  - 所有 stats 读取和比较改为显式 `|| return 1`，并新增 synthetic artifact
    regression，确认 physical PASS core 从 4 变成 3 时校验器必定失败。
  - `.pass` marker 现在绑定 rows/dim/chunk/workers/band/staging/job/retry/max-retries
    和额外 pipeline args 的完整签名；缓存命中还会重新验证 stdout、SFU stats、
    DMA summary、输出大小和输出 SHA-256，旧 marker 或损坏产物不会再直接返回
    CACHED PASS。
  - 默认 artifact stamp 增加纳秒与 PID，避免并发 runner 共享 sweep root。
  - focused distributed tests `10 passed`；五文件静态 suite `112 passed`；shell
    syntax 和 `git diff --check` 均通过。
  - 第一次重跑在受限 sandbox 内被 Open MPI 本机网络接口初始化拒绝，SST 尚未
    启动；切换到获批的 sandbox 外执行后，修复后的校验器重新完成 8/8 real
    PASS，512/1024 golden 仍为 0 mismatch，所有 DMA retry/exhausted 为 0。
  - 随后同 root 重入 8 点，全部输出 `skip validated PASS`；manifest 最后 8 行
    均为 `PASS,0,...,CACHED`，证明缓存依赖签名匹配和完整产物重校验。
  - 新增同尺寸 corrupted-output regression，确认即使 tensor 文件仍存在且大小
    正确，只要 SHA-256 与 marker 不一致也会拒绝缓存；pipeline args 改变同样会
    生成不同 point signature。
  - runner-owned `--softmax-c-file` 作为 pipeline args 之后的最终 CLI 参数固定，
    附加参数不能把真实输出重定向到校验器未读取的路径；新增参数顺序回归覆盖。

- 2026-07-13：完成 Phase 4A modeled reduction transport observability：
  - SFU 新增 `distributed_reduction_transport` 参数，默认 `shared` 保持旧行为；
    `modeled_noc` 模式仍复用当前 shared in-SST reducer，但在 max/sum 两个
    reduction 边界记录 message-equivalent request/response 统计。
  - 新增四个统计项：`sfu_reduction_max_requests`,
    `sfu_reduction_max_responses`, `sfu_reduction_sum_requests`,
    `sfu_reduction_sum_responses`。每个 worker-row 的 max submit、max ready、
    sum submit、sum ready 各计一次；轮询 ready 不会重复计 response。
  - `cpu_builder.py` 通过 `GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT` 向 SFU
    child 传递该参数；distributed scaling runner 默认使用 `modeled_noc`，并将
    transport 纳入 point signature。
  - runner artifact validator 现在要求四个 reduction transport 计数都等于
    `rows * worker_cores`，从而把 distributed reducer 的 request/response
    生命周期纳入真实 SST 产物验收。
  - review 修复：distributed scaling runner 明确只接受 `modeled_noc`/
    `noc_model`/`noc` transport；传 `shared` 会在入口 exit 2，避免合法
    shared SFU 模式与本 runner 的 message-counter validator contract 混用。
    SFU 构造函数也会对未知 `distributed_reduction_transport` 字符串 fatal，
    不再静默退回 `shared`。
  - focused unittest：`test_sfu_primitive_core.py`,
    `test_cpu_builder_sfu_mount.py`, `test_sfu_workload_scaffold.py` 合计 97 tests
    通过；`bash -n run_sfu_unified_job_distributed_scaling.sh` 通过。
  - SST element 在 `build/sst-elements/src/sst/elements/golem` 重新编译通过，并
    确认 `.libs/libgolem.so` 已包含新参数、新统计字符串和 invalid-transport
    fatal 字符串。
  - real smoke artifact：
    `tests/artifacts/sweeps/sfu_unified_job_modeled_noc_smoke_20260713_reviewfix`。
    配置 `rows=16, dim=512, chunk=256, worker_cores=4, band_cores=4`，manifest
    `PASS,0,...,artifact_validation=PASS`；golden checked 8192、0 mismatch、
    `max_abs_diff=3.90064624e-10`。
  - active core0-3 每核记录 max/sum request/response 均为 16，总计 64；
    inactive core4-15 均为 0。`sfu_partial_submits=128`,
    `sfu_partial_done=64`，DMA read/write issue 与 completion 均为 64，
    retry/exhausted/write retry 仍为 0。
  - 这一阶段只建立 reduction traffic 的可观测 contract，尚未把 partial
    通过 SST `SimpleNetwork` 真实发送；正式 NoC latency/contention 建模仍是
    下一阶段任务。

## 2026-07-13 Phase 4B Explicit NoC Session Start

- Re-read the unified-job architecture, Phase 4B plan, local task log, findings,
  and workload README. The first implementation cut remains queue-backed
  `explicit_noc`: shared reducer math remains the functional reference while
  request/response accounting moves to typed internal transport messages.
- Acceptance remains the canonical `16:512:4:4` real SST smoke with golden
  correctness, four reduction counter totals of `rows * worker_cores`, and
  clean DMA lifecycle stats.

### Task 1 Complete: Explicit Transport Mode Scaffold

- Added `DistributedReductionTransport::ExplicitNoC`, exact `explicit_noc`
  parsing, and `explicitDistributedReductionEnabled()` while preserving the
  default shared mode, existing modeled-NoC aliases, and fatal handling for
  unknown transport strings.
- TDD evidence: the new source-level assertion failed before the parser was
  added, then `python3 -m unittest .../test_sfu_primitive_core.py -v` passed
  26/26. A separate task review found no Critical, Important, or Minor issue.

### Task 2 Complete: Typed Reduction Message Queue

- Added the private `DistributedReductionMessageKind` and
  `DistributedReductionMessage` payload with max/sum request/response kinds,
  identity fields, value, and FIFO enqueue/drain helpers. The queue is not yet
  wired into the reducer path, so Phase 4A execution and counter behavior stay
  unchanged until Task 3.
- TDD evidence: the structural test failed on the missing message token before
  implementation and then `test_sfu_primitive_core.py` passed 27/27. Separate
  task review found no Critical, Important, or Minor issue.

### Task 3 Complete: Queue-Backed Explicit Transport Boundary

- Under `explicit_noc`, typed max/sum request messages are enqueued and drained
  into the existing shared reducer; reducer-ready values are then enqueued and
  drained as typed responses before row state consumes them. Shared reducer
  math remains the functional reference.
- Request counters record at explicit enqueue and response counters at explicit
  drain. Modeled-NoC retains its Phase 4A direct accounting, and shared remains
  uncounted. The response stage advances after first delivery, preventing
  repeated `wait` polling from adding a second response.
- TDD evidence: the core test first failed on the missing explicit path, then
  the three focused suites passed 99/99. Separate task review found no blocking
  issue; only the expected real-SST runtime verification remains.

### Task 4 Complete: Explicit-NoC Observability Runner Profile

- The distributed scaling runner accepts `explicit_noc` alongside the existing
  modeled-NoC aliases and still rejects `shared`; its point signature carries
  `reduction_transport=explicit_noc` for artifact/cache identity.
- TDD evidence: the focused runner assertion failed before the allowlist change,
  then the workload scaffold suite passed 67/67; `bash -n` and the requested
  explicit-NoC dry-run passed. Task review found no blocking issue.
- Minor follow-up recorded for final cleanup: update the shared-rejection text
  so it names the full modeled/explicit observability runner contract.

### Phase 4B Explicit-NoC Queue Smoke Complete

- Rebuilt `build/sst-elements/src/sst/elements/golem/.libs/libgolem.so` after
  synchronizing `sfu.cc` and `sfu.h`; the rebuilt library exports the explicit
  transport predicate and queue helper symbols.
- Canonical real SST artifact:
  `tests/artifacts/sweeps/sfu_unified_job_explicit_noc_smoke_20260713` with
  `rows=16, dim=512, chunk=256, worker_cores=4, band_cores=4` and
  `distributed_reduction_transport=explicit_noc`. Manifest records
  `PASS,0,...,artifact_validation=PASS`; run summary is `407.503 us` simulated
  and `83s` wall time.
- Offline logits golden verification of the runner-owned output recorded
  `checked=8192`, `mismatches=0`, and `max_abs_diff=3.90064624e-10`.
- Active core0-core3 each record 16 max requests, 16 max responses, 16 sum
  requests, and 16 sum responses; every per-kind total is 64
  (`rows * worker_cores`). `sfu_partial_submits=128` and
  `sfu_partial_done=64` also match the shared-reducer lifecycle.
- DMA lifecycle is clean: read/write issue and completion are all 64, read/write
  bytes are 32768 each, and timeout retry, exhausted, and write-timeout retry
  are all 0. The first standalone verifier invocation omitted its CLI-required
  `--a-file/--b-file` arguments despite logits reference; rerunning with the
  artifact inputs produced the PASS above.
- Final regression after aligning the runner diagnostic test with the expanded
  modeled/explicit transport contract: 100 focused tests pass, `bash -n` and
  `git diff --check` pass, and a whole-Phase-4B read-only review found no
  blocking issue.

### Task 4 Baseline Final Retry (Blocked)

- 2026-07-14 exact SFU-disabled 64x64x64 fp32 GEMM command was launched outside
  the sandbox with the requested archived CPU architecture script and all four
  SFU environment variables unset. The runner generated fresh sample tensors,
  HBM backing files, and compiled the guest binary using the absolute compiler
  `/data/lzq/packages/install/riscv64_musl_toolchain/bin/riscv64-linux-musl-g++`.
- SST launch was selected as the absolute binary
  `/data4/jjgong/local/sstcore/bin/sst`, but it failed before simulation start:
  `error while loading shared libraries: libpython3.13.so.1.0: cannot open
  shared object file: No such file or directory`.
- Therefore no `VERIFY-C` result, DMA lifecycle statistics, or SFU/reduction
  inactivity evidence exists for run `run_20260714_191750_3305034`. Stopped on
  this launch-environment failure without changing the command or source.

### Task 4 Baseline After Generic-Runner Fixes (Blocked)

- 2026-07-14 reran the original SFU-disabled fp32 `64x64x64` GEMM baseline
  outside the sandbox with `GOLEM_ARCH_SCRIPT=small/mvm_noc_softmax_cpu/ncores_selfcom_dma_softmax_archive.py`,
  group manager, ctrl link, and WCP disabled, and all `GOLEM_SFU_*` and
  reduction-transport variables unset. The generic runner generated fresh
  sample tensors/HBM files and compiled with the required absolute compiler
  `/data/lzq/packages/install/riscv64_musl_toolchain/bin/riscv64-linux-musl-g++`.
- The SST/LD fix is effective: SST started and instantiated the archived
  architecture, but Vanadis aborted during wire-up because the archive resolved
  the guest executable as
  `tests/architecture/small/mvm_noc_int_array/riscv64/test_noc_dma` rather than
  the built `tests/small/mvm_noc_int_array/riscv64/test_noc_dma`.
- Run `run_20260714_195218_3348967` therefore has no `VERIFY-C` result, DMA
  lifecycle totals, or post-simulation SFU/reduction inactivity statistics. The
  archived log contains no SFU or reduction activity before the ELF-path fatal;
  stopped on this failure without retrying or changing source.

### Task 4 Original GEMM Default Baseline (Blocked)

- 2026-07-14 ran the requested exact no-override baseline as
  `env -u GOLEM_SFU_ENABLE -u GOLEM_SFU_STANDALONE_SOFTMAX -u GOLEM_SFU_JOB_SOFTMAX -u GOLEM_SFU_PRIMITIVE_SOFTMAX -u GOLEM_ARCH_SCRIPT GOLEM_GROUP_MANAGER_ENABLE=0 GOLEM_CTRL_LINK_ENABLE=0 GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0 bash src/sst/elements/golem/tests/run_noc_dma_pipeline.sh --gemm-m 64 --gemm-n 64 --gemm-k 64 --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 --dtype fp32 --tensor-source sample --verify-c` outside the sandbox. With `GOLEM_ARCH_SCRIPT` unset, the generic runner selected its normal `architecture/ncores_selfcom_dma_ctrl.py` default.
- The guest compiler command used
  `/data/lzq/packages/install/riscv64_musl_toolchain/bin/riscv64-linux-musl-g++`
  with all `64x64x64` GEMM/block defines plus
  `-DGOLEM_GROUP_MANAGER_ENABLE=0`, `-DGOLEM_CTRL_LINK_ENABLE=0`, and
  `-DGOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0`. SST was launched as
  `/data4/jjgong/local/sstcore/bin/sst --num-threads=1 architecture/ncores_selfcom_dma_ctrl.py`.
- Run `run_20260714_200412_3356946` failed during SST Python architecture
  construction: `RuntimeError: control endpoint missing while
  GOLEM_CTRL_LINK_ENABLE=1`. This contradicts the runner's printed value and
  the requested process environment (`0`), and occurred before guest execution.
- Consequently `VERIFY-C` did not execute and no DMA read/write issue,
  completion, byte, or retry statistics exist. The sole SST log
  (`artifacts/logs/test_default_run_20260714_200412_3356946.log`) contains no
  `SFU`, `softmax`, `reduction`, or DMA lifecycle activity before the fatal;
  stopped on this failure without retrying or modifying source.

### Task 4 Clean Default GEMM Baseline (Incomplete)

- 2026-07-14 invoked the exact requested baseline command with
  `GOLEM_SFU_ENABLE`, `GOLEM_SFU_STANDALONE_SOFTMAX`,
  `GOLEM_SFU_JOB_SOFTMAX`, `GOLEM_SFU_PRIMITIVE_SOFTMAX`,
  `GOLEM_ARCH_SCRIPT`, `GOLEM_GROUP_MANAGER_ENABLE`,
  `GOLEM_CTRL_LINK_ENABLE`, and `GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE`
  explicitly unset. No group/control/WCP values were supplied. The runner
  selected its default `architecture/ncores_selfcom_dma_ctrl.py`, whose guest
  is `tests/small/mvm_noc_int_array/riscv64/test_noc_dma`.
- The sandboxed attempt stopped in Open MPI initialization because it could not
  enumerate a network interface. The outside-sandbox retry reached and
  completed SST simulation for run `run_20260714_201618_3364519`:
  `Simulation is complete, simulated time: 228.468 us` in
  `artifacts/logs/test_default_run_20260714_201618_3364519.log`.
- The launcher was killed by its 60-second external command limit during
  `[4/4] Unpacking C tensor from HBM output`. Therefore `VERIFY-C` did not run
  and this is not a passing end-to-end baseline. The captured DMA evidence is
  core 4: `write_issue_count=1`, `write_bytes_total=16384`,
  `write_completion=1`, `wait_count=1`, and all timeout-retry/exhausted
  counters zero. The completed SST log has no `SFU`, `softmax`, or `reduction`
  entry. Stopped on this incomplete run without source changes or further
  retries.

## 2026-07-14 SimpleNetwork Reduction Transport Closure

### Real Explicit-NoC Anchor PASS

- The first real `SimpleNetwork` reduction smoke is recorded at
  `tests/artifacts/sweeps/sfu_unified_job_simple_network_smoke_20260714_runner_gate_fix`.
  The canonical point is `rows=16, dim=512, chunk=256, worker_cores=4`, and
  `band_cores=4` with `distributed_reduction_transport=explicit_noc`.
- Manifest outcome is `PASS,0,...,artifact_validation=PASS`; the logits golden
  checker reports `checked=8192`, `mismatches=0`, and the four distributed
  max/sum request/response counters total 64 each (`rows * worker_cores`).
- The real event path carries 256 reduction events: SFU's exported
  `sfu_reduction_transport_received` statistic totals 256, matching two
  request and two response messages for every worker-row. Runtime GlobalMemory
  diagnostics record `immediate=256`, `queued=0`, and `received=256`.
- DMA lifecycle is clean: read/write issue and completion each total 64, read
  and write bytes each total 32768, and timeout retry/exhausted/write retry are
  all zero.

### Integration Repairs Required by the Real Smoke

- `ReductionTransportEvent` inherits SST's pooled event base and cannot be
  allocated with `new (std::nothrow)`; use ordinary `new`.
- `golem.cc` is the ELI aggregation translation unit. It must include
  `globalmemory/globalmemory.h`; otherwise a rebuilt `libgolem.so` contains
  implementation code but stale GlobalMemory ELI metadata.
- `memHierarchy` also instantiates the GlobalMemory ELI builder through
  `memNICBase.h`. After changing GlobalMemory ELI declarations, rebuild and
  install both `memHierarchy` and `golem`; rebuilding only `libgolem.so`
  leaves stale installed `libmemHierarchy.so` metadata active at runtime.
- GlobalMemory nested-subcomponent statistics are not emitted to
  `stats_selfcom.txt` in this configuration. The explicit runner therefore
  uses the exported SFU transport-receive total as its authoritative artifact
  gate, while retaining GlobalMemory counters as runtime diagnostics.

### Default GEMM Non-Regression PASS

- The original GEMM baseline must use its normal default architecture
  `architecture/ncores_selfcom_dma_ctrl.py` and guest
  `small/mvm_noc_int_array/riscv64/test_noc_dma`. Do not route it through the
  obsolete `small/mvm_noc_softmax_cpu` archive or force historical no-ctrl
  variables.
- The final default `64x64x64` fp32 `run_noc_dma_pipeline.sh --verify-c`
  invocation completed with exit status 0; SST simulated time is `228.468 us`.
  The run summary and DMA summary were written, timeout retry/exhaustion are
  zero, and its stats contain no `sfu_`, `reduction`, or
  `gmem_reduction` activity.
- Generic runner hardening now makes this baseline reproducible in a
  non-interactive shell: it uses the configured RISC-V musl toolchain,
  absolute SST binary, documented Conda/SST dynamic-library path, and preserves
  an explicitly supplied architecture script across preset loading.

### Current Boundary and Next Matrix

- The shared reducer remains a same-process functional oracle. Cross-rank abort
  notification and bounded reuse of an aborted reduction identity require a
  future protocol generation/abort-event extension and are outside this cut.
- The next performance matrix varies `worker_cores`, `band_cores`, and
  `GOLEM_SFU_REDUCTION_VN`, while retaining the `16:512:4:4` anchor and its
  golden, reduction, transport, DMA, and default-GEMM gates.
