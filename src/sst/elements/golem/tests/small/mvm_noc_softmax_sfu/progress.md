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
