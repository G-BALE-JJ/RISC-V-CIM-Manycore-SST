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
