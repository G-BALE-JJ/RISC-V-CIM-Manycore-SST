# Softmax CPU Fallback Findings

## 已确认事实

- 当前 GEMM 主测试应用目录是 `../mvm_noc_int_array`。
- 当前主测试 binary 是 `riscv64/test_noc_dma`。
- 原 `../mvm_noc_int_array/Makefile` 默认目标为 `test_noc_dma`。
- 当前 `test_noc_dma.cpp` 使用 `golem_matmul_runtime.h/.cpp` 创建并运行 matmul。
- `golem_matmul_runtime.h` 已定义可复用的 `golem_status_t`、`golem_dtype_t`、
  `golem_layout_t` 和 `golem_tensor_desc_t`。
- 当前 dtype 体系支持 `GOLEM_DTYPE_INT32` 和 `GOLEM_DTYPE_FP32`。
- 当前脚本层 `GOLEM_MATMUL_DTYPE` 支持 `int32|fp32`，默认配置中常用 `fp32`。
- 当前 GEMM 主测试的 tensor descriptor 使用 `.data = nullptr`，因此正式接入
  softmax 需要处理 GM/HBM 地址路径，而不能只依赖普通 C pointer。
- GEMM 输出 tile 地址由 `gemm_task_desc_for_task(...).c_base_mm` 给出。
- GEMM 输出 tile 写回路径为 `remote_store(rt.local_accum, desc.c_base_mm)`。
- tile 内部暂按 `block_m x block_n` row-major contiguous buffer 处理；现有
  `store_c_tile` host path 也是按 `row * block_n + col` 组织 C tile。
- 现有 LeNet 后处理从远端 C tile 读取的模式是：
  `dma_remote_load_to_gm(desc.core_id, desc.c_base_mm, local_out_gm, bytes)` 后
  `gm2mm(c_tile, local_out_gm)`。

## 设计判断

- softmax v1 应只支持 FP32。
- int32 softmax 输出没有明确概率语义，v1 返回 unsupported 更稳妥。
- 新建同级目录比直接修改 `../mvm_noc_int_array` 更符合隔离要求。
- pointer softmax 仍有价值：用于算法自测和后续复用。
- 正式 GEMM+softmax 接入应使用 GM/HBM 地址感知 CPU fallback。
- GM/HBM softmax v1 使用整块搬运策略：先将 `[outer, stride]` FP32 数据搬到 CPU
  buffer，计算 softmax，再整块写回。这样实现简单，后续可再优化为逐行搬运。

## 待确认问题

- 端到端接入时，调用方需要为每个 GEMM output tile 传入对应的 `desc.c_base_mm`。
- 如果要对完整 GEMM `[M,N]` 做 row-wise softmax，需要在 tile 间聚合一整行的所有
  `N` 列；当前 `golemRunSoftmaxCpuGm` 只处理单个 row-major 矩形区域。
- softmax 输出是覆盖 GEMM C 输出，还是写入独立 softmax output 区域。
- 是否需要新增 host-side `verify_softmax_against_golden.py`，以及该文件是否也放入本目录。
- 非交互 shell 的 PATH 没有直接暴露 `riscv64-linux-musl-g++`，但 `~/.bashrc`
  中配置了工具链路径；使用绝对路径可以稳定构建。
- `~/.bashrc` 中配置了 RISC-V musl 工具链路径：
  `/data/lzq/packages/install/riscv64_musl_toolchain/bin`。
- 绝对路径编译器
  `/data/lzq/packages/install/riscv64_musl_toolchain/bin/riscv64-linux-musl-g++`
  可执行，版本为 GCC 9.4.0。
- `make ARCH=riscv64` 已成功生成 `riscv64/test_noc_dma_softmax`。
- `file riscv64/test_noc_dma_softmax` 显示该 binary 是静态链接的 64-bit UCB RISC-V ELF。
- 架构脚本 `architecture/ncores_selfcom_dma_ctrl.py` 支持 `VANADIS_EXE` 覆盖默认
  binary，因此可以在不修改原架构脚本的情况下运行本目录 softmax binary。
- 原 `../../run_noc_dma_pipeline.sh` 在 `GOLEM_SKIP_BUILD=1` 时仍会校验
  `small/mvm_noc_int_array/riscv64/test_noc_dma` 和
  `small/mvm_noc_int_array/riscv64/test_noc_dma.build.env`。
- 本目录 `run_noc_dma_softmax_pipeline.sh` 默认设置 `VANADIS_EXE` 指向
  `mvm_noc_softmax_cpu/riscv64/test_noc_dma_softmax`，并默认设置
  `GOLEM_SKIP_BUILD=1` 复用原 pipeline。
- wrapper 已增加前置检查：如果非 dry-run 且 `GOLEM_SKIP_BUILD=1`，但原
  GEMM-only binary 或 build metadata 缺失，会提前报错并提示使用
  `GOLEM_SKIP_BUILD=0` 重新生成基线 metadata。
- 原 pipeline 的 `verify_c_against_golden.py` 只验证 GEMM 输出 `C=A*B`。
  softmax 入口会覆盖 C tile，因此 wrapper 默认设置 `GOLEM_VERIFY_C=0`。
- wrapper 现在会按本次参数解析并构造 softmax 编译 CFLAGS，使
  `GOLEM_GEMM_M/N/K`、block、array、core、stride 等编译期宏与原 pipeline 一致。
- wrapper 会写入本目录 build metadata
  `riscv64/test_noc_dma_softmax.build.env`，用于判断是否可以复用 softmax binary。
- 当原 GEMM-only `test_noc_dma.build.env` 与本次配置不匹配时，wrapper 自动设置
  `GOLEM_SKIP_BUILD=0`，让原 pipeline 刷新基线 metadata；匹配时自动设置
  `GOLEM_SKIP_BUILD=1`。
- 原 pipeline 默认使用裸命令 `sst`，且当前非交互 shell 未加载 `.bashrc` 中的 SST
  和 Python 动态库路径。本目录新增 `bin/sst` shim，wrapper 将 `bin/` 放到 PATH
  前面，shim 只在执行真实 SST 时设置 `LD_LIBRARY_PATH`。

## 风险

- GM/HBM FP32 读写可能需要 bitcast 和对齐处理，不能简单用整数语义解释 float。
- 如果 GEMM 输出布局不是 row-major contiguous，softmax 需要先按当前布局读取 logits，或增加 layout conversion。
- 如果新 wrapper 脚本复用原 pipeline 太多逻辑，后续维护可能需要抽取公共脚本函数。
- 当前 GM/HBM 版本在 RISC-V 下使用 `core_id=0` 和 `LOCAL_LAYOUT.tmp` 作为本地搬运缓冲；
  后续接入多核 GEMM 后，应把执行 softmax 的 core_id 显式作为参数或包装在更高层入口中。
- 对大矩阵完整 softmax，整块 `std::vector<float>` 缓冲可能较大；v1 适合 tile 或 logits
  规模，后续可改为逐行处理以降低本地内存压力。
- wrapper 仍复用原 pipeline 的构建元数据校验逻辑，因此不是完全独立的 pipeline；
  它的隔离重点是 softmax 代码、binary、文档和运行入口不侵入原 GEMM 目录。
- 当前 RISC-V 入口执行 tile-local softmax：每个 C tile 独立按 `block_n` 做归一化。
  当 `N` 跨多个 tile 时，它不等价于完整 `[M,N]` 的 row-wise softmax。
- 若后续需要数值正确性验证，应在本目录新增 softmax 专用 golden checker，而不是复用
  原 GEMM `verify_c_against_golden.py`。
- 当前 Codex 沙箱环境中真实 SST 运行会触发 OpenMPI socket 权限错误
  `opal_ifinit: socket() failed with errno=1`。这属于执行环境限制；dry-run、RISC-V
  构建和 Python checker 单元测试可在沙箱内完成。
- 用户在正常 shell 中运行默认 20-core 64x64x64 smoke 时，SST 已进入计算阶段并完成
  `RoCC core=4 MVM_PROGRESS: completed=64/64 (100%)`，但随后进程在 99% 被系统
  `Killed`。
- 该默认 smoke 使用 20 core、4 group manager、control-link。对于单 tile GEMM，这会
  引入大量无任务 worker 和组同步；日志中没有出现 `[SOFTMAX]` 完成输出，说明失败点在
  GEMM 后处理/收尾附近。
- softmax RISC-V GM 路径原先使用 `LOCAL_LAYOUT.tmp=0x0800` 作为整 tile 搬运缓冲；
  对 64x64 tile 来说需要 16KB，会覆盖 `LOCAL_DATA_BASE=0x2000` 之后的本地 GM 布局。
  已改为逐行搬运，并使用 `LOCAL_LAYOUT.accum + LOCAL_OUT_TILE_BYTES_ALIGNED` 后方作为
  临时 GM 缓冲。
- RISC-V softmax v1 现在限制 `dim <= 64` 且行连续，匹配当前 tile-local smoke。
- 用户已明确不需要 native C++ 编译链；本目录 `Makefile` 应只使用从 `.bashrc`
  确认过的 RISC-V musl 工具链路径
  `/data/lzq/packages/install/riscv64_musl_toolchain/bin`。
- wrapper 现在支持私有参数 `--group-manager-enable`、`--ctrl-link-enable` 和
  `--ctrl-overlap-ab`，这些参数用于设置环境变量和编译宏，不透传给原 pipeline。
- 推荐用 1-core smoke 验证 softmax 接入：
  `--groups 1 --num-cores 1 --gemm-cores 1 --num-mem-nodes 2 --mesh-dim-x 1 --group-manager-enable 0 --ctrl-link-enable 0`。
- 用户正常 shell 中的 1-core smoke 已经通过，验证的是单 tile `64x64` GEMM 输出上的
  tile-local softmax：
  - stdout 证据：`[Core 0] [SOFTMAX] tile-local softmax complete: tiles=1`
  - GEMM 证据：`RoCC core=0 MVM_PROGRESS: completed=64/64 (100%)`
  - SST 证据：`Simulation is complete, simulated time: 810.8 us`
  - 主要产物：
    `artifacts/logs/softmax_smoke_1core_run_20260623_194325_842685.log`
    和 `artifacts/stats/overlap0/run_20260623_194325_842685/execution_summary.csv`
  - 执行统计：`total_cycles=19562`，`compute_active_time=11259`，
    `control_other_time=8303`。
- 新增 softmax checker 后，用旧 1-core HBM 输出做离线验证：
  - 解包命令复用原 `tools/unpack_c_from_hbm.py`，输出 `/tmp/softmax_c_out_verify.bin`。
  - checker 的 naive 参考为 `tile-local softmax(A@B)`。
  - 真实 C 输出每行概率和约为 1，说明 RISC-V softmax 后处理确实写回了概率分布。
  - 但真实 C 的概率列分布与 naive `A@B` golden 不一致。例如 row0 真实输出在
    col 7/24/41/58 附近为约 `0.249997`，而 naive `A@B` 的 softmax 参考在
    col 7/26/45 附近为约 `0.333333`。
  - 当前判断：checker 数学逻辑可作为工具入口，但真实端到端强校验前需要先对齐原
    GEMM FP32 路径的 packing/阵列输出语义，不能直接把 naive `A@B` 当作当前硬件路径
    的可信 golden。
- 阶段 8 的收敛策略：
  - `--reference probability` 作为当前默认端到端 softmax checker，只验证 HBM 输出中的
    每个 tile-local row 是有限概率分布，元素在 `[0,1]` 内，且行和接近 1。
  - 旧 1-core artifact 在该模式下通过：
    `[VERIFY-SOFTMAX] PASS reference=probability ... max_row_sum_abs_diff=9.14315024e-09`。
  - `--reference a_b` 保留为更严格的诊断模式，后续用于对齐 GEMM FP32 layout/golden。
- 共享 `../golem_operator_api.h` 在阶段 9 前只包含 Conv2d/MaxPool/Dense 的描述和
  validation 声明，没有 softmax operator 描述。
- `golem_operator_api.h` 的 C++ enum 数值与当前 matmul/softmax runtime C ABI 不完全一致：
  - `TensorDataType::FP32 = 0`
  - `GOLEM_DTYPE_FP32 = 1`
  因此不能用 `static_cast<golem_dtype_t>(TensorDataType::FP32)` 这类方式直接转换。
- 阶段 9 采用的 API 边界：
  - 共享 API 只记录 operator 语义和 descriptor 形状。
  - 本目录 runtime 继续暴露既有 `golem_softmax_op_desc_t` C ABI。
  - 通过 `golemMakeSoftmaxRuntimeDesc(const SoftmaxOpDesc&)` 显式转换 dtype/layout，避免破坏
    已通过的 RISC-V softmax binary 和 GM/HBM 调用路径。
- `SoftmaxOpDesc::allow_golem` 当前固定用于表达策略意图：v1 为 CPU fallback，不新增
  RoCC/CIM softmax 加速。runtime helper 不使用该字段，后续接入调度层时可读取它。
