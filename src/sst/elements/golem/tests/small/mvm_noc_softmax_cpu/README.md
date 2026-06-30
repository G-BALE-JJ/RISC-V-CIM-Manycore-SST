# MVM NoC Softmax CPU Fallback

本目录是当前 GEMM 主测试的隔离式 softmax 扩展工程。目标是在尽量不修改
`../mvm_noc_int_array` 的前提下，复用其 RISC-V 应用结构、Golem GEMM
runtime、dtype/layout 定义和 SST/Vanadis 运行环境，新增由 CPU 执行的
FP32 softmax kernel。

## 目标

- 保留原 GEMM 主测试目录 `../mvm_noc_int_array` 作为 GEMM-only 基线。
- 在本目录新增 `test_noc_dma_softmax`，形成 GEMM + CPU softmax 的独立测试入口。
- softmax v1 使用 `GOLEM_DTYPE_FP32`、`GOLEM_LAYOUT_ROW_MAJOR`、二维 row-major 张量。
- 正式接入路径使用 GM/HBM 地址感知版本，从 GEMM 输出地址读取 logits，由 RISC-V CPU 计算 softmax，再写回指定输出地址。
- 所有 softmax 工程相关文档都保存在本目录，包括计划、发现、进度、设计说明和后续 README 更新。

## 与原项目的隔离策略

本目录采用同级隔离方式：

```text
tests/small/
  mvm_noc_int_array/        # 原 GEMM-only 主测试，尽量不修改
  mvm_noc_softmax_cpu/      # 新 softmax CPU fallback 工程
```

本工程允许通过 include 和编译输入复用 `../mvm_noc_int_array` 中的公共代码：

- `golem_matmul_runtime.h`
- `golem_matmul_runtime.cpp`
- `pipeline_config.h`
- `gm_config.h`
- `operators.h`
- `core_bind.h`

但 softmax 自身的 runtime、测试入口、计划和说明文档应放在本目录内。

阶段 9 起，softmax 也在共享 `../golem_operator_api.h` 中登记了声明级 operator API：

- `GolemOperatorKind::SOFTMAX`
- `SoftmaxOpDesc`
- softmax 的 `validate_op_desc` / `validate_compatibility` 声明

注意该共享 API 的 C++ enum 数值与 `../mvm_noc_int_array/golem_matmul_runtime.h` 中的
runtime C ABI enum 不完全一致。本目录通过 `golemMakeSoftmaxRuntimeDesc` 显式转换
`SoftmaxOpDesc` 到 `golem_softmax_op_desc_t`，不直接 cast dtype/layout。

## 计划文件

- `design.md`: 架构设计和约束。
- `task_plan.md`: 实现任务拆分和状态。
- `findings.md`: 代码阅读发现、接口判断、风险记录。
- `progress.md`: 实施过程记录、验证命令和结果。

## 初始实现边界

softmax v1 的建议边界：

- 支持 `GOLEM_DTYPE_FP32`。
- 不支持 `GOLEM_DTYPE_INT32`，返回 `GOLEM_STATUS_UNSUPPORTED`。
- 支持 `axis = -1` 或 `axis = 1`。
- 支持二维 row-major contiguous 布局。
- 支持 pointer 核心函数用于自测。
- 正式端到端路径使用 GM/HBM 地址版本。

## 后续预期文件

```text
mvm_noc_softmax_cpu/
  README.md
  design.md
  task_plan.md
  findings.md
  progress.md
  Makefile
  test_noc_dma_softmax.cpp
  test_softmax_operator_api_compile.cpp
  golem_softmax_runtime.h
  golem_softmax_runtime.cpp
  softmax_config.h
```

## 工具链

本目录的 `Makefile` 固定使用以下 RISC-V musl 工具链路径：

```text
/data/lzq/packages/install/riscv64_musl_toolchain/bin
```

默认 `ARCH=riscv64` 时，C++ 编译器为：

```text
/data/lzq/packages/install/riscv64_musl_toolchain/bin/riscv64-linux-musl-g++
```

本工程当前不使用 native C++ 编译链。`Makefile` 只支持 RISC-V binary：

```bash
make
make ARCH=riscv64
```

如果传入 `ARCH=native`，`Makefile` 会直接报错，避免误用本机 `g++` 或其他 native
编译环境。Python checker 可在本机解释器下运行，但它不依赖 native C++ 编译链。

## 运行入口

本目录提供隔离 wrapper：

```bash
./run_noc_dma_softmax_pipeline.sh
```

它会：

1. 按本次 pipeline 参数在本目录构建 `riscv64/test_noc_dma_softmax`。
2. 设置 `VANADIS_EXE` 指向该 binary。
3. 自动判断原 GEMM-only build metadata 是否匹配；匹配时设 `GOLEM_SKIP_BUILD=1`，
   不匹配时设 `GOLEM_SKIP_BUILD=0` 让原 pipeline 刷新 metadata。
4. 默认设置 `GOLEM_VERIFY_C=0`，避免用 GEMM golden 校验已被 softmax 覆盖的 C 输出。
5. 固定 RISC-V 工具链、SST 路径和 SST 运行所需 Python/SST 动态库路径。
6. 调用原 `../../run_noc_dma_pipeline.sh`。

该方式不修改原 `../mvm_noc_int_array` 和原 pipeline。原架构脚本已支持通过
`VANADIS_EXE` 覆盖默认应用 binary。

### dry-run

建议先运行 dry-run 确认配置：

```bash
./run_noc_dma_softmax_pipeline.sh --dry-run
```

wrapper 会先打印实际传给 SST 架构脚本的应用：

```text
[SOFTMAX] VANADIS_EXE=.../mvm_noc_softmax_cpu/riscv64/test_noc_dma_softmax
```

原 pipeline 的 dry-run 输出仍可能显示：

```text
skip build and reuse small/mvm_noc_int_array/riscv64/test_noc_dma
```

这是原 pipeline 对 `GOLEM_SKIP_BUILD=1` 的构建阶段提示，不代表 SST 会执行
GEMM-only binary。SST 架构脚本最终使用 `VANADIS_EXE`，因此实际应用仍是本目录的
`test_noc_dma_softmax`。

### 与原 pipeline 的一个限制

为了尽量不修改原 pipeline，wrapper 复用 `../../run_noc_dma_pipeline.sh`。该脚本在
`GOLEM_SKIP_BUILD=1` 时仍会校验原 GEMM-only 构建产物：

```text
small/mvm_noc_int_array/riscv64/test_noc_dma
small/mvm_noc_int_array/riscv64/test_noc_dma.build.env
```

如果这两个文件不存在，wrapper 会提前报错并提示处理方式。可以使用以下命令让原
pipeline 重新生成这份 build metadata：

```bash
GOLEM_SKIP_BUILD=0 ./run_noc_dma_softmax_pipeline.sh <pipeline 参数>
```

即使 `GOLEM_SKIP_BUILD=0` 触发原 GEMM-only binary 构建，SST 实际执行的应用仍由
`VANADIS_EXE` 指向本目录的 `riscv64/test_noc_dma_softmax`。

当前 wrapper 默认会自动完成这件事：当本次参数与旧 metadata 不匹配时，自动把
`GOLEM_SKIP_BUILD` 置为 `0`；当 metadata 已匹配时，自动置为 `1`。

wrapper 还提供本目录内的 `bin/sst` shim。原 pipeline 仍调用命令名 `sst`，但 PATH
优先解析到该 shim，由 shim 只在启动真实 SST 时设置：

```text
LD_LIBRARY_PATH=/data4/jjgong/miniconda3/lib:/data4/jjgong/local/sstcore/lib:...
```

这样可以满足 `libpython3.13.so.1.0` 依赖，同时避免把 conda lib 全局注入到整个
pipeline 的 bash 进程。

### 当前语义边界

当前端到端入口执行的是 GEMM 后的 tile-local softmax：

- 每个 worker core 对自己负责的 C tile 调用 `golemRunSoftmaxCpuGmForCore`。
- softmax 输入和输出地址默认都使用 `GemmTaskDescriptor::c_base_mm`，即覆盖该 tile 的 GEMM 输出。
- 每个 tile 按 `[block_m, block_n]` row-major 计算 `axis=-1` softmax。

这还不是完整 GEMM 输出矩阵 `[M, N]` 的全行 softmax。如果一行跨多个 N tile，完整
row-wise softmax 需要先跨 tile 聚合该行所有 logits，再统一归一化。该部分留给后续
阶段实现。

因此当前 wrapper 默认关闭原 pipeline 的 `verify_c_against_golden.py`。该校验只检查
`C=A*B`，而 softmax 入口会覆盖 C tile；后续若要做数值正确性验证，应在本目录新增
softmax 专用 golden checker。

### smoke run

建议优先使用单 core、单 tile 配置做端到端 smoke。该配置关闭 group manager 和
control-link，避免一个 64x64x64 单 tile 测试启动 20 个 core 后引入额外组同步开销：

```bash
./run_noc_dma_softmax_pipeline.sh \
  --groups 1 --num-cores 1 --gemm-cores 1 \
  --num-mem-nodes 2 --mesh-dim-x 1 \
  --num-arrays 64 --array-in 64 --array-out 64 \
  --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --group-manager-enable 0 --ctrl-link-enable 0 \
  --log softmax_smoke_1core.log
```

`--group-manager-enable` 和 `--ctrl-link-enable` 是本 wrapper 支持的隔离参数。wrapper
会用它们设置对应环境变量和编译期宏，但不会把这些私有参数透传给原 pipeline。

该 1-core smoke 已在用户正常 shell 中完成端到端验证。关键证据：

```text
[Core 0] [SOFTMAX] tile-local softmax complete: tiles=1
RoCC core=0 MVM_PROGRESS: completed=64/64 (100%)
Simulation is complete, simulated time: 810.8 us
```

对应产物：

```text
artifacts/logs/softmax_smoke_1core_run_20260623_194325_842685.log
artifacts/stdout/overlap0/run_20260623_194325_842685/
artifacts/stats/overlap0/run_20260623_194325_842685/execution_summary.csv
artifacts/stats/overlap0/run_20260623_194325_842685/dma_summary.csv
```

本次 `execution_summary.csv` 的主要统计：

```text
total_cycles=19562
compute_active_time=11259
control_other_time=8303
```

### softmax 数值校验

本目录新增实验性 checker：

```bash
python3 verify_softmax_tile_against_golden.py \
  --dtype fp32 \
  --a-file data/a.bin \
  --b-file data/b.bin \
  --c-file /path/to/unpacked_softmax_c.bin \
  --m 64 --n 64 --k 64 \
  --block-m 64 --block-n 64
```

wrapper 支持后置验证开关：

```bash
./run_noc_dma_softmax_pipeline.sh <smoke 参数> --verify-softmax
```

该开关会在原 pipeline 成功后先调用原项目的
`tools/unpack_c_from_hbm.py` 解包 HBM C 输出，再调用本目录 checker。

默认验证模式是：

```text
--reference probability
```

该模式不假设 GEMM 输出等于 naive `A@B`，只验证 HBM 输出中每个 tile-local row 是合法
softmax 概率分布：所有值有限、位于 `[0,1]` 容差范围内、行和接近 1。已有 1-core
artifact 在该模式下通过：

```text
[VERIFY-SOFTMAX] PASS reference=probability dtype=fp32 checked=4096 bad_rows=0 max_row_sum_abs_diff=9.14315024e-09
```

更严格的诊断模式是：

```text
--reference a_b
```

该模式比较 naive `tile-local softmax(A@B)`。当前旧 1-core artifact 在该模式下仍不
通过，因为 GEMM GM fast path 的本地累加/写回布局与 naive row-major `A@B` 的列分布
尚未完全对齐。该问题留给后续 GEMM FP32 layout/golden 对齐阶段，不阻塞当前 CPU
fallback softmax 的概率分布校验。

如果要复现默认 20-core 配置，可以使用：

```bash
./run_noc_dma_softmax_pipeline.sh \
  --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --log softmax_smoke.log
```

当前观察到默认 20-core 配置能完成 GEMM MVM 阶段，但在收尾阶段被系统 `Killed`；
单 core smoke 更适合作为 softmax 接入验证。

在当前 Codex 沙箱中，真实 SST 运行仍会因为 OpenMPI 初始化 socket 被拒绝而失败：

```text
opal_ifinit: socket() failed with errno=1
```

这属于运行环境权限限制。dry-run、RISC-V 构建和 Python checker 单元测试可在沙箱内
执行；真实 SST smoke 需要在正常 shell/终端里执行上述命令。
