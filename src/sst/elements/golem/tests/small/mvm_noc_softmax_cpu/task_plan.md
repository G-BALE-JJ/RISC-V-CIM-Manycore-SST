# Softmax CPU Fallback Task Plan

## 目标

在独立目录 `mvm_noc_softmax_cpu` 中实现 GEMM + CPU softmax 测试工程，尽量不修改
原 `../mvm_noc_int_array` GEMM 主测试目录。所有 softmax 相关代码、README、计划、
发现和进度记录保存在本目录。

## 当前状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 1. 工程目录和文档骨架 | complete | 创建隔离目录和项目文档 |
| 2. softmax API 和测试设计 | complete | 已定义 header、测试入口和预期行为 |
| 3. pointer softmax TDD | complete | 已完成 FP32 pointer softmax 实现；当前工程不再使用 native C++ 编译链 |
| 4. GM/HBM softmax 接入 | complete | 已确认 GEMM tile 输出地址模式并实现 GM/HBM 地址版本 |
| 5. 独立 Makefile | complete | 已固化 RISC-V musl 工具链；Makefile 仅支持 `ARCH=riscv64` |
| 6. pipeline wrapper | complete | 已新增本目录 wrapper，支持参数感知构建、metadata 自动刷新和 SST shim |
| 7. 验证和文档收尾 | complete | 构建、自测、1-core dry-run 和用户正常 shell 端到端 smoke 通过 |
| 8. softmax checker | complete | `--verify-softmax` 默认验证真实 HBM 输出为合法 tile-local 概率分布；严格 `softmax(A@B)` 模式保留为后续 GEMM layout/golden 对齐诊断 |
| 9. softmax operator API 整理 | complete | 在共享 `../golem_operator_api.h` 增加 softmax operator 描述，并在本目录提供 API descriptor 到 runtime C ABI descriptor 的显式转换 |

## 文件规划

| 文件 | 动作 | 职责 |
|---|---|---|
| `README.md` | 已创建 | 工程入口说明、隔离策略、文件索引 |
| `design.md` | 已创建 | 架构设计、dtype/layout、接入原则 |
| `task_plan.md` | 已创建 | 任务拆分和状态跟踪 |
| `findings.md` | 已创建 | 代码阅读发现和风险记录 |
| `progress.md` | 已创建 | 实施日志和验证结果 |
| `golem_softmax_runtime.h` | 已创建 | softmax op desc 和 runtime API |
| `golem_softmax_runtime.cpp` | 已创建 | FP32 pointer softmax 实现和 GM/HBM stub |
| `softmax_config.h` | 已创建 | softmax 编译期配置 |
| `test_noc_dma_softmax.cpp` | 已创建 | 独立 softmax pointer 自测入口 |
| `Makefile` | 已创建 | 独立构建 `test_noc_dma_softmax` |
| `.gitignore` | 已创建 | 忽略本目录构建产物 |
| `verify_softmax_tile_against_golden.py` | 已创建 | 比较解包 C 与 tile-local `softmax(A@B)` golden |
| `test_verify_softmax_tile_against_golden.py` | 已创建 | checker 的最小单元测试 |
| `test_softmax_operator_api_compile.cpp` | 已创建 | RISC-V 编译期 smoke test，验证共享 softmax API 和本目录 runtime descriptor 转换 |

## 设计决策

| 决策 | 结果 | 原因 |
|---|---|---|
| 隔离目录 | 使用 `mvm_noc_softmax_cpu` | 保留原 GEMM 主测试目录 |
| 正式接入方式 | GM/HBM 地址版本 | 当前 GEMM 主路径 `.data = nullptr`，实际数据不走普通 pointer |
| 核心算法 | 保留 pointer 版本 | 便于 RISC-V 小数组自测和算法复用 |
| dtype v1 | 只支持 FP32 | softmax 输出是概率，int32 无明确语义 |
| 硬件路径 | 不新增 RoCC/CIM softmax | 当前目标是不做定制加速 |
| 文档位置 | 全部放本目录 | 便于后续统一管理 |

## 后续任务明细

### 阶段 2: softmax API 和测试设计

- [x] 创建 `golem_softmax_runtime.h`，定义 `golem_softmax_op_desc_t`。
- [x] 声明 `golemRunSoftmaxCpu` 和 `golemRunSoftmaxCpuGm`。
- [x] 创建 `softmax_config.h`，定义默认 enable 和 axis。
- [x] 在 `findings.md` 记录需要确认的 GEMM 输出地址和 stride。

### 阶段 3: pointer softmax TDD

- [x] 创建 `test_noc_dma_softmax.cpp` 的小数组自测模式。
- [x] 先让构建失败于缺少 `golemRunSoftmaxCpu`，确认测试覆盖 API。
- [x] 实现 FP32 pointer softmax。
- [x] 验证 `[1, 2, 3]` 输出近似 `[0.09003057, 0.24472848, 0.66524094]`。
- [x] 记录命令和结果到 `progress.md`。

### 阶段 4: GM/HBM softmax 接入

- [x] 阅读 `../mvm_noc_int_array/pipeline_config.h` 和 GEMM 输出写回路径。
- [x] 明确 C 输出地址、布局和 stride。
- [x] 实现 FP32 GM/HBM 读写 helper。
- [x] 实现 `golemRunSoftmaxCpuGm`。
- [x] 记录地址假设和风险到 `findings.md`。

### 阶段 5: 独立 Makefile

- [x] 创建 `Makefile`。
- [x] 编译本目录 `test_noc_dma_softmax.cpp` 和 `golem_softmax_runtime.cpp`。
- [x] 添加 `-I../mvm_noc_int_array` 和 `-lm`。
- [x] 固化 RISC-V musl 工具链路径。
- [x] 禁用 native C++ 构建入口，避免依赖本机 `g++`。
- [x] 保持原 `../mvm_noc_int_array/Makefile` 不变。

### 阶段 6: pipeline wrapper

- [x] 新增本目录 `run_noc_dma_softmax_pipeline.sh`。
- [x] 使用新 binary `small/mvm_noc_softmax_cpu/riscv64/test_noc_dma_softmax`。
- [x] 通过 `VANADIS_EXE` 覆盖架构脚本默认应用 binary。
- [x] 保持原 `tests/run_noc_dma_pipeline.sh` 作为 GEMM-only 基线。
- [x] 记录 `GOLEM_SKIP_BUILD=1` 仍会校验原 GEMM-only binary 和 build metadata 的限制。
- [x] 默认关闭原 GEMM `verify_c_against_golden.py`，避免校验 softmax 覆盖后的 C 输出。
- [x] wrapper 按本次参数传入 softmax 编译期宏，避免 binary 与 pipeline 配置不一致。
- [x] wrapper 自动判断/刷新原 GEMM-only build metadata。
- [x] 新增本目录 `bin/sst` shim，隔离设置 SST 运行所需动态库路径。
- [x] wrapper 支持私有 `--group-manager-enable`，便于构造 1-core smoke。
- [x] wrapper 过滤 `--group-manager-enable`、`--ctrl-link-enable`、`--ctrl-overlap-ab`，避免原 pipeline unknown option。

### 阶段 7: 验证和文档收尾

- [x] 运行最小构建验证。
- [x] 运行 RISC-V 构建和 wrapper dry-run 验证。
- [x] 运行 wrapper dry-run 和 1-core dry-run 验证。
- [x] 运行 GEMM+softmax SST 端到端验证。
- [x] 更新 `README.md` 的使用方法。
- [x] 更新 `progress.md` 和 `findings.md`。

### 阶段 8: softmax checker

- [x] 新增 `verify_softmax_tile_against_golden.py`。
- [x] 新增 checker 单元测试，覆盖正确 softmax 输出通过、错误输出失败。
- [x] wrapper 支持私有参数 `--verify-softmax` 和 `--softmax-c-file`。
- [x] wrapper 后置复用原 `tools/unpack_c_from_hbm.py` 生成 C dump。
- [x] checker 新增 `--reference probability`，验证每个 tile-local row 为有限概率分布。
- [x] `--verify-softmax` 默认使用 `probability` 模式，旧 1-core HBM artifact 已通过。
- [x] 保留 `--reference a_b` 作为严格 `tile-local softmax(A@B)` 诊断模式。
- [ ] 后续阶段对齐真实 FP32 GEMM 输出与 naive `A@B` 的 packing/阵列语义。

### 阶段 9: softmax operator API 整理

- [x] 阅读共享 `../golem_operator_api.h`，确认当前只有 Conv/Pool/Dense 描述。
- [x] 在共享 API 中新增 `GolemOperatorKind::SOFTMAX`。
- [x] 在共享 API 中新增 `SoftmaxOpDesc`，描述 `outer/dim/axis/dtype/layout`。
- [x] 声明 softmax 的 `validate_op_desc` 和 `validate_compatibility` overload，保持与现有 operator API 风格一致。
- [x] 在本目录 `golem_softmax_runtime.h` 增加 `golemMakeSoftmaxRuntimeDesc` 转换 helper。
- [x] 用 RISC-V 编译期 smoke test 验证共享 API descriptor 能转换为 `golem_softmax_op_desc_t`。
- [x] 保持 runtime C ABI 不变，避免影响已通过的 RISC-V 入口和 GM/HBM softmax。

## 错误记录

| 错误 | 处理 |
|---|---|
| 初次 RED 构建失败在缺少 `golem_softmax_runtime.cpp` 文件，而不是 API 未定义 | 先创建 header 和空 cpp，再重新 RED，最终得到 `golemRunSoftmaxCpu` 未定义的有效链接失败 |
| 初次未通过当前非交互 shell 的 PATH 找到 RISC-V 编译器 | 从 `~/.bashrc` 确认工具链路径后，在 Makefile 中使用绝对路径 `/data/lzq/packages/install/riscv64_musl_toolchain/bin/riscv64-linux-musl-g++`，RISC-V 构建已通过 |
| 用户要求不再需要 native 编译链 | Makefile 固定 `ARCH=riscv64`，禁用 `ARCH=native`，文档改为只推荐 `.bashrc` 中确认的 RISC-V musl 工具链路径 |
| 查询 pipeline 参数时使用 `grep -n "--m\|--n\|--k\|block"`，pattern 开头的 `--m` 被 grep 当成选项 | 改用 `rg -n -- "--m|--n|--k|block" ...`，查询成功 |
| 64x64x64 smoke run 失败：原 `test_noc_dma.build.env` 是旧配置 `1024/1024/128` | wrapper 改为参数感知构建并自动设置 `GOLEM_SKIP_BUILD=0` 刷新旧 metadata；刷新后 dry-run 显示可复用 |
| 原 pipeline 重建 baseline 时 `riscv64-linux-musl-g++: Permission denied` | wrapper 将 `/data/lzq/packages/install/riscv64_musl_toolchain/bin` 加入 PATH |
| `sst: error while loading shared libraries: libpython3.13.so.1.0` | 新增 `bin/sst` shim，只在真实 SST 进程中设置 conda Python/SST lib 的 `LD_LIBRARY_PATH` |
| Codex 沙箱运行 SST 出现 OpenMPI `socket() failed with errno=1` | 属于沙箱权限限制；真实 SST smoke 需在用户正常 shell 中运行 |
| 默认 20-core 64x64x64 smoke 在 GEMM MVM 100% 后 99% 处被系统 `Killed` | 记录为默认多核/组同步 smoke 风险；推荐改用 1-core smoke 做接入验证 |
| RISC-V softmax GM 路径使用 `LOCAL_LAYOUT.tmp=0x0800` 做 16KB tile 缓冲，可能覆盖本地 GM 布局 | 改为逐行搬运，RISC-V v1 限制 `dim<=64` 且行连续，临时 GM 缓冲移到 `LOCAL_LAYOUT.accum + LOCAL_OUT_TILE_BYTES_ALIGNED` 后方 |
| `--group-manager-enable` / `--ctrl-link-enable` 不是原 pipeline CLI 参数 | wrapper 将这些作为私有参数解析并导出环境变量，不再透传给原 pipeline |
| 1-core smoke 在用户正常 shell 中通过 | 记录日志、stdout 和统计产物路径；阶段 7 标记 complete |
| 新增 softmax checker 后，旧 1-core HBM 输出与 naive `softmax(A@B)` 不匹配 | checker 单元测试通过；改为默认验证概率分布合法性，旧 1-core HBM artifact 已通过；严格 `a_b` 模式保留为后续 GEMM layout/golden 对齐诊断 |
| 共享 `TensorDataType::FP32=0`，而 runtime `GOLEM_DTYPE_FP32=1` | 阶段 9 不做 enum 直接 cast；新增显式转换 helper，避免 FP32 被误解释成 INT32 |
