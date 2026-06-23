# Softmax CPU Fallback Design

## 背景

当前 GEMM 主测试位于 `../mvm_noc_int_array`，主程序为 `test_noc_dma.cpp`，
构建产物为 `riscv64/test_noc_dma`。该路径通过 `golem_matmul_runtime` 复用
Golem/RoCC/CIM GEMM 实现，GEMM tensor descriptor 在主测试中使用
`.data = nullptr`，表示实际数据路径由当前 GM/HBM/RoCC 地址布局驱动，而不是
普通 host pointer。

新增 softmax 的要求是：

- 与当前 GEMM 主测试保持相同架构和 dtype 体系。
- 目前不做 softmax 硬件加速。
- softmax 由被仿真的 RISC-V CPU 执行。
- 尽可能隔离原项目，新增工程文件和文档统一放在本目录。

## 推荐架构

采用同级新目录 `mvm_noc_softmax_cpu`：

```text
test_noc_dma_softmax
  -> golemCreateMatmulKernel / golemRunMatmul
  -> Golem/RoCC/CIM GEMM
  -> golemRunSoftmaxCpuGm
  -> RISC-V CPU scalar softmax
```

原 `../mvm_noc_int_array/test_noc_dma` 保持 GEMM-only 基线。

## 模块边界

### `test_noc_dma_softmax.cpp`

独立 RISC-V 应用入口。职责：

- 复用原 GEMM runtime 创建并运行 matmul。
- 在 GEMM 完成后根据配置调用 softmax。
- 不修改原 `test_noc_dma.cpp`。

### `golem_softmax_runtime.h/.cpp`

softmax runtime。职责：

- 定义 `golem_softmax_op_desc_t`。
- 提供 pointer 版本 CPU softmax，用于小数组自测。
- 提供 GM/HBM 地址版本 CPU softmax，用于正式接 GEMM 输出。
- 复用 `golem_status_t`、`golem_dtype_t`、`golem_layout_t` 和 `golem_tensor_desc_t`。

### `softmax_config.h`

softmax 编译期配置。职责：

- 定义 `GOLEM_SOFTMAX_ENABLE` 默认值。
- 定义 `GOLEM_SOFTMAX_AXIS` 默认值。
- 定义 softmax 输出地址相关的默认配置或编译期检查。

### `Makefile`

独立构建入口。职责：

- 构建 `riscv64/test_noc_dma_softmax`。
- 编译本目录 softmax runtime。
- 编译并链接 `../mvm_noc_int_array/golem_matmul_runtime.cpp`。
- 不修改原 `../mvm_noc_int_array/Makefile`。

## 数据类型策略

softmax v1 复用现有 dtype 枚举：

```cpp
GOLEM_DTYPE_INT32 = 0
GOLEM_DTYPE_FP32 = 1
```

v1 行为：

- `GOLEM_DTYPE_FP32`: 支持。
- `GOLEM_DTYPE_INT32`: 返回 `GOLEM_STATUS_UNSUPPORTED`。

理由：softmax 输出是概率分布，天然需要浮点表示。int32 输出没有明确概率语义。
后续如果需要，可以新增 `int32 logits -> fp32 probabilities`，但不作为 v1 范围。

## 布局和 axis

softmax v1 支持：

- `GOLEM_LAYOUT_ROW_MAJOR`
- `ndim = 2`
- `axis = -1` 或 `axis = 1`
- 每行独立 softmax：`[outer, dim]`

稳定 softmax 算法：

```text
max_val = max(row)
sum = sum(exp(x - max_val))
y = exp(x - max_val) / sum
```

## 地址接入原则

正式端到端路径优先使用 GM/HBM 地址版本：

```cpp
golemRunSoftmaxCpuGm(
    const golem_softmax_op_desc_t* op_desc,
    uint64_t input_gm_addr,
    uint64_t output_gm_addr,
    int64_t input_stride,
    int64_t output_stride);
```

该接口从 GEMM 输出地址读取 FP32 logits，计算后写回 softmax 输出地址。实现时需要
确认当前 GEMM 输出 C 的地址、布局和 stride。

当前实现约束：

- 单次调用处理一个 row-major 矩形区域 `[outer, dim]`。
- RISC-V 路径先整块读取到 CPU buffer，再计算并整块写回。
- 当前 RISC-V 低层实现使用 `core_id=0` 和 `LOCAL_LAYOUT.tmp` 作为本地搬运缓冲。
- 如果后续要由任意 GEMM worker 直接执行 softmax，应新增显式 `core_id` 参数或
  上层包装函数。
- 完整矩阵 row-wise softmax 若跨多个 GEMM N tile，需要先在上层聚合一整行的所有列，
  或增加跨 tile 的 softmax 调度。

pointer 版本保留用于小数组自测：

```cpp
golemRunSoftmaxCpu(
    const golem_softmax_op_desc_t* op_desc,
    const golem_tensor_desc_t* input,
    const golem_tensor_desc_t* output);
```

## 验证策略

第一阶段验证：

- RISC-V 程序内构造小 FP32 数组。
- 调用 pointer softmax。
- 检查输出近似 `[0.09003057, 0.24472848, 0.66524094]`。

第二阶段验证：

- GEMM 完成后调用 GM/HBM softmax。
- host Python 只作为 golden/reference 验证，不作为实际 softmax 执行路径。
- 验证每行 sum 接近 1，输出与 numpy softmax golden 在容差内一致。

## 非目标

- v1 不新增 RoCC softmax 指令。
- v1 不修改 Golem CIM array 硬件模型。
- v1 不支持 int32 softmax 输出。
- v1 不重构原 `mvm_noc_int_array`。
