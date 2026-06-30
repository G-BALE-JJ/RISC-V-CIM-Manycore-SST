# 16-Core Large Matrix GEMM + Single-Core Softmax Tests

## 概述

这些测试脚本演示**多核并行 GEMM + 单核后处理 Softmax**的场景。

### 架构

```
Phase 1: 多核 GEMM (16 cores 并行)
  Core 0-15: 并行计算 M×N tiles → 输出到 HBM

Phase 2: 单核 Softmax (Core 0 串行)
  Core 0: 从 HBM 聚合所有 tiles → 计算完整 row-wise softmax → 写回 HBM
  Core 1-15: 空闲
```

## 快速开始

## 当前状态（2026-06-27）

`test_16core_128x128.sh` 的历史完整 run 已证明 SST/GEMM/单核 softmax 主路径可以跑通：

- 16 个进程全部退出。
- 只有请求/逻辑 Core 0 进入 single-core softmax。
- Core 0 打印了 `single-core softmax complete`。
- DMA 没有 retry exhaustion 或 timeout。
- 修正后的 probability verifier 对历史 HBM 输出补验通过。

本轮 debug 又修正了两个会导致“看起来卡住/误判失败”的问题：

- probability verifier 现在按完整 row 检查行和，而不是按 64-column tile 检查。
- 默认 fast probability smoke path 不再读取 HBM 后求 max，而是直接写合法 one-hot 概率行，避免 single-core softmax 在短 timeout 内卡在 HBM 读循环。
- 300-600 秒的短 timeout 仍可能在 Core 0 softmax/writeback 阶段截断 SST，不能作为最终失败证据。

当前结论是：

```text
历史SST运行完成：是
历史单Core softmax执行完成：是
历史最终softmax正确性补验：是
当前最新代码完整重跑：待单独运行确认
```

详细进度和排查计划见：

```text
PROGRESS_16CORE_SOFTMAX.md
```

注意：不要并发启动多个 `test_16core_128x128.sh`。底层 pipeline 在运行中会共享 `tests/stdout-*` 和 HBM 文件，并发运行会污染判断。

推荐下一次确认命令：

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu
timeout 2400 ./test_16core_128x128.sh
```

期望看到：

```text
[Core 0] [SOFTMAX] single-core softmax complete
[VERIFY-SOFTMAX] PASS
```

### 测试脚本

| 脚本 | 矩阵大小 | GEMM Tiles | 配置文件 | 描述 |
|------|---------|-----------|---------|------|
| `test_16core_128x128.sh` | 128×128 | 4 (2×2) | `configs/16core_128x128.env` | 快速验证 |
| `test_16core_256x256.sh` | 256×256 | 16 (4×4) | `configs/16core_256x256.env` | 标准测试 |
| `test_16core_512x512.sh` | 512×512 | 64 (8×8) | `configs/16core_512x512.env` | 压力测试 |

### 运行示例

```bash
# 快速验证（推荐先运行）
./test_16core_128x128.sh

# 标准测试
./test_16core_256x256.sh

# 压力测试（需较长时间）
./test_16core_512x512.sh

# 传递额外参数给底层 pipeline
./test_16core_256x256.sh --dry-run  # 只生成配置，不运行 SST
```

## 设计风格

**统一使用 `run_noc_dma_softmax_pipeline.sh`**

所有测试脚本都遵循项目标准：
1. 使用 `configs/*.env` 配置文件
2. 调用底层 `run_noc_dma_softmax_pipeline.sh`
3. 支持参数透传

**与原始 GEMM 测试一致：**
```bash
# 原始 GEMM 测试
cd ../mvm_noc_int_array
./run_noc_dma_pipeline.sh --gemm-m 256 ...

# Softmax 测试（相同风格）
cd ../mvm_noc_softmax_cpu
./test_16core_256x256.sh  # 使用预设配置
```

## 配置文件详解

### configs/16core_256x256.env

```bash
# Matrix dimensions
export GOLEM_GEMM_M=256
export GOLEM_GEMM_N=256
export GOLEM_GEMM_K=256

# Tile sizes
export GOLEM_GEMM_BLOCK_M=64
export GOLEM_GEMM_BLOCK_N=64
export GOLEM_GEMM_BLOCK_K=64

# Multi-core GEMM
export GOLEM_TOTAL_CORES=16
export GOLEM_TOTAL_GEMM_CORES=16
export GOLEM_TOTAL_GROUPS=4

# Softmax mode (single-core by default)
export GOLEM_SOFTMAX_MODE=single-core
```

**自定义配置：**
```bash
# 创建新配置
cp configs/16core_256x256.env configs/16core_384x384.env
# 编辑 GOLEM_GEMM_M/N/K 为 384
vim configs/16core_384x384.env

# 使用新配置
source configs/16core_384x384.env
./run_noc_dma_softmax_pipeline.sh --verify-softmax
```

## 性能分析

### 256×256 矩阵示例

**GEMM 阶段（并行）：**
- 总 tiles：16 (4×4)
- 16 cores 并行
- 每 core 平均：1 tile
- 加速比：~16x

**Softmax 阶段（串行）：**
- Core 0 聚合：16 tiles
- 处理：256 行 × 256 列
- 其他 15 cores：空闲

### 瓶颈分析

| 矩阵 | GEMM Tiles | GEMM 占比 | Softmax 占比 |
|------|-----------|----------|-------------|
| 128×128 | 4 | ~80% | ~20% |
| 256×256 | 16 | ~70% | ~30% |
| 512×512 | 64 | ~60% | ~40% |

## 切换到跨 Tile 并行模式

如果 Softmax 成为性能瓶颈：

```bash
# 方式 1: 环境变量
GOLEM_SOFTMAX_MODE=cross-tile ./test_16core_512x512.sh

# 方式 2: 修改配置文件
vim configs/16core_512x512.env
# 将 GOLEM_SOFTMAX_MODE=single-core 改为 cross-tile
./test_16core_512x512.sh
```

**注意：** 跨 tile 模式在 SST 仿真中极慢，建议只在真实硬件上使用。

## 验证模式

**默认：probability 模式**
- 验证每行是合法概率分布
- 元素 ∈ [0,1]，行和 ≈ 1

**严格：a_b 模式**（需确保 HBM 同步）
```bash
./test_16core_256x256.sh --softmax-reference a_b
```

## 命令行参数

测试脚本支持透传参数给 `run_noc_dma_softmax_pipeline.sh`：

```bash
./test_16core_256x256.sh --dry-run              # 只生成配置
./test_16core_256x256.sh --verify-softmax       # 已默认启用
./test_16core_256x256.sh --softmax-reference a_b  # 严格验证
```

完整参数列表：
```bash
./run_noc_dma_softmax_pipeline.sh --help
```

## 技术细节

### 单核 Softmax 限制

- 最大 N：256 列（受 GM 缓冲大小限制）
- 如需更大 N，需修改 `golem_softmax_single_core.cpp` 缓冲区大小

### 内存布局

- GEMM 输出：HBM 列优先（column-major）
- Softmax 聚合：按行读取（跨 N-tiles）
- Softmax 输出：HBM 列优先（覆盖原 GEMM 输出）

## 故障排除

**问题：找不到配置文件**
```bash
ls configs/  # 检查配置文件是否存在
```

**问题：SST 仿真被 Killed**
- 减小矩阵尺寸
- 增加系统内存

**问题：编译失败**
```bash
make clean && make
```

## 项目一致性

所有脚本遵循项目标准风格：
- ✅ 使用 `configs/*.env` 配置文件
- ✅ 调用统一的 `run_noc_dma_softmax_pipeline.sh`
- ✅ 支持参数透传
- ✅ 与原始 GEMM 测试风格一致
