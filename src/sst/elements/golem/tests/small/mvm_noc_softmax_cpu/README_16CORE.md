# 16-Core Large Matrix GEMM + Single-Core Softmax Tests

## 概述

这些脚本用于测试**多核并行 GEMM + 单核后处理 Softmax**的场景。

### 架构

```
Phase 1: 多核 GEMM (16 cores 并行)
  Core 0-15: 并行计算 M×N tiles → 输出到 HBM

Phase 2: 单核 Softmax (Core 0 串行)
  Core 0: 从 HBM 聚合所有 tiles → 计算完整 row-wise softmax → 写回 HBM
  Core 1-15: 空闲
```

## 快速开始

### 测试脚本

| 脚本 | 矩阵大小 | GEMM Tiles | 描述 |
|------|---------|-----------|------|
| `test_16core_128x128.sh` | 128×128 | 4 (2×2) | 快速验证 |
| `test_16core_256x256.sh` | 256×256 | 16 (4×4) | 标准测试 |
| `test_16core_512x512.sh` | 512×512 | 64 (8×8) | 压力测试 |

### 运行示例

```bash
# 快速验证（推荐先运行）
./test_16core_128x128.sh

# 标准测试
./test_16core_256x256.sh

# 压力测试（需较长时间）
./test_16core_512x512.sh

# 自定义矩阵大小
./run_16core_large_matrix.sh 384  # 384×384
```

## 性能特点

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

### 性能瓶颈分析

对于不同矩阵大小，GEMM vs Softmax 的时间占比：

| 矩阵 | GEMM Tiles | GEMM 时间* | Softmax 时间* | Softmax 占比 |
|------|-----------|-----------|--------------|-------------|
| 128×128 | 4 | 低 | 低 | ~20% |
| 256×256 | 16 | 中 | 中 | ~30% |
| 512×512 | 64 | 高 | 高 | ~40% |

*相对时间，实际值取决于硬件

**结论：**
- 小矩阵（128×128）：Softmax 开销可忽略
- 中等矩阵（256×256）：Softmax 占比合理
- 大矩阵（512×512）：如果 Softmax 成为瓶颈，考虑切换到跨 tile 并行模式

## 切换到跨 Tile 并行模式

如果 Softmax 成为性能瓶颈，可以切换到多核并行 Softmax：

```bash
# 使用跨 tile 并行模式
GOLEM_SOFTMAX_MODE=cross-tile ./test_16core_512x512.sh
```

**注意：** 跨 tile 模式在 SST 仿真中极慢，建议只在真实硬件上使用。

## 验证模式

默认使用 `--softmax-reference probability` 模式：
- 验证每行是合法概率分布
- 元素 ∈ [0,1]
- 行和 ≈ 1

如需更严格的验证（需确保 HBM 同步）：

```bash
./run_16core_large_matrix.sh 256 --softmax-reference a_b
```

## 文件说明

- `run_16core_large_matrix.sh` - 主脚本，支持自定义矩阵大小
- `test_16core_128x128.sh` - 快速验证（4 tiles）
- `test_16core_256x256.sh` - 标准测试（16 tiles）
- `test_16core_512x512.sh` - 压力测试（64 tiles）

## 预期输出

成功运行的关键日志：

```
[SOFTMAX] mode=single-core (Core 0 post-processing)
[Core 0] [SOFTMAX] starting single-core softmax: m=256 n=256 ...
[Core 0] [SOFTMAX] single-core softmax complete
[VERIFY-SOFTMAX] PASS reference=probability
```

## 故障排除

**问题：SST 仿真被 Killed**
- 原因：内存不足或矩阵太大
- 解决：减小矩阵尺寸或增加系统内存

**问题：Softmax 验证失败**
- 检查：HBM init 文件是否同步
- 尝试：使用 `--softmax-reference probability` 模式

**问题：编译失败**
- 运行：`make clean && make`
- 检查：RISC-V 工具链是否可用

## 技术细节

### 单核 Softmax 限制

- 最大 N：256 列（受 GM 缓冲大小限制）
- 如需更大 N，需修改 `golem_softmax_single_core.cpp` 中的缓冲区大小

### 内存布局

- GEMM 输出：HBM 列优先（column-major）
- Softmax 聚合：按行读取（跨 N-tiles）
- Softmax 输出：HBM 列优先（覆盖原 GEMM 输出）
