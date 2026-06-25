# 16-Core GEMM + Single-Core Softmax - 最终使用指南

## ✅ 已验证配置

所有配置已经过测试和修复，可以直接使用。

## 🚀 快速开始

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST/build/sst-elements/src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu

# 快速验证（推荐首次运行）
./test_16core_128x128.sh

# 标准测试
./test_16core_256x256.sh

# 压力测试
./test_16core_512x512.sh
```

## 📊 测试配置

| 脚本 | 矩阵 | GEMM Tiles | 核心 | 预计时间 |
|------|------|-----------|------|---------|
| `test_16core_128x128.sh` | 128×128 | 4 (2×2) | 16 | 2-5 分钟 |
| `test_16core_256x256.sh` | 256×256 | 16 (4×4) | 16 | 10-20 分钟 |
| `test_16core_512x512.sh` | 512×512 | 64 (8×8) | 16 | 30-60 分钟 |

## 🔧 配置文件详解

### configs/16core_128x128.env

```bash
# 矩阵维度
GOLEM_GEMM_M=128
GOLEM_GEMM_N=128
GOLEM_GEMM_K=128

# 16核配置
GOLEM_TOTAL_CORES=16
GOLEM_TOTAL_GEMM_CORES=16
GOLEM_TOTAL_GROUPS=4

# 拓扑约束（关键！）
# 公式：(num_mem_nodes - 1) <= mesh_dim_x
# 9个内存节点 - 1个OS节点 = 8个数据节点
GOLEM_NUM_MEMORY_NODES=9
GOLEM_MESH_DIM_X=8
GOLEM_MESH_DIM_Y=8

# Softmax模式
GOLEM_SOFTMAX_MODE=single-core

# 禁用group manager（避免自动添加CPU行）
GOLEM_GROUP_MANAGER_ENABLE=0
GOLEM_CTRL_LINK_ENABLE=0
```

## ⚙️ 关键约束说明

### 1. 内存拓扑约束

```
DATA_MEM_NODES = num_mem_nodes - 1  # 减去1个OS节点
约束：DATA_MEM_NODES <= mesh_dim_x

示例：
  num_mem_nodes=9
  9 - 1 = 8 data nodes
  需要：mesh_dim_x >= 8 ✅
```

### 2. Group Manager 约束

**重要：** 当启用 `GROUP_MANAGER_ENABLE=1` + `CTRL_LINK_ENABLE=1` + `TOTAL_CORES=16` 时，系统会**自动添加 mesh_dim_x 个额外核心**，导致核心数不匹配。

**解决方案：**
- 方案 A（推荐）：禁用 group manager（已采用）
- 方案 B：使用 20 核配置（需要调整）

## 💡 运行示例

### 基本运行

```bash
./test_16core_128x128.sh
```

### Dry-run（不运行SST）

```bash
./test_16core_256x256.sh --dry-run
```

### 切换到跨Tile并行模式

```bash
GOLEM_SOFTMAX_MODE=cross-tile ./test_16core_128x128.sh
```

### 严格验证模式

```bash
./test_16core_256x256.sh --softmax-reference a_b
```

## 📈 架构说明

```
Phase 1: 16核并行GEMM
  ┌─────────────────────────────────┐
  │ Core 0-15: 并行计算M×N tiles    │
  │ 输出分散在HBM的不同tile区域      │
  └─────────────────────────────────┘
          ↓
Phase 2: Core 0单核Softmax
  ┌─────────────────────────────────┐
  │ Core 0:                         │
  │   1. 从HBM读取所有N-tiles        │
  │   2. 聚合完整行                  │
  │   3. 计算row-wise softmax       │
  │   4. 写回HBM                    │
  │                                 │
  │ Core 1-15: 空闲                 │
  └─────────────────────────────────┘
```

## 🎯 预期输出

成功运行时的关键日志：

```
==========================================
16-Core 128x128 Test (Quick Validation)
==========================================
Config: configs/16core_128x128.env
GEMM: 16 cores parallel (4 tiles)
Softmax: Core 0 single-core post-processing

[SOFTMAX] mode=single-core (Core 0 post-processing)
[Core 0] [SOFTMAX] starting single-core softmax: m=128 n=128 ...
[Core 0] [SOFTMAX] single-core softmax complete
[VERIFY-SOFTMAX] PASS reference=probability
```

## 🔍 故障排除

### 错误：数据节点数(X) 不能大于 --mesh-dim-x(Y)

**原因：** 拓扑约束不满足

**解决：** 确保 `(num_mem_nodes - 1) <= mesh_dim_x`

```bash
# 检查配置
cat configs/16core_128x128.env | grep "MEMORY_NODES\|MESH_DIM"

# 应该看到：
# GOLEM_NUM_MEMORY_NODES=9
# GOLEM_MESH_DIM_X=8    # 9-1=8 ✅
```

### 错误：当前 GroupCtrlEndpoint 管理器...active_workers 不匹配

**原因：** Group manager 自动添加了额外CPU行

**解决：** 使用禁用 group manager 的配置（已修复）

### 错误：SST被Killed

**原因：** 内存不足或矩阵太大

**解决：** 
- 使用更小的矩阵（128×128）
- 增加系统内存

## 📝 自定义配置

创建新的矩阵尺寸：

```bash
# 复制现有配置
cp configs/16core_256x256.env configs/16core_384x384.env

# 编辑矩阵维度
vim configs/16core_384x384.env
# 修改：
#   GOLEM_GEMM_M=384
#   GOLEM_GEMM_N=384
#   GOLEM_GEMM_K=384

# 使用新配置
source configs/16core_384x384.env
./run_noc_dma_softmax_pipeline.sh --verify-softmax
```

## ✨ 完整功能列表

- ✅ 16核并行GEMM
- ✅ 单核后处理Softmax
- ✅ 完整row-wise softmax（语义正确）
- ✅ 运行时模式切换（single-core / cross-tile）
- ✅ 自动验证（probability模式）
- ✅ 配置文件驱动
- ✅ 统一项目风格

## 🎓 进阶使用

### 性能分析

查看GEMM vs Softmax时间占比：

```bash
./test_16core_256x256.sh 2>&1 | grep "cycles\|time"
```

### 切换验证模式

```bash
# 默认：probability模式（验证概率分布）
./test_16core_256x256.sh

# 严格：a_b模式（验证数值精度）
./test_16core_256x256.sh --softmax-reference a_b
```

### 后台运行

```bash
nohup ./test_16core_512x512.sh > test.log 2>&1 &
tail -f test.log
```

## 📚 相关文档

- `README_16CORE.md` - 完整技术文档
- `findings.md` - 实现细节和已知限制
- `configs/*.env` - 预设配置文件
