# MVM NoC Softmax SFU 设计文档

## 目标

本项目目标是把 softmax 从当前的 CPU fallback 路径迁移到硬件建模的 SFU
路径上。SFU 表示 Special Function Unit，不只服务于 softmax；softmax 是第一个
接入的特殊函数类操作。

从长期架构看，SFU 是 Golem 中面向特殊数学函数和非矩阵主算子的通用执行单元。
它应能逐步承载 `exp`、`log`、`reciprocal`、`rsqrt`、`sigmoid`、`tanh`、
`softmax`、`layernorm`、`gelu` 等操作。当前 small test 只要求打通
softmax，是因为现有迁移目标是替代 CPU fallback softmax。

第一版 SFU softmax 必须支持 GEMM 输出矩阵 `C[M, N]` 的完整 row-wise softmax，
而不是 tile-local softmax。

当前目标架构基于已有 Golem manycore 流程：

- GEMM 生成形状为 `block_m x block_n` 的 C tile。
- 当 `N > block_n` 时，同一行会跨多个 N tile。
- SFU 必须在这种跨 tile 情况下保持完整 row-wise softmax 的数值正确性。

本实验相关文件统一放在：

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
```

实际 SST 组件源码后续放在：

```text
src/sst/elements/golem/sfu/
```

## 当前 Tile 形状带来的影响

当前 softmax CPU 配置基本使用：

```text
block_m = 64
block_n = 64
GEMM worker cores = 16
```

示例：

| 矩阵规模 | Tile 网格 | C tile 总数 | 每 core 的 tile 数 | 每行跨几个 N tile |
| --- | --- | ---: | ---: | ---: |
| 64 x 64 | 1 x 1 | 1 | core0: 1 | 1 |
| 128 x 128 | 2 x 2 | 4 | 前 4 个 core 各 1 个 | 2 |
| 256 x 256 | 4 x 4 | 16 | 16 个 core 各 1 个 | 4 |
| 512 x 512 | 8 x 8 | 64 | 16 个 core 各 4 个 | 8 |

只有当 `N == block_n` 时，tile-local softmax 才等价于完整 row-wise softmax。例如：

- `64 x 64, block_n=64`：每行只在一个 tile 内，tile-local 正确。
- `256 x 256, block_n=64`：每行跨 4 个 N tile，tile-local 不正确。
- `512 x 512, block_n=64`：每行跨 8 个 N tile，tile-local 不正确。

因此，本目录的 SFU 方案从第一版开始就采用跨 tile reduction，不再把
tile-local softmax 作为最终正确性目标。

## 组件结构

SFU 是通用 Special Function Unit，组件名不应包含 `softmax`。

```text
Vanadis CPU
   |
   v
RoCCAnalogInt / RoCCAnalogFloat
   |-- array              -> MVMComputeArray
   |-- global_memory      -> GlobalMemory
   |-- sfu                -> SFU
   |-- group_ctrl         -> optional
   |-- request_scheduler  -> optional
```

Python 架构脚本中的 RoCC slot：

```python
cpu_rocc.setSubComponent("sfu", "golem.SFU")
```

组件文件：

```text
src/sst/elements/golem/sfu/sfu.h
src/sst/elements/golem/sfu/sfu.cc
```

构建注册：

- 在 `golem.cc` 中 include `sfu/sfu.h`。
- 在 `Makefile.am` 的 `libgolem_la_SOURCES` 中加入 `sfu/sfu.h` 和 `sfu/sfu.cc`。

## SFU 通用职责与操作分层

SFU 不应被建模成只会执行 softmax 的专用模块。推荐把 SFU 操作分为两层：

```text
SFU primitive ops:
  exp
  log
  reciprocal / divide approximate
  rsqrt
  max / sum reduction helper

SFU fused ops:
  softmax
  sigmoid
  tanh
  layernorm
  gelu
```

第一版只对 workload 暴露 fused softmax 指令，原因是它能直接替换当前 CPU
fallback 路径，并且避免在 RISC-V workload 中暴露过多尚未使用的 primitive
RoCC 指令。softmax 内部仍应按 SFU primitive 的思想组织：`max reduction`、
`exp`、`sum reduction`、`reciprocal/divide` 和 normalize。这样后续新增
standalone `exp`、`log`、`reciprocal` 或 layernorm 时，不需要重命名组件或推翻
RoCC slot 结构。

## Online Softmax 协议

正式方案采用 online softmax 的跨 tile 统计合并，而不是先全行 max、再全行 sum 的
传统三遍算法。

每个 tile SFU 先对自己负责的 row fragment 计算局部统计量：

```text
tile_m = max(tile_row)
tile_l = sum(exp(x - tile_m))
```

row-owner/reducer 收到同一 global row 的多个 tile 统计后，用 online softmax 公式合并：

```text
m_new = max(m_old, tile_m)
l_new = l_old * exp(m_old - m_new)
      + tile_l * exp(tile_m - m_new)
```

所有 N tile 的统计合并完成后，row-owner 发布：

```text
global_m = m_final
global_l = l_final
```

每个 tile SFU 再重新读取或复用自己的 tile 数据，执行归一化：

```text
y = exp(x - global_m) / global_l
```

## 跨 Tile 数据流

如果一行被切成多个 N tile：

```text
Row r:
[ tile n0 ][ tile n1 ][ tile n2 ][ tile n3 ]
    |          |          |          |
   SFU0       SFU1       SFU2       SFU3
```

online softmax 数据流：

```text
每个 tile 的 SFU:
  计算 tile_m / tile_l
      |
      v
row-owner / reducer:
  使用 online update 合并所有 tile 统计
      |
      v
row-owner / reducer:
  发布 global_m / global_l
      |
      v
每个 tile 的 SFU:
  normalize(row fragment, global_m, global_l)
```

对 `256 x 256, block_n=64`：

```text
每行 N tile 数 = 256 / 64 = 4
每行需要 4 组 (tile_m, tile_l) partial stats
```

对 `512 x 512, block_n=64`：

```text
每行 N tile 数 = 512 / 64 = 8
每行需要 8 组 (tile_m, tile_l) partial stats
```

## 借鉴 GEMM DMA 的竞争与死锁控制

可以参考 GEMM DMA / request scheduler 的机制，但不能直接照搬 CPU prototype 中
非原子 HBM reduction buffer 的写法。

可以借鉴的 GEMM DMA 机制：

```text
1. inflight 限制
   限制每个 core / 每个 reducer 同时挂起的 SFU partial 请求数量。

2. credit 机制
   reducer 或 manager 给 producer SFU 发 credit，有 credit 才能继续提交 partial。

3. submit / done 队列
   producer 提交 tile stats，reducer 合并完成后产生 done，producer 再进入 normalize。

4. retry / timeout
   对等待 global_m/global_l 的请求设置重试或超时诊断，避免静默死锁。

5. batch submit / batch done
   将多个 row 的 tile stats 打包提交，降低控制消息数量。
```

不能照搬的旧方式：

```text
load current max/sum
compute new max/sum
store back
```

这种 HBM read-modify-write 在多核并发下会丢更新，DMA retry 只能保证传输完成，
不能保证数值更新原子性。

正式 SFU 设计采用显式 row ownership：

- 每个 global row 映射到一个 reducer owner。
- producer SFU 向 owner 提交该行 fragment 的 `(tile_m, tile_l)`。
- owner 等待 `n_tiles_per_row` 个 partial 都到齐。
- owner 用 online softmax 公式合并得到 `global_m/global_l`。
- owner 发布结果，producer SFU 获得结果后进入 normalize。

第一版 owner 映射：

```text
owner_core(row) = row % active_worker_cores
```

这个映射确定性强，便于调试。后续可以根据 NoC 拓扑或数据位置优化 locality。

## SFU 操作集合

对 RISC-V workload 暴露的高层 fused 操作：

```text
sfu.softmax_begin
sfu.softmax_wait
```

SFU 内部或后续可暴露的 primitive 操作：

```text
sfu.tile_stats       # 计算 tile_m/tile_l
sfu.merge_stats      # row-owner online update
sfu.normalize        # 用 global_m/global_l 归一化
sfu.wait
```

第一版只对软件暴露 fused softmax 指令，把 primitive 阶段封装在 SFU 内部。
保留 primitive 名称，是为了后续扩展到其他 SFU 操作，例如：

- layernorm 中的局部规约和归一化；
- standalone `exp`、`log`、`reciprocal`、`rsqrt`；
- sigmoid；
- tanh；
- reciprocal / divide approximate。

## 数据搬运

SFU 应沿用当前 Golem pipeline 的地址空间约定：

- 输入是 GEMM 产生的 C tile，位于 HBM/GM 地址空间。
- 输出第一版可以原地覆盖 C，后续可支持单独 output base。
- tile layout 遵循当前 GEMM 输出 layout。
- reduction 临时状态属于 SFU 协议，不应依赖 CPU 本地数组。

online softmax 方案仍然需要 normalize 阶段访问 tile 数据。第一版可以重新从
HBM/GM 读取 C tile；后续如果要建模片上缓冲，可让 SFU 缓存 tile logits 或缓存
`exp(x - tile_m)`。

## 统计项

SFU 需要单独暴露统计项，便于区分 softmax 开销来源：

```text
sfu_ops_issued
sfu_softmax_rows
sfu_softmax_tiles
sfu_cycles_tile_stats
sfu_cycles_merge_stats
sfu_cycles_normalize
sfu_cross_tile_wait_cycles
sfu_partial_submits
sfu_partial_done
sfu_credit_stalls
sfu_retry_events
```

这些统计项用于拆分：

- 本地 SFU 数学计算开销；
- row-owner online merge 开销；
- 等待和同步开销；
- credit / inflight stall；
- 数据搬运开销。

## 正确性要求

SFU softmax 输出必须匹配完整 row-wise softmax：

```text
softmax(row[i]) = exp(x[i] - max(full_row)) /
                  sum_j exp(x[j] - max(full_row))
```

online softmax 的 `(global_m, global_l)` 必须与完整行统计等价：

```text
global_m = max(full_row)
global_l = sum_j exp(x[j] - global_m)
```

每一行都必须满足：

- 输出值全部为 finite；
- 输出值在 `[0, 1]` 容差范围内；
- 行和接近 1；
- 对确定性输入，能通过 `softmax(A @ B)` 的数值 golden 对比。

本目录不接受 tile-local softmax 作为最终正确性目标。

## 实现范围

设计分为两层：

1. 组件层：
   - 新增 `golem.SFU` SST 子组件。
   - 给 RoCC 增加 `sfu` slot。
   - 增加 softmax 命令处理、online stats merge 和 SFU 统计项。
   - 借鉴 GEMM DMA 的 inflight、credit、submit/done、retry 机制，避免竞争和死锁。

2. 测试 workload 层：
   - 新建隔离 small test：`mvm_noc_softmax_sfu`。
   - 复用现有 GEMM pipeline 的配置风格和 tile 形状。
   - 将 CPU fallback softmax 调用替换为触发 SFU 的 RoCC 指令。

## 待确认的设计决策

实现前还需要确认：

1. row-owner 消息第一版是否用 SFU 管理的全局共享 reducer state 建模，
   还是直接通过显式 SimpleNetwork link 建模。
2. 第一版只支持原地覆盖输出，还是同时支持 input/output 分离。
3. `sfu.softmax_begin` 一次描述一个 tile、一个 row block，还是一个 core 负责的所有 tile。
4. SFU 的 credit manager 是独立于 request scheduler，还是复用 request scheduler 的控制路径。

推荐第一版选择：

1. 先使用确定性的 SFU-managed reducer state，保证正确性和可调试性；
   后续需要 NoC 流量建模时再接显式网络消息。
2. 第一版先支持原地覆盖 C。
3. 一个命令先描述一个 tile；等正确性稳定后再增加 batch 命令。
4. 第一版在 SFU 内部实现轻量 credit/inflight 控制；后续再评估是否接入
   request scheduler。
