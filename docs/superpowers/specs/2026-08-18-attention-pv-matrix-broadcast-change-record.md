# Attention PV 矩阵广播架构变更记录

**状态：** 实验性硬件机制已实现，并通过 E2/E3 验证

**日期：** 2026-08-18

**范围：** 多核融合 Attention 的 PV 矩阵编程路径

## 1. 变更定性与结论

本次修改在单个 worker 的 array buffer 与多个计算阵列之间新增了一个建模的本地
广播路径，并让 PV 数据流使用该路径。因此，它属于**实验性硬件架构变更**，不是在
原硬件不变条件下完成的纯软件调度优化。

后续报告必须区分两种配置：

- **原始架构基线：** 关闭 PV 矩阵广播。同一份 V 矩阵通过独立的 buffer transfer
  分别写入每个阵列。
- **启用广播的实验架构：** 开启 PV 矩阵广播。一次建模的 buffer transfer 将同一份
  V 矩阵分发给所有活动的 query-row 阵列。

广播机制为显式可选项，默认关闭。启用广播得到的性能结果不得表述为原架构未修改时
的性能。

## 2. 原 PV 路径中的数据复用机会

一个 16-row query block 使用同一 worker 内的 16 个阵列。在一个 PV dimension
panel 内：

- 每个阵列使用不同的一行 Softmax 概率 P；
- 所有阵列使用相同的 V 矩阵 panel；
- E3 的 V 矩阵 payload 为 `16 output dims x 128 array inputs x FP32`，即
  8,192 bytes。

原始路径只从 Local GM 读取一次 V panel，但随后将完全相同的矩阵依次写入阵列
0 到 15：

```text
从 Local GM 读取一个 V panel
  -> 将 V 写入 array 0
  -> 将 V 写入 array 1
  -> ...
  -> 将 V 写入 array 15
  -> 分别写入 16 个不同的 P 输入向量
  -> 执行 16 次 PV 阵列运算
```

每次 `programMatrixAsync()` 都会占用有界 array-buffer transfer service，并按完整
矩阵统计 bytes 和 transfer cycles。因此，虽然矩阵内容完全相同，原路径仍对同一
payload 收取了 16 次传输成本。

E3 中每个 worker 执行 1,024 组 PV 矩阵编程。原路径因此产生
`1,024 x 16 = 16,384` 次独立的 V 矩阵传输。

## 3. 启用广播后的 PV 路径

新路径构造活动阵列 ID 集合，并只调用一次 `programMatrixGroupAsync()`：

```text
从 Local GM 读取一个 V panel
  -> V 只通过 array buffer 传输一次
  -> 在本地将 V 广播到 array 0...15
  -> 分别写入 16 个不同的 P 输入向量
  -> 执行 16 次 PV 阵列运算
```

group 操作先检查所有目标阵列 ID，再以 `matrix.size() * elemBytes` 入队一次 transfer；
transfer 完成时更新所有目标阵列的矩阵状态。输入向量编程和阵列计算仍然逐阵列执行，
没有改变 PV 的算术工作量。

对 E3 而言，该机制将每个 worker 的 16,384 次独立 V 矩阵传输替换为 1,024 次
group broadcast，消除了 15,360 次重复矩阵传输。

## 4. 具体代码修改

修改范围限制在阵列编程接口、两个后端、Attention PV 控制路径、配置传递和验证：

| 文件 | 修改内容 |
|---|---|
| `src/sst/elements/golem/array/computeArray.h` | 增加抽象接口 `programMatrixGroupAsync()`。 |
| `src/sst/elements/golem/array/mvmComputeArray.h` | 建模一次 buffer transfer，并在完成时将矩阵送到所有选定的 MVM 阵列。 |
| `src/sst/elements/golem/array/crossSimComputeArray.h` | 为 CrossSim 后端实现相同的 group 语义。 |
| `src/sst/elements/golem/rocc/roccAnalog.h` | 增加默认关闭的参数、广播统计量和 PV group-programming 分支。 |
| `src/sst/elements/golem/rocc/roccAnalogFloat.h` | 声明 float RoCC 参数和统计量。 |
| `src/sst/elements/golem/rocc/roccAnalogInt.h` | 声明 integer RoCC 参数和统计量。 |
| `src/sst/elements/golem/tests/architecture/cpu_builder.py` | 将 `GOLEM_ATTENTION_PV_MATRIX_BROADCAST` 传入 RoCC 组件。 |
| `src/sst/elements/golem/tests/small/muticore_attention/run_fused_attention_scale.sh` | 增加显式选项 `--pv-matrix-broadcast`。 |
| `src/sst/elements/golem/tests/small/muticore_attention/verify_fused_attention_scale_stats.py` | 默认配置要求广播数为零；启用时检查精确广播次数。 |
| `src/sst/elements/golem/tests/small/muticore_attention/test_fused_attention_e1_contract.py` | 固定显式启用、默认关闭和源码集成契约。 |

原有的串行 `programMatrixAsync()` 路径完整保留；未提供广播选项时继续使用原路径。

## 5. 修改前后实测对比

下表均为 1 GHz accelerator clock 下的 accelerator completion cycles。两个优化
测试均使用单 SST 线程，并设置了有界 wall-time timeout。

| 规模点 | 负载 | 原始架构基线 | 启用广播 | Cycle 降幅 | 加速比 |
|---|---|---:|---:|---:|---:|
| E2 | `B1,H1,S256,D64` | 97,177 | 75,181 | 21,996（22.63%） | 1.293x |
| E3 | `B1,H1,S1024,D128` | 3,568,975 | 2,241,546 | 1,327,429（37.19%） | 1.592x |

core 19 代表性 worker 的 array-buffer 统计如下：

| 规模点 | 指标 | 原始架构基线 | 启用广播 | 变化 |
|---|---|---:|---:|---:|
| E2 | Requests | 2,624 | 2,144 | -480 |
| E2 | Bytes | 2,895,872 | 929,792 | -67.89% |
| E2 | Transfer cycles | 47,872 | 16,672 | -65.17% |
| E3 | Requests | 75,264 | 59,904 | -15,360 |
| E3 | Bytes | 163,807,232 | 37,978,112 | -76.82% |
| E3 | Transfer cycles | 2,634,752 | 653,312 | -75.20% |

E3 的聚合 `all_softmax_to_pv` 区间从 2,838,528 cycles 降到 857,088 cycles，
减少的 1,981,440 cycles 与 array-buffer 建模传输周期的减少量完全一致。

端到端 cycles 的下降小于 1,981,440，原因是在线流水会重叠部分工作，而且矩阵编程
缩短后暴露了此前被隐藏的等待。另一个原因是 `inter_tile_pv_to_next_qk` 本身包含
下一 tile 的 KV/QK 准备和中间输出 DMA，因此新的关键路径有一部分被 milestone
归入该区间。

## 6. 正确性与验证结果

该硬件机制改变数据搬运时序，不改变算术或 tensor layout：

| 规模点 | 检查输出数 | Mismatch | 最大绝对误差 |
|---|---:|---:|---:|
| E2 | 16,384 | 0 | `4.4967521108628394e-09` |
| E3 | 131,072 | 0 | `1.5966506542461692e-09` |

2026-08-18 完成的其他验证：

- 45/45 fused-Attention C1/D1-D4/E1 contract tests 通过；
- E2 每个 worker 精确记录 32 次广播；
- E3 每个 worker 精确记录 1,024 次广播；
- array-buffer rejection 保持为零，high-water mark 保持为一；
- Python 语法检查和 `git diff --check` 通过；
- build/install `libgolem.so` SHA-256 一致：
  `333c44966eab1fa0f614332e428845341e72a36a6f39b95bd2c5491533e3b4c9`。

验证产物：

- E2 原始架构：`/data4/jjgong/tmp/fused_attention_e2_cycle_breakdown_20260818`
- E2 启用广播：`/data4/jjgong/tmp/fused_attention_e2_pv_broadcast_20260818`
- E3 原始架构：`/data4/jjgong/tmp/fused_attention_e3_cycle_breakdown_20260818`
- E3 启用广播：`/data4/jjgong/tmp/fused_attention_e3_pv_broadcast_20260818`

## 7. 当前模型边界

当前模型可以验证本地 fan-out 的功能和时序收益，但尚未详细建模物理广播网络。
现阶段假设为：

- 一次 array-buffer 矩阵传输能够同时服务所有选定阵列；
- 增加接收阵列不会增加 transfer bytes 或 service cycles；
- 暂不计入额外的扇出延迟、仲裁、布线拥塞、面积和能耗；
- 所有选定阵列在 transfer 完成时原子地收到矩阵。

因此，当前 cycle 降幅只对“已建模的广播实验架构”有效，不能视为完整的物理成本/性能
结论。在将其用于最终 GPU 对比前，需要定义广播宽度、最大 fan-out、带宽、仲裁、附加
延迟和面积/能耗成本，并通过敏感性分析验证加入现实广播成本后加速是否仍然成立。

## 8. 复现方式

关闭广播的原始架构基线：

```bash
bash src/sst/elements/golem/tests/small/muticore_attention/run_fused_attention_scale.sh \
  --scale-point e3 \
  --artifact-root /data4/jjgong/tmp/fused_attention_e3_baseline \
  --timeout 1200
```

启用广播的实验架构：

```bash
bash src/sst/elements/golem/tests/small/muticore_attention/run_fused_attention_scale.sh \
  --scale-point e3 \
  --pv-matrix-broadcast \
  --artifact-root /data4/jjgong/tmp/fused_attention_e3_pv_broadcast \
  --timeout 1200
```

## 9. 暂缓事项

本次变更不包含任何 `inter_tile_pv_to_next_qk` 优化。下一项建议工作仅为诊断：将该
聚合区间拆分为输出 DMA、下一 tile 的 KV/V 准备、QK 矩阵/输入编程，以及 worker
状态切换等待。得到测量数据并单独评审后，才决定是否进行下一项架构修改。
