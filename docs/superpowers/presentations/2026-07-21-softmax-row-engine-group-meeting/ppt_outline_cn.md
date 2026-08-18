# Softmax Row Engine 组会汇报提纲

建议时长：18--20 分钟。主线是“旧方案为什么慢 -> 架构如何改变 -> 实验如何证明 -> 下一步为什么是 EXP”。

| 页 | 标题与作用 | 建议时间 |
| --- | --- | ---: |
| 1 | 标题：给出系统、任务和本次阶段主题 | 0.5 min |
| 2 | 旧 distributed-column 路径：解释 16 workers/row 的控制与归约膨胀 | 1.5 min |
| 3 | 三轮架构演进：先建立 whole-system 进展，再引出 accelerator 指标 | 1.0 min |
| 4 | 总体架构：单 tensor job、16 Row Engines、4 HBM nodes | 1.5 min |
| 5 | Row Engine 微架构：vector/reduction/EXP/DMA 资源解耦 | 1.5 min |
| 6 | Scratchpad 数据流：一次读、一次写、EXP 中间值本地复用 | 1.0 min |
| 7 | 四上下文流水：解释 32.8% stage overlap | 1.2 min |
| 8 | NoC/HBM 布局：连续 band striping 与四节点带宽分担 | 1.0 min |
| 9 | ABI 与完成语义：DMA ACK 驱动 completion | 1.0 min |
| 10 | 测试设置：参数、golden、lifecycle 和周期口径 | 1.2 min |
| 11 | 三个负实验：worker、burst、polling 都不是关键解 | 1.5 min |
| 12 | Target timeline：17.6k DMA 路径与 66.1k compute 路径 | 1.5 min |
| 13 | Shape scaling：16/64/256/1024 四点稳定增长 | 1.0 min |
| 14 | GPU 量级参考：强调口径边界，不做不公平性能宣称 | 1.0 min |
| 15 | 当前瓶颈与下一步：4-lane EXP -> 16-result/cycle EXP2 | 1.5 min |
| 16 | 总结：架构、数据流、正确性、性能和下一阶段 gate | 0.8 min |

## 汇报时应重点强调

1. `66,062 cycles` 是 descriptor acceptance 到 accelerator completion 的主要指标。
2. `72,409 cycles` 是排除诊断输出后的 guest kernel window。
3. `598,221 whole-system cycles` 来自可比较的非扩展诊断版本，不与 clean instrumentation 的 whole-system 时间混用。
4. stage active cycles 可以重叠，`16,384 + 65,536 + 16,384` 不能直接当作 critical-path latency。
5. `16.9k cycles` 是下一版 16-wide EXP 模型估算，不是已经测得的结果。

## 可按时间删减

- 12 分钟版本：保留 1--7、10--13、15--16，跳过 8、9、14。
- 10 分钟版本：保留 1、2、4、5、10、12、13、15、16。
