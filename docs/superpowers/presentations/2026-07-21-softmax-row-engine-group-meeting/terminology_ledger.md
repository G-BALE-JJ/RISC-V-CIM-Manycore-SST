# Terminology Ledger

| 术语 | 汇报中的固定含义 |
| --- | --- |
| Row Engine | 每个物理 SFU endpoint 内的行级 Softmax 状态机与资源模型 |
| tensor job | coordinator guest 发出的单个 tensor-level SFU job |
| row context | 一个 16 KiB row buffer 及其 DMA/compute/store 状态 |
| accelerator observed cycles | descriptor acceptance 到 completion observed 的 SST accelerator-cycle 窗口 |
| modeled critical cycles | Row Engine 吞吐与 pipeline 参数计算出的 ready-cycle 关键路径 |
| clean kernel window | guest 在诊断输出之后开始，到 accelerator wait 返回的 `rdcycle` 窗口 |
| whole-system cycles | 整个 SST 模拟区间按 2.3 GHz 换算；不得与 kernel cycle 混用 |
| stage active cycles | 某资源在所有行上的累计活跃周期，可与其他 stage 重叠 |
| temporal span | 从最早 stage start 到最晚 stage end 的关键时间跨度 |
| band striping | 以 64 行连续 band 为单位，轮转分配到四个 data HBM node |
| EXP+SUM | 计算 `exp(x-max)` 并在同一 pass 内累加 row sum |
| GPU reference cycles | 由 nominal SM clock 和 measured time 估算的参考值，不是 GPU cycle counter |
