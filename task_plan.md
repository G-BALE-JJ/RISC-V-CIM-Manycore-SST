# Attention Local Adaptation

## Goal
让 LLM 项目的 attention 相关 workload 使用本地构建的 SST/Golem，并逐步支持与 GEMM 一致的 CPU 多核/MPI 仿真和结果校验。

## Phases
- [completed] 1. 盘点 attention/softmax 入口、架构配置和构建契约
- [completed] 2. 接入单 rank 本地库运行
- [completed] 3. 接入 MPI rank 切分、DRAMSim3 隔离和统计合并
- [completed] 4. 完成 Softmax/attention 端到端小规模验证
- [completed] 5. 回归测试并记录运行方法

## Next Step
任务完成；当前只维护 FlashAttention scale/archive E3 基线，E4/E5 作为显式压力 profile。

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
