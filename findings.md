# Findings

- GEMM 主 runner 已支持 `--mpi-ranks`、`sst.simple` partitioner、每 rank DRAMSim3 输出目录和统计合并。
- 当前 active attention 入口仅为 `small/muticore_attention/run_flash_attention.sh`，默认执行 E3 scale/archive profile。
- 旧 standalone Softmax workload、materialized attention 和阶段性 runner 已从工作树删除，不再作为构建或测试依赖。
- 之前的 `Bad system call` 发生在受限执行环境中，属于 seccomp 拦截交叉编译器 syscall，不是源码错误。
- FlashAttention scale/archive 路径保持单 rank；MPI query-block 分区不属于当前 active scope。
