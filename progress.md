# Progress

## 2026-09-02
- 开始 attention 本地适配工作。
- 已确认 attention/softmax workload 文件存在，且 MPI hooks 当前集中在 GEMM 主 runner 和 `architecture/ncores_selfcom_dma_ctrl.py`。
- QK^T attention runner 已接入 MPI 参数并保持本地 LLM runner 链路。
- materialized attention 已向 QK/PV GEMM 阶段透传 MPI 参数。
- attention 套件回归测试：100 tests passed。
- standalone Softmax 和 materialized attention 路径已删除，当前只保留融合 FlashAttention scale/archive 路径。
- shell 语法和 attention 套件回归通过；当前环境在 RISC-V worker 交叉编译时出现 `Bad system call`，属于执行环境限制，主机 SST/Golem 本地库编译已成功。
- 在正常主机 syscall 环境复现并修复 Softmax worker 链接错误：`LDFLAGS_STATIC=-static -no-pie`；修正后的 `make -j2` 已成功。
- 修复后产物为静态 RISC-V ELF；attention 套件重新验证为 100/100，通过 shell 语法和 `git diff --check`。
- 旧 standalone/materialized 阶段已删除；历史验证结果不再作为 active workflow。
- scale attention E3（S1024,D128，4 manager + 16 worker）真实本地通过：`Fused Attention E3 PASS`。
- scale archive architecture 暂不支持安全 MPI query-block 分区；runner 已对 `GOLEM_MPI_RANKS>1` 增加明确门禁。
