# FlashAttention Query-Block MPI Adaptation

## Goal
在不改变 GEMM 测试路径和行为的前提下，让当前 FlashAttention scale/archive workload 支持 SST MPI query-block 分区，并完成单 rank 与多 rank 数值、生命周期及统计验证。

## Phases
- [completed] 1. 验证既有 GEMM MPI 契约并梳理 scale/archive 与 ctrl 架构差异
- [completed] 2. 设计并添加 Attention MPI 失败契约测试
- [completed] 3. 实现 query-block MPI 分区、完成事件汇总和统计隔离
- [completed] 4. 运行 E2 单 rank/2-rank 数值与生命周期对照
- [completed] 5. 运行 E3 MPI 回归并固化独立基线
- [completed] 6. 完成代码审查、文档和最终验证

## Next Step
实现与本机验收均已完成；下一步由仓库维护者审阅 diff 后提交并推送分支。

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| `rg` 将以 `--verify-c` 开头的 pattern 识别为参数 | 1 | 后续使用 `rg -- 'pattern'`，不重复错误命令 |
| 新契约的 archive 路径使用了错误的 `parents[2]` | 1 | 修正为指向 `tests/` 的 `parents[1]` 后重新建立红灯 |
| `run_flash_attention.sh --profile e2` 被主入口拒绝 | 1 | 主入口按既有约束保持 E3/E4/E5；内部验证直接调用 scale runner 的 E2 profile |
| 显式 `setRank()` 后 E2 双 rank 仍为 15/5 自动划分 | 1 | 本机 SST 源码确认 `sst.simple` 覆盖配置放置；改用专为预分区配置提供的 `sst.self` |
| 切换 `sst.self` 后报 `Bad rank: 4294967295` | 1 | `sst.self` 要求所有顶层组件预先分配；为 CPU 子系统、NoC 路由器、OS 和内存节点补齐显式 rank |
| 完整放置后 `GlobalMemory` 注册统计量报 ELI 缺失 | 1 | 回溯确认混载本地 `libgolem` 与 `/data/shun` 旧 element；Attention 入口用本地 install 完整库目录覆盖 SST 全局默认 |
| 纯本地 element 首次运行找不到 `libdramsim3.so` | 1 | 不再覆盖通用 runner 的动态库组合变量，仅固定 SST element 搜索路径 |
| 官方入口以模块方式加载契约测试时找不到同目录 verifier | 1 | 基于测试文件路径显式加入本地模块目录，直接运行与模块加载均通过 |
| 受限环境禁止 Open MPI 创建本机通信 socket | 2 | 确认失败发生在 `MPI_Init` 前，并在获准的本机环境完成 1-rank/4-rank 实测 |
