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

## 2026-09-03
- 开始 FlashAttention scale/archive query-block MPI 移植。
- 硬约束：不修改 GEMM 测试路径或测试语义；先验证既有 MPI 基础设施，再只在 Attention 边界实现适配。
- 完成第一轮差异调查：archive 架构无 MPI 放置，Attention guest 已按 4 manager 划分 query band。
- 完整对照 archive 与参考 ctrl：MPI 分区由 `sst.simple` 自动完成，archive 需补齐 MPI 环境、NoCut 条件和 DRAMSim3 输出隔离。
- GEMM MPI 与 RISC-V toolchain 契约共 9 项通过；未修改 GEMM 测试路径。
- 新增 Attention MPI runner/archive 契约；首次执行发现并修正测试自身的 pathlib 层级错误。
- Attention MPI 契约已红绿验证：scale runner 显式传 rank，archive 支持 MPI NoCut 条件与 DRAMSim3 node 隔离。
- 实现仅修改 Attention scale runner 和 archive 架构，未修改 GEMM runner/测试路径。
- E2 单 rank 和 2-rank 实测均 PASS，数值输出、生命周期与完成周期一致；MPI 统计合并及 DRAMSim3 node 隔离正常。
- 检查 rank 原始统计后发现自动分区不按 query manager group 共置；回到实现阶段增加 Attention 专用显式放置。
- 首次显式放置 E2 双 rank：数值与生命周期仍通过，但运行后分区校验失败；原始统计证明 `sst.simple` 覆盖了 `setRank()`。
- 查阅本机 SST `selfpart.h`/`simplepart.h`，确认 Attention 显式放置必须使用 `sst.self`；开始进行单变量 partitioner 修复。
- `sst.self` 首次启动准确暴露未赋值组件：`Bad rank: 4294967295`；确认 CPU builder 的多个顶层缓存/总线/TLB 和 NoC 路由器也需要显式 rank。
- 在 Attention archive 内统一放置每个完整 CPU 子系统及其本地路由器，未修改共享 CPU builder 或 GEMM runner。
- 完整放置已越过 SST rank 检查，但组件实例化发现 `gmem_reduction_send_immediate` ELI 缺失；回溯显示本地 `libgolem` 与 `/data/shun/.../libmerlin` 被同时加载。
- 确认污染源是 SST core 全局配置中的旧 element 安装路径；Attention runner 开始强制使用本工作树完整 install 目录和覆盖式 `--lib-path`。
- 首次库隔离运行不再加载 `/data/shun`，但过度覆盖 `SST_SOFTMAX_LD_LIBRARY_PATH` 排除了 DRAMSim3；撤掉该覆盖，继续由通用 runner 拼接 DRAMSim3 动态库路径。
- E2 纯本地双 rank 实测 PASS：数值、生命周期、统计合并、DRAMSim3 隔离和 query-block 10/10 核心映射全部通过。
- E2 纯本地单 rank 实测 PASS；与双 rank 的结果 JSON、生命周期 JSON、四个 HBM 输出文件完全一致。
- E2 4-rank 实测 PASS：四个 manager/worker group 各占一个 rank，5 cores/rank。
- E3 2-rank 实测 PASS，并新增 `baseline/e3/mpi2/result.json` 与 `scripts/test_flash_attention.sh --mpi-ranks 2` 冻结回归入口。
- 正式 E3 MPI 测试命令完整通过，包括 7 项源契约、仿真、数值、生命周期、分区及冻结基线匹配。
- 收尾审计发现 scale runner 仍隐式执行 guest `make`；将所有 FlashAttention guest 构建移至本地构建脚本，测试路径只检查并运行已有产物。
- 本地增量构建/安装实际通过，`[5/5]` guest 构建阶段确认 E2-E5 产物均就绪。
- 根据独立审查：隔离专用 SST args、运行前清理关键派生文件、扩展 router/rank-file 校验、增加 shape/非物化冻结门禁，并加入 E3 4-rank 基线入口。
- E3 4-rank 正式冻结回归通过：131072 项、0 mismatch、20 个 core 与 28 个 router 全部按 query group 正确放置，4 个 rank 统计文件完整且无陈旧文件。
- E3 单 rank 正式冻结回归通过，与 MPI 版本保持相同数值误差上限和 3561012-cycle 生命周期。
- 最终回归：Attention 13 项、GEMM MPI 4 项、RISC-V toolchain 5 项全部通过；shell/Python 语法、baseline JSON、`git diff --check` 全部通过，GEMM 三个关键路径无 diff。
- 根据最终复审补齐构图证据：E3 4-rank 运行时 manifest 覆盖 200 个顶层组件，OS、OS L1、目录、内存、CPU 子系统和路由器均无缺失、多余或错放。
- 最终复审无 Critical/Important；其余命名、帮助文本和完整性负向测试三个 Minor 也已处理。
