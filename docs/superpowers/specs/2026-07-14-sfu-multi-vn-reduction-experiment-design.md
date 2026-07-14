# SFU Explicit-NoC Multi-VN Reduction 实验设计

## 目标

在不改变 GEMM 默认行为和 unified-job softmax 数学路径的前提下，将 standalone
softmax 网络从临时的 `num_vns=1` 恢复为项目默认的 `num_vns=3`，使 SFU
explicit-NoC reduction 可以分别运行在 VN0、VN1 和 VN2，并确保不同 VN 的实验
产物不会互相覆盖或错误命中缓存。

## 当前问题

基础 archive 拓扑的 Merlin、GlobalMemory 和目录 MemNIC 均配置为 3 VN。当前
softmax shim 将目录 MemNIC 的 `num_vns` 改成 1，以强制 Golem DMA response 使用
VN0。

基础 archive 已向目录 MemNIC 传入 `golem_dma_response_vn`，但当前
`memNICBase.h` 没有读取它，而是按 `num_vns >= 2 ? 1 : 0` 推导 DMA response VN。
因此直接删除单 VN shim 会把 DMA response 移到 VN1，并重新暴露历史 DMA 等待路径。

distributed scaling runner 目前也没有把 reduction VN 纳入 run ID、manifest 和
marker signature。直接在同一 root 扫 VN0/1/2 会产生 artifact identity 冲突。

## 方案选择

采用共享 MemNIC 的向后兼容参数化方案：

1. `memNICBase` 显式读取可选的 `golem_dma_response_vn`。
2. 未配置时仍使用现有推导规则，保持默认 GEMM 和其他 SST workload 行为不变。
3. 配置值必须小于 `num_vns`，越界时在初始化阶段明确失败。
4. standalone softmax shim 保持统计和环境转发逻辑，但不再把目录 MemNIC 改为
   单 VN；它显式将 DMA response 固定到 VN0。

不采用 softmax 专用 MemNIC 分支，因为它会复制共享网络、初始化和 endpoint
发现逻辑。不采用 SFU 独立网络端口，因为它偏离已验证的 GlobalMemory transport
bridge，并扩大拓扑与 GEMM 隔离风险。

## 网络映射

恢复后的 standalone softmax 拓扑为：

```text
num_vns = 3
Golem DMA request  -> VN0
Golem DMA response -> VN0（softmax 显式配置）
SFU reduction      -> GOLEM_SFU_REDUCTION_VN（0、1 或 2）
```

这三个 VN 共用物理 Merlin 链路，但拥有独立的虚拟队列和流控状态。VN sweep 比较的
是 reduction 与其他流量共享或隔离逻辑队列后的效果，不应解释为三套独立物理 NoC。

## 组件修改

### MemNIC DMA Response VN

`MEMNICBASE_ELI_PARAMS` 增加 `golem_dma_response_vn` 文档项。构造时先计算当前默认值：

```text
default_response_vn = num_vns >= 2 ? 1 : 0
```

再通过参数读取覆盖值。读取完成后检查：

```text
golem_dma_response_vn < num_vns
```

越界配置必须 fatal，禁止静默回退或取模。该参数只影响 Golem DMA bridge 生成的
read/write completion event，不改变普通 MemHierarchy coherence response 的 VN。

### Standalone Softmax Architecture Shim

shim 不再将匹配到的目录 MemNIC `num_vns` 从 3 改成 1，保留 3 VN，并继续补充
softmax 所需的 network buffer、drain limit、统计开关和 guest 环境转发。DMA
response VN 使用 `GOLEM_DMA_RESPONSE_VN`，本实验 runner 固定为 0。

shim 必须验证目标源码片段确实被替换；若基础 archive 结构变化导致匹配失败，应在
配置阶段报错，而不是静默运行不完整拓扑。

### Multi-VN Runner Artifact Identity

distributed scaling runner 增加 `REDUCTION_VN`，默认读取
`GOLEM_SFU_REDUCTION_VN`，未设置时为 0。它必须：

- 验证值为 0、1 或 2；
- 在 run ID 中加入 `_vn${REDUCTION_VN}`；
- 在 manifest 增加 `reduction_vn` 列；
- 在 marker signature 增加 `reduction_vn`；
- 在子进程环境中显式传递 `GOLEM_SFU_REDUCTION_VN`；
- 在 dry-run 输出中显示 VN。

这会使新 run ID 与既有 VN0 artifact 不同。旧 marker 不迁移、不重命名，也不在新
多 VN root 中复用。

## 错误处理

- `golem_dma_response_vn >= num_vns`：MemNIC 初始化 fatal。
- reduction VN 不在 0..2：runner 在启动 SST 前返回配置错误。
- reduction VN 超出 GlobalMemory endpoint 的 `num_vns`：保留 GlobalMemory
  初始化 fatal，作为第二层防御。
- 任一 VN smoke 的 golden、reduction counter、transport receive 或 DMA gate
  失败：停止 sweep，不进入性能比较。

## 测试策略

按 TDD 顺序增加 focused tests：

1. MemNIC 静态/构造语义测试先证明显式 response VN 当前未被读取，再实现参数读取、
   默认兼容和越界检查。
2. shim 测试要求 3 VN 保留且 DMA response VN 可配置，先观察旧单 VN替换导致失败。
3. runner 测试要求 VN 出现在 run ID、manifest、signature 和 child environment，
   并验证不同 VN signature 不同、非法 VN 在 dry-run 前失败。
4. 运行现有 softmax focused Python tests，确保旧 artifact validation contract 未变。

## 构建与真实 SST 验证

由于修改 `memNICBase.h`，必须重新构建并安装 `memHierarchy`，再重新构建并链接
`golem/libgolem.so`。构建临时目录使用 `/data4/jjgong/.tmp`。

真实 SST 顺序为：

1. VN0 anchor：`rows=16, dim=512, worker=4, band=4`，确认恢复 3 VN 后现有
   correctness 与 transport contract 不回归。
2. 同一 anchor 分别运行 VN1、VN2；每点使用独立 run ID 和全新 artifact root。
3. 三点均要求 golden 8192/0、四类 reduction counter 各 64、transport receive
   256、DMA read/write issue 与 completion 各 64、retry/exhausted 为 0。
4. 汇总 simulated time、transport latency、inbox high-water 和 queued send。
5. 最后运行原始 `run_noc_dma_pipeline.sh --verify-c` 默认 GEMM，不提供 softmax、
   SFU、architecture、ctrl 或 group override。要求 GEMM 正确性通过、DMA 生命周期
   完整且 reduction activity 为 0。

任一 GEMM 非回归失败都阻止本阶段完成。

## 非目标

- 不修改 `SFUJobDesc` ABI 或 softmax 数学算法。
- 不回到 primitive/batch softmax 主线。
- 不为 SFU 新增独立物理网络。
- 不扩展 rows、dim、chunk、worker/band 矩阵；本阶段只隔离 reduction VN 变量。
- 不删除或覆盖已有 VN0 scaling artifacts。
