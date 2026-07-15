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

另外，archive 源码目前无条件传入 `golem_dma_response_vn` 的默认字符串 `0`；
一旦 MemNIC 开始消费该参数，所有 archive/no-ctrl workload 的实际响应 VN 都会从
历史派生值 VN1 改成 VN0。这个兼容性变化必须显式处理，不能只依赖默认 GEMM 回归。

## 方案选择

采用共享 MemNIC 的向后兼容参数化方案：

1. `memNICBase` 显式读取可选的 `golem_dma_response_vn`。
2. 未配置时仍使用现有推导规则，保持默认 ctrl-link GEMM 行为不变。
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
GlobalMemory request                       -> VN0
GlobalMemory ordinary READ response        -> VN1（保持现状）
directory MemNIC Golem DMA completion      -> VN0（softmax 显式配置）
GlobalMemory SFU reduction request/response -> GOLEM_SFU_REDUCTION_VN（0、1 或 2）
```

这三个 VN 共用物理 Merlin 链路，但拥有独立的虚拟队列和流控状态。VN sweep 比较的
是 reduction 与其他流量共享或隔离逻辑队列后的效果，不应解释为三套独立物理 NoC。
其中 VN0 会与 DMA request 和 softmax DMA completion 共享，VN1 可能与
GlobalMemory ordinary READ response 共享，VN2 在当前 workload 中预计最少共享流量；
VN1/VN2 不是天然等价的控制组。

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

越界配置必须 fatal，禁止静默回退或取模。该参数只影响目录 MemNIC 的 Golem DMA
bridge 生成的 read/write completion event，不改变 `GlobalMemory::response_vn`
（普通 `NetworkDataEvent::READ` response 仍为 VN1）或 DMA request VN0。

为保留 archive 的既有实际行为，基础
`architecture/archive/ncores_selfcom_dma.py` 必须将未设置环境变量时的参数默认改为
legacy-derived VN1；环境变量显式设置时才允许覆盖它。softmax shim/runner 再显式
传入 VN0。必须增加 archive/no-ctrl 回归，证明这不是静默的全局行为变化。

### Standalone Softmax Architecture Shim

shim 不再将匹配到的目录 MemNIC `num_vns` 从 3 改成 1，保留 3 VN，并继续补充
softmax 所需的 network buffer、drain limit、统计开关和 guest 环境转发。DMA
response VN 使用 `GOLEM_DMA_RESPONSE_VN`，本实验 runner 固定为 0。shim 的源码
替换必须对目标片段做 exactly-once 检查：匹配数不是 1 时立即失败，并分别检查
`num_vns=3` 保留、buffer/drain 参数注入和 DMA VN 注入均成功。

若基础 archive 结构变化导致匹配失败，应在配置阶段报错，而不是静默运行不完整拓扑。

### Multi-VN Runner Artifact Identity

distributed scaling runner 增加 `REDUCTION_VN`，默认读取
`GOLEM_SFU_REDUCTION_VN`，未设置时为 0。它必须：

- 验证值为 0、1 或 2；
- 在 run ID 中加入 `_vn${REDUCTION_VN}`；
- 在 manifest 增加 `reduction_vn` 列；
- 在 marker signature 增加 `reduction_vn`；
- 在子进程环境中显式传递 `GOLEM_SFU_REDUCTION_VN`；
- 在子进程环境中显式传递固定的 `GOLEM_DMA_RESPONSE_VN=0`；
- 将 `num_vns=3` 和 `dma_response_vn=0` 写入 manifest/signature；
- 在 dry-run 输出中显示 VN。

本阶段的 VN sweep 入口必须强制
`GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc`；VN1/VN2 不得在
`modeled_noc` 下生成“通过”产物。runner 应在创建 SST 前拒绝非法 reduction VN、
非 explicit transport 和继承的非零 DMA response VN。

每个真实点还必须留下可解析的 resolved-topology 证据：
`num_vns=3,reduction_vn=N,dma_response_vn=0,globalmemory_response_vn=1`。
仅有 run ID、golden 和聚合 counters 不足以证明消息实际走了目标 VN。建议开启
`GOLEM_GM_VERBOSE=2` 并解析 GlobalMemory mapping 行，同时由 MemNIC 输出 DMA
bridge response VN，并开启 `GOLEM_DMA_TRACE=1` 抽查 completion request 的
`vn=0`；该证据应进入 artifact validator，而不是作为可选诊断。

这会使新 run ID 与既有 VN0 artifact 不同。旧 marker 不迁移、不重命名，也不在新
多 VN root 中复用。一个 root 内的 HBM、inputs、manifest 和 run-summary 仍是共享
状态，因此 runner 必须对 root 加互斥锁；不同 VN 建议使用不同的新 root，禁止并发
写同一 root。manifest 已存在时必须逐字校验新 schema header，旧 15 列 schema 直接
fail-fast，不能追加缺少 `reduction_vn,num_vns,dma_response_vn` 的新记录。

## 错误处理

- `golem_dma_response_vn >= num_vns`：MemNIC 初始化 fatal。
- reduction VN 不在 0..2：runner 在启动 SST 前返回配置错误。
- resolved `reduction_vn` 与 manifest 请求值不一致、`dma_response_vn != 0` 或
  resolved topology 其他字段与 manifest 不一致：artifact validation 失败。
- reduction VN 超出 GlobalMemory endpoint 的 `num_vns`：保留 GlobalMemory
  初始化 fatal，作为第二层防御。
- 任一 VN smoke 的 golden、reduction counter、transport receive 或 DMA gate
  失败：停止 sweep，不进入性能比较。

## 测试策略

按 TDD 顺序增加 focused tests：

1. MemNIC 参数测试覆盖：`num_vns=3` 缺省值仍为 VN1、显式 0 生效、显式 2 生效、
   显式 3 初始化 fatal，以及 `num_vns=1` 时缺省值为 VN0。
2. shim 测试要求 3 VN 保留、DMA response VN 可配置、exactly-once 替换，并验证
   archive legacy default 不被无意改成 VN0。
3. runner 测试要求 VN 出现在 run ID、manifest、signature 和 child environment，
   并验证不同 VN signature 两两不同、非法 VN/transport/继承 DMA VN 在 dry-run
   前失败、旧 manifest header 被拒绝、同一 root lock 能阻止并发写入。
4. validator 行为测试要求 resolved topology 与 manifest 不符时失败，并要求
   explicit-NoC 的 GlobalMemory send/receive/rejected runtime diagnostics 可解析；
   nested stats 不出现在 CSV 时不能静默跳过 runtime evidence。
5. 运行现有 softmax focused Python tests，更新所有旧 run ID fixture 为 VN-aware
   形式，确保旧 artifact validation contract 未被错误放宽。

## 构建与真实 SST 验证

由于修改 `memNICBase.h`，必须重新构建并安装 `memHierarchy`，再重新构建并链接
`golem/libgolem.so`。构建临时目录使用 `/data4/jjgong/.tmp`。构建后用
`sst-info memHierarchy.MemNIC` 确认新 ELI 参数存在，并检查运行时实际加载的
`libmemHierarchy.so`/`libgolem.so` 路径和 build identity，防止旧库造成“golden PASS
但 VN 参数未生效”。

真实 SST 顺序为：

1. 保存改动前的 default ctrl-link GEMM baseline identity；先做 VN0 anchor：
   `rows=16, dim=512, worker=4, band=4`，确认恢复 3 VN 后现有 correctness、
   resolved topology 与 transport contract 不回归。
2. 同一 anchor 分别运行 VN1、VN2；每个 VN 使用独立新 root、相同输入/HBM SHA、
   相同 guest binary 和两套 SST library identity。
3. 三点均要求 golden 8192/0、四类 reduction counter 各 64、transport receive
   256、DMA read/write issue 与 completion 各 64、retry/exhausted 为 0。
4. 汇总 simulated time、transport latency、inbox high-water 和 queued send。
5. 最后用清洁环境运行已验证的 64x64x64 fp32 canonical GEMM command：显式 `env -u`
   清除所有 `GOLEM_SFU_*`、`GOLEM_SFU_REDUCTION_VN`、`GOLEM_DMA_RESPONSE_VN`、
   `GOLEM_ARCH_SCRIPT`、softmax、group/control/WCP 覆盖变量，只保留矩阵尺寸、
   `--tensor-source sample` 和 `--verify-c`。要求 runner exit 0、日志明确出现
   `[VERIFY-C] PASS` 和 `Simulation is complete`，DMA issue/completion/bytes/retry
   与既有 baseline 一致，reduction activity 为零（统计项缺失或值为零均可，但
   不得出现非零 reduction event）。另外运行一次 archive/no-ctrl GEMM smoke，
   验证 legacy response VN1 兼容性。

canonical GEMM 的矩阵参数固定为 `--gemm-m 64 --gemm-n 64 --gemm-k 64
--gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 --dtype fp32
--tensor-source sample --verify-c`；不得把历史 softmax archive 或 no-ctrl
architecture 作为默认 GEMM 证据。

任一 GEMM 非回归失败都阻止本阶段完成。

## 非目标

- 不修改 `SFUJobDesc` ABI 或 softmax 数学算法。
- 不回到 primitive/batch softmax 主线。
- 不为 SFU 新增独立物理网络。
- 不扩展 rows、dim、chunk、worker/band 矩阵；本阶段只隔离 reduction VN 变量。
- 不删除或覆盖已有 VN0 scaling artifacts。
