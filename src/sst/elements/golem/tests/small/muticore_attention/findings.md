# Attention 架构与模型真实性审计

**审计日期：** 2026-07-22
**范围：** Golem 的 SFU、per-Core GlobalMemory、RoCC、ComputeArray、
WorkerCommandProcessor、manager/control plane，以及它们连接的 NoC/HBM/cache。

## 判断标准

“实际模拟”不等于必须写 RTL，也不等于禁止使用 C++ vector。一个 C++ 对象可以
表示硬件寄存器、SRAM 或队列，但必须同时满足：

- 容量有上限并由配置或结构确定；
- 读写端口、带宽、latency、queue 和 arbitration 明确；
- 数据移动有 bytes、开始/完成事件和资源占用；
- completion 不能早于最后一个真实因果事件；
- 跨 core 数据不能通过进程级共享对象绕过 NoC。

功能算术可以继续在 C++ 中计算，只要结果在建模的资源完成前不可见，且其存储
和搬运没有被隐藏。

## P0：fused Attention 前必须整改

### 1. per-Core GlobalMemory 本地访问没有 SRAM 时序

证据：`globalmemory/globalmemory.cc:507-550` 的 `wr_to_globalmem()` 和
`rd_from_globalmem()` 直接对 `storage` 执行 `std::copy`；参数
`globalMemTransLatency` 建立了 self link，但本地读写没有使用它。RoCC、SFU、
WCP、GroupCtrl 和 RequestScheduler 都可以同步调用这两个接口。

影响：同一周期可以有任意数量、任意字节的本地访问；Array、SFU、DMA 和控制
访问不会争用端口，fused kernel 的 local-memory 性能将失真。

整改：扩展 `GlobalMemoryAPI`，提供异步 local read/write request，至少建模
read/write port 数、bytes/cycle、base latency、queue depth、client/tag、
completion callback 和统计。`storage` 继续作为数据内容，但不能决定完成时间。

### 2. SFU Row Engine 隐式拥有整行私有存储

证据：`sfu/sfu.h:403-411` 的每个 context 包含 `std::vector<float> values`；
`sfu/sfu.cc:1945-1955` 从 Local GM 一次性复制整行；`sfu/sfu.cc:2006-2070`
在该 vector 上完成 scale/mask、MAX、EXP/SUM、NORMALIZE，并直接构造输出 DMA
payload。`rowMax` 和 `rowSum` 已具有寄存器语义，但 `values` 相当于未充分建模的
大容量 Row Buffer。

影响：三阶段之间没有 Local GM bytes、port contention 或容量压力；SFU 可以
隐式保存完整 4096-element row。

整改：SFU 内保留固定数量 context register，包含 `m/l/inv_sum/scale/mask/
row/stage/valid`；另设容量受限的 lane FIFO/register。完整 S/P tile 位于 per-Core
GlobalMemory，各阶段通过异步 chunk 访问推进。

### 3. legacy SFU primitive/tile 路径是同步读算写

证据：`sfu/sfu.cc:868-987` 的 `issueSoftmaxTile()`、`issuePrimitive()` 和
`issuePrimitiveBatch()` 在 issue 调用内读取输入、执行 `std::exp/log/sqrt/...`
并写回输出；这些路径没有统一占用 Row Engine vector/EXP resource，也没有 Local
GM 时序。

影响：如果 Attention 或性能测试误用 legacy API，结果功能正确但 latency 近似
为零，并可绕过 tensor Row Engine 的资源冲突。

整改：新 fused/性能路径禁止使用这些同步 API。后续若继续支持性能测试，应把
它们迁移到同一个 SFU resource scheduler；否则明确标记为 functional-only。

### 4. RoCC 本地搬运使用固定延迟加直接复制

**2026-07-23 状态：measured GEMM 路径已整改（C0.3c）。** blocking 与 batch
GM2IMAT/GM2IVEC 现在组合异步 Local GM read 和 bounded Array programming；
OVEC2GM 组合 bounded Array readout 和异步 Local GM write。固定延迟参数仅为旧
配置兼容保留，不再决定 completion；整数 output 使用原始字节 readout，避免
`int64_t -> double -> int64_t` 的精度损失。

证据：`rocc/roccAnalog.h:1334-1450` 的 OVEC2GM、GM2IVEC、GM2IMAT 在固定
cycle 数后直接访问 GlobalMemory 和 Array vector。默认 runner 将三类延迟设为
10 cycles，和 payload bytes、Local GM 端口、Array buffer 端口无关。异步 load
在 `rocc/roccAnalog.h:2100-2190` 也采用同类 fixed-ready-cycle 模型。

影响：64 B vector 与数 KiB matrix 可以具有相同延迟；SFU 和 Array 同时访问
Local GM 时没有竞争。

合同测试禁止上述路径恢复同步 GM/Array API 或 `ready_cycle`。

### 5. 当前 coordinator/worker 物理映射不能直接用于 manager

证据：已验收 Softmax runner 使用 `--group-manager-enable 0`，Core 0 同时是
coordinator 和 worker。runtime 把 `coordinator_core`/`owner_core` 设为 executor
core；`sfu/sfu.cc:2202` 使用 `band % worker_cores` 作为消息目标 Core ID。开启
dedicated manager 后，worker physical ID 会后移，该假设不成立。

整改：manager descriptor/dispatch 持有版本化 worker map。manager-core RoCC
control FSM 只维护 job、phase、completion bitmap；worker-local RoCC/SFU 执行
数据流。manager SFU datapath 不接收 compute dispatch。legacy Core 0 路径保留
到新路径通过回归。

## P1：相关路径用于性能结论前必须整改

### 6. WorkerCommandProcessor 存在未声明的 panel 和 partial-C 存储

**2026-07-23 状态：WCP measured path 已整改（C0.3a/C0.3b）。** `partialCTiles_` 已删除，partial C 已
迁移到有地址、有容量和异步端口时序的 per-Core GlobalMemory；operand panel 也已
改为 WCP client 的分块异步 Local GM read。`activeMatPayload_`/
`activeVecPayload_` 目前只在 callback 完成后短暂承载 Array transfer 数据；实际
programming/readout 已进入有界 Array buffer scheduler。

原始证据（整改前）：`workercmdproc/workercmdproc.h` 曾将 Local GM 的
matrix/vector panel 同步复制到 `activeMatPayload_`/`activeVecPayload_`，并按
reuse 数动态分配 `partialCTiles_`，再在这些 vector 与 Array output vector 之间
直接复制。当前合同测试禁止恢复 operand 同步读取和 `partialCTiles_`。

影响：GEMM partial C 和 operand panel 获得了没有地址、容量、端口或时序的隐式
本地存储。fused Attention 若复用 WCP 保存 Oacc，会重复 SFU 的同类问题。

整改：operand panel 留在 per-Core GlobalMemory 并通过建模的 Array buffer port
读取；partial C 优先保存在 Array accumulator，spill 时分配 Local GM Oacc 区。

### 7. ComputeArray 只建模 compute latency，未建模 buffer programming

**2026-07-23 状态：measured WCP 与 legacy RoCC 路径已整改（C0.3b/C0.3c）。** ComputeArray 已提供有界、
按字节计时的异步 operand programming/output read/output write，并输出 per-core
统计。legacy 同步 item/vector API 只为未迁移的 functional 路径兼容保留，新的
fused 路径及 measured GEMM 路径不得调用。

证据：`array/mvmComputeArray.h:78-98` 通过 self event 建模 MVM completion；但
`setMatrixItem()`、`setVectorItem()`、`getOutputVector()` 和
`moveOutputToInput()` 在 `:103-168` 立即访问或复制内部 vector。

评价：MVM host dot-product 本身可以保留为功能实现，当前按 MAC/CU 和 pipeline
推导的 compute cycles 也可继续使用。缺口是 matrix/input/output buffer 的容量、
端口、programming/readout bandwidth、occupancy 和 stall。

整改：在 ComputeArray API 后增加 bounded async buffer operation；MVM 只能在
operand ready 后开始，输出只能在 compute completion 后通过 output port 读取。

### 8. 静态 reducer 可以绕过跨核传输

**2026-07-23 状态：manager tensor Row Engine 路径已整改（C0.4）。** manager
RoCC 通过 GlobalMemory explicit-NoC transport 分发 physical worker band 并接收
唯一 completion；该路径不读取 static reducer map。RoCC 是 reduction handler
唯一持有者，并把非 manager message 转发给 SFU，避免 manager/SFU 双重注册。

证据：`sfu/sfu.cc:50-90` 的 `softmaxReducerRows()`、
`distributedSoftmaxReducerRows()` 等是进程级 static map。当前已验收 tensor
Row Engine 的 band dispatch/completion 使用 `explicit_noc`，但 legacy shared/
modeled reduction 路径仍可通过这些 map 共享状态。

整改：任何测量的跨核 reduction/coordination 必须经过 explicit NoC message 或
独立的有限容量 reduction component。static map 仅可作为单核 functional oracle，
并应由运行时契约阻止进入性能模式。

### 9. GroupCtrl/RequestScheduler mailbox 绕过 Local GM 时序

证据：`groupctrl/groupctrl.cc:541-565` 和
`requestscheduler/requestscheduler.cc:1111-1125` 通过同步 Local GM helper 轮询和
更新 mailbox。两组件自身已有 link latency、queue、credit 和 issue budget，
缺口集中在 mailbox 接口。

整改方案二选一并固定：把 mailbox 定义为组件内部有限端口 control register，
或把它作为普通 Local GM client 进入统一 arbitration。不能继续既占用 GM 地址、
又零时延访问。

### 10. DMA 的本地源/目的端口尚未完整建模

**2026-07-23 状态：Softmax Row Engine 与 WCP final-C 路径已整改。** WCP 最终
输出先落入完整 tile 大小的 `local_out`，再由 address-based DMA 经 Local GM read
port 发送，并等待 HBM ACK。其他 legacy payload DMA caller 仍需按是否进入性能
路径逐项清理。

证据：HBM DMA packet、NoC、MemController 和 completion ACK 已建模；但
`dma_write_to_host()` 接受一个已经构造好的 C++ payload，调用者可以绕过 Local
GM read port。DMA read response 到达后也通过同步 `wr_to_globalmem()` 落地。

整改：新增以 Local GM 地址为源/目的的 DMA API。DMA 只有在本地读出/写入完成
后才能发送/确认；本地端口占用和 NoC/HBM completion 均进入因果链。

## 已有充分模型，继续复用

- NoC：`architecture/noc_builder.py` 使用 Merlin `hr_router`/mesh，已建模 link、
  xbar、flit、buffer、VN 和 stall。
- HBM：architecture 使用 memHierarchy Directory/MemController 和 DRAMSim3
  backend，GlobalMemory DMA 通过 NoC 收发并等待 completion。
- CPU cache/memory：Vanadis 加 memHierarchy L1/L2、MemNIC 和 Directory 路径。
- GroupCtrl/RequestScheduler 的网络 link、queue、credit 和 issue budget 已有模型；
  只需修正本地 mailbox 边界。

## P2：fused 功能验收后的校准项

- ComputeArray 当前用理想 host dot product 生成结果，并按
  `ceil(input/mac_per_cu_per_cycle)+pipeline_depth` 给出 latency。发布目标 CIM
  性能结论前，需要用明确的 datapath/RTL/论文参数校准 MAC、pipeline、ADC/DAC
  或数字累加假设。
- SFU 当前用 `std::exp`、`std::sqrt` 等生成理想 FP32 功能结果。若后续评估模型
  精度或近似硬件，需要定义 approximation、rounding、overflow/underflow 和
  accumulation precision；这不阻塞首个 FP32 fused timing path。
- 当前没有 area/energy 模型。只报告 cycles、traffic、utilization 时可以不做；
  一旦声称能效或面积优势，必须补充经来源校准的模型。

## 整改顺序

1. GlobalMemory async local-access scheduler。
2. SFU bounded register/lane context，并迁移 tensor Row Engine 数据访问。
3. RoCC/Array 本地搬运接口；同步处理 DMA 本地端口。
4. manager coordinator 与显式 worker topology map。
5. 首个 S32 fused case。
6. 若 fused executor 使用 WCP，则先移除 WCP hidden panel/partial-C storage。
7. 在发布最终性能结论前补齐 Array buffer programming/readout 模型。
