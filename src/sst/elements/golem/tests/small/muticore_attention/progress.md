# Multicore Attention Progress

## 2026-07-22：架构决策与模型审计落盘

已完成：

- 保留 Phase A 的 native-K `QK^T` 和 Phase B 的 S64/D64 materialized
  non-causal/causal 验收结果。
- 确认每个 core 的 `golem.GlobalMemory` 是真实 SST subcomponent 和本地存储，
  后续 fused Attention 不再新增一块重复的 RoCC/SFU scratchpad。
- 确认 SFU 需要内部 Context Register File 和 lane FIFO/register；完整 S/P tile
  放在 per-Core GlobalMemory。
- 决定新 fused 架构使用 dedicated manager coordinator 和 worker-local RoCC
  executor；coordinator 落在 manager-core RoCC control FSM，manager SFU
  datapath 不参与计算。当前 worker Core 0 coordinator 仅作为 legacy Softmax
  回归路径。
- 完成 Golem 组件模型真实性初审，并在 `findings.md` 中记录 P0/P1 缺口、证据和
  整改顺序。
- 更新双语权威实施计划，将下一阶段改为 Phase C0 physical-model foundation。

当前有效 Softmax 基线保持不变：

- `1024x4096` actual accelerator completion：66,958 cycles。
- 66,061 cycles 仅为 analytical compute reference。
- 因果链仍是 input DMA -> MAX -> EXP/SUM -> NORMALIZE -> output DMA ACK ->
  unique band completion。

本次只修改文档，没有修改 SFU、GlobalMemory、RoCC、Array、WCP 或 runner，因而
没有触发 `libgolem` rebuild、focused runtime tests 或 GEMM regression。下一步从
GlobalMemory async local-access focused tests 开始。

## 2026-07-23：完成 Phase C0.1/C0.2

已完成：

- 为 per-Core `GlobalMemory` 增加统一异步本地访问调度器，按 read/write port、
  bytes/cycle、base latency、queue depth 和 max request bytes 建模，并输出
  machine-readable 请求数、字节数、拒绝数、队列高水位和排队周期。
- 把 Softmax Row Engine 的完整行 `context.values` 替换为 16-lane bounded
  buffer 与标量/阶段/tag 寄存器；MAX、EXP/SUM、NORMALIZE 均通过 Local GM
  异步 callback 推进。
- input DMA response 先按最大本地请求大小落入 Local GM，最后一个 landing
  callback 后才发布 input-ready；output DMA 直接从 Local GM 地址读取，并保持
  ACK -> unique band completion 因果关系。
- 完整重编并安装 `libgolem` 与 `libmemHierarchy`。由于 `globalmemory.h` 同时被
  两个库包含，只重编 `libgolem` 会产生 ABI 不一致；后续修改该头文件必须继续
  全量重编这两个库。

验证结果：

- focused Local GM + Row Engine tests：29/29 PASS。
- SFU Softmax discovery：22/22 PASS；multicore Softmax：12/12 PASS；
  multicore Attention：14/14 PASS。
- `muticore_softmax` 与 `muticore_attention` guest build PASS。
- `16x64` smoke：1024/1024 checked、0 mismatch，391 cycles。
- `16x4096`：65,536/65,536 checked、0 mismatch，5,345 cycles。
- `1024x4096`：4,194,304/4,194,304 checked、0 mismatch，max abs diff
  `5.72476014e-11`；actual accelerator completion **139,750 cycles**。
- `1024x4096` 的六个逐行因果事件各 1024 次，16 个 band completion 唯一；
  每核 Local GM read/write 为 `4,194,304/3,145,728` bytes，reject=0，
  high-water=4。
- canonical `64x64x64` FP32 GEMM regression：`VERIFY-C PASS`，64 个采样
  mismatch=0，max abs diff=0。首次沙箱内运行只因 OpenMPI socket 权限失败，
  允许本地 MPI 通信后同参数运行通过。

基线解释：

- 66,958 cycles 保留为加入真实 Local GM 时序前的冻结 legacy 基线。
- 66,061 cycles 继续仅作为 analytical compute reference。
- 139,750 cycles 是当前实现的端到端 accelerator completion；增加量主要来自
  input landing 和三遍 SFU Local GM 访问的端口/带宽排队，不能静默覆盖旧基线。

下一步是 Phase C0.3：迁移 fused 关键路径中的 RoCC/Array/WCP 本地搬运。
当前 WCP 的 `activeMatPayload_`、`activeVecPayload_` 和 `partialCTiles_` 仍是
host-side storage shortcut；RoCC 的部分 gm2imat/gm2ivec/ovec2gm 路径仍使用
固定延迟加同步复制。完成这些迁移并再次通过 GEMM 回归后，再进入 Phase C0.4
manager coordinator 与显式 worker mapping。

## 2026-07-23：完成 Phase C0.3a WCP operand/partial-C 迁移

已完成：

- 为 WCP 增加 callback 驱动的 operand load 状态机。matrix/vector panel 按 Local
  GM 最大请求大小分块异步读取，队列拒绝时后续 tick 重试，所有 callback 完成后
  才编程 Array 并开始 compute。
- 删除 `partialCTiles_` host-side 容器。partial C 以
  `local_accum_gm_addr + tile_index * partial_tile_bytes` 为地址异步 spill/reload，
  并新增 store wait phase 保证窗口推进晚于最后一个 write callback。
- partial tile 数从固定 reuse 窗口改为
  `min(reuse_M, M_tiles) * min(reuse_N, N_tiles)`；主 runner 与 Softmax wrapper
  使用相同公式计算 per-Core Local GM stride。
- `GlobalMemoryAPI` 暴露已配置的 local max request bytes，WCP 不再硬编码分块。
- 新增 `test_wcp_local_memory_contract.py`，锁定 WCP async client、禁止 operand
  同步读取并禁止恢复 `partialCTiles_`。

验证结果：

- Attention discovery：16/16 PASS；WCP 合同：2/2 PASS；Softmax pipeline
  wrapper focused tests：30/30 PASS；Attention guest build PASS。
- 完整重编并安装最新版 `libgolem`；build/install SHA-256 一致：
  `5827b0320bfc9e274240e3a9f7b6fe1c36cc3882c210290311a2930ac0391914`。
- 最新二进制的 `64x64x512` FP32 GEMM：`VERIFY-C PASS`，sampled=64、
  mismatch=0、max abs diff=0，system latency=6,530 cycles。
- core 4 Local GM：76 reads / 311,296 bytes，12 writes / 49,152 bytes，
  reject=0，high-water=2。其中 262,144 read bytes 是 8 组 matrix/vector operand，
  49,152 read bytes 和 49,152 write bytes 分别是 3 次 reload 与 spill。
- 全量 SFU Python discovery 的功能相关失败已通过 wrapper 修复消除；剩余 9 个
  error 均为当前 Python 环境缺少 `matplotlib` 的报告绘图测试，本阶段未安装依赖，
  也未修改任何报告、PPT、Draw.io 或图像素材。

当前边界：

- `activeMatPayload_`/`activeVecPayload_` 仅作为异步 callback 的短暂传输和 Array
  programming buffer，不再代表零时延 Local GM 副本；但 Array programming 本身
  仍是同步接口，因此 C0.3 不能标记完成。
- GEMM request-scheduler 的 panel DMA landing 尚未统一计入本地 DMA write 统计；
  本次 Local GM write 统计只覆盖 WCP partial-C spill，不能解读为全部本地写流量。
- 下一步继续 C0.3b：先增加 Array bounded async programming/readout，再把最终
  output writeback 改为 Array -> Local GM -> address-based DMA，最后迁移 legacy
  RoCC GM2IMAT/GM2IVEC/OVEC2GM。

## 2026-07-23：完成 Phase C0.3b Array/WCP 异步数据路径

已完成：

- 在 `ComputeArray` 内复用独立 self-link，实现按 base latency、bytes/cycle、port
  和 queue depth 建模的有界 buffer transfer scheduler；功能算术仍复用现有 MVM/
  CrossSim 实现。
- WCP matrix/vector programming 改为逐 Array 异步提交；program callback 后该
  Array 才开始 compute，队列拒绝由后续 tick 重试。
- partial-C spill 先通过 async Array readout 捕获，reload 后通过 async Array
  output write 恢复；WCP 中已无 `setMatrixItem`、`setVectorItem` 和
  `getOutputVector` 调用。
- final writeback 改为 Array readout -> Local GM full-tile landing -> Local GM
  source DMA -> HBM ACK，ACK 前不推进当前 reuse/task。
- 修正 `local_out` 容量：由单 output vector 扩为完整 C tile；主 runner 与 Softmax
  wrapper 同步更新 stride 公式，避免覆盖 partial accumulator。
- 新增 `test_array_buffer_async_contract.py`，并更新 C0.3a 合同的函数锚点。

验证结果：

- Attention discovery 19/19 PASS；其中 C0.3b 合同 3/3、C0.3a WCP 合同 2/2。
- Softmax pipeline wrapper 30/30 PASS；Attention guest build PASS。
- `libgolem` 完整重编、安装成功；最终 build/install SHA-256 一致：
  `5322b78af0c98334a805b9a3713220b668c1f5df8f3e96a836d6035e116bfba7`。
- 最终二进制 `64x64x64` FP32：`VERIFY-C PASS`，sampled=64、mismatch=0、
  max abs diff=0，system latency=18,397 cycles；core 4 Array buffer 128 requests /
  1,081,344 bytes、reject=0、high-water=1。
- `64x64x512` FP32 partial case：`VERIFY-C PASS`，sampled=64、mismatch=0、
  max abs diff=0，system latency=143,033 cycles。3 次 spill/reload 与最终 Local GM
  landing/DMA 的读写字节数全部闭合，说明 output window 未覆盖 accumulator。

基线解释：

- C0.3a 的 6,530-cycle `64x64x512` 结果保留为 Array buffer 时序加入前的参考，
  不是当前架构完成时间。当前 143,033 cycles 的主要增加来自此前同步完成的 512
  次大块 Array operand programming。
- Softmax 的重构前 66,958-cycle legacy 基线和当前 139,750-cycle Local GM Row
  Engine 结果均未修改，本阶段也未触碰汇报、PPT、Draw.io 或图像素材。

下一步为 Phase C0.3c：迁移 legacy RoCC GM2IMAT/GM2IVEC/OVEC2GM 固定延迟路径。
完成后再判断 Phase C0.3 是否可整体关闭并进入 manager coordinator mapping。

## 2026-07-23：完成 Phase C0.3c legacy RoCC 异步搬运

已完成：

- GM2IMAT/GM2IVEC 的 blocking 与 batch 路径统一为分块异步 Local GM read，再经
  bounded Array matrix/input programming；不再使用 fixed-ready-cycle、同步 GM
  read 或逐元素 Array write。
- OVEC2GM 改为 bounded Array async readout，再分块异步写入 Local GM；blocking
  RoCC response 晚于最后一个 Local GM write callback。
- 独立代码审查发现 `vector<double>` readout 会破坏超过 `2^53` 的 int64 输出；
  最终实现增加 byte-exact async readout，legacy OVEC2GM 不再经过浮点中转。
- ComputeArray/MVM/CrossSim 增加独立 matrix/input programming API，继续复用
  C0.3b 的 port、bandwidth、queue depth 和 byte-scaled latency 模型。
- 新增 3 项 RoCC async local-transfer 合同测试。

验证结果：

- Attention discovery 22/22、RoCC/SFU integration 9/9、Softmax wrapper 30/30
  全部通过；`libgolem` build/install SHA-256 一致：
  `446b991b40c49334a118ec6164e07f81624e3ee911c2f6d0460fe9eeaf7b08d8`。
- legacy `64x64x64` FP32：`VERIFY-C PASS`，sampled=64、mismatch=0、max abs
  diff=0，worker total=71,224 cycles。core 4 Array 为 192 requests / 1,081,344
  bytes；Local GM 为 320 reads / 1,064,960 bytes、64 writes / 16,384 bytes。
- canonical WCP `64x64x64` FP32：`VERIFY-C PASS`，system latency=18,397 cycles；
  core 4 Array 128 requests / 1,081,344 bytes，Local GM 12 reads / 49,152 bytes、
  4 writes / 16,384 bytes，与 C0.3b 结果一致。

调试边界：

- 额外的 16 维 legacy probe 暴露了既有 batch 布局限制：HBM B-vector slot 按
  256B 对齐，而指令按 raw vector bytes 前进，导致只在每四个 Array 读到有效
  向量。64 维 raw vector 恰为 256B，因此权威回归不受影响。该问题不属于 C0.3c
  callback 改造，但在以后支持非 64 维 legacy batch 前必须显式传递 stride 或统一
  packing contract。
- legacy 64 维正确性通过，但逐 tick 扫描 64 个 load 会积累大量 queue-rejection
  统计；这是后续 event-driven retry/公平仲裁优化项，不阻塞 C0.3 功能关闭。

Phase C0.3 已完成。下一步进入 C0.4：manager coordinator FSM 与显式 worker
slot -> physical Core ID topology map，并保留当前 worker/Core 0 legacy 回归路径。

## 2026-07-23：完成 Phase C0.4 manager coordinator 与物理 worker mapping

已完成：

- manager RoCC 增加独立 tensor issue/wait opcode、异步 metadata FSM、版本化
  topology map 校验、physical worker dispatch 和唯一 completion bitmap。
- worker dispatch message 显式携带 `workerCore`；manager completion 由 RoCC
  消费，其余 reduction transport 继续转发 SFU。manager SFU datapath 保持空闲。
- runner 增加 manager opt-in 与 manager/worker 数量参数；默认 legacy 16-core
  配置不变。解析器按 RoCC manager stats 与 worker SFU stage stats 联合验证因果链。
- RoCC Float/Int ELI 补齐 manager 统计声明，避免运行前 statistic registration
  fatal；新增 manager coordinator 合同测试。

验证结果：

- manager `16x64`：checked=1,024、mismatch=0、max abs diff
  `3.29887216e-09`，accelerator latency=418 cycles，4 次 band dispatch/唯一
  completion，16 行完整通过 input DMA -> MAX -> EXP/SUM -> NORMALIZE -> output
  DMA ACK，contract pass。
- legacy `64x64`：checked=4,096、mismatch=0、accelerator latency=470 cycles，
  16 次唯一 completion，contract pass。首次回归暴露 argv logical ID 被误用作
  physical Core ID；恢复 `sched_getcpu()` 物理映射后通过。
- canonical WCP `64x64x64`：`VERIFY-C PASS`，sampled=64、mismatch=0、max abs
  diff=0，system latency=18,397 cycles；DMA timeout/write/send retry 均为 0，
  manager statistics 全为 0。
- focused suites：manager/Softmax 合同 16/16、SFU transport 33/33、Softmax
  row-engine 22/22、GM/RoCC/wrapper 46/46、multicore Softmax 12/12、Attention
  30/30；两个 guest build 通过。
- 完整 SFU discovery 的其余 9 个 error 均来自缺少 `matplotlib` 的绘图测试；
  本阶段未安装依赖，也未修改报告、PPT、Draw.io 或图片。
- `libgolem` build/install SHA-256：
  `0877acd167f44e499c1d79892596c3d410a7142c432b04ac3e91d8413ed66514`。

Phase C0 已关闭。下一步是 Phase C1 单 KV tile fused Attention：
`S32,D64,Br16,Bc32`。

## 2026-07-23：完成 Phase C1 单 KV tile fused Attention

已完成：

- 增加版本化 Attention descriptor、manager issue/wait、显式 worker dispatch 与
  unique completion。manager core 0 不执行 SFU/Array 数据面，worker core 1 在
  output DMA ACK 后才发送最终 completion。
- 固定 C1 数据流为 K/V DMA once；每个 query block 执行 Q DMA -> QK Array ->
  Local GM S -> local SFU scale/max/exp-sum/normalize -> Local GM P -> PV Array ->
  Local GM O -> output DMA ACK。S/P 不分配 HBM 地址。
- Local GM window 为 26,752 bytes，覆盖 Q/K/V、复用的 S/P、O 和 128-byte
  metadata；SFU 使用已有 bounded context/lane 和异步 Local GM 访问。
- 增加 C1 guest、runner、HBM Q/K/V preload、全量 Attention verifier 和精确 SST
  统计校验器。

验证结果：

- `B1,H1,S32,D64,Br16,Bc32` non-causal：checked=2,048、mismatch=0、max abs
  error=`1.3586599793141696e-08`；S/P logical HBM traffic=0。
- manager issue/complete=1/1；worker QK/PV Array ops=64/128；SFU jobs=2、
  rows=32；manager SFU jobs=0；S/P HBM bytes=0。首次全量运行发现第二个 query
  block 的 panel 未复位而只有 48 次 QK，修复后计数和数值均闭合，并将这些计数
  固化为 runner gate。
- functional focused tests：Attention 37/37、SFU/GlobalMemory/RoCC 218/218、
  multicore Softmax 12/12。另一次包含报告绘图的宽泛 SFU 运行只有 4 项因环境缺少
  `matplotlib` 未执行；本阶段未安装绘图依赖，也未修改汇报素材。
- Softmax `16x64` smoke：checked=1,024、mismatch=0、max abs diff
  `3.29887216e-09`、contract pass、accelerator completion=394 cycles。
- canonical WCP `64x64x64` FP32 GEMM：`VERIFY-C PASS`、sampled=64、mismatch=0、
  max abs diff=0、system latency=18,397 cycles；Attention activity 全为 0。
- `libgolem` build/install SHA-256：
  `26a97dab361632df9f99a8bbad52eeed91f0e8d8a2786d8993cd86181145bdcd`；
  `libmemHierarchy` build/install SHA-256：
  `b0edd9f3bc6b46c0f59e7ddbb58701290d9a5614072c9edf68b7633e3c28d43d`。

调试边界：

- 首次 GEMM 回归误把 `mesh-dim-x` 设为 5，与 4-group/4-worker-slot 合同不符，
  因此在 setup 阶段缺少 `req_in_3`；恢复 canonical `mesh-dim-x=4` 后通过，未发生
  GEMM 算术失败。
- C1 只有一个 KV tile，使用稳定三阶段 Softmax；它尚未证明 FlashAttention 的
  多 tile online recurrence、causal mask、partial tile 或 extreme logits。

Phase C1 已关闭。下一步为 Phase D 的 `S64,D64,Br16,Bc32` 两 KV tile online
`(m,l,O)` recurrence，然后按 causal、partial tile、extreme logits 顺序扩展。

## 2026-07-23：完成 Phase D1 双 KV tile online recurrence

已完成：

- `B1,H1,S64,D64,Br16,Bc32` non-causal 使用四个 query block、两个 KV tile。
  K/V 每项只从 HBM 搬入一次，S/P 始终位于 worker per-Core GlobalMemory。
- SFU 固定容量 context 保存 running `(m,l)`；第二 tile 返回旧 Oacc 和当前 tile
  的缩放系数。RoCC 从 Local GM 恢复旧 Oacc，经 Array `writeOutputAsync()`
  写入输出寄存器，再用 accumulate 模式完成 online O recurrence。
- QK 每次计算前强制恢复 Array overwrite mode，防止上一 query block 的 PV
  accumulate 状态污染 score。SFU 物理 row context 重新分配逻辑行时同步重绑
  online row identity，使默认 4-context 的 C1 仍能分批处理 16 行。

验证结果：

- D1 checked=4,096、mismatch=0、max abs error=`7.568611652339352e-09`；
  QK/PV ops=256/512，SFU jobs/rows=8/128，S/P HBM bytes=0，RSQRT count=1。
- C1 重新验证 checked=2,048、mismatch=0、max abs error=
  `1.3586599793141696e-08`。
- Attention focused tests 43/43，SFU/GlobalMemory focused tests 68/68。
- Softmax `16x64` smoke checked=1,024、mismatch=0、completion=394 cycles。
- canonical WCP `64x64x64` GEMM `VERIFY-C PASS`，system latency=18,397 cycles。
- `libgolem` build/install SHA-256：
  `5cfa9f165bef7c79985b488ff3bbde8a76fb9b8f3289093a238b342254968f8d`。

Phase D1 已关闭。下一步是 Phase D2 causal：完全位于未来的 KV tile 应跳过，
对角 tile 在 SFU scale 后、row-max 前逐元素 mask；完成后再进入 partial tile 和
extreme logits。

## 2026-07-23：完成 Phase D2 causal 与 future-tile skipping

已完成：

- causal flag 由 guest descriptor 经 manager/worker RoCC 传递至 SFU；manager 和
  worker 都拒绝 causal bit 以外的未知 flag。
- worker 根据当前 query block 的全局末行计算有效 KV tile 数。前两个 query block
  跳过 key tile 1，后两个 query block 遍历两个 key tile；因此两个完全 future tile
  不执行 QK、SFU 或 PV。对角 tile 仍由 SFU 在 scale 后、row-max 前逐元素 mask。
- runner 增加 `--causal 0|1`，按模式选择 D1/D2 guest、输出 oracle 和精确统计门禁；
  guest Makefile 增加独立 `fused_attention_d2_causal` 目标。

验证结果：

- D2 causal checked=4,096、mismatch=0、max abs error=
  `2.3585739111764426e-08`；QK/PV ops=192/384、SFU jobs/rows=6/96、
  scaled/masked elements=3,072/992、S/P HBM bytes=0、RSQRT count=1。
- D1 non-causal 重新通过 checked=4,096、mismatch=0、max abs error=
  `7.568611652339352e-09`；C1 重新通过 checked=2,048、mismatch=0。
- Attention discovery 46/46，focused RoCC/SFU/GlobalMemory 38/38。
- Softmax `16x64` smoke checked=1,024、mismatch=0、completion=394 cycles。
- canonical WCP `64x64x64` GEMM `VERIFY-C PASS`，sampled=64、mismatch=0、
  max abs diff=0。
- `libgolem` build/install SHA-256 均为
  `c780d93bf8498872828513ceb4831b6f0ec180279571a9bb13174c40ef634d33`。

Phase D 尚未关闭。下一步是 Phase D3 partial shape：
`Sq=20,Skv=70,D64`；随后执行 extreme-logit SST 验收。

## 2026-07-23：完成 Phase D3 partial query/key tile

已完成：

- fused worker 的 query block、key tile 和 16-key panel 使用 ceil division，并在
  QK、SFU、PV、output DMA 各阶段传播真实尾块长度。
- `Sq=20` 的尾部 query block 只搬运和写回 4 行；`Skv=70` 的尾部 key tile 只
  搬运 6 行 K/V。Array 固定端口所需 padding 在 operand buffer 中补零，SFU 只处理
  `rows=4`、`cols=6`，padding 不参与 max、sum 或输出。
- QK 仅启动每个 panel 的有效 key Array，PV 仅启动有效 query Array；runner 新增
  `--partial` 和独立 `fused_attention_d3_partial` guest，并以精确统计作为门禁。

验证结果：

- D3 checked=1,280、mismatch=0、max abs error=`7.499891131745873e-09`；
  QK/PV ops=140/240、SFU jobs/rows=6/60、scaled elements=1,400、S/P HBM
  bytes=0、RSQRT count=1。
- D1 non-causal、D2 causal 和 C1 真实 SST 全部重新通过，数值与既有结果一致。
- Attention discovery 49/49、SFU focused 22/22、Local GM/RoCC/Array 13/13、
  multicore Softmax 12/12。
- Softmax `16x64` checked=1,024、mismatch=0、completion=394 cycles；canonical
  WCP `64x64x64` GEMM `VERIFY-C PASS`、sampled=64、mismatch=0、max abs diff=0、
  system latency=18,397 cycles。
- `libgolem` build/install SHA-256 均为
  `904a5dd1a205a93c7ace72fddc7f13c79d60a618657ba1131369f062bb43b4d9`。

Phase D 尚未关闭。下一步是 Phase D4 extreme-logit fused SST；通过后才能关闭
Phase D 并进入 Phase E 的多 worker prefill mapping。

## 2026-07-23：完成 Phase D4 extreme-logit fused SST

已完成：

- 增加确定性 extreme-logit profile。Q 只在第 0 维取 1；前 32 个 K 的第 0 维
  取 -800，后 32 个取 +800，配合 `1/sqrt(64)=1/8` 后两个 KV tile 的 score
  分别恰为 -100 和 +100，强制 running max 跃迁 200。
- generator 同时检查 Q/K/V 和缩放 score 有限；Attention oracle 检查最终输出
  有限。runner 增加互斥的 `--extreme-logits` 模式，复用 D1 fused guest 与 D1
  精确活动统计，不引入测试专用硬件路径。

验证结果：

- D4 checked=4,096、mismatch=0、max abs error=0；QK/PV ops=256/512、SFU
  jobs/rows=8/128、S/P HBM bytes=0、RSQRT count=1。
- 默认 D1 同轮真实 SST 回归 checked=4,096、mismatch=0、max abs error=
  `7.568611652339352e-09`，精确活动统计通过。
- Attention discovery 51/51 通过。
- 本增量未修改 SFU、GlobalMemory、RoCC 或其他共享组件，故未重编 `libgolem`，
  也未重复执行 GEMM/Softmax 回归；上一阶段已验收的安装库保持不变。

Phase D 已关闭。下一步是 Phase E：4 manager + 16 worker prefill mapping。

## 2026-07-23：完成 Phase E1 S256 多 manager/worker fused SST

已完成：

- descriptor 在保持 128 bytes 的前提下增加 global query-row、每 node KV row 数和
  node stride；manager 从显式 topology map 取得 physical worker Core ID，不再把
  worker slot 当作 Core ID。
- 4 个 manager（core 0-3）各自协调 4 个 worker（core 4-19），每个 worker 负责
  一个 16-row query block。manager 使用 expected slot/core bitmap 拒绝重复或错误
  completion，并等待本组 4 个 worker 完成。
- Q/O 和 K/V 都以 64-row band 分布在 HBM node 1-4。K/V 在 worker 端按 32-row
  tile 流入 Local GM，完整 `S256` K/V 不驻留在单个 64 KiB window 中。
- 新增 scale guest、HBM striped initializer、runner、完整输出 verifier 和精确 stats
  verifier。16-worker 压力下 runner 使用 DMA read retry ticks/count=`4096/32`。

验证结果：

- `B1,H1,S256,D64,Br16,Bc32` fused SST：checked=16,384、mismatch=0、max abs
  error=`4.4967521123807225e-09`，输出来自 HBM node 1-4，S/P HBM bytes=0。
- 每个 manager issue/complete=1/1、manager SFU jobs=0；每个 worker QK/PV=
  256/512、SFU jobs/rows=8/128、scaled elements=4,096、RSQRT count=1。
- D1 回归 checked=4,096、mismatch=0、max abs error=`7.568611652339352e-09`。
- canonical WCP `64x64x64` FP32 GEMM 回归 `VERIFY-C PASS`、sampled=64、
  mismatch=0、max abs diff=0。
- Attention discovery 57/57、SFU focused 22/22、multicore Softmax 12/12
  通过；`libgolem` build/install SHA-256 均为
  `cc884ddba01f6426bf6651f973e7704ce5b415cddc4af881c575480beb919319`。

Phase E 尚未关闭。当前是 4 个 manager-level band completion；下一步 Phase E2
实现跨 manager/root aggregation，在全部 16 个 worker 的 output DMA ACK 后产生
一次 tensor-level completion，然后才进入 `S1024,D128`。

## 2026-07-24：完成 Phase E2 单一 tensor-level completion

已完成：

- descriptor 在保持 128 bytes 的前提下增加 root manager、manager slot 和 manager
  count；scale guest 的四个 manager 共享 tensor job/tag。
- 增加独立 `AttentionManagerComplete` transport。每个 manager 先完成本组 4 个
  worker 的唯一 completion 汇聚，manager 1-3 再通过 NoC 报告给 root core 0；root
  通过 manager bitmap 收齐 slot 0-3 后才结束 wait。
- stats verifier 新增强制门禁：四个 manager local band completion 各 1；root 收到
  4 个 manager completion；全系统只能有一个 tensor completion，且必须位于 core 0。
- scale runner/guest 统一使用 `fused_attention_e2_s256_d64` run-id 和
  `fused_attention_scale` 名称，避免 E1/E2 artifact 混淆。

验证结果：

- 最终 E2 artifact：`/data4/jjgong/tmp/fused_attention_e2_s256_d64_final`。
  checked=16,384、mismatch=0、max abs error=`4.4967521123807225e-09`，S/P HBM
  bytes=0；manager band=`1/1/1/1`、root received=4、tensor completion=`1/0/0/0`。
- D1 single-manager 回归 checked=4,096、mismatch=0、max abs error=
  `7.568611652339352e-09`；canonical GEMM `VERIFY-C PASS`、sampled=64、
  mismatch=0、max abs diff=0。
- Attention discovery 60/60、SFU focused 22/22、multicore Softmax 12/12 通过。
- `libgolem` build/install SHA-256：
  `6df59c6be54a163915b86eb89049b388638745d154ef70f19d625431e226fa9a`。

Phase E control-plane 汇聚已完成；整个 Phase E 仍需完成规模阶梯。下一步 Phase E3
是 `B1,H1,S1024,D128` fused SST。

## 2026-07-24：完成 Phase E3 S1024/D128 fused SST

已完成：

- RoCC Attention worker 改为动态 `head_dim`，并允许一个 dispatch 顺序处理多个
  16-row query block。E3 每个 worker 处理 64 个 query row，PV 使用 8 个 D128
  dimension panel，K/V 仍按 32-row tile 流式进入 per-Core GlobalMemory。
- E3 的 Q/K/V/S/P/O Local GM window 加安全区为 51,328 bytes，小于既有 64 KiB
  上限；S/P 只存在于本地窗口，不分配或回写 HBM。
- scale guest、HBM initializer、输出 verifier 和 stats verifier 均按 E2/E3 profile
  参数化；runner 默认运行 E3，E2 使用 `--scale-point e2`。

验证结果：

- E3 artifact：`/data4/jjgong/tmp/fused_attention_e3_s1024_d128_final`。
  checked=131,072、mismatch=0、max abs error=
  `1.5966506535956479e-09`，S/P HBM bytes=0。
- 四个 manager band=`1/1/1/1`、root received=4、tensor completion=
  `1/0/0/0`。每个 worker QK/PV ops=`4,096/16,384`、SFU jobs/rows=
  `128/2,048`、scaled elements=`65,536`、RSQRT count=1。
- E2 与 D1 真实 SST 回归均通过。Attention discovery 64/64、SFU focused 43/43、
  GlobalMemory async 7/7、multicore Softmax 12/12 通过。canonical WCP
  `64x64x64` FP32 GEMM `VERIFY-C PASS`、sampled=64、mismatch=0、max abs diff=0。
- `libgolem` build/install SHA-256：
  `7eb6a002dac3881d91fbb6040cbb595d3ea47116e847b3bf08cdf91f23c055b0`。

Phase E3 已关闭。下一步 Phase E4 是 `B1,H1,S2048,D128`；先完成容量、地址、
计数契约和 dry-run，再启动真实 SST。

## 2026-07-24：完成 Phase E4 S2048/D128 fused SST

已完成：

- RoCC worker/manager descriptor 增加 E4 shape，继续复用 E3 的动态 D128 路径、
  32-row K/V streaming 和 51,328-byte bounded Local GM window。
- 新增 `scale-e4` guest、runner profile 和精确 stats profile。每个 manager 管理
  512 个 query row，每个 worker 顺序处理 8 个 query block 和 64 个 KV tile。
- scale runner 默认点切换为 E4，默认 watchdog 为 7,200 秒；E2/E3 仍可通过
  `--scale-point` 显式运行。

验证结果：

- E4 artifact：`/data4/jjgong/tmp/fused_attention_e4_s2048_d128_final`。
  checked=262,144、mismatch=0、max abs error=
  `1.181374809935097e-09`，S/P HBM bytes=0。
- manager band=`1/1/1/1`、root received=4、tensor completion=`1/0/0/0`。
  每个 worker QK/PV ops=`16,384/65,536`、SFU jobs/rows=`512/8,192`、
  scaled elements=`262,144`、RSQRT count=1。
- E3 与 D1 真实 SST 回归通过。Attention discovery 67/67、SFU focused 43/43、
  GlobalMemory async 7/7、multicore Softmax 12/12 通过。canonical WCP
  `64x64x64` FP32 GEMM `VERIFY-C PASS`、sampled=64、mismatch=0、max abs diff=0。
- `libgolem` build/install SHA-256：
  `d37308086cf25cb26e3861de5d82ad8590e682ae9d5e30a5cac189a89fa05125`。

Phase E4 已关闭。下一步 Phase E5 是 `B1,H1,S4096,D128`；先完成契约、dry-run
和运行成本门禁，再启动预计约为 E4 四倍工作量的真实 SST。
