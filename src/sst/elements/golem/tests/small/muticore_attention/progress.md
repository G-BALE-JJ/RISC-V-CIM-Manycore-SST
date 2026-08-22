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

## 2026-08-19：完成 K/V 并行 DMA 优化（E2/E3 回归）

本次优化针对已测得的 `inter_tile_pv_to_next_qk` 瓶颈。原实现中，每个
Attention worker 在流式 KV 模式下先发起 K tile DMA，等待 K 完成后才发起 V tile
DMA；两块数据写入独立的 `kLocal`/`vLocal` 缓冲区，GlobalMemory 本身支持多个
inflight DMA，因此两路传输可以并发发起。

实现方式：

- `AttentionWorkerState` 新增 `attentionKvLoadsPending` 完成计数器。
- `loadAttentionKeyTile()` 同时提交 K/V 两个 DMA 请求，不改变地址、大小、burst
  或内存控制器参数。
- 新增 `completeAttentionKvLoad()`：任一路失败立即结束 worker；两路 callback
  都返回成功后，计数器归零，才记录 `KvLoad` 阶段并进入 QK。
- 新增契约测试，确保源码包含并行发起和 join 语义；完整 Attention 契约测试
  `77/77` 通过。

周期与正确性对比（PV matrix broadcast 配置）：

| 负载 | 优化前 accelerator cycles | 优化后 accelerator cycles | 降低 |
|---|---:|---:|---:|
| E2 `S=256,D=64` | 75,181 | 70,757 | 4,424（5.89%） |
| E3 `S=1024,D=128` | 2,241,546 | 2,179,516 | 62,030（2.77%） |

- E2 checked=16,384，mismatch=0；E3 checked=131,072，mismatch=0。
- E2 慢核 KV 阶段由 24,007 降至 19,916 cycles。
- E3 慢核 KV 阶段由 801,313 降至 739,364 cycles，降低 7.73%。
- 两次运行均无 DMA timeout/retry，inter-tile breakdown 守恒检查通过。

结论：并行 DMA 已生效，但总周期收益低于 KV 阶段收益，说明共享内存路径和
QK 编程/计算阶段仍是主要限制。E3 优化后慢核 inter-tile breakdown 为：KV
739,364 cycles、QK matrix programming 262,128、QK input programming 73,152、
QK compute/readout 171,196。当前不继续扩大到 E4/E5，下一步先基于这些数据评估
QK 数据流和跨 tile 调度是否值得优化。

本次 artifact：

- `/data4/jjgong/tmp/fused_attention_e2_kv_parallel_20260819`
- `/data4/jjgong/tmp/fused_attention_e3_kv_parallel_20260819`

## 2026-08-19：QK 数据流下一步评估

对 E3 优化后统计和 QK 控制流逐项核对，得到以下结论：

1. `qk_matrix_program` 每个 KV tile 固定为 2,064 cycles。D128 的一份 Q matrix
   为 `16x128x4=8,192 bytes`，Array buffer 带宽为 64 bytes/cycle，加 1 cycle
   基础延迟后单次传输为 129 cycles。当前实现把完全相同的 Q matrix 串行写入
   16 个 array，因此恰好是 `129x16=2,064 cycles`。该项在所有 worker 上恒定，
   不是 NoC/HBM 拥塞造成的波动。
2. ComputeArray 已有 `programMatrixGroupAsync()`，并已由 PV matrix broadcast
   路径实际使用。把它复用于 QK 不需要新增硬件模型，但明确依赖已有的 array
   matrix broadcast 能力；它属于“利用现有硬件机制消除重复搬运”，不是纯软件
   调度优化。
3. E3 每个 worker 有 128 个 QK tile。将 16 次串行 Q matrix programming 改为
   一次 group programming，理论减少
   `(2,064-129)x128=247,680 cycles`。若其他关键路径不变，accelerator cycles
   可由 2,179,516 降至约 1,931,836，即约 11.36% 的上界收益。E2 对应上界为
   7,800 cycles，约当前总周期的 11.02%。
4. 更彻底的数据流方案是把 QK 从“一个 array 对应一个 key”转置为“一个 array
   对应一个 query”：每个 16-key panel 将 K 组织成 matrix 并广播，Q 作为各
   array 的 input。这样每个 array 输出同一 query 的 16 个连续 score，可将当前
   逐 scalar、跨 stride 的 score 写回改成连续 64-byte 写回。按现有端口参数估算，
   它在完成 QK broadcast 后仍可能额外减少约 1,003 cycles/KV tile（E3 约 128k
   cycles），但会同时改变矩阵布局、输入映射、panel 循环和 SFU score 写入，验证
   面显著更大。

决策建议：先实现显式、可开关的 QK matrix broadcast，并只重测 E2/E3。只有实测
收益接近预测且 `qk_compute_readout` 仍是主要固定瓶颈时，再实施 QK 数据流转置。
不先运行 E4/E5，也不直接引入新的 array 端口或 scatter-write 硬件。

## 2026-08-19：完成 QK matrix broadcast 优化

已按上述决策实现并验证：

- 新增默认关闭的 `attention_qk_matrix_broadcast` 参数和 runner 选项
  `--qk-matrix-broadcast`；关闭时保留原逐 array programming 路径。
- 启用时，每个 KV tile 使用既有 `programMatrixGroupAsync()` 将同一 Q matrix
  一次写入当前 active arrays。两个 16-key panel 复用该 matrix，不重复广播。
- 新增 `attention_qk_matrix_broadcasts` 统计；verifier 强制每个 worker 的广播次数
  等于 KV tile jobs（E2=8，E3=128）。完整 Attention 契约测试 `78/78` 通过。

实测结果（同时启用 PV matrix broadcast 和 K/V 并行 DMA）：

| 负载 | QK broadcast 前 | QK broadcast 后 | 本次降低 |
|---|---:|---:|---:|
| E2 `S=256,D=64` | 70,757 | 65,017 | 5,740（8.11%） |
| E3 `S=1024,D=128` | 2,179,516 | 1,936,507 | 243,009（11.15%） |

- E2 checked=16,384、mismatch=0；E3 checked=131,072、mismatch=0。
- E2 慢核 inter-tile QK matrix programming 由 7,280 降至 455 cycles。
- E3 慢核该阶段由 262,128 降至 16,383 cycles，即 127 transitions × 129
  cycles；本次总周期收益达到预测上界的约 98.1%。
- 相对 QK/KV 两项优化前的 E3 `2,241,546 cycles`，当前累计减少 305,039
  cycles（13.61%）。
- 两次运行的活动计数、数值结果和 inter-tile 周期守恒均通过；没有继续运行
  E4/E5。

优化后 E3 慢核 inter-tile 主要阶段为：KV 744,036 cycles、QK compute/readout
171,196、QK input programming 73,152、QK matrix programming 16,383。下一步若
继续降低原理周期，应评估前述 QK 数据流转置能否消除 scalar score 写回，而不应
继续优化已经降到 0.85% 总周期的 QK matrix programming。

本次 artifact：

- `/data4/jjgong/tmp/fused_attention_e2_qk_broadcast_20260819`
- `/data4/jjgong/tmp/fused_attention_e3_qk_broadcast_20260819`

## 2026-08-19：完成 QK 转置数据流优化

在 QK matrix broadcast 基础上实现了默认关闭的
`attention_qk_dataflow_transpose`（runner：`--qk-dataflow-transpose`）：

- 原数据流由一个 array 对应一个 key，Q 为公共 matrix、K 为各 array input；输出
  需要将每个 array 的 query score 逐 scalar、跨 stride 写回。
- 转置后一个 array 对应一个 query。每个 16-key panel 将 K 组织为 matrix 并广播，
  Q 在每个 KV tile 开始时编程为各 query array 的 input，并跨两个 panel 保留。
- 每个 array 输出同一 query 的 16 个连续 score，直接以一次 64-byte Local GM
  write 写入 score row。未增加 Array API、Local GM 端口或 scatter-write 硬件。
- transpose 自动启用既有 QK matrix broadcast；关闭时完整保留原数据流。

实测结果（同时启用 PV broadcast 和 K/V 并行 DMA）：

| 负载 | 转置前 cycles | 转置后 cycles | 本次降低 |
|---|---:|---:|---:|
| E2 `S=256,D=64` | 65,017 | 59,057 | 5,960（9.17%） |
| E3 `S=1024,D=128` | 1,936,507 | 1,782,364 | 154,143（7.96%） |

- E2 checked=16,384、mismatch=0；E3 checked=131,072、mismatch=0。
- E3 慢核 QK compute/readout 由 171,196 降至 82,169 cycles（52.0%）；QK
  input programming 由 73,152 降至 18,288（75.0%）。
- panel 级 K matrix broadcast 使 QK matrix programming 由 16,383 增至 32,893
  cycles，但远小于 input/readout 的节省。
- E3 每 worker 的 QK array ops 仍为 4,096；matrix broadcasts 由每 KV tile 一次
  变为每 panel 一次，即 256 次。活动计数和 inter-tile 周期守恒通过。
- 相对 K/V 与 QK 优化前的 E3 `2,241,546 cycles`，当前累计减少 459,182
  cycles（20.48%）。

优化后 E3 慢核 inter-tile 仍由 KV 718,173 cycles 主导；QK compute/readout 已降至
82,169，matrix/input 分别为 32,893/18,288。下一步不应继续扩大 E4/E5，而应先
分析 KV 并行收益受限和 manager completion skew 增大的原因，确认是否存在内存
节点负载不均或 worker 映射问题。

本次 artifact：

- `/data4/jjgong/tmp/fused_attention_e2_qk_transpose_20260819`
- `/data4/jjgong/tmp/fused_attention_e3_qk_transpose_20260819`

## 2026-08-19：完成 KV tile rotation 访存调度优化

对 QK 转置后的 E3 慢核继续分析，确认剩余 KV 瓶颈不是 DRAM 或 NoC 吞吐不足：

- 16 个 worker 的 DMA read/write 次数和字节数完全相同；没有 queued send、retry
  或 timeout。内存 queue delay p99=1 cycle，NoC output stall=0。
- 4 个数据 HBM 位于 top row 的 `rtr_0..3`，20 个 CPU 位于下方 5 行。原状态机
  的所有 worker 都以相同顺序遍历 `node1 -> node2 -> node3 -> node4`，使瞬时 K/V
  请求集中到同一 HBM 节点和路径；最终字节均匀不代表运行时流量均匀。
- E3 原路径的 manager local-complete skew 为 245,413 cycles；逐核 KV 时间和
  DMA RTT 呈强烈的列方向梯度。把 HBM 从 top 移到 bottom 的 E2 对照使最慢 worker
  从右侧列转为 `core16`，且 cycles 从 59,057 增至 68,711，证明关键因素是拓扑
  路径和同步访问顺序，而非计算量不均。

实现方式：

- 新增默认关闭的 `attention_kv_tile_rotation` 参数和 runner 选项
  `--kv-tile-rotation`。
- worker 同时维护 `keyTileOrdinal`（online Softmax/PV 处理顺序）和物理
  `keyTile`（K/V 地址、keyBegin 和 causal mask）。每个 manager/worker 组合按
  `(ownerCore + workerSlot)` 轮转首个 HBM band，之后循环遍历全部物理 tile。
- online recurrence 的首次初始化、最终归一化和结束条件继续使用 ordinal；物理
  key 范围不变。没有新增 Array、DMA、NoC、Local GM 或 HBM 硬件机制，只改变
  已有 K/V tile 的访问次序。
- 开关默认关闭，旧路径保留，可做严格 A/B。

严格 A/B 结果（均启用 PV matrix broadcast、QK transpose 和 K/V 并行 DMA）：

| 负载 | rotation 前 cycles | rotation 后 cycles | 本次降低 |
|---|---:|---:|---:|
| E2 `S=256,D=64` | 59,057 | 32,352 | 26,705（45.22%） |
| E3 `S=1024,D=128` | 1,782,364 | 1,195,277 | 587,087（32.94%） |

- E2 checked=16,384、mismatch=0；E3 checked=131,072、mismatch=0。
- E2 慢核 inter-tile KV 从 24,056 降至 4,325 cycles，manager complete skew
  从 17,837 降至 2,588 cycles。
- E3 慢核 inter-tile KV 从 718,173 降至 141,538 cycles，manager complete skew
  从 245,413 降至 10,067 cycles。
- E3 相对本轮 Attention 优化前的 2,241,546 cycles，累计减少 1,046,269
  cycles（46.68%）。

对比纪律：第一次 rotation E2 试跑未开启 PV matrix broadcast，得到 63,552
cycles，不能与开启 PV broadcast 的 59,057-cycle 历史 artifact 比较。补齐相同
开关后才得到上述 32,352-cycle 严格 A/B。后续所有 cycle 表必须同时记录
PV broadcast、QK broadcast/transpose 和 KV rotation 开关。

本次 artifact：

- 拓扑对照：
  `/data4/jjgong/tmp/fused_attention_e2_qk_transpose_bottom_hbm_20260819`
- E2 严格 A/B 优化结果：
  `/data4/jjgong/tmp/fused_attention_e2_pv_qk_transpose_kv_rotation_20260819`
- E3 严格 A/B 优化结果：
  `/data4/jjgong/tmp/fused_attention_e3_pv_qk_transpose_kv_rotation_20260819`

下一步不运行 E4/E5。先以 E3 的 1,195,277 cycles 为新基线，重新拆分慢核的
`all_softmax_to_pv=857,088`、`inter_tile=298,333`（其中 KV=141,538、QK
compute/readout=82,169）以及 query-block output/query reload，判断下一项应优化
PV 数据流还是跨 query-block 调度。

## 2026-08-19：完成 PV V-tile panel 复用数据流优化

瓶颈归因确认 `beginAttentionPvPanel()` 的旧路径在每个 16-column output panel
开始前，都会从 Local GM 重新读取完整的 V tile；但相邻 panel 只是在同一 V tile
中选择不同的 16 个 head-dimension columns。E2 的一个 V tile 被重复读取 4 次，
E3 被重复读取 8 次。

实现方式：

- worker state 新增 `vPayload`，在每个 KV tile 的第 0 个 PV panel 读取一次完整
  V tile；后续 panel 直接从该 payload 选择对应的 16 个维度并编程 Array matrix。
- 新增默认关闭的 `attention_pv_v_tile_reuse` 参数、
  `GOLEM_ATTENTION_PV_V_TILE_REUSE` 环境变量和 runner 选项
  `--pv-v-tile-reuse`，保留旧路径用于严格 A/B。
- 没有增加 V Local GM 容量、带宽、端口或 Array API；这是消除重复读取的数据流
  优化。缓存是 RoCC worker 已经读取到的 tile payload，不改变 HBM/KV DMA、QK、
  online Softmax 或 PV 累加顺序。

每 worker 的 PV V Local-GM 读取量：

- E2：`8 tiles * 4 panels * 8 KiB = 256 KiB` 降为 `64 KiB`，减少 75%。
- E3：`128 tiles * 8 panels * 16 KiB = 16 MiB` 降为 `2 MiB`，减少 87.5%。

严格 A/B 结果（均启用 PV matrix broadcast、QK transpose、K/V 并行 DMA 和
KV tile rotation）：

| 负载 | V tile reuse 前 cycles | V tile reuse 后 cycles | 本次降低 |
|---|---:|---:|---:|
| E2 `S=256,D=64` | 32,352 | 29,198 | 3,154（9.75%） |
| E3 `S=1024,D=128` | 1,195,277 | 965,933 | 229,344（19.19%） |

- E2 checked=16,384、mismatch=0；E3 checked=131,072、mismatch=0。
- E3 慢核 `all_softmax_to_pv` 从 857,088 降至 624,128 cycles，减少
  232,960 cycles；与总周期收益 229,344 接近，验证收益来自目标 PV 路径。
- E3 `inter_tile_pv_to_next_qk` 为 301,596 cycles，对比原 298,333 基本不变；
  其中 KV=141,982、QK matrix/input/compute-readout=32,893/18,288/82,169，
  说明优化没有把瓶颈转移或隐藏到下一 tile 准备阶段。
- manager local-complete skew 为 9,444 cycles，对比原 10,067 基本稳定。
- 相对本轮 Attention 优化前的 E3 2,241,546 cycles，当前累计减少
  1,275,613 cycles（56.91%）。

本次 artifact：

- `/data4/jjgong/tmp/fused_attention_e2_v_tile_reuse_explicit_20260819`
- `/data4/jjgong/tmp/fused_attention_e3_v_tile_reuse_explicit_20260819`

下一步仍不运行 E4/E5。E3 当前最大可见阶段是
`all_softmax_to_pv=624,128 cycles`，其次是 inter-tile 301,596 cycles。先增加 PV
内部的分项统计，区分 matrix programming、P input、output restore、compute 和
output read/write；只有确认最大分项后，才决定是否优化 P 输入搬运或跨 KV tile
的 PV output restore。

## 2026-08-20：整合跨阶段 MVM/Softmax overlap 优化计划

当前目标从“继续优化单阶段搬运”扩展为“让不同 KV tile 的阶段时间重叠”，但必须
区分真实 overlap 和阶段统计：同一个 worker 当前仍严格执行

```text
K/V load -> QK MVM -> score read/write -> Softmax -> PV MVM -> output restore
         -> next KV tile
```

QK MVM 与 Softmax 不能直接对同一 tile overlap，因为 Softmax 依赖完整的
`S[16,32]`；Softmax 与 PV MVM 也不能直接 overlap，因为 PV 依赖完整的 `P[16,32]`。
当前已经存在的并行只有 worker 间并行、16 个 Array 间并行和 K/V 双 DMA 并行。

因此新的优化路线按风险排序：

1. **F0 观测和上界模型**：增加 KV、QK、SFU、PV、Oacc 的 issue/complete tick，
   记录 tile ordinal 和 buffer id，先确认当前关键路径与可隐藏的 KV load。E3
   当前 `inter_tile_pv_to_next_qk=301,596`，其中 KV load=141,982；这 141,982
   是双缓冲预取的主要理论收益上界，不能把 QK 和 Softmax 的依赖周期直接相减。
2. **F1 K/V 双缓冲预取**：两个固定 Local GM KV buffer。当前 buffer 执行 QK、
   Softmax、PV 时，下一物理 KV tile DMA 到另一个 buffer；tile 完成后交换 buffer。
   `keyTile` 继续负责物理地址，`keyTileOrdinal` 继续负责 online softmax 顺序。
   必须处理 buffer busy、DMA 失败和 causal skip，并保留默认关闭的回退路径。
3. **F2 QK panel/Softmax overlap（条件实施）**：只有 SFU 支持按 score panel
   增量更新 `(m,l)` 才进行。当前 `issueAttentionTile()` 是整 tile 接口，不能直接
   宣称已经支持该流水。实现前必须解决 running-max 跃迁和 partial/causal 语义。
4. **F3 Softmax/PV overlap（低优先级）**：只有 SFU 能输出可消费的 P panel、PV
   能进行 panel 累加，并有双 P buffer 和 ready/credit 协议时才实施。online max/l
   的重新缩放可能改变整个 tile 的比例，因此该阶段可能需要 SFU ABI 和 PV 累加
   协议修改，风险高于 F1。
5. **F4 规模门禁**：每一步都先用 E2（180 秒 watchdog）和 E3（600 秒 watchdog）
   做严格 A/B；只有 cycle、墙钟时间、mismatch、buffer stall、manager skew 和
   阶段时间守恒都通过，才重新评估 E4/E5。超时立即停止并分析，不继续等待。

权威恢复入口和每个阶段的接口、验收条件已同步到 `task_plan.md` 的 Phase F0-F4。

## 2026-08-20：完成 Phase F0 tile-pipeline 细粒度观测

实现了不改变执行顺序的 observation-only 计时。每个 Attention tile 从 K/V load
开始，到最后一个 PV output 写回 Local GM 结束，使用同一连续时间轴记录 11 个
互斥阶段。stats verifier 同时检查每阶段 Count 与 tile jobs 相等，并要求
`tile_total_ticks == sum(tile_phase_ticks)`。

严格基线结果（全部启用 PV broadcast、QK transpose、KV rotation 和 V reuse）：

| 负载 | cycles | tile samples/worker | checked | mismatch | 时间守恒 |
|---|---:|---:|---:|---:|---|
| E2 `S256,D64` | 29,198 | 8 | 16,384 | 0 | unattributed=0 |
| E3 `S1024,D128` | 965,933 | 128 | 131,072 | 0 | unattributed=0 |

E3 慢 worker（core19）的 tile path：

| 阶段 | cycles | tile path 占比 |
|---|---:|---:|
| KV load | 145,408 | 15.27% |
| Q local read | 16,640 | 1.75% |
| QK matrix program | 66,304 | 6.97% |
| QK input program | 18,432 | 1.94% |
| QK compute/readout | 49,664 | 5.22% |
| Softmax | 31,367 | 3.30% |
| PV matrix program | 165,376 | 17.37% |
| PV input program | 196,608 | 20.65% |
| PV output restore | 63,488 | 6.67% |
| PV compute | 133,120 | 13.98% |
| PV output read/write | 65,536 | 6.88% |
| 合计 | 951,943 | 100% |

结论：期望的 MVM/Softmax overlap 在方向上成立，但当前 Softmax 只占 tile path 的
3.30%，完全隐藏也不是最大收益项。F1 仍先做 K/V 双缓冲，因为可隐藏 KV load
上界为 145,408 cycles（15.27%）；随后若实现 QK panel/Softmax overlap，需要把
31,367 cycles 作为收益上界。PV 五项合计 624,128 cycles（65.56%），后续仍需与
F2/F3 一起评估，不能只优化 Softmax。

本次 artifact：

- `/data4/jjgong/tmp/fused_attention_e2_phase_f0_stats_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_phase_f0_stats_20260820`

## 2026-08-20：完成 Phase F1 K/V 双缓冲预取

实现了默认关闭的跨 tile DMA/compute overlap。每个 Attention worker 具有两组
固定 K/V Local GM buffer：首 tile 正常加载到 active buffer，QK、Softmax 和 PV
消费它时，下一物理 K/V tile DMA 到另一个 buffer；当前 tile 完成后交换 buffer，
然后继续预取。逻辑 `keyTileOrdinal` 与物理 `keyTile` 分离，保持 online softmax
顺序并兼容 KV tile rotation。

容量与观测：

- D64 双缓冲 window=43,136 bytes；D128 window=84,096 bytes（`0x14880`）。
- runner 开关为 `--kv-double-buffer`，默认仍走 64 KiB 单 buffer 路径。
- 新增 prefetch tiles/hits/waits 统计和 verifier 守恒约束。

严格结果（同时开启 PV broadcast、QK transpose、KV rotation、V reuse）：

| 负载 | F0 cycles | F1 cycles | 降低 | prefetch hit/wait | 正确性 |
|---|---:|---:|---:|---:|---:|
| E2 `S256,D64` | 29,198 | 26,944 | 2,254（7.72%） | 99 / 13 | 16,384 checked，0 mismatch |
| E3 `S1024,D128` | 965,933 | 849,298 | 116,635（12.08%） | 1,984 / 0 | 131,072 checked，0 mismatch |

E3 慢 worker 的 tile KV 阶段从 145,408 降至 15,189 cycles；inter-tile 从
301,596 降至 182,163 cycles，时间守恒仍为 unattributed=0。PV 五阶段仍合计
624,709 cycles，Softmax 为 31,950 cycles；下一步应先做 F2 panel-level online
softmax 接口和数值语义审计，而不是直接运行更大负载。

验证：Attention 83/83、SFU 22/22、multicore Softmax 12/12、guest build 和
canonical `64x64x64` GEMM 均通过。`libgolem` build/install SHA-256 均为
`5cc8c2ce945dc1f72f1ba4654efc4b26e3d21e5dc38be7d4ed6d3ff0799f2019`。

artifact：

- `/data4/jjgong/tmp/fused_attention_e2_phase_f1_kv_double_buffer_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_phase_f1_kv_double_buffer_20260820`
- `/data4/jjgong/tmp/attention_f1_gemm_regression_20260820`

## 2026-08-20：完成 Phase F2 QK/Softmax panel overlap 可行性审计

本阶段没有修改生产代码，也没有运行新的 SST。代码审计确认 QK transpose 路径
每个 32-key tile 产生两个 16-key panel，但 RoCC 要等全部 panel 写完 Local GM
才提交唯一的整 tile `issueAttentionTile()`。SFU 内部虽按 16 lanes 分块，仍执行
完整 MAX -> EXP/SUM -> NORMALIZE 三遍，并且同时只允许一个 tensor worker op。

不能直接把 panel 当作伪 tile：后续 panel 提高 running max 时，先前 panel 的 P
需要重新缩放；现有结果只返回 tile 级 `oldOutputScale`，无法修正已写 P。要保持
数学正确，要么只提前执行 MAX scan，要么同时引入 panel PV、P buffer credit 和
Oacc 重缩放，后者已经是 F2+F3 联合架构改造。

F1 E3 的完整 Softmax 为 31,950/849,298 cycles（3.762%）。按现有 row-stage tick
的服务时间比例，MAX 占 25.425%，MAX-only overlap 估计约 8,123 cycles（总周期
0.956%）；这只是基于已有 artifact 的代理上界。PV input 为 196,608 cycles
（23.15%），优先级显著更高。

决策：F2 以“已评估、不实施”关闭。下一步先设计 PV input 的并行 issue/批量编程
优化；F3 联合 panel ABI 暂缓，E4/E5 继续冻结。

## 2026-08-20：完成 Phase G1 PV input 两级流水

实现了默认关闭的 `attention_pv_input_pipeline`。此前每一行 P 必须等待
`Local GM read` 和 `Array input program` 都完成后才读取下一行；现在 Array 接受
当前行编程请求后立即发起下一行 Local GM read，使两个既有单端口资源形成两级
流水。实现没有增加 batch API 或理想化带宽，也没有改变 P/PV/Oacc 数值路径。

完成条件由 `attentionPvInputsPending` 统一 join：计数在一行读取开始前增加，在该行
Array callback 后减少；只有所有行已发起且 pending 清零才进入 output restore。
`attention_pv_input_pipeline_rows` 对 E2/E3 的每个 worker 分别精确为 512/16,384。

严格结果（PV broadcast、QK transpose、KV rotation、V reuse、KV double buffer
全部保持开启）：

| 负载 | 优化前 cycles | 优化后 cycles | 降低 | PV input 前后 | 正确性 |
|---|---:|---:|---:|---:|---:|
| E2 `S256,D64` | 26,944 | 26,571 | 373（1.38%） | 4,096 -> 2,656 | 16,384 checked，0 mismatch |
| E3 `S1024,D128` | 849,298 | 799,873 | 49,425（5.82%） | 196,608 -> 150,528 | 131,072 checked，0 mismatch |

E3 的 PV input 减少 46,080 cycles（23.44%），是总周期收益的主要来源；慢 worker
tile path=779,473 cycles，11 阶段 unattributed=0。仿真均在 watchdog 内结束，没有
异常长 cycle，也没有运行 E4/E5。

回归结果：Attention 84/84、SFU row-engine 22/22、multicore Softmax 12/12；
canonical `64x64x64` FP32 GEMM sampled=64、mismatch=0、max abs diff=0。
build/install `libgolem.so` SHA-256 均为
`50c34631d1ad20d9021dfbbd2afcdac9344f046841da96a99c9fafae1d05bac1`。

artifact：

- `/data4/jjgong/tmp/fused_attention_e2_pv_input_pipeline_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_pv_input_pipeline_20260820`
- `/data4/jjgong/tmp/attention_pv_input_pipeline_gemm_20260820`

下一步 G2 评估并尝试 Softmax/PV-matrix overlap：QK readout 完成后，在 SFU 计算
Softmax 的同时提前把 V 编程到 PV Array。E3 当前 Softmax=31,926、PV matrix=
165,376 cycles，可隐藏上限约为 31,926 cycles（总周期 3.99%）。实施前先证明 Array
ownership、V buffer 生命周期和失败回滚，不重新开启 panel-level ABI 改造。

## 2026-08-20：完成 Phase G2 Softmax/PV matrix overlap

新增默认关闭的 `attention_pv_matrix_softmax_overlap`。QK readout 完成且 SFU 接受
整 tile Softmax 后，RoCC 立即用 active V buffer 编程首个 PV panel。两个完成位
`attentionSoftmaxComplete`/`attentionPvMatrixComplete` 做 join：矩阵先完成为 hit，
SFU 先完成为 wait；只有二者均完成才读取 P 并进入 PV input。后续 V panels 沿用
原路径，online softmax 数值语义和 SFU ABI 均未修改。

阶段计时保持互斥：并行区首先归入 Softmax，SFU 先结束时才将残余矩阵时间归入
PV matrix，因此 11 阶段仍严格守恒。

| 负载 | 优化前 cycles | 优化后 cycles | 降低 | 正确性 |
|---|---:|---:|---:|---:|
| E2 `S256,D64` | 26,571 | 26,408 | 163（0.61%） | 16,384 checked，0 mismatch |
| E3 `S1024,D128` | 799,873 | 790,516 | 9,357（1.17%） | 131,072 checked，0 mismatch |

E2 全系统 128 tiles 全部 hit。E3 全系统 2,048 tiles 中 411 hit、1,637 wait，hit
率 20.1%，每个 worker 均满足 `tiles=hits+waits=128`。E3 慢 worker 的 PV matrix
从 165,376 降至 116,340 cycles，但 Softmax 归因从 31,926 增至 64,917，说明 V
读取/编程与 SFU 竞争共享 Local GM，实际收益被明显抵消。manager local-complete
skew 从 14,489 增至 16,400 cycles，但总 cycle 仍下降，生命周期和阶段守恒有效。

验证：Attention 85/85、SFU row-engine 22/22、multicore Softmax 12/12；canonical
`64x64x64` FP32 GEMM sampled=64、mismatch=0、max abs diff=0。build/install
`libgolem.so` SHA-256 均为
`3aa9a5b3ee6827c8774d2ceac457af6fd7d4dc3fe0d38648376e1bd6b10ad3c4`。

artifact：

- `/data4/jjgong/tmp/fused_attention_e2_pv_matrix_softmax_overlap_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_pv_matrix_softmax_overlap_20260820`
- `/data4/jjgong/tmp/attention_pv_matrix_softmax_overlap_gemm_20260820`

下一步 G3 评估 PV output restore 两级流水：把下一 array 的 Oacc Local GM read 与
当前 array 的 `writeOutputAsync` 重叠，使用 pending join 后才启动 PV compute。
E3 当前 restore=63,488 cycles；先审计不同 array output ownership，再做默认关闭的
E2/E3 严格 A/B。E4/E5 继续冻结。

## 2026-08-20：完成 Phase G3 PV output restore 两级流水

新增默认关闭的 `attention_pv_restore_pipeline`。非首 key tile 中，当前 Array 的
Oacc 在 Local GM 读取并缩放后提交到 `writeOutputAsync`；请求被有界 Array buffer
queue 接受后立即读取下一 Array 的 Oacc，从而重叠单 Local GM read port 和单
Array-buffer port。没有增加硬件端口或使用同步 shortcut。

`attentionPvRestoresPending` 覆盖每行从 read 发起到 output write callback 的完整
生命周期；只有全部行已提交且 pending 清零才启动 PV compute。首 key tile 仍不做
restore。每个 worker 的流水行统计精确为 E2=448、E3=15,872。

| 负载 | 优化前 cycles | 优化后 cycles | 降低 | restore 前后 | 正确性 |
|---|---:|---:|---:|---:|---:|
| E2 `S256,D64` | 26,408 | 26,168 | 240（0.91%） | 1,792 -> 952 | 16,384 checked，0 mismatch |
| E3 `S1024,D128` | 790,516 | 759,338 | 31,178（3.94%） | 63,488 -> 33,728 | 131,072 checked，0 mismatch |

E3 restore 减少 29,760 cycles（46.88%），与总收益 31,178 cycles 一致；慢 worker
tile path=735,604 cycles，11 阶段 unattributed=0，manager local-complete skew
从 16,400 降至 14,680 cycles。没有异常长 cycle，E4/E5 未运行。

验证：Attention 86/86、SFU row-engine 22/22、multicore Softmax 12/12；canonical
`64x64x64` FP32 GEMM sampled=64、mismatch=0、max abs diff=0。build/install
`libgolem.so` SHA-256 均为
`ee725b2dff73acc7602c67e9b84876825f821bbcd7f1a47f009d182b369c7e92`。

artifact：

- `/data4/jjgong/tmp/fused_attention_e2_pv_restore_pipeline_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_pv_restore_pipeline_20260820`
- `/data4/jjgong/tmp/attention_pv_restore_pipeline_gemm_20260820`

下一步 G4 先审计 PV output read/write 流水。E3 当前该阶段为 70,658 cycles；需要
证明下一 Array 的 readout 能与当前 Local GM write 安全重叠，并建立 write pending
join/backpressure，之后才决定是否实现。若单 Local GM write port 不支持安全排队，
转而分析 PV input 150,528 和 PV matrix 116,357 cycles 的剩余资源瓶颈。

## 2026-08-20：完成 Phase G4 PV output read/write 流水

新增默认关闭的 `attention_pv_output_pipeline`。Array readout 返回一行 16 个 FP32
结果后，直接向真实 GlobalMemory Local write FIFO 提交 64-byte 写；请求被接受即
发起下一 Array readout，从而重叠 Array buffer port 与单 Local GM write port。
`attentionPvOutputWritesPending` join 全部写回，队列拒绝时保留单行 payload/address
并逐 tick 重试，没有扩展通用单槽 `attentionLocalWrite`，也没有增加硬件端口。

独立审查发现 LocalMemory API 允许同步 callback；最终版本在提交前预建 pending/
index，拒绝时回滚，并在返回后校验 job/phase/panel，消除了同步 callback 下的悬空
state 风险。每 worker 精确流水行数为 E2=512、E3=16,384；Local GM queue
rejected=0，worker high-water=17--18（队列深度 32）。

| 负载 | 优化前 cycles | 优化后 cycles | 降低 | output read/write 前后 | 正确性 |
|---|---:|---:|---:|---:|---:|
| E2 `S256,D64` | 26,168 | 25,915 | 253（0.97%） | 2,621 -> 1,173 | 16,384 checked，0 mismatch |
| E3 `S1024,D128` | 759,338 | 729,683 | 29,655（3.91%） | 70,658 -> 39,287 | 131,072 checked，0 mismatch |

E3 局部阶段减少 31,371 cycles（44.40%），慢 worker tile path=705,176 cycles，
11 阶段 unattributed=0，无异常 cycle。manager local-complete skew 从 14,680 增至
19,046 cycles，后续需结合 worker 分配分析。E4/E5 未运行。

验证：Attention 87/87、SFU row-engine 22/22、multicore Softmax 12/12；canonical
`64x64x64` FP32 GEMM sampled=64、mismatch=0、max abs diff=0。

artifact：

- `/data4/jjgong/tmp/fused_attention_e2_pv_output_pipeline_syncsafe_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_pv_output_pipeline_syncsafe_20260820`
- `/data4/jjgong/tmp/attention_pv_output_pipeline_gemm_20260820`

下一步 G5 审计按 Array 提前启动 PV compute，使已完成 input/restore 的 Array 计算
与后续 Array 的 input programming 重叠。当前 E3 最大阶段为 PV input=150,528、
PV compute=133,120、PV matrix=116,291 cycles。若独立 ready/Array-done/accumulate
语义无法安全闭合，则不新增机制，转向共享 Local GM 竞争和 manager skew 分析。

## 2026-08-20：完成 Phase G5 按 Array 提前启动 PV compute

新增默认关闭的 `attention_pv_early_compute`。G5 没有增加计算单元：PV matrix 仍
broadcast 到全部 Array，MVM 仍由原 Array 模型执行。改变的是启动时序：首 key tile
的每个 Array 在自身 P input programming callback 后立即计算；后续 key tile 在
自身 Oacc restore output-write callback 后立即计算。因此前部 Array 的 MVM 可以与
后部 Array 的 input/restore 准备重叠。

为避免争用单 Array-buffer port，所有 input/restore 完成前不做 output readout。
barrier 完成后按 Array index 输出；若当前 Array 尚未 done，reader 等待对应 callback
唤醒。所有 Local GM output writes 完成后才推进 panel/tile。QK done 路径未改变，
每 worker 的提前计算统计精确为 E2=512、E3=16,384。

| 负载 | 优化前 cycles | 优化后 cycles | 降低 | 正确性 |
|---|---:|---:|---:|---:|
| E2 `S256,D64` | 25,915 | 25,705 | 210（0.81%） | 16,384 checked，0 mismatch |
| E3 `S1024,D128` | 729,683 | 699,750 | 29,933（4.10%） | 131,072 checked，0 mismatch |

E3 慢 worker tile path=678,049 cycles；PV input=150,528、restore=33,728、compute=
99,200、output read/write=46,683，11 阶段 unattributed=0。manager local-complete
skew 从 19,046 降至 13,034 cycles，没有异常 cycle。E4/E5 未运行。

验证：Attention 88/88、SFU row-engine 22/22、multicore Softmax 12/12；canonical
`64x64x64` FP32 GEMM sampled=64、mismatch=0、max abs diff=0。独立审查未发现
Critical/Important；残余测试缺口是 contract test 尚未确定性注入 done 早于 barrier、
乱序 done 和 output wait/wakeup，E2/E3 集成覆盖降低了风险。

artifact：

- `/data4/jjgong/tmp/fused_attention_e2_pv_early_compute_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_pv_early_compute_20260820`
- `/data4/jjgong/tmp/attention_pv_early_compute_gemm_20260820`

下一步 G6 先审计 PV operand delivery。E3 的 PV input 仍为 150,528 cycles，PV
matrix 约 116k cycles；先量化 Local GM read、Array-buffer 单端口占用和调度空洞，
再决定双 bank matrix/input programming 是否符合真实硬件，或能否在不复制 matrix
流量的前提下融合 operand programming。没有资源依据前不直接新增硬件端口；继续
只跑 E2/E3，不运行 E4/E5。

## 2026-08-20：完成 Phase G6 PV operand delivery 审计，实验未合入

审计发现 E3 的 PV 有效 K 宽度只有 32，但物理 Array input 宽度为 128，因此每行
仍按 512 B/9 cycles 搬运，75% 是 padding。matrix 与 input 尾部不能同时省略；候选
实验仅压缩 input 前缀，继续完整编程 PV matrix，使矩阵零尾部屏蔽 QK 遗留 input。

实验把 E3 每 worker 的 Array-buffer bytes 从 22,249,472 降到 15,958,016，transfer
cycles 从 403,712 降到 305,408，PV input 阶段从 150,528 降到 52,224 cycles。
局部 98,304 cycles 的理论收益完全兑现，但全系统结果没有改善：

| 负载 | G5 cycles | 实验 cycles | 变化 | 正确性 |
|---|---:|---:|---:|---:|
| E2 `S256,D64` | 25,705 | 25,481 | -224（-0.87%） | 16,384 checked，0 mismatch |
| E3 `S1024,D128` | 699,750 | 700,212 | +462（+0.07%） | 131,072 checked，0 mismatch |

直接观测到的是 compact-input 使 KV prefetch ready slack 坍缩：E3 的 hit/wait 从
G5 的 1,313/671 变为 338/1,646；全局 DMA strict 平均 RTT 也从 3,522 增至
4,257 cycles，慢核 KV-load 归因从约 15k 增至约 120k cycles，manager
local-complete skew 从 13,034 增至 16,690 cycles。这与 worker 更同步地离开 PV、
进入下一 KV tile 后产生共享回包压力的解释一致，但当时尚无 MemNIC response queue
直接统计，不能把“KV 读取突发”写成已闭合的唯一因果。数值、精确活动统计和阶段
守恒均通过，但最终 cycle 退化，因此按项目边界撤回 `attention_pv_compact_input`
的生产代码、配置、统计和测试。当前可交付架构仍停留在 G5；E4/E5 未运行。

负结果 artifact：

- `/data4/jjgong/tmp/fused_attention_e2_pv_compact_input_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_pv_compact_input_20260820`

下一步 G7 先审计共享 KV prefetch burst/worker 同步，按 HBM node 和 worker 对齐请求
时刻、queue wait 与 completion skew，并评估现有 credit 或 worker-slot stagger。
只有全局 KV 等待下降后，才重新组合 compact input 做 E2/E3 A/B。

## 2026-08-20：完成 Phase G7 KV prefetch slack 与 MemNIC 回包队列观测

G7 只增加观测，不改变事件顺序、credit、队列深度或 worker 调度。RoCC 新增每次
双缓冲 KV prefetch 的 DMA 完成延迟、ready-to-consume 提前量和 consume-to-ready
等待量；MemNIC highlink 新增 read response 的 attempted/immediate/enqueued/drained、
queue high-water 和排队等待统计。verifier 同时要求 `dma count=tiles`、
`ready-lead count=hits`、`wait count=waits`，防止统计漏样本。

G5 全开关复测保持逐 cycle 一致：E2=25,705、E3=699,750；分别检查 16,384 和
131,072 个 FP32 输出，mismatch=0。E3 在 300 秒看门狗下约 115 秒完成，未运行
E4/E5。

E3 共 1,984 次 prefetch：1,313 hit、671 wait。累计 prefetch DMA 延迟为
8,503,336 cycles（平均 4,286），ready lead 为 1,872,410 cycles（平均 1,426），
实际 consumer wait 为 118,193 cycles（平均 176、最大 1,485）。慢核 core19 的
124 次 prefetch 中 85 hit/39 wait，累计 wait=6,366 cycles。这里的累计值跨 16 个
并行 worker，不能直接加到端到端 699,750 cycles 上。

MemNIC 证明确有显著且随规模增长的 response backpressure：E2 的 272 个 read
response 中 38 个入队（14.0%），high-water=3，平均入队等待约 261 cycles；E3 的
4,160 个 response 中 3,485 个入队（83.8%），high-water=7，平均入队等待约
1,798 cycles，单次最大 4,121 cycles，结束时 pending=0。该结果证明 G5/E3 已存在
共享回包队列压力，并与 G6 的 prefetch slack 坍缩解释相容；但 G6 负结果 artifact
没有这些新队列计数，因此尚不能量化 compact-input 对队列本身的增量。

artifact：

- `/data4/jjgong/tmp/fused_attention_e2_g7_observe_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_g7_observe_20260820`

下一步 G8 做最小因果 A/B：仅在实验分支恢复 compact-input，保持 G7 观测开启，
比较 G5/compact 的 response enqueued、queue wait、prefetch DMA/lead/wait 和最终
cycle。只有确认 compact 确实放大队列压力后，才评估确定性 worker-slot stagger；
若队列增量不成立，则转向“consumer 提前但 producer launch 未提前”的 prefetch
距离问题。暂不调 credit、不增加端口，也不运行 E4/E5。

## 2026-08-20：完成 Phase G8 compact-input 因果 A/B

G8 恢复默认关闭的 `attention_pv_compact_input` 实验开关。开启时 Array input
programming 接受非空、且不超过物理宽度的前缀，只更新有效 `keyCols` lane；PV
matrix 仍完整编程并将无效 K 行写零，因此遗留 input 尾部不参与计算。没有增加
端口、queue 或 credit。默认关闭时仍要求并传递完整 input vector。

恢复实现精确复现旧 G6：E2=25,481 cycles，E3=700,212 cycles；分别检查
16,384/131,072 个输出，mismatch=0。E3 在 300 秒 watchdog 下约 115 秒完成。

| E3 指标 | G7/G5 | G8 compact | 变化 |
|---|---:|---:|---:|
| 端到端 cycles | 699,750 | 700,212 | +462（+0.07%） |
| PV input cycles | 150,528 | 52,224 | -98,304（-65.31%） |
| prefetch hit / wait | 1,313 / 671 | 338 / 1,646 | hit rate 66.2% -> 17.0% |
| 平均 prefetch DMA cycles | 4,286 | 4,981 | +16.2% |
| 平均 consumer wait cycles | 176 | 1,024 | +481.2% |
| response 入队数 | 3,485 / 4,160 | 3,809 / 4,160 | 83.8% -> 91.6% |
| 平均 response queue wait | 1,799 | 2,375 | +32.1% |
| response high-water / 最大等待 | 7 / 4,121 | 9 / 5,487 | 均恶化 |

compact 使每 worker 的 Array-buffer bytes 从 22,249,472 降至 15,958,016，
transfer cycles 从 403,712 降至 305,408，且 high-water 从 11 降至 1；局部 Array
资源确实改善。与此同时，累计 consumer wait 从 118,193 增至 1,685,193 cycles，
manager local-complete skew 从 13,034 增至 16,690 cycles。累计等待跨 16 个并行
worker，不等同于端到端 cycle，但与最终退化方向一致。

因此 G6 的机制判断现已在同配置 A/B 中闭合：compact 一方面缩短当前 tile，减少
下一 tile prefetch 的 ready slack；另一方面使 worker 请求相位更集中，MemNIC
response queue 的入队率、平均等待、high-water 和最大等待全部上升。E2 的 queue
压力没有明显恶化，仍能保留 224-cycle 收益，说明该问题具有规模相关性。

artifact：

- `/data4/jjgong/tmp/fused_attention_e2_g8_compact_observe_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_g8_compact_observe_20260820`

当前交付配置仍为 G5；compact 开关仅为后续实验保留且默认关闭。下一步 G9 先做
确定性 worker-slot 相位错开的小范围 A/B，目标是降低 E3 response queue wait，
同时检查是否损伤 prefetch ready lead。若 queue 明显下降但 consumer wait 仍高，
则停止调 stagger，转向把 prefetch launch 提前一个安全阶段。暂不调 credit、不增加
硬件端口，也不运行 E4/E5。

## 2026-08-20：完成 Phase G9 worker-slot stagger 实验，机制撤回

G9 实现过一个默认关闭、且仅与 compact-input 配套的启动错峰实验：全局 worker
槽位由 `query_row_begin / worker_rows` 确定，相邻槽位间隔 128 RoCC cycles，最大
启动跨度为 1,920 cycles。它只延迟首次 Q/K/V DMA，不修改后续 tile 状态机，也
没有增加 queue、credit 或端口。

首次试跑错误地把 RoCC cycle 数直接加到 SST timebase tick 上；本配置 1 cycle=
1,000 ticks，因而 128 实际仅为 0.128 cycle/slot，E2/E3 与 G8 逐 cycle 相同。该
无效结果未用于决策。focused contract 随后加入 `LastTickCycle` 单位约束，修正后
重新构建并覆盖原 G9 artifact，再做正式 E2/E3。

| 指标 | G8 compact | G9 stagger-128 | 变化 |
|---|---:|---:|---:|
| E2 cycles | 25,481 | 27,094 | +1,613 |
| E3 cycles | 700,212 | 702,693 | +2,481 |
| E3 response 入队 | 3,809 / 4,160 | 3,746 / 4,160 | 91.6% -> 90.0% |
| E3 平均 response queue wait | 2,375 | 2,350 | -1.1% |
| E3 prefetch hit / wait | 338 / 1,646 | 448 / 1,536 | hit 增加 110 |
| E3 累计 consumer wait | 1,685,193 | 1,645,902 | -2.3% |
| E3 平均 consumer wait | 1,024 | 1,072 | +4.7% |
| E3 manager local-complete skew | 16,690 | 18,502 | +1,812 |

E2/E3 分别检查 16,384/131,072 个输出，mismatch=0。错峰确实略微降低了 aggregate
response queue 压力和 wait 次数，但没有降低一次真实 miss 的等待严重度，并把
一次性启动延迟暴露在最慢 worker 的关键路径上；E3 既未优于 G8，也未优于 G5 的
699,750 cycles。因此 G9 不满足“queue、consumer wait、总 cycle 同时改善”的验收
条件，生产代码、ELI 参数、runner/env 链和 contract 已全部撤回，不继续扫描其他
stagger 数值。

正式 artifact：

- `/data4/jjgong/tmp/fused_attention_e2_g9_stagger128_observe_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_g9_stagger128_observe_20260820`

下一步 G10 先审计 double-buffer 的最早安全 prefetch launch point。后续 tile 当前
已在 buffer swap 后立即启动下一次 prefetch，因此必须先证明首 tile load、K/V 两个
子请求完成时序或现有空闲 buffer 中仍有可提前空间；若两 buffer ownership 已把
launch 推到最早，则不伪造“提前”机制，而转向 response 服务顺序/rotation 映射。
compact 继续默认关闭，交付配置仍为 G5；E4/E5 继续冻结。

## 2026-08-20：完成 Phase G10 双 buffer ownership 与最早预取点审计

G10 没有修改生产代码，也没有运行新的 SST。静态状态机审计确认：首个 key tile 的
K/V DMA 全部完成后，`completeAttentionKvLoad()` 已立即发出下一 tile prefetch；后续
tile 在 `activateAttentionKvPrefetch()` 完成 buffer swap 后，也立即向刚释放的另一个
buffer 发出下一次 prefetch。在“当前 tile 始终占有 active buffer”的 tile-atomic
模型下，现有 launch 已经最早，单纯移动 `launchAttentionKvPrefetch()` 调用没有安全
空间。

首 tile 可以在 K 或 V 子请求各自完成时分别尝试发出下一 tile 的同类子请求，但 E3
每 worker 有 4 个 query block、每 block 31 次 prefetch；该优化最多只覆盖全系统
`16 * 4 = 64` 次，即 `64 / 1,984 = 3.23%`。它还会在初始 K/V 未完全就绪时增加并发
请求，与 G8 已观测到的 response queue backpressure 冲突，因此拒绝作为下一机制。

审计同时找到一个不增加物理 KV buffer 的候选安全窗口：QK 最后一个 panel 的 K
本地读取 callback 返回后，当前 K 区域不再被使用；启用 `attention_pv_v_tile_reuse`
时，第一个 PV panel 把 V 复制到现有 `vPayload` 后，当前 V 区域也不再被使用。如果
此时另一个 buffer 中的 tile N+1 已完全 ready，就可以把已释放的旧 buffer 用于
tile N+2，使两个物理 buffer 同时驻留两个未来 tile。这个 gate 不应在 N+1 仍 pending
时发出 N+2，否则会让非紧急请求与关键的 N+1 回包竞争。

该候选不是简单改一处 launch：当前 worker 只有一组
`prefetchedKeyTileOrdinal/prefetchedKeyTile/pending/ready/timing` 元数据，无法同时表示
N+1 与 N+2 的归属和完成状态。实现前必须改为按 buffer 记录 descriptor，并处理
query-block 尾部、DMA 失败和 swap 后 descriptor 保留。G8 慢核 core10 的当前 tile
扣除 KV wait 后平均约 4,329 cycles，而 prefetch DMA 平均约 4,907 cycles，说明增加
安全 lead 可能有价值；但现有统计没有记录 V release 时刻，也不知道 N+1 在该时刻
ready 的覆盖率，不能据此承诺端到端收益。

Phase G10 因此以“发现可证明安全的两级驻留窗口，但暂不实现”关闭。下一步 G11 只
增加观测：记录每 tile 的 K release、V release、N+1 ready-at-release、release 到
tile boundary 的可用 lead，以及满足“两者均释放且 N+1 ready”的 N+2 候选次数。
先在 compact E2/E3 下量化机会；只有覆盖率和理论 lead 足以解释 consumer wait，才
进入按 buffer descriptor 的机制实现。不得在 N+1 pending 时抢发 N+2，不增加
buffer、credit、queue 或端口；compact 仍默认关闭，交付仍为 G5，E4/E5 继续冻结。

## 2026-08-20：完成 Phase G11 两级预取释放窗口观测

G11 只增加观测，不改变 DMA 发起、callback 顺序或计算状态机。每个 tile 记录最后
一次 K local read 完成和首个 PV panel 将 V 复制到 `vPayload` 的时刻；当 K/V 均
释放且 N+1 prefetch 已 ready 时，记录 N+2 eligibility。eligibility 可以由 release
或稍后到达的 N+1 ready 触发，tile boundary 再记录实际可用 lead。verifier 要求
K/V release 数量与 tile 数一致、candidate 不超过每 query block 最后两个 tile 之外
的理论上限、ready-at-release 不超过 candidate，且 lead count 精确等于 candidate。

正式 compact 结果逐 cycle 复现 G8，证明统计没有扰动事件时序：E2=25,481 cycles，
E3=700,212 cycles；分别检查 16,384/131,072 个输出，mismatch=0。两者都在 300 秒
watchdog 内完成，E3 host 时间约一分钟，没有触发 1.25x cycle 调查边界。

| 指标 | E2 | E3 |
|---|---:|---:|
| N+2 理论候选上限 | 96 | 1,920 |
| 实际 candidate | 12（12.50%） | 305（15.89%） |
| N+1 ready-at-release | 3 | 27 |
| release 后、boundary 前 ready | 9 | 278 |
| 平均可用 lead | 518 cycles | 1,099 cycles |
| 最大可用 lead | 891 cycles | 2,784 cycles |
| 平均 K / V release | 1,181 / 1,540 cycles | 1,764 / 2,445 cycles |

E3 candidate 分布在全部 16 个 worker，每核 13--26 个。慢核 core10 有 18 个，平均
lead=1,091 cycles；其 102 次 prefetch miss 平均 wait=992 cycles，因此候选窗口在
关键核上与实际等待同量级。全系统 candidate lead 累计 335,241 cycles，相当于 G8
累计 consumer wait 1,685,193 cycles 的 19.9%。该比例是无额外 queue 干扰时的
机会规模，不是可直接相加到端到端 cycle 的收益承诺。

构建/安装 `libgolem.so` SHA-256 均为
`787fd479bf69872fae80b35dea40c1347918143dee4edde696782ba801e92dc2`；Attention
focused tests 92/92 通过。正式 artifact：

- `/data4/jjgong/tmp/fused_attention_e2_g11_release_observe_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_g11_release_observe_20260820`

准备阶段曾有一次 E2 命令漏掉 `--pv-matrix-broadcast`，得到 50,414 cycles，已按
配置错误拒绝；正式运行覆盖了 lifecycle、stats 和输出文件，但 E2 `run_summary.csv`
仍保留该无效首行，最后一行及当前 `attention_lifecycle.json` 才是权威 G11 结果。

Phase G11 以“窗口覆盖有限但关键核 lead 足以进入机制 A/B”关闭。下一步 G12 实现
默认关闭的 per-buffer descriptor 两级驻留：只在 K/V 均释放、N+1 已 ready 且 N+2
存在时向旧 active buffer 发出 N+2，N+1 pending 时绝不抢发。先做 focused 状态/计数
测试，再跑 compact E2/E3。E3 必须低于 G5 的 699,750 cycles，且 response queue、
consumer wait 和正确性不能退化；否则撤回机制。仍不增加 buffer、credit、queue 或
端口，E4/E5 继续冻结。

## 2026-08-21：完成 Phase G12 两级 KV 驻留 A/B，机制撤回

G12 曾把单组 prefetch 元数据改为两个 per-buffer descriptor，并在当前 tile 的 K/V
均释放、N+1 ready 且 N+2 存在时，向已释放的 active buffer 提前发出 N+2。实现未
增加物理 buffer、credit、queue 或端口，DMA callback 用 buffer identity 和 job ID
隔离乱序与旧 worker 回调；另加实际 N+2 发起计数，区分候选窗口和真实机制触发。

开发中发现 build tree 保存的是 RoCC 头文件副本，而不是仓库源码的 VPATH/symlink。
最初重编仍使用 G11 副本，因此得到的 E2=25,481、E3=700,212 只是旧库逐 cycle
复现，且新增发起统计没有出现在 stats 中；这两次探针明确作废。同步源码到 build
副本并确认新统计进入 `libgolem.so` 后，正式结果为：

| 负载 | G11 compact 基线 | G12 | 变化 | 正确性 |
|---|---:|---:|---:|---:|
| E2 `S256,D64` | 25,481 | 25,471 | -10（-0.04%） | 16,384 checked，0 mismatch |
| E3 `S1024,D128` | 700,212 | 706,255 | +6,043（+0.86%） | 131,072 checked，0 mismatch |

G12 实际提前发出 E2 21 次、E3 538 次 N+2 prefetch，证明机制生效。E3 慢核
prefetch wait 从 101,145 降到 99,225 cycles，系统累计 consumer wait 也降低约
8.1%；但 miss 平均等待从约 1,024 增到 1,107 cycles，manager local-complete skew
从 16,690 增到 21,431 cycles。慢核 inter-tile KV 降低约 1,167 cycles 的同时，
output DMA、query load 分别增加约 3,612、2,642 cycles，关键路径转移后端到端退化。
MemNIC 总 response 入队和平均 queue wait 没有整体恶化，但部分节点 high-water 从
7 升到 9，说明更早发起改变了共享回包时序，却没有形成稳定的关键路径收益。

按预设 `E3 < 699,750` 门槛拒绝 G12，并撤回 per-buffer descriptor、实际发起统计、
参数和 runner 开关；不做 launch 阈值 sweep。交付仍为 G5：E2=25,705、
E3=699,750，compact 与 G12 都不进入默认配置。最终 build/install 库 SHA-256 一致为
`2fe2e09107e4b64e48e2ee827b2abcd562f5f4f893bfd2ffff1f6f0a57b30be8`，该二进制与
撤回前已验证的 G11/G5 库逐字节一致。

正式 artifact：

- `/data4/jjgong/tmp/fused_attention_e2_g12_real_20260820`
- `/data4/jjgong/tmp/fused_attention_e3_g12_real_20260821`

撤回后 G5 E2 复核 artifact 为
`/data4/jjgong/tmp/fused_attention_e2_g12_rollback_g5_20260821`，结果精确恢复
25,705 cycles，16,384 checked、mismatch=0。

Phase G12 以“机制有效但端到端退化，完整撤回”关闭。下一步不再增加 KV lookahead
距离；先做 G13 只读关键路径归因，比较 G5 与 G12 的 manager skew、query/output
DMA 和各 HBM 节点 response 服务顺序，判断是否存在不增加资源的 demand-response
优先级方案。没有确定性优先级契约前不修改 MemNIC，E4/E5 继续冻结。

## 2026-08-21：完成 Phase G13 只读关键路径归因

G13 使用已存在的 G5/G7 观测 artifact 与正式 G12 artifact 做离线对比，没有重新
运行 SST，也没有修改生产代码。两组结果均为 `PASS` 且数值检查为 0 mismatch。

| 指标 | E3 G5/G7 | E3 G12 | 变化 |
|---|---:|---:|---:|
| 端到端 cycles | 699,750 | 706,255 | +6,505（+0.93%） |
| 慢核 | core19 | core10 | 关键核发生迁移 |
| manager local-complete skew | 13,034 cycles | 21,431 cycles | +8,397 |
| inter-tile PV→next-QK | 180,112 cycles | 292,021 cycles | +111,909 |
| inter-tile KV load | 11,435 cycles | 112,392 cycles | +100,957 |
| inter-tile query load | 5,848 cycles | 11,306 cycles | +5,458 |
| inter-tile output DMA | 10,958 cycles | 18,138 cycles | +7,180 |
| final PV→output ACK | 290 cycles | 278 cycles | -12 |
| KV prefetch wait | 6,367 cycles | 99,225 cycles | +92,858 |

G12 的 lookahead 确实降低了部分慢核 KV 等待，但提前发起的 N+2 请求与 query、output
DMA 共享相同的 MemNIC/HBM 回包资源，导致关键核的 KV load 和 query/output load
同时变长；因此收益没有转化为端到端周期下降。G12 的 manager skew 增大和慢核从
core19 迁移到 core10 进一步说明这是共享 response 服务顺序变化，而不是 PV 计算
本身变慢。E2 只获得 10-cycle 的 G12 微小收益，不能改变 E3 的拒绝结论。

G13 结论：当前首要瓶颈是 demand-response 竞争和服务顺序，不是继续增加 KV
lookahead 距离。下一步只定义可观测、可回归的确定性优先级契约（当前 tile 的
consumer-critical read 高于非关键 N+2 prefetch；query/output DMA 的优先级和
tie-break 固定），先用 trace/replay 或最小 fake queue 验证公平性、无死锁和
response 顺序，再决定是否修改 MemNIC。不得增加 queue、credit、port 或物理 buffer；
E4/E5 继续冻结，正式交付仍为 G5（E2=25,705，E3=699,750）。

已加入不接入生产代码的最小契约模型与测试：
`demand_response_priority_contract.py` 和
`test_demand_response_priority_contract.py`。该模型只验证有限 trace 的确定性排序
与 exactly-once completion，不代表 MemNIC 已经采用该策略；后续若进入实现阶段，
必须在真实 response queue 上复用相同规则并增加竞争压力测试。

随后完成一次 E2 `GOLEM_DMA_TRACE=1` trace-only 运行，artifact 为
`/data4/jjgong/tmp/fused_attention_e2_g13_dma_trace_20260821`。该运行没有打开 G5
优化开关，端到端为 `91,288 cycles`，因此不作为性能 A/B 或正式基线。trace 记录了
272 个 DMA read response，其中 23 个进入 MemNIC retry queue，最大 queue high-water
为 17；但每条事件只有地址、长度和按 worker 重置的 request ID，没有 query/output/
consumer/prefetch semantic kind，无法离线验证优先级契约。下一步必须先给 DMA trace
增加不改变时序的 request-kind 元数据，再做一次完整 G5 E2 trace；在此之前不修改
MemNIC 的调度行为。

## 2026-08-21：G13 DMA semantic kind trace 完成

已加入非时序性的 `DmaRequestKind` 元数据并沿 MemNIC response 路径透传。编号为
`1=AttentionQuery`、`2=AttentionKv`、`3=AttentionKvPrefetch`，没有改变 queue、
credit 或 response arbitration。完整 G5 E2 artifact 为
`/data4/jjgong/tmp/fused_attention_e2_g13_kind_trace_20260821c`。

结果保持正式基线：`25,705 cycles`，`16,384 checked`，`mismatches=0`。运行时 trace
早期按全部 `kind=` 行统计得到 16/36/296，其中包含 enqueue 重复行。现已补齐
`4=AttentionOutput`，并加入 `dma_kind_trace.py` 按唯一完成事件类型
`send READ_RESP/WRITE_COMPLETE` 解析。最终 G5 E2 artifact 为
`/data4/jjgong/tmp/fused_attention_e2_g13_output_kind_20260821b`，仍为 `25,705 cycles`、
0 mismatch；机器可复现统计为 Query 16、KV 32、KV prefetch 224、Output 16、Unknown 0。
这完成了四类 DMA 观测闭环，但仍未改变 MemNIC arbitration。下一步用有限压力
trace/replay 验证 priority contract，E4/E5 继续冻结。

## 2026-08-21：G13 有限竞争 trace/replay 通过

新增 `dma_priority_replay.py`，复用 `demand_response_priority_contract.py` 的
consumer > query > output > prefetch 顺序。模型把真实 G5 E2 的 288 个唯一 DMA 完成
事件按 cycle 压缩为到达流，每个 replay tick 只服务一个当前已到达 response；不修改
生产 MemNIC。三档 `arrival_quantum` 压力结果如下：

| quantum | issued/completed | exactly-once/drained | queue high-water | prefetch max wait |
|---:|---:|---|---:|---:|
| 100,000 | 288/288 | true/true | 65 | 80 ticks |
| 1,000,000 | 288/288 | true/true | 263 | 266 ticks |
| 10,000,000 | 288/288 | true/true | 286 | 285 ticks |

四类完成数始终为 consumer 32、query 16、output 16、prefetch 224。由此可确认固定
tie-break、有限任务 exactly-once、排空和有限 trace 无饥饿。边界是：严格优先级对
无限持续的高优先级流量不保证 prefetch 公平性；当前结论仅适用于有限 Attention
任务。下一步可进入最小 MemNIC arbitration A/B，但必须先做 E2，周期或正确性退化即
回退；E2 通过后才允许 E3，E4/E5 继续冻结。

## 2026-08-21：G13 最小 MemNIC priority A/B 完成

实现了默认关闭的 `GOLEM_DMA_RESPONSE_PRIORITY_ENABLE`：只在已有 DMA response retry
queue 中按 consumer > query > output > prefetch 选择下一个请求，同级保持 FIFO；没有
增加 queue、credit、port 或 buffer。新增 `priority_reorders` 统计实际越过次数。

| 配置 | E2 cycles | E3 cycles | E3 实际重排 | 数值检查 |
|---|---:|---:|---:|---|
| priority off | 25,705 | 699,750 | 0 | E2/E3 均 0 mismatch |
| priority on | 25,705 | 699,133 | 6 | E2/E3 均 0 mismatch |

E3 净减少 617 cycles（0.088%），manager skew 减少 506，query load 减少 638，output
DMA 减少 1,546；但 KV load 增加 3,676、prefetch wait 增加 3,688，prefetch DMA 增加
27,240。说明 6 次重排会改变全局到达时序，但没有稳定改善目标 consumer 路径，净收益
太小，不足以成为正式架构机制。

决策：保留代码作为默认关闭的实验/诊断开关，不纳入 G5 正式配置；正式结果仍是
E2=25,705、E3=699,750。停止扩大到 E4/E5。下一步应回到请求产生端，寻找能同时降低
KV load/prefetch wait 和端到端周期的结构性方案，而不是继续调整 response queue 顺序。

## 2026-08-21：建立 PyTorch V100 GPU baseline

新增 `gpu_baseline/`，固定与正式 E3 相同的 `B1,H1,S1024,D128,FP32`、非 causal、
`1/sqrt(128)` scale 和项目 Q/K/V 生成公式。主机环境为 2 张 Tesla V100-SXM2-32GB，
PyTorch 2.10.0+cu128，CUDA 可用。使用默认
`torch.nn.functional.scaled_dot_product_attention`，50 次 warmup、200 次 CUDA Event
计时；不包含 allocation、H2D 和 correctness reference。

旧 GPU kernel-only 运行的 median/p95 为 0.2048/0.2335 ms，显式 GPU reference 最大误差
1.91e-9；该数字不再作为端到端 SST/GPU 结论。正式 SST G5 E3 为 699,750 cycles @ 1 GHz。
旧 GPU cycle 估算为 264,192，但只覆盖 kernel，且 GPU 时钟是动态的。

随后按端到端对齐要求修改 `gpu_baseline/benchmark_attention.py`：计时区间现在包含
Q/K/V H2D、SDPA、输出 D2H 和 host synchronization，输入构造、一次性 buffer 分配和
correctness reference 仍在计时外；CUDA event kernel-only latency 仅保留为诊断字段。
当前工作区无法访问 CUDA，新的端到端 V100 结果尚未重跑；现有 JSON 中的旧数字仍是
kernel-only 结果，不得用于新的 SST/GPU 端到端结论。
