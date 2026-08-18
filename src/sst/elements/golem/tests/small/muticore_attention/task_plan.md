# Attention 架构改造任务计划

权威实施计划：
`docs/superpowers/plans/2026-07-22-attention-flash-attention-implementation.md`。
本文件只保存当前阶段、已经锁定的决策和恢复入口，避免与主计划重复。

## 已锁定决策

- per-Core `golem.GlobalMemory` 是 worker 唯一的 tile 级本地存储。
- SFU 内部只保留容量受限的 Context Register File 和 lane FIFO/register。
- manager core 负责 Softmax/Attention 控制平面；worker-local RoCC 负责本地
  Array/SFU/GlobalMemory 数据平面。
- coordinator 是 manager-core RoCC control FSM；manager SFU datapath 不参与
  row/tile compute。
- worker slot 必须通过显式 topology map 转换为 physical Core ID。
- 当前 `1024x4096` Softmax 的 66,958-cycle 路径作为 legacy 回归基线，不在
  新 manager 路径验收前删除或替换。
- 加入 Local GM 时序后，周期数允许出现可解释变化；66,958 保持为重构前冻结
  基线，不能被静默覆盖。
- NoC、HBM 和 cache 继续使用 Merlin、memHierarchy 和 DRAMSim3；本轮不重建。

## 阶段状态

- [x] Phase A：原生 K 的 `QK^T` SST 路径与 partial key 验证。
- [x] Phase B：S64/D64 non-causal、causal materialized Attention 验收。
- [x] 模型真实性初审与整改分级，见 `findings.md`。
- [x] Phase C0.1：为 GlobalMemory 增加统一异步本地访问与端口仲裁。
- [x] Phase C0.2：把 SFU 整行私有 vector 改为 bounded context/lane 模型。
- [x] Phase C0.3：迁移 fused 关键路径的 RoCC/Array/WCP 本地搬运接口。
- [x] Phase C0.4：实现 manager coordinator 和显式 worker mapping。
- [x] Phase C1：通过 `S32,D64,Br16,Bc32` 单 KV tile fused SST。
- [x] Phase D1：通过 `S64,D64,Br16,Bc32` 两 KV tile non-causal online recurrence。
- [x] Phase D2：通过同一双 KV tile 路径的 causal mask 与 future-tile skip。
- [x] Phase D3：通过 `Sq20,Skv70,D64` partial query/key tile。
- [x] Phase D4：通过跨 KV tile running-max 跃迁的 extreme-logit fused SST。
- [x] Phase D：通过 S64 两 KV tile online recurrence、causal、partial、extreme。
- [x] Phase E1：通过 4 manager + 16 worker 的 `S256,D64` prefill mapping。
- [x] Phase E2：增加跨 manager 汇聚与单一 tensor-level completion。
- [x] Phase E3：通过 4 manager + 16 worker 的 `S1024,D128` fused 规模点。
- [x] Phase E4：通过 4 manager + 16 worker 的 `S2048,D128` fused 规模点。
- [ ] Phase E5：通过 4 manager + 16 worker 的 `S4096,D128` fused 规模点。

### Phase C0.1/C0.2 验收结果（2026-07-23）

- `GlobalMemoryAPI` 已提供带 client/tag/callback 的异步本地读写；模型包含独立
  read/write port、bytes/cycle、base latency、queue depth、最大请求大小和
  queue/high-water/bytes 统计。
- Softmax input DMA 必须先完成 Local GM landing；MAX、EXP/SUM、NORMALIZE
  通过 16-lane bounded buffer 分块访问 Local GM；output DMA 从 Local GM 地址
  读取，ACK 后才允许 unique completion。
- `1024x4096` 实测 accelerator completion 为 **139,750 cycles**，完整校验
  `4,194,304/4,194,304`、mismatch=0、max abs diff=`5.72476014e-11`。
- 1024 行的 input DMA ready、MAX、EXP/SUM、NORMALIZE、output DMA ACK 均为
  1024 次；unique band completion 为 16 次。每核 Local GM read/write bytes
  为 `4,194,304/3,145,728`，queue reject=0，queue high-water=4。
- 66,958 cycles 仍是重构前冻结 legacy 基线；66,061 仍仅是 analytical
  compute reference。139,750 cycles 是加入 Local GM 端口/带宽/落盘时序后的
  当前端到端结果，三者不得互相替代。
- focused tests、两个 guest build 和 canonical `64x64x64` GEMM regression
  均通过；GEMM `VERIFY-C` mismatch=0、max abs diff=0。

### Phase C0.3a 验收结果（2026-07-23）

- WCP operand panel 不再通过同步 `rd_from_globalmem()` 读取；matrix/vector 均按
  `GlobalMemoryAPI::localMaxRequestBytes()` 分块，经 `localReadAsync()` 和
  `LocalMemoryClient::WCP` callback 完成后才允许 Array compute 开始。
- 删除无地址、无容量的 `partialCTiles_`。跨 resident-K window 的 partial C 现在
  spill/reload 到 `local_accum_gm_addr`，最后一次异步 write callback 完成前 WCP
  不得推进窗口；reload callback 全部完成后才恢复 Array accumulator。
- partial-C 空间按当前 GEMM 的实际 M/N tile 数与 reuse 上限的较小值分配。定向
  `64x64x512` case 的每核 Local GM stride 为 `549,440 bytes`，不再按最大
  `4x4` reuse 窗口无条件预留。
- 最新 `libgolem` 已重编并安装；build/install SHA-256 均为
  `5827b0320bfc9e274240e3a9f7b6fe1c36cc3882c210290311a2930ac0391914`。
- `64x64x512` FP32 定向回归通过：8 个 K tile、resident K=2，触发 3 次
  16 KiB spill 和 3 次 16 KiB reload；`VERIFY-C` sampled=64、mismatch=0、
  max abs diff=0。core 4 的 Local GM 统计为 read `76/311,296 bytes`、write
  `12/49,152 bytes`、reject=0、high-water=2，字节数与 operand + partial-C
  搬运闭合。
- C0.3 **尚未完成**：Array buffer programming/readout 仍是立即操作；最终
  output writeback 仍需改成 Array -> Local GM -> address-based DMA；legacy RoCC
  GM2IMAT/GM2IVEC/OVEC2GM 固定延迟路径也尚未迁移。

### Phase C0.3b 验收结果（2026-07-23）

- `ComputeArray` 已增加有界 buffer transfer scheduler，默认 1-cycle base
  latency、64 B/cycle、1 port、queue depth 64；请求数、bytes、reject、high-water
  和 transfer cycles 均按 core 输出。
- MVM/CrossSim Array 提供 `programOperandsAsync`、`readOutputAsync` 和
  `writeOutputAsync`。WCP 不再调用 `setMatrixItem`、`setVectorItem` 或
  `getOutputVector`，operand programming、partial-C capture/restore 和最终 readout
  都等待 Array callback。
- 最终 C 路径改为 Array async read -> `local_out_gm_addr` async write ->
  `dma_write_from_globalmem_to_host` -> HBM ACK -> worker advance；不再使用
  fire-and-forget host payload。
- `local_out` 从单 vector scratch 扩为完整 C tile，防止 16 KiB final tile 覆盖
  partial-C accumulator。`64x64` 的 per-Core Local GM stride 相应变为
  `565,568 bytes`。
- 最终 `64x64x64` FP32 回归：`VERIFY-C` sampled=64、mismatch=0、max abs
  diff=0，system latency=`18,397 cycles`。core 4 Array buffer 为 128 requests /
  1,081,344 bytes；Local GM 为 12 reads / 49,152 bytes、4 writes / 16,384
  bytes，reject=0。
- `64x64x512` partial 回归：`VERIFY-C` sampled=64、mismatch=0、max abs
  diff=0，system latency=`143,033 cycles`。Array buffer 的 960 requests /
  8,634,368 bytes 与 512 operand program + 192 partial read + 192 partial restore
  + 64 final read 完全闭合；Local GM 327,680 read bytes / 65,536 write bytes 与
  operand、3 次 spill/reload 和 final DMA 完全闭合。
- 最新 `libgolem` build/install SHA-256 均为
  `5322b78af0c98334a805b9a3713220b668c1f5df8f3e96a836d6035e116bfba7`。
- C0.3 仍未整体完成：legacy RoCC GM2IMAT/GM2IVEC/OVEC2GM 尚未组合这些异步
  Local GM/Array 接口，下一步为 Phase C0.3c。

### Phase C0.3c 验收结果（2026-07-23）

- legacy RoCC GM2IMAT/GM2IVEC/OVEC2GM 不再由固定延迟后同步复制完成。
  matrix/vector load 先分块请求 `LocalMemoryClient::RoCC`，再经独立的 bounded
  Array matrix/input programming port；output store 先异步读取 Array，再分块写入
  Local GM。blocking 命令等待最后一个 callback，batch 命令仍提交即返回。
- MVM/CrossSim Array 新增 `programMatrixAsync` 和 `programInputAsync`，与 C0.3b
  的 combined programming/readout 共用同一有界 buffer scheduler。队列拒绝只
  表示 backpressure，由后续 tick 重试。
- legacy output 使用 `readOutputBytesAsync` 保留 int64 原始位模式，不经过
  `double` 中转；避免绝对值超过 `2^53` 时静默丢失精度。
- 新增 `test_rocc_async_local_transfer_contract.py`。Attention discovery 22/22、
  RoCC/SFU integration 9/9、Softmax wrapper 30/30 均通过。
- `libgolem` 完整重编并安装；build/install SHA-256 均为
  `446b991b40c49334a118ec6164e07f81624e3ee911c2f6d0460fe9eeaf7b08d8`。
- legacy `64x64x64` FP32 回归通过：`VERIFY-C` sampled=64、mismatch=0、
  max abs diff=0，worker total=`71,224 cycles`。core 4 为 192 次 Array transfer、
  320 次 Local GM read 和 64 次 Local GM write，覆盖 batch matrix/vector load 与
  64 次 output store。
- canonical WCP `64x64x64` FP32 仍为 `18,397 cycles`，`VERIFY-C` mismatch=0；
  Array/Local GM 请求和字节数与 C0.3b 完全一致。因此 C0.3 已关闭，下一步进入
  Phase C0.4 manager coordinator 和显式 worker topology map。

### Phase C0.4 验收结果（2026-07-23）

- 新增 V3 tensor params 与固定大小、版本化的 worker topology map；worker slot
  不再隐式等于 Core ID，而是由 manager RoCC 映射到唯一 physical Core ID。
- manager coordinator 是独立 RoCC FSM：异步读取 descriptor/params/topology，
  通过 GlobalMemory/NoC 分发 band，按 worker bitmap 接收唯一 completion。manager
  SFU datapath 不执行 row compute；wait 只在全部 worker 的 output DMA ACK 之后返回。
- GlobalMemory reduction transport handler 由 RoCC 唯一持有：manager completion
  进入 coordinator FSM，其他 reduction message 转发给本地 SFU。legacy opcode 和
  SFU coordinator 路径保留。
- manager `16x64` smoke：`VERIFY-SFU-SOFTMAX PASS`，checked=1,024、mismatch=0、
  max abs diff=`3.29887216e-09`；4 次 band dispatch、4 次唯一 completion、16 次
  input DMA/MAX/EXP-SUM/NORMALIZE/output DMA ACK，contract pass，accelerator
  latency=`418 cycles`。
- legacy `64x64` smoke：checked=4,096、mismatch=0、16 次唯一 completion、
  contract pass，accelerator latency=`470 cycles`。回归中发现 argv logical ID 与
  Vanadis physical Core 不同会导致跨 GM base 写入；guest 已恢复以
  `sched_getcpu()` 选择 executor，只有查询无效时才回退 argv。
- canonical WCP `64x64x64` FP32 仍为 `18,397 cycles`，`VERIFY-C` sampled=64、
  mismatch=0、max abs diff=0；DMA timeout/write/send retry 均为 0，manager/SFU
  新路径 activity 为 0。
- `libgolem` build/install SHA-256 一致：
  `0877acd167f44e499c1d79892596c3d410a7142c432b04ac3e91d8413ed66514`。
- 功能 focused tests 与两个 guest build 全部通过。完整 SFU discovery 中另有 9 个
  绘图测试因当前 Python 环境缺少 `matplotlib` 未执行；未修改任何报告或图像素材。

Phase C0 已完成。下一步进入 Phase C1：实现并验收
`S32,D64,Br16,Bc32` 的单 KV tile fused Attention SST 数据流。

### Phase C1 验收结果（2026-07-23）

- 新增 128-byte `GolemAttentionDescV1`、独立 Attention manager issue/wait
  opcode、manager FSM，以及显式 `AttentionDispatch/AttentionComplete` NoC 消息。
  manager core 0 只执行控制面，physical worker core 1 执行全部数据面；最终唯一
  completion 晚于两个 query block 的 output DMA ACK。
- worker 对 K/V 各执行一次 HBM DMA；每个 16-row query block 执行 Q DMA、两次
  QK key-panel Array 计算、local SFU scale/Softmax、四次 PV dim-panel Array 计算，
  再从 Local GM 发起 O DMA。S 与 P 共用同一 2,048-byte Local GM 窗口，HBM 中
  没有 S/P 区域。
- per-Core Attention window 精确为 `26,752 bytes`：Q 4,096、K 8,192、V 8,192、
  S/P 2,048、O 4,096、metadata 128 bytes。SFU 继续使用 bounded context/lane，
  不保存完整行私有 vector。
- 最终 `B1,H1,S32,D64,Br16,Bc32` non-causal SST：checked=2,048、mismatch=0、
  max abs error=`1.3586599793141696e-08`。精确统计门禁通过：manager issue/complete
  `1/1`、QK/PV Array ops `64/128`、SFU jobs/rows `2/32`、S/P HBM bytes=0，
  manager SFU jobs=0。
- 回归中发现第二个 query block 继承前一块 PV 的 panel index，导致 QK 只执行
  48 次；在 query-block 入口显式清零 panel 后恢复 64 次并通过全量数值校验。
  runner 现已硬性校验上述统计，避免只凭部分数值通过掩盖阶段缺失。
- functional focused tests：Attention 37/37、SFU/GlobalMemory/RoCC 218/218、
  multicore Softmax 12/12。Softmax `16x64` smoke checked=1,024、mismatch=0、
  contract pass、accelerator completion=394 cycles。
- canonical WCP `64x64x64` FP32 GEMM：`VERIFY-C PASS`、sampled=64、mismatch=0、
  max abs diff=0，system latency 仍为 `18,397 cycles`；Attention stats 全为 0。
- `libgolem` 与 `libmemHierarchy` 已重编并安装，build/install SHA-256 分别一致为
  `26a97dab361632df9f99a8bbad52eeed91f0e8d8a2786d8993cd86181145bdcd` 和
  `b0edd9f3bc6b46c0f59e7ddbb58701290d9a5614072c9edf68b7633e3c28d43d`。

Phase C1 已完成。下一步进入 Phase D：先实现 `S64,D64,Br16,Bc32` 的两个 KV
tile online `(m,l,O)` recurrence，再增加 causal、partial tile 和 extreme logits。

### Phase D1 验收结果（2026-07-23）

- worker 对 K/V 各 DMA 一次；四个 16-row query block 分别遍历两个 32-key
  tile。每个 tile 执行 QK Array -> Local GM S/P -> SFU online `(m,l)` -> PV
  Array，只有最后一个 key tile 后才执行 output DMA 和 query-block 推进。
- SFU 使用 16 个固定容量 online row context 保存 `(m,l)`；首 tile 由
  `m=-inf,l=0` 开始，第二 tile 计算 `alpha/beta`，并把旧 Oacc 乘
  `alpha/l_new` 后通过 Array output buffer 恢复，再以 accumulate 模式加入
  `beta/l_new * P_tile V_tile`。RSQRT(D) 每个 Attention job 只执行一次。
- `B1,H1,S64,D64,Br16,Bc32` non-causal SST 全量通过：checked=4,096、
  mismatch=0、max abs error=`7.568611652339352e-09`；QK/PV Array ops=
  `256/512`、SFU jobs/rows=`8/128`、S/P HBM bytes=0、RSQRT count=1。
- 调试中修复两处共享资源生命周期问题：PV accumulate mode 不得泄漏到下一
  query block 的 QK；4-context C1 中物理 SFU context 复用时必须重绑定新的
  global row。对应契约测试已加入，C1 与 D1 均重新通过。
- focused tests：Attention 43/43，SFU/GlobalMemory 68/68。Softmax `16x64`
  smoke checked=1,024、mismatch=0、accelerator completion=394 cycles；canonical
  WCP `64x64x64` GEMM `VERIFY-C PASS`、system latency=18,397 cycles。
- `libgolem` 已重编并安装，build/install SHA-256 均为
  `5cfa9f165bef7c79985b488ff3bbde8a76fb9b8f3289093a238b342254968f8d`。

### Phase D2 验收结果（2026-07-23）

- Attention descriptor 的 causal flag 已从 guest 经 manager/worker RoCC 传至 SFU；
  未知 flag 会被拒绝。SFU 在 scale 后、row-max 前按全局 query/key 位置逐元素
  mask，不改变 online `(m,l,O)` recurrence。
- worker 按 query block 计算有效 KV tile 数。query block 0/1 只执行 key tile 0，
  query block 2/3 执行 key tile 0/1；两个完全位于未来的 tile 不发起 QK、SFU 或
  PV 操作。四个对角边界共 mask `992` 个 score 元素。
- `B1,H1,S64,D64,Br16,Bc32` causal SST 全量通过：checked=4,096、mismatch=0、
  max abs error=`2.3585739111764426e-08`；QK/PV Array ops=`192/384`、SFU
  jobs/rows=`6/96`、scaled/masked elements=`3,072/992`、S/P HBM bytes=0、
  RSQRT count=1。
- D1 non-causal 与 C1 分别重新通过全量 4,096/2,048 输出校验；Softmax
  `16x64` smoke mismatch=0、accelerator completion=394 cycles。canonical WCP
  `64x64x64` GEMM `VERIFY-C PASS`、sampled=64、mismatch=0、max abs diff=0。
- Attention discovery 46/46，focused RoCC/SFU/GlobalMemory 38/38；`libgolem`
  已重编并安装，build/install SHA-256 均为
  `c780d93bf8498872828513ceb4831b6f0ec180279571a9bb13174c40ef634d33`。

### Phase D3 验收结果（2026-07-23）

- RoCC 的 query block、key tile 和 key panel 数改为 ceil division；每个阶段都使用
  当前块的真实 `queryRows`/`keyCols`。尾部 Q 只 DMA 4 行，尾部 K/V 只使用 6 行；
  固定 Array 端口所需的零填充只存在于 operand buffer，不进入 SFU max/sum。
- Array 仅启动有效实例：两个 query block、三个 key tile 共执行 QK/PV ops
  `140/240`，而不是按 `16x32` 满块补算。SFU 执行 6 jobs、60 rows，并恰好 scale
  `20x70=1,400` 个有效 score。
- `B1,H1,Sq20,Skv70,D64,Br16,Bc32` non-causal SST 全量通过：checked=1,280、
  mismatch=0、max abs error=`7.499891131745873e-09`；S/P HBM bytes=0、RSQRT
  count=1。
- D1、D2、C1 真实 SST 均重新通过；Attention discovery 49/49、SFU focused
  22/22、Local GM/RoCC/Array 13/13、multicore Softmax 12/12。Softmax `16x64`
  completion=394 cycles；canonical GEMM `VERIFY-C PASS`、system latency=18,397
  cycles。
- `libgolem` 已重编并安装，build/install SHA-256 均为
  `904a5dd1a205a93c7ace72fddc7f13c79d60a618657ba1131369f062bb43b4d9`。

### Phase D4 验收结果（2026-07-23）

- 增加确定性 extreme-logit 输入 profile：每个 query 的缩放 score 在第一个 KV
  tile 恰为 `-100`、第二个 KV tile 恰为 `+100`，强制 online running max 跨
  tile 跃迁 `200`；Q/K/V、score 和最终 Attention 输出均要求有限。
- runner 增加互斥的 `--extreme-logits` 模式，复用已验收的 D1 fused guest 和
  精确活动计数门禁，因此没有增加第二套硬件路径。
- `B1,H1,S64,D64,Br16,Bc32` extreme-logit SST 全量通过：checked=4,096、
  mismatch=0、max abs error=0；QK/PV ops=`256/512`、SFU jobs/rows=`8/128`、
  S/P HBM bytes=0、RSQRT count=1。
- 默认 D1 同轮回归通过：checked=4,096、mismatch=0、max abs error=
  `7.568611652339352e-09`，精确统计保持不变。
- 本增量只修改测试生成器、runner 和契约测试，未修改 SFU、GlobalMemory、RoCC
  或其他共享组件，因此不需要重编 `libgolem`，也不需要重复 GEMM/Softmax 回归。

Phase D 已关闭。下一步进入 Phase E：把当前单 manager/单 worker fused 数据流
扩展为 4 manager + 16 worker 的 prefill mapping，并保持每个 query block 的
output DMA ACK 先于 unique completion。

### Phase E1 验收结果（2026-07-23）

- `B1,H1,S256,D64,Br16,Bc32` 使用 4 个 dedicated manager（core 0-3）和
  16 个显式映射 worker（core 4-19）。每个 manager 管理 64 个 query row，向
  4 个 worker 各分发一个 16-row block；completion 按 topology slot/core bitmap
  去重，且只在对应 worker 的 output DMA ACK 后返回。
- Q/O 按 64-row band 分布在 HBM node 1-4；K/V 也按每 node 64 rows block
  stripe。worker 每次只把一个 32-row K/V tile 流入 per-Core GlobalMemory，避免
  将完整 `S256` K/V 放入 64 KiB Attention window；S/P HBM bytes 保持为 0。
- 真实 fused SST 全量通过：checked=16,384、mismatch=0、max abs error=
  `4.4967521123807225e-09`。每个 manager issue/complete=`1/1`；每个 worker
  QK/PV ops=`256/512`、SFU jobs/rows=`8/128`、scaled elements=`4,096`、
  RSQRT count=1。
- 16 worker 并发下 HBM read response 压力会超过小规模默认重试窗口；E1 runner
  明确使用 retry ticks/count=`4096/32`，但不放宽 output 数值、活动计数或 completion
  因果门禁。
- RoCC 修改后已重编并安装 `libgolem`，build/install SHA-256 均为
  `cc884ddba01f6426bf6651f973e7704ce5b415cddc4af881c575480beb919319`。D1
  `S64,D64` 回归重新通过；canonical WCP `64x64x64` FP32 GEMM
  `VERIFY-C PASS`、sampled=64、mismatch=0、max abs diff=0。
- Attention discovery 57/57、SFU focused 22/22、multicore Softmax 12/12
  通过。

Phase E 尚未关闭：当前产生 4 个 manager-level band completion。下一步 Phase E2
增加 root/跨 manager completion aggregation，确保全部 16 个 worker 的 output DMA
ACK 后只产生一次 tensor-level completion，再继续 `S1024,D128` 规模点。

### Phase E2 验收结果（2026-07-24）

- 128-byte Attention descriptor 利用原保留区显式携带 `tensor_root_core`、
  `tensor_manager_slot` 和 `tensor_manager_count`，ABI 大小不变。scale guest 的四个
  manager 使用同一 tensor job/tag，manager core 0 是 root，manager slot 0-3 与
  dedicated manager core 0-3 一一对应。
- 新增独立 `AttentionManagerComplete` NoC 消息。每个 manager 先通过本组 4 个
  worker 的 physical-core/slot bitmap；非 root 再向 core 0 报告 band completion。
  root 使用第二级 manager bitmap 拒绝错误或重复 slot，收齐四个 band 后才完成
  software wait 并增加一次 tensor completion。
- 最终 `B1,H1,S256,D64,Br16,Bc32` fused SST：checked=16,384、mismatch=0、
  max abs error=`4.4967521123807225e-09`，S/P HBM bytes=0。四个 manager 的 local
  band completion 各为 1；core 0 收到 4 个 manager completion；全系统仅 core 0
  的 `attention_tensor_jobs_completed=1`，其他 core 均为 0。
- D1 single-manager 回归 checked=4,096、mismatch=0、max abs error=
  `7.568611652339352e-09`。Attention discovery 60/60、SFU focused 22/22、
  multicore Softmax 12/12 通过；canonical WCP `64x64x64` FP32 GEMM
  `VERIFY-C PASS`、sampled=64、mismatch=0、max abs diff=0。
- `libgolem` 已重编并安装，build/install SHA-256 均为
  `6df59c6be54a163915b86eb89049b388638745d154ef70f19d625431e226fa9a`。

Phase E 的 control-plane completion 条件已满足，但规模阶梯尚未完成。下一步
Phase E3 扩展到 `B1,H1,S1024,D128`，必须继续保持单一 tensor completion、S/P
HBM bytes=0、Local GM window 有界以及数值/活动计数门禁。

### Phase E3 验收结果（2026-07-24）

- worker 现在支持动态 `head_dim` 和一个 dispatch 内的多个 16-row query block。
  `D128` 使用 8 个 PV dimension panel；每个 worker 顺序处理 4 个 query block，
  同时只保留一个 32-row K/V tile。Q/K/V/S/P/O 加安全区的 Attention Local GM
  window 为 51,328 bytes，仍小于既有 64 KiB 上限。
- 真实 `B1,H1,S1024,D128,Br16,Bc32` fused SST 全量通过：checked=131,072、
  mismatch=0、max abs error=`1.5966506535956479e-09`，S/P HBM bytes=0。
  四个 manager band completion 各为 1，root 收到 4 个 manager completion，
  tensor completion=`1/0/0/0`。
- 每个 worker 的精确活动计数为 QK/PV ops=`4,096/16,384`、SFU jobs/rows=
  `128/2,048`、scaled elements=`65,536`、RSQRT count=1；16 个 worker 全部通过。
- E2 `S256,D64` 与 D1 `S64,D64` 真实 SST 回归均保持 mismatch=0；Attention
  discovery 64/64、SFU focused 43/43、GlobalMemory async 7/7、multicore
  Softmax 12/12 通过。canonical WCP `64x64x64` FP32 GEMM `VERIFY-C PASS`、
  sampled=64、mismatch=0、max abs diff=0。
- `libgolem` 已重编并安装，build/install SHA-256 均为
  `7eb6a002dac3881d91fbb6040cbb595d3ea47116e847b3bf08cdf91f23c055b0`。

Phase E3 已关闭。下一步 Phase E4 扩展到 `B1,H1,S2048,D128`；在启动真实
SST 前先增加容量/地址/精确计数契约和 dry-run，并保持 E3 的 completion、Local GM
与零 S/P HBM 门禁。

### Phase E4 验收结果（2026-07-24）

- `B1,H1,S2048,D128,Br16,Bc32` 继续使用 4 manager + 16 worker。每个 manager
  管理 512 个 query row，每个 worker 顺序处理 8 个 16-row query block 和 64 个
  KV tile；Attention Local GM window 仍固定为 51,328 bytes。
- 真实 fused SST 全量通过：checked=262,144、mismatch=0、max abs error=
  `1.181374809935097e-09`，S/P HBM bytes=0。四个 manager band completion 各为
  1，root 收到 4 个 manager completion，tensor completion=`1/0/0/0`。
- 每个 worker 的精确活动计数为 QK/PV ops=`16,384/65,536`、SFU jobs/rows=
  `512/8,192`、scaled elements=`262,144`、RSQRT count=1；16 个 worker 全部通过。
- E3 `S1024,D128` 与 D1 `S64,D64` 真实 SST 回归保持 mismatch=0；Attention
  discovery 67/67、SFU focused 43/43、GlobalMemory async 7/7、multicore
  Softmax 12/12 通过。canonical WCP `64x64x64` FP32 GEMM `VERIFY-C PASS`、
  sampled=64、mismatch=0、max abs diff=0。
- `libgolem` 已重编并安装，build/install SHA-256 均为
  `d37308086cf25cb26e3861de5d82ad8590e682ae9d5e30a5cac189a89fa05125`。

Phase E4 已关闭。下一步 Phase E5 是 `B1,H1,S4096,D128`；真实 SST 工作量约为
E4 的 4 倍，因此先增加 E5 契约、dry-run、运行成本和 watchdog 门禁，再决定是否
在当前资源上执行完整规模点。

## Phase C0 验证边界

进入首个 fused SST 前必须证明：

1. Local GM latency 随 bytes 增长，并可观察 read/write port contention。
2. 同核 Local GM 访问不产生 NoC packet，跨核访问仍走 NoC。
3. SFU full row/tile 不再由无界 C++ vector 隐式保存。
4. fused 路径没有直接 GM/array `memcpy` completion shortcut。
5. manager 不搬运 S/P；映射中的 worker completion 唯一且晚于 output ACK。
6. shared-component focused tests、`libgolem`、Softmax smoke 和 canonical GEMM
   regression 全部通过。

## 恢复入口

继续实现前依次阅读：

1. `src/sst/elements/golem/tests/small/muticore_softmax/PROJECT_HANDOFF.md`
2. 本文件
3. `findings.md`
4. `progress.md`
5. 权威实施计划的 4.5、4.7、Phase C 和 12.1 节
