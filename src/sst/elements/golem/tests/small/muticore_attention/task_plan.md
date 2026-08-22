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
- [x] Phase F0：建立 QK/Softmax/PV/DMA 的细粒度流水观测和 overlap 上界模型。
- [x] Phase F1：为每个 worker 增加 K/V 双缓冲，先实现下一 KV tile DMA 与当前
  QK/Softmax/PV 的预取 overlap。
- [x] Phase F2：在 F1 回归通过后，评估 QK panel 与 Softmax 的 panel-level overlap；
  只有 SFU 接口支持增量 `(m,l)` 更新时才实现该阶段。
- [ ] Phase F3：评估 Softmax/PV 的生产者-消费者 overlap；不满足带宽、buffer
  容量和 online recurrence 条件时保持串行路径。
- [ ] Phase F4：在 E2/E3 验证收益稳定后，再决定是否执行 E4/E5 规模回归。

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

## Phase F：QK/Softmax/PV 跨阶段流水化计划（2026-08-20）

目标是让不同 Attention tile 的阶段发生时间重叠，优先降低当前 E3 的端到端
`965,933 cycles`。当前实现已经有 worker 间并行、16 个 Array 间并行和 K/V 两路
DMA 并行，但单个 worker 仍是：

```text
K/V load -> QK MVM -> score read/write -> Softmax -> PV MVM -> output restore
         -> next KV tile
```

同一 worker 内 QK MVM 与 Softmax 不能直接重叠，因为 Softmax 依赖完整的
`S[16,32]`；Softmax 与 PV MVM 也不能直接重叠，因为 PV 依赖完整的 `P[16,32]`。
因此 Phase F 先做跨 tile 的 DMA/compute overlap，再评估 panel-level 算法和接口
修改，避免把“统计阶段重叠”误认为真实硬件流水。

### F0：观测、约束和理论上界

1. 在 `AttentionWorkerState` 中增加阶段统计：KV DMA、QK matrix/input/compute、
   score read/write、SFU、PV matrix/input/compute、Oacc restore/read/write。
2. 为每个 worker 记录 tile ordinal、buffer id、issue tick、complete tick，验证
   不同阶段是否真实时间重叠，并检查每个 buffer 没有读写冲突。
3. 根据 E3 的 128 个 KV tile 计算 overlap 上界：当前 KV load=141,982 cycles，
   QK/Softmax/PV 是串行关键路径；双缓冲的理论收益上限是隐藏可重叠的 KV load，
   不能把 QK 与 Softmax 的数据依赖周期直接相减。
4. 验收：E2/E3 输出 mismatch=0，tile 顺序和 online `(m,l)` 结果保持一致，
   新增统计满足时间守恒；任何仿真超过硬 watchdog 立即停止并分析。

### F1：K/V 双缓冲预取（第一实现目标）

1. 将单一 `kLocal/vLocal` 扩展为两个固定大小的 KV buffer；每个 buffer 包含
   `K[32,D]` 和 `V[32,D]`，Local GM window 需要重新计算并增加容量检查。
2. 当前 tile 在 buffer A 执行 QK、Softmax、PV 时，DMA engine 将物理顺序中的
   下一 tile 预取到 buffer B；当前 tile 完成后交换 A/B。
3. 预取地址必须继续使用 `keyTile`（物理 tile），SFU/PV recurrence、first/final
   tile 判定继续使用 `keyTileOrdinal`。
4. 不允许覆盖仍被 QK/PV 使用的 buffer；预取失败或 buffer ownership 不满足
   不变量时 fail-fast，稳定回退由默认关闭的单 buffer 开关路径提供。
5. 验收顺序：契约/dry-run -> E2 180 秒 watchdog -> E3 600 秒 watchdog；比较
   `inter_tile_pv_to_next_qk` 和总 cycles。若收益小于统计误差或出现 buffer stall，
   保留功能开关但不进入 E4/E5。

### F2：QK panel 与 Softmax 的增量 overlap（条件阶段）

1. 先确认 SFU 是否能接受一个 KV tile 的多个 score panel，并在 panel 到达时增量
   更新每行 `(m,l)`；当前 `issueAttentionTile()` 的整 tile 接口不足以实现该阶段。
2. 若接口可扩展，设计双 score panel buffer：QK panel n 写入 buffer n，SFU 消费
   buffer n，同时 QK 生成 panel n+1。
3. 必须处理 running-max 跃迁：后续 panel 发现更大 max 时，要重新缩放此前的
   `l` 和已生成的 P/partial PV，不能简单地把 panel softmax 结果相加。
4. 验收必须覆盖 D4 extreme-logit、D2 causal、D3 partial tile；任何数值差异先
   关闭 overlap，不得以牺牲 online softmax 正确性换周期。

### F3：Softmax/PV overlap（条件阶段）

1. 只有 SFU 能输出可消费的 P panel，并且 PV 允许 panel 级累加时才考虑；否则
   保持当前“完整 P tile 后启动 PV”的串行路径。
2. 需要至少两个 P buffer，以及明确的 P panel ready/consume credit，防止 SFU
   覆盖 PV 尚未读取的 P 数据。
3. 需要重新处理 Oacc scale：online max/l 更新可能改变整个 tile 的归一化比例，
   不能在 scale 尚未稳定时提交不可逆 PV 累加。
4. 该阶段的优先级低于 F1，因为它可能要求修改 SFU ABI、P storage 语义和 PV
   累加协议，风险明显更高。

### F4：规模回归门禁

1. F1/F2/F3 任一阶段都必须先在 E2/E3 完成严格 A/B，并记录所有 overlap 开关。
2. 只有 E3 周期、manager skew、Local GM/Array 请求和结果校验均稳定后，才重新
   评估 E4；E5 仍需显式 `--allow-expensive` 和 watchdog。
3. 不再使用“仿真跑完即可”的验收标准；必须同时报告 cycle、仿真墙钟时间、
   watchdog、数值 mismatch、buffer stall 和阶段时间守恒。

### Phase F0 验收结果（2026-08-20）

- RoCC 为每个完成的 Attention tile 记录连续时间轴：KV load、Q local read、QK
  matrix/input/compute-readout、Softmax、PV matrix/input/restore/compute/output
  read-write，共 11 个互斥阶段；verifier 强制检查总时长等于所有阶段之和。
- 每个阶段即使耗时为 0 也写入一次样本，因此每 worker 的统计 Count 必须等于
  SFU Attention jobs：E2=8，E3=128。
- E2 保持 29,198 cycles，checked=16,384、mismatch=0；慢 worker tile total=
  26,011 cycles，8/8 样本齐全，unattributed=0。
- E3 保持 965,933 cycles，checked=131,072、mismatch=0；慢 worker tile total=
  951,943 cycles，128/128 样本齐全，unattributed=0。
- E3 分项为：KV 145,408；Q local 16,640；QK matrix/input/compute-readout
  66,304/18,432/49,664；Softmax 31,367；PV matrix/input/restore/compute/output
  165,376/196,608/63,488/133,120/65,536 cycles。
- 因此同一 tile 的 QK/Softmax overlap 即使完全隐藏 Softmax，理论上也只覆盖
  31,367 cycles（tile path 的 3.30%）；F1 双缓冲可尝试隐藏的 KV load 上界为
  145,408 cycles（15.27%）。最大剩余区域仍是 PV 合计 624,128 cycles（65.56%）。
- F0 artifact：
  `/data4/jjgong/tmp/fused_attention_e2_phase_f0_stats_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_phase_f0_stats_20260820`。

### Phase F1 验收结果（2026-08-20）

- 新增默认关闭的 `attention_kv_double_buffer`。每个 worker 使用两组独立的
  `K[32,D]`/`V[32,D]` Local GM buffer；首 tile 前台加载，执行当前 tile 的
  QK/Softmax/PV 时预取下一物理 tile，完成后交换 active/prefetch buffer。
- `keyTileOrdinal` 仍维护 online softmax 的逻辑顺序，预取地址通过独立的物理
  `keyTile` 计算，兼容既有 KV tile rotation。DMA 任一路失败或 buffer ownership
  不满足不变量时 fail-fast；关闭开关时保留原单 buffer 路径。
- E1/D64 双缓冲 window 为 43,136 bytes，E3/E4/E5 D128 为 84,096 bytes；scale
  runner 启用 `--kv-double-buffer` 时显式使用 `0x14880`，默认仍为 `0x10000`。
- 新增 `attention_kv_prefetch_tiles/hits/waits`；verifier 要求每 worker 的预取数
  等于 `jobs-qblocks`，且 `hits+waits` 与消费数相等。

严格 A/B（其余优化均开启）：

| 负载 | F0 baseline cycles | F1 cycles | 降低 | checked/mismatch |
|---|---:|---:|---:|---:|
| E2 `S256,D64` | 29,198 | 26,944 | 2,254（7.72%） | 16,384 / 0 |
| E3 `S1024,D128` | 965,933 | 849,298 | 116,635（12.08%） | 131,072 / 0 |

- E2 全系统预取 112 tile：99 hit、13 wait；E3 预取 1,984 tile：1,984 hit、
  0 wait。E3 慢 worker 的 KV tile-path 从 145,408 降至 15,189 cycles，
  `inter_tile_pv_to_next_qk` 从 301,596 降至 182,163 cycles。
- E3 剩余最大阶段仍是 PV 五项合计 624,709 cycles；Softmax 为 31,950 cycles。
  因此 F1 已关闭，但这不等于已经实现 QK/Softmax panel overlap。
- Attention 83/83、SFU 22/22、multicore Softmax 12/12 通过；canonical
  `64x64x64` FP32 GEMM sampled=64、mismatch=0、max abs diff=0；`libgolem`
  build/install SHA-256 均为
  `5cc8c2ce945dc1f72f1ba4654efc4b26e3d21e5dc38be7d4ed6d3ff0799f2019`。
- artifact：
  `/data4/jjgong/tmp/fused_attention_e2_phase_f1_kv_double_buffer_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_phase_f1_kv_double_buffer_20260820`。

Phase F1 已关闭。下一步是 F2 可行性门：先审计 SFU 是否能进行 panel 级 online
`(m,l)` 更新，以及 PV/Oacc 是否允许后续 max 变化时重新缩放；在接口和数值语义
明确前不修改执行路径，也不运行 E4/E5。

### Phase F2 可行性结论（2026-08-20）

结论：**不实施独立的 QK-panel/Softmax 流水**。现有接口不能安全增量消费 panel，
而安全的最小 ABI 扩展收益不足以优先于 PV 路径。

1. RoCC 只有在两个 16-key QK panel 均计算并写完 `S[16,32]` 后，才调用一次
   `issueAttentionTile()`。`AttentionTileRequest` 只有整 tile 地址、rows/cols 和
   单个完成 callback，没有 panel ready/finalize/consume 协议。
2. SFU 虽以 16-lane chunk 访问 Local GM，但每行仍严格执行整 tile MAX、整 tile
   EXP/SUM、整 tile NORMALIZE 三遍；`tensorWorkerOps_` 非空时拒绝新请求，因此
   不能在同一 tile 内并存 panel producer 和 finalize operation。
3. online `(m,l)` 在完整 tile EXP/SUM 后才更新，并只返回每行
   `oldOutputScale`。若把 panel 当成伪 tile，panel 1 提高 running max 后，panel 0
   已写出的 P 权重需要重新缩放；除非 PV 同时按 panel 消费并重缩放 Oacc，否则
   数值语义错误。这会把 F2 强制扩展为高风险的 F2+F3 联合改造。
4. 安全的独立方案只能新增 `ingestMaxPanel()`/`finalizeAttentionTile()`：QK panel
   完成后让 SFU 提前做 MAX，最终 panel 后仍对整 tile执行 EXP/SUM 和 NORMALIZE。
   它需要 panel generation、buffer ownership、causal/partial 和 skip-MAX ABI，
   但只能隐藏 MAX 子阶段。

收益判断基于 F1 E3：

- 总周期 849,298；完整 Softmax 31,950 cycles，仅占 3.762%，这是 F2 的绝对上限。
- 由 2,048 行的 stage start/done tick 求和，MAX/EXP-SUM/NORMALIZE 服务时间为
  79,608/180,855/52,643 cycles。MAX 占三阶段服务时间 25.425%；用该比例估计，
  MAX-only overlap 约 8,123 cycles，即总周期约 0.956%。这是代理估计，不是新的
  SST 实测值。
- PV matrix/input/restore/compute/output 仍为
  165,376/196,608/63,488/133,120/66,117 cycles，其中单独 PV input 就占总周期
  23.15%，明显高于 F2 的全部理论空间。

因此 F2 以“已评估、不实施”关闭，没有修改生产代码，也没有启动新 SST。下一步
先设计低风险的 PV input 并行 issue/批量编程方案；F3 Softmax/PV panel overlap
保留为后续联合 ABI 研究，不作为当前实现目标，E4/E5 继续冻结。

### Phase G1 验收结果：PV input 两级流水（2026-08-20）

- 新增默认关闭的 `attention_pv_input_pipeline`。原路径逐行串行执行
  `P Local GM read -> Array input program -> wait`；新路径在当前行的 Array input
  请求被接受后，立即用独立的 Local GM read port 读取下一行，使第 `i+1` 行读取
  与第 `i` 行 Array 编程重叠。没有新增 Array batch API，也没有改变 P、PV 或 Oacc
  的数据布局和数值语义。
- `attentionPvInputsPending` 在每行 Local GM read 开始前递增，在对应 Array program
  callback 后递减，因此覆盖 read 和 program 两个阶段。只有所有行均已提交且
  pending=0，才进入 PV output restore；这避免“最后一行仍在读取、前一行 callback
  已耗尽”导致的提前完成竞态。
- 新增 `attention_pv_input_pipeline_rows`，verifier 要求开启时每 worker 精确为
  E2=512、E3=16,384，关闭时为 0。

严格 A/B（F1 的全部优化保持开启）：

| 负载 | F1 cycles | G1 cycles | 总周期降低 | PV input 降低 | checked/mismatch |
|---|---:|---:|---:|---:|---:|
| E2 `S256,D64` | 26,944 | 26,571 | 373（1.38%） | 4,096 -> 2,656（35.16%） | 16,384 / 0 |
| E3 `S1024,D128` | 849,298 | 799,873 | 49,425（5.82%） | 196,608 -> 150,528（23.44%） | 131,072 / 0 |

- E3 慢 worker tile path 为 779,473 cycles，11 阶段守恒、unattributed=0；流水
  行计数在 16 个 worker 上均为 16,384。总周期收益主要由 PV input 的 46,080
  cycles 降低解释，没有出现异常长 cycle。
- Attention 84/84、SFU row-engine 22/22、multicore Softmax 12/12 通过；canonical
  `64x64x64` FP32 GEMM sampled=64、mismatch=0、max abs diff=0。build/install
  `libgolem.so` SHA-256 均为
  `50c34631d1ad20d9021dfbbd2afcdac9344f046841da96a99c9fafae1d05bac1`。
- artifact：
  `/data4/jjgong/tmp/fused_attention_e2_pv_input_pipeline_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_pv_input_pipeline_20260820`、
  `/data4/jjgong/tmp/attention_pv_input_pipeline_gemm_20260820`。

Phase G1 已关闭。下一步 G2 优先评估“Softmax 与 PV V-matrix programming
overlap”：QK readout 完成后 Array 已不再执行 QK，而 SFU 正在生成 P，此时可提前
把已驻留 Local GM 的 V 编程到 PV Array。E3 中 Softmax=31,926 cycles、PV matrix
program=165,376 cycles，因此可隐藏量上限约 31,926 cycles（总周期 3.99%）。先审计
Array ownership、V buffer 生命周期和 SFU failure rollback，再实现默认关闭开关并做
E2/E3 严格 A/B；不恢复高风险 panel ABI，也不运行 E4/E5。

### Phase G2 验收结果：Softmax/PV matrix overlap（2026-08-20）

- 新增默认关闭的 `attention_pv_matrix_softmax_overlap`。SFU 接受整 tile Softmax
  后，RoCC 立即读取 active V tile 并编程首个 PV panel；后续 PV panels 保持原顺序，
  没有改变 SFU ABI、P layout 或 online `(m,l)` 语义。
- `attentionSoftmaxComplete` 与 `attentionPvMatrixComplete` join 两个异步操作。矩阵
  先完成记为 hit，SFU 先完成记为 wait 并只把矩阵剩余尾部计入 PV matrix 阶段。
  任一路失败仍 fail-fast；另一条晚到 callback 在 worker 已释放后直接返回。
- 关键路径统计仍互斥守恒：overlap 开始后继续计入 Softmax；SFU 完成时若矩阵未
  完成，才切换到 PV matrix；二者均完成后进入 PV input。

严格 A/B（G1 及此前全部优化保持开启）：

| 负载 | G1 cycles | G2 cycles | 降低 | checked/mismatch |
|---|---:|---:|---:|---:|
| E2 `S256,D64` | 26,571 | 26,408 | 163（0.61%） | 16,384 / 0 |
| E3 `S1024,D128` | 799,873 | 790,516 | 9,357（1.17%） | 131,072 / 0 |

- E2 全系统 128/128 tiles 为 hit。E3 为 2,048 tiles、411 hits、1,637 waits，hit
  率 20.1%；所有 worker 均满足 `tiles=hits+waits=128`。
- E3 慢 worker 的 PV matrix 尾部从 165,376 降至 116,340 cycles，减少 49,036；
  但 overlap 对共享 Local GM 施压，Softmax 关键路径归因从 31,926 增至 64,917。
  因此净收益远低于无争用理论上限 31,926 cycles，这是架构资源竞争而非统计遗漏。
- E3 manager local-complete skew 从 14,489 增至 16,400 cycles；生命周期顺序、
  11 阶段守恒和数值结果仍全部有效，没有出现异常长 cycle。
- Attention 85/85、SFU row-engine 22/22、multicore Softmax 12/12、canonical
  `64x64x64` FP32 GEMM 均通过。build/install `libgolem.so` SHA-256 均为
  `3aa9a5b3ee6827c8774d2ceac457af6fd7d4dc3fe0d38648376e1bd6b10ad3c4`。
- artifact：
  `/data4/jjgong/tmp/fused_attention_e2_pv_matrix_softmax_overlap_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_pv_matrix_softmax_overlap_20260820`、
  `/data4/jjgong/tmp/attention_pv_matrix_softmax_overlap_gemm_20260820`。

Phase G2 已关闭。下一步 G3 优先评估 PV output restore 两级流水：当前非首 key
tile 逐 array 串行执行 `Oacc Local GM read -> scale -> Array output write -> wait`，
可复用 G1 的 pending join，使下一 array 的 Oacc read 与当前 Array write 重叠。
E3 restore 当前为 63,488 cycles；先确认 `writeOutputAsync` 对不同 array 的独立
ownership，再用默认关闭开关做 E2/E3 A/B。E4/E5 继续冻结。

### Phase G3 验收结果：PV output restore 两级流水（2026-08-20）

- 新增默认关闭的 `attention_pv_restore_pipeline`。非首 key tile 原先逐 array 串行
  执行 `Oacc Local GM read -> scale -> writeOutputAsync -> wait`；新路径在当前 Array
  output write 被有界 buffer queue 接受后立即读取下一 Array 的 Oacc，使单 Local GM
  read port 与单 Array-buffer port 重叠，没有增加端口数或绕过传输延迟。
- `attentionPvRestoresPending` 在每行 read 开始前递增，在对应 output write callback
  后递减。只有全部行均已提交且 pending=0 才启动 PV compute，避免最后一次 read
  尚未完成时提前计算。首 key tile 不需要 restore，继续直接设置 overwrite mode。
- verifier 要求每 worker 的 `attention_pv_restore_pipeline_rows` 精确为 E2=448、
  E3=15,872；两个实测 artifact 的 16 个 worker 均满足。

严格 A/B（G2 及此前全部优化保持开启）：

| 负载 | G2 cycles | G3 cycles | 降低 | restore 前后 | checked/mismatch |
|---|---:|---:|---:|---:|---:|
| E2 `S256,D64` | 26,408 | 26,168 | 240（0.91%） | 1,792 -> 952 | 16,384 / 0 |
| E3 `S1024,D128` | 790,516 | 759,338 | 31,178（3.94%） | 63,488 -> 33,728 | 131,072 / 0 |

- E3 restore 减少 29,760 cycles（46.88%），基本解释 31,178 cycles 的总收益；
  慢 worker tile path=735,604 cycles，11 阶段 unattributed=0。manager local-complete
  skew 从 16,400 降至 14,680 cycles，没有发现异常 cycle 或新负载失衡。
- Attention 86/86、SFU row-engine 22/22、multicore Softmax 12/12、canonical
  `64x64x64` FP32 GEMM 均通过。build/install `libgolem.so` SHA-256 均为
  `ee725b2dff73acc7602c67e9b84876825f821bbcd7f1a47f009d182b369c7e92`。
- artifact：
  `/data4/jjgong/tmp/fused_attention_e2_pv_restore_pipeline_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_pv_restore_pipeline_20260820`、
  `/data4/jjgong/tmp/attention_pv_restore_pipeline_gemm_20260820`。

Phase G3 已关闭。下一步 G4 先做 PV output read/write 流水可行性审计：当前 E3
阶段为 70,658 cycles，路径为 `Array readOutputAsync -> Local GM write -> wait`。
只有证明下一 Array readout 可与当前 Local GM write 并行、且单 Local GM write port
有明确的 backpressure/join 语义后才实现；否则转向剩余的 PV input/matrix 资源
竞争分析。继续只跑 E2/E3，不运行 E4/E5。

### Phase G4 验收结果：PV output read/write 流水（2026-08-20）

- 审计确认 Array `readOutputAsync` 有界排队，GlobalMemory 有独立 Local write FIFO、
  单 write port、32-entry 总队列和完成 callback；但原 `attentionLocalWrite` 只有一份
  transfer state，不能承载多笔在途写。因此 G4 只在 PV output 路径直接提交固定
  64-byte Local write，不修改共享传输 helper，也不增加端口。
- 新增默认关闭的 `attention_pv_output_pipeline`。当前行写请求被真实 Local write
  FIFO 接受后立即发起下一 Array readout；`attentionPvOutputWritesPending` 在全部
  写回完成前阻止 panel/tile 前进。若队列拒绝，payload/address 保留并由 RoCC
  时钟重试，index/stat 不前进。
- 独立审查发现 `GlobalMemoryLocal` 允许同步 callback。最终实现改为提交前预建
  pending/index，拒绝时事务式回滚，并在 API 返回后重新验证 job/phase/panel，
  避免同步完成导致悬空 state。修复后 E2/E3 cycle 与修复前事件化结果完全一致。
- verifier 要求每 worker 的 `attention_pv_output_pipeline_rows` 精确为 E2=512、
  E3=16,384；实测 16 个 worker 全部满足。Local GM queue rejected 均为 0，worker
  queue high-water 为 17--18，低于深度 32。

严格 A/B（G3 及此前全部优化保持开启）：

| 负载 | G3 cycles | G4 cycles | 降低 | PV output read/write 前后 | checked/mismatch |
|---|---:|---:|---:|---:|---:|
| E2 `S256,D64` | 26,168 | 25,915 | 253（0.97%） | 2,621 -> 1,173 | 16,384 / 0 |
| E3 `S1024,D128` | 759,338 | 729,683 | 29,655（3.91%） | 70,658 -> 39,287 | 131,072 / 0 |

- E3 PV output read/write 减少 31,371 cycles（44.40%），基本解释 29,655 cycles
  的全局收益；慢 worker tile path=705,176 cycles，11 阶段 unattributed=0，没有
  异常长 cycle。manager local-complete skew 从 14,680 增至 19,046 cycles，记录为
  后续负载均衡风险，但不抵消总 cycle 收益。
- Attention 87/87、SFU row-engine 22/22、multicore Softmax 12/12、canonical
  `64x64x64` FP32 GEMM 均通过。最终 artifact：
  `/data4/jjgong/tmp/fused_attention_e2_pv_output_pipeline_syncsafe_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_pv_output_pipeline_syncsafe_20260820`、
  `/data4/jjgong/tmp/attention_pv_output_pipeline_gemm_20260820`。

Phase G4 已关闭。下一步 G5 先审计按 Array 提前启动 PV compute 的可行性：E3
剩余最大阶段为 PV input=150,528、PV compute=133,120、PV matrix=116,291 cycles。
只有证明每个 Array 的 matrix/input/restore ready 可以独立触发 compute，且不会破坏
output accumulate 与 Array-done join，才实现 input/compute overlap；否则转向 PV
matrix 与 Softmax 的共享 Local GM 竞争和 manager completion skew 分析。仍只跑
E2/E3，不运行 E4/E5。

### Phase G5 验收结果：按 Array 提前启动 PV compute（2026-08-20）

- 审计确认仅把现有“一次启动全部 Array”改成逐 Array 调用不会产生收益：16 个
  Array 原本已并行计算，关键路径仍由最后 ready 的 Array 决定。G5 因此新增默认
  关闭的 `attention_pv_early_compute`，把每个 Array 的 compute 启动点前移到其真实
  operand ready callback，而不是增加 compute 单元或缩短 MVM 固有延迟。
- 首个 key tile 在该 Array 的 P input programming callback 完成后启动；后续 key
  tile 在该 Array 的 Oacc restore `writeOutputAsync` callback 完成后启动。matrix
  仍使用现有 broadcast 路径，compute 仍是原 Array 模型。
- `attentionPvPreparationComplete` 作为全体 input/restore barrier。barrier 前允许已
  ready Array 计算，但不读取 output，避免 readout 与尚未完成的 input/restore 竞争
  单 Array-buffer port。barrier 后按 Array index 读取；若当前 Array 尚未完成则等待
  其 done callback。全部 Local GM output writes join 后才推进 panel/tile。
- QK Array-done 路径保持不变；`arraysPending` 与逐 Array pending 成对维护。verifier
  要求每 worker 的 `attention_pv_early_compute_arrays` 精确为 E2=512、E3=16,384，
  两个 artifact 的 16 个 worker 均满足。

严格 A/B（G4 及此前全部优化保持开启）：

| 负载 | G4 cycles | G5 cycles | 降低 | checked/mismatch |
|---|---:|---:|---:|---:|
| E2 `S256,D64` | 25,915 | 25,705 | 210（0.81%） | 16,384 / 0 |
| E3 `S1024,D128` | 729,683 | 699,750 | 29,933（4.10%） | 131,072 / 0 |

- E3 慢 worker tile path=678,049 cycles，阶段为 PV input=150,528、restore=33,728、
  compute=99,200、output read/write=46,683，11 阶段 unattributed=0。manager
  local-complete skew 从 19,046 降至 13,034 cycles；没有异常长 cycle。
- Attention 88/88、SFU row-engine 22/22、multicore Softmax 12/12、canonical
  `64x64x64` FP32 GEMM 均通过；GEMM sampled=64、mismatch=0、max abs diff=0。
- artifact：
  `/data4/jjgong/tmp/fused_attention_e2_pv_early_compute_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_pv_early_compute_20260820`、
  `/data4/jjgong/tmp/attention_pv_early_compute_gemm_20260820`。

Phase G5 已关闭。下一步 G6 是 PV operand delivery 瓶颈审计，而不是直接新增端口。
E3 剩余 PV input=150,528 cycles，且 PV matrix 仍占约 116k cycles；先拆分 Local GM
读取、Array-buffer 单端口占用和 callback/调度空洞，再评估双 bank matrix/input
programming 是否符合目标硬件，或是否存在不复制 matrix 流量的 operand fusion。
只有资源模型和收益上限成立才实现默认关闭的机制并做 E2/E3 A/B；E4/E5 继续冻结。

### Phase G6 验收结果：PV operand delivery 审计与负结果（2026-08-20）

- 审计确认 PV 的有效 `keyCols=32`，但 E3 Array 物理输入宽度为 128。每行 input
  原本搬运 512 B（9 buffer cycles），其中 96 个 padding 元素不参与 PV。matrix
  与 input 不能同时省略尾部，因为 QK 会在同一 Array 留下非零旧值；只压缩 input
  前缀是安全候选，完整 PV matrix 已将无效 K 行写零。
- 实验性 `attention_pv_compact_input` 把 PV input 改为只写 32 元素，不新增端口。
  E3 每 worker Array-buffer bytes 从 22,249,472 降到 15,958,016，transfer cycles
  从 403,712 降到 305,408；PV input 阶段从 150,528 降到 52,224 cycles，局部减少
  98,304 cycles（65.31%）。E2 PV input 从 2,656 降到 1,632 cycles。
- 端到端严格 A/B 未通过验收：

| 负载 | G5 cycles | compact-input cycles | 变化 | checked/mismatch |
|---|---:|---:|---:|---:|
| E2 `S256,D64` | 25,705 | 25,481 | -224（-0.87%） | 16,384 / 0 |
| E3 `S1024,D128` | 699,750 | 700,212 | +462（+0.07%） | 131,072 / 0 |

- E3 中更短的 PV 路径伴随 KV prefetch slack 坍缩：hit/wait 从 G5 的
  1,313/671 变为 338/1,646，全局 DMA strict 平均 RTT 从 3,522 增至
  4,257 cycles；慢核 KV-load 归因由约 15k 增至约 120k cycles，manager
  local-complete skew 从 13,034 增至 16,690 cycles。结果与更同步地进入下一 KV
  tile 后产生共享回包压力的解释一致，但 G6 当时没有 MemNIC response queue 直接
  计数，因此该解释尚不是闭合的唯一因果。11 阶段仍 unattributed=0。
- 按 cycle-first 边界，compact-input 生产代码、开关、统计和测试均已撤回，当前生产
  状态仍是 G5。负结果 artifact 保留：
  `/data4/jjgong/tmp/fused_attention_e2_pv_compact_input_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_pv_compact_input_20260820`。

Phase G6 以“已审计、实验拒绝”关闭。下一步 G7 审计共享 KV prefetch burst 和
worker 同步：按 HBM node/worker 对齐 KV 请求时间、queue wait 与 completion skew，
评估现有 credit 或确定性 worker-slot stagger 是否能平滑突发。只有先证明全局等待
下降，才把 compact input 作为配套机制重新做 E2/E3 A/B；不单独增加 Array 端口，
E4/E5 继续冻结。

### Phase G7 验收结果：KV prefetch slack 与 MemNIC response queue（2026-08-20）

- 仅增加观测，不改变调度。RoCC 记录每次 prefetch 的 issue、K/V 均 ready 和
  consume 时刻，导出 DMA latency、ready lead、consumer wait；verifier 强制
  `dma count=tiles`、`ready-lead count=hits`、`wait count=waits`。
- MemNIC highlink 记录 read response attempted/immediate/enqueued/drained、队列
  high-water、累计/最大 queue wait；结束时仅对有 DMA response 的组件输出，且
  E2/E3 的 pending 均为 0。
- G5 全开关 E2/E3 分别保持 25,705/699,750 cycles，输出检查 16,384/131,072，
  mismatch=0，证明统计没有扰动模拟事件时序。E3 由 300 秒 watchdog 保护，约
  115 秒完成；E4/E5 未运行。
- E3 的 1,984 次 prefetch 中 1,313 hit、671 wait；平均 DMA latency=4,286、平均
  ready lead=1,426、平均 consumer wait=176 cycles，最大 consumer wait=1,485。
  慢核 core19 为 85 hit/39 wait，累计 wait=6,366 cycles。所有累计时间跨并行
  worker，只用于压力归因，不等同于端到端 cycle。
- MemNIC 从 E2 的 38/272 response 入队（14.0%、high-water=3、平均等待约
  261 cycles）增长为 E3 的 3,485/4,160（83.8%、high-water=7、平均等待约
  1,798 cycles、最大 4,121）。这证明 G5/E3 已有显著 response backpressure，
  但旧 G6 artifact 没有同一队列统计，尚不能直接给出 compact-input 的队列增量。
- artifact：`/data4/jjgong/tmp/fused_attention_e2_g7_observe_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_g7_observe_20260820`。

Phase G7 以“观测链闭合、G6 增量因果待测”关闭。下一步 G8 仅在实验代码中恢复
compact-input 并保持 G7 统计，做一次 E2/E3 严格 A/B。若 response queue 与
prefetch wait 同时恶化，再以确定性 worker-slot stagger 做最小机制实验；若队列
不恶化，则转向 prefetch launch distance。不得先调 credit、增加端口或运行 E4/E5。

### Phase G8 验收结果：compact-input 因果 A/B（2026-08-20）

- 恢复默认关闭的 `attention_pv_compact_input`。Array `programInputAsync` 接受
  非空、且不超过物理宽度的 input prefix，并只覆盖该前缀；普通路径仍传完整
  vector。PV matrix 继续完整清零无效 K 行，所以 input 尾部保留不会影响结果。
- E2/E3 精确复现旧 G6 的 25,481/700,212 cycles，输出检查 16,384/131,072，
  mismatch=0；E3 受 300 秒 watchdog 保护，约 115 秒完成。E4/E5 未运行。
- E3 PV input 从 150,528 降到 52,224 cycles，但 prefetch hit/wait 从
  1,313/671 变为 338/1,646；平均 DMA latency 从 4,286 增至 4,981，平均实际
  consumer wait 从 176 增至 1,024 cycles。
- 同一次 A/B 中，MemNIC response 入队由 3,485/4,160（83.8%）增至
  3,809/4,160（91.6%），平均 queue wait 从 1,799 增至 2,375 cycles，
  high-water 从 7 增至 9，最大等待从 4,121 增至 5,487 cycles。结束时均
  drained=all、pending=0。
- Array 局部 bytes 22,249,472 -> 15,958,016、transfer cycles
  403,712 -> 305,408，证明局部优化兑现。与之同时发生的 prefetch slack 坍缩和
  MemNIC queue 放大解释了为何 98,304-cycle 局部减少没有形成端到端收益，最终反而
  +462 cycles。累计 queue/prefetch wait 跨并行 worker，不能直接加到总 cycle。
- E2 queue 基本不变（38 -> 39 个 response 入队，平均等待 261 -> 243 cycles），
  且端到端减少 224 cycles，确认同步/回包压力是 E3 规模下才显著的系统效应。
- artifact：`/data4/jjgong/tmp/fused_attention_e2_g8_compact_observe_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_g8_compact_observe_20260820`。

Phase G8 以“因果 A/B 闭合、compact 不进入默认配置”关闭。实验开关保留为默认
关闭，交付配置仍为 G5。下一步 G9 先审计并实现最小确定性 worker-slot phase
stagger，只在 compact 实验中做 E2/E3 A/B。验收要求 response queue wait 与
consumer wait 同时下降且 E3 总 cycle 优于 699,750；否则撤回 stagger，转向更早
prefetch launch。不得以增大 credit/queue 或新增端口掩盖问题，E4/E5 继续冻结。

### Phase G9 验收结果：worker-slot stagger 被拒绝（2026-08-20）

- 实验采用全局 worker slot × 128 RoCC cycles 的一次性启动延迟，仅在 compact
  开启时生效。开发中发现并修复了 cycle 与 SST timebase tick 的单位混用；第一次
  无效的逐 cycle 相同结果不进入性能决策。
- 修正后 E2=27,094、E3=702,693 cycles，输出检查 16,384/131,072，mismatch=0。
  相比 G8 分别退化 1,613/2,481 cycles，E3 也比 G5 慢 2,943 cycles。
- E3 response 入队 3,809 -> 3,746，平均 queue wait 2,375 -> 2,350 cycles；
  prefetch hit 338 -> 448，累计 consumer wait 下降 2.3%，说明错峰有轻微削峰效果。
  但平均 consumer wait 1,024 -> 1,072，manager skew 16,690 -> 18,502，关键路径
  没有兑现收益。
- 按预设三重门槛拒绝 G9，并撤回全部 stagger 生产代码和配置链；不做参数 sweep。
  artifact 保留在 `/data4/jjgong/tmp/fused_attention_e{2,3}_g9_stagger128_observe_20260820`。

Phase G9 以“假设部分成立、端到端失败、机制撤回”关闭。下一步 G10 先做 prefetch
launch-distance/双 buffer ownership 审计：确认首次 tile 的 K/V 子请求或后续 swap
前是否存在安全提前点。若现有两 buffer 已使后续 launch 最早，则转向 response
服务顺序或 rotation 映射，不能通过增加 buffer、credit、queue 或端口掩盖瓶颈。
compact 默认关闭，交付仍为 G5，E4/E5 继续冻结。

### Phase G10 验收结果：最早 launch 点与两级驻留候选（2026-08-20）

- 在现有 tile-atomic ownership 下，首 tile 在 K/V 均完成后立即 prefetch N+1；
  后续 tile 在 swap 后立即用另一个 buffer prefetch N+1。现有单级 prefetch 没有可
  通过移动调用获得的更早安全点。
- 首 tile 按 K/V 子请求分别提前最多覆盖 E3 的 64/1,984 次 prefetch（3.23%），且
  会在初始 load 未完成时增加共享 response 压力，拒绝实现。
- 存在同容量的两级驻留候选：当前 tile 的最后 K read callback 已返回、首 PV panel
  已把 V 复制到 `vPayload`，并且另一个 buffer 的 N+1 已 ready 时，可将旧 active
  buffer 用于 N+2。N+1 pending 时禁止发 N+2，避免延迟关键请求。
- 当前仅有一组 prefetch ordinal/tile/pending/ready/timing，不能表达两个未来 tile；
  若实现，必须改成按 buffer descriptor 管理。G8 core10 的非 KV-wait tile 时间约
  4,329 cycles，prefetch DMA 平均约 4,907 cycles，候选可能有 lead 收益，但目前
  缺少 V-release 时刻和 ready-at-release 覆盖率，不能直接进入机制实现。

Phase G10 以只读审计关闭，无生产代码和配置变化、无新增 SST。下一步 G11 先增加
release/ready-at-release/available-lead 观测并仅跑受 watchdog 保护的 compact E2/E3。
只有候选覆盖率与理论 lead 足以覆盖已测 consumer wait，才实现 per-buffer descriptor
两级驻留；否则转向 response 服务顺序/rotation 映射。不得增加 buffer、credit、
queue 或端口，E4/E5 继续冻结。

### Phase G11 验收结果：release-window 证据支持机制 A/B（2026-08-20）

- 新增 K/V release timing、N+1 ready-at-release、N+2 candidate 和 candidate-to-
  boundary lead 统计；eligibility 取 K/V 均释放与 N+1 ready 两事件中的较晚时刻。
  verifier 检查 release 精确计数、candidate 上限和 candidate/lead 守恒。
- 正式 compact E2/E3 为 25,481/700,212 cycles，与 G8 逐 cycle 相同；输出分别
  checked=16,384/131,072、mismatch=0，证明观测无调度扰动。
- E2 candidate=12/96（12.50%），平均/最大 lead=518/891 cycles。E3 candidate=
  305/1,920（15.89%），其中 ready-at-release=27、release 后 ready=278，平均/最大
  lead=1,099/2,784 cycles。
- E3 全部 worker 都有机会（每核 13--26）；慢核 core10 有 18 个 candidate，平均
  lead=1,091 cycles，对应 miss 平均 wait=992 cycles。累计 candidate lead=335,241，
  是累计 consumer wait=1,685,193 的 19.9%，足以进行机制实验，但不是端到端收益。
- Attention tests 92/92 通过；build/install `libgolem.so` SHA-256 均为
  `787fd479bf69872fae80b35dea40c1347918143dee4edde696782ba801e92dc2`。
- artifact：`/data4/jjgong/tmp/fused_attention_e{2,3}_g11_release_observe_20260820`。
  E2 `run_summary.csv` 首行来自一次漏开 PV broadcast 的无效探针，权威结果为当前
  lifecycle/stats 和最后一行 summary。

Phase G11 关闭。下一步 G12 实现默认关闭的 per-buffer descriptor 两级驻留，仅在
K/V 已释放、N+1 ready、N+2 存在时提前发出 N+2；N+1 pending 时禁止抢发。先验证
descriptor ownership、query-block 尾部、失败路径与无重复 DMA，再做 compact E2/E3
A/B。验收要求 E3 < 699,750 cycles，且 response queue、consumer wait、数值正确性
均不退化；否则撤回。不得增加 buffer、credit、queue 或端口，E4/E5 继续冻结。

### Phase G12 验收结果：两级 KV 驻留被拒绝（2026-08-21）

- per-buffer descriptor 机制在 K/V release 且 N+1 ready 后向旧 active buffer 发出
  N+2，不增加物理存储或传输资源；实际触发 E2 21 次、E3 538 次。
- 正式 compact E2=25,471 cycles，较 G11 25,481 降低 10；E3=706,255 cycles，
  较 G11 700,212 增加 6,043，较 G5 699,750 增加 6,505。两者输出均 mismatch=0。
- E3 累计 consumer wait 降低约 8.1%，但平均 miss wait 约 1,024 -> 1,107 cycles，
  manager skew 16,690 -> 21,431 cycles；慢核 KV 改善被 query/output DMA 延迟抵消。
- 最初 E2=25,481/E3=700,212 的两次运行加载了 build tree 中未同步的 G11 头文件
  副本，新增统计缺失，已判为无效探针，不参与性能结论。
- 按验收门槛撤回 descriptor、统计、参数与 runner 开关；最终交付仍为 G5，最终
  build/install SHA-256 均为
  `2fe2e09107e4b64e48e2ee827b2abcd562f5f4f893bfd2ffff1f6f0a57b30be8`。
- 有效 artifact：`/data4/jjgong/tmp/fused_attention_e2_g12_real_20260820`、
  `/data4/jjgong/tmp/fused_attention_e3_g12_real_20260821`。
- 撤回后的 G5 E2 复核为 25,705 cycles、mismatch=0，artifact：
  `/data4/jjgong/tmp/fused_attention_e2_g12_rollback_g5_20260821`。

Phase G12 关闭。下一步 G13 仅归因 G5/G12 的 manager skew、query/output DMA 与
response 服务顺序；先形成 demand-response 优先级的确定性契约，再决定是否修改
MemNIC。不得继续扩大 lookahead、增加 queue/credit/port 或运行 E4/E5。

### Phase G13 验收结果：G5/G12 关键路径只读归因（2026-08-21）

- 使用既有 G5/G7 与正式 G12 lifecycle/stats artifact 离线比较；未重新运行 SST，
  没有生产代码改动。
- E3 G12 从 G5 的 699,750 增至 706,255 cycles（+0.93%），数值检查仍为 0 mismatch。
- 关键核 inter-tile KV load 从 11,435 增至 112,392 cycles，query load 从 5,848
  增至 11,306，output DMA 从 10,958 增至 18,138；manager local-complete skew
  从 13,034 增至 21,431 cycles。慢核由 core19 迁移到 core10。
- 结论是 N+2 lookahead 改变了共享 MemNIC/HBM response 服务顺序，降低局部 KV
  等待但放大 query/output 竞争；不是 PV compute 的计算吞吐瓶颈。G12 因此不应继续
  增大 lookahead 距离。
- G13 关闭。下一步先写出 demand-response priority contract 并做 trace/replay 或
  最小 fake-queue 验证，再决定是否改 MemNIC；不得增加 queue、credit、port 或
  physical buffer，E4/E5 继续冻结，交付保持 G5。

G13 的第一步已完成：新增独立的 `demand_response_priority_contract.py` 与
`test_demand_response_priority_contract.py`。它定义 consumer → query → output →
prefetch 的固定等级、issue sequence/request ID tie-break 和有限 trace 的
exactly-once 检查；该模型不改变 MemNIC。下一步是把同一契约映射到可导出的真实
response trace，并用压力 trace 检查关键读不会被 N+2 prefetch 延迟；在此之前不改
MemNIC，也不运行 E4/E5。

补充的 E2 trace-only 运行表明当前 MemNIC 日志缺少 request-kind 元数据：272 个 read
response 中有 23 个排队、high-water=17，但仅能看到地址/长度/request ID，不能可靠
区分 query、output 和 prefetch。该运行因关闭 G5 开关得到 91,288 cycles，仅作日志
格式探针。下一步先增加非时序性的 kind 标记与 focused trace parser，再用完整 G5
配置复跑 E2；kind 标记验证前不修改 response arbitration。

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

### G13 当前完成点与下一步

- 已完成 DMA read semantic kind 的非时序性透传和真实 trace 验证。
- G5 E2：25,705 cycles，16,384 项检查，0 mismatch。
- 已补齐 `AttentionOutput` 和 `dma_kind_trace.py`；按唯一完成事件统计，Query/KV/KV
  prefetch/Output 为 16/32/224/16，Unknown=0。旧的 16/36/296 包含 enqueue 重复行，废弃。
- priority contract 尚未接入 MemNIC arbitration。下一步用有限压力 trace/replay 验证
  确定性排序、无死锁、exactly-once completion；通过后才评估最小化调度改动，
  不增加 queue/credit/port，E4/E5 保持冻结。

### G13 有限压力验证结果

- `dma_priority_replay.py` 已用真实 G5 E2 的 288 个唯一完成事件做三档到达压缩。
- 三档均完成 288/288、exactly-once、最终排空；queue high-water 为 65/263/286，
  prefetch 最大等待为 80/266/285 replay ticks。
- 已证明有限任务无饥饿；不声称严格优先级对无限高优先级流量具有公平性。
- 下一步允许实现最小 MemNIC arbitration A/B：先 E2，保持 25,705-cycle 基线和
  0 mismatch；通过后再决定是否运行 E3。不得增加物理资源，E4/E5 继续冻结。

### G13 MemNIC priority A/B 决策

- 默认关闭的 priority 开关已实现；只重排已有 retry queue，同级 FIFO，无新增资源。
- E2 off/on 均为 25,705 cycles，实际重排 0。
- E3 off/on 为 699,750/699,133 cycles，实际重排 6，均 131,072 checked、0 mismatch。
- 虽净改善 617 cycles（0.088%），但 KV load +3,676、prefetch wait +3,688，未稳定改善
  consumer 关键路径。
- 结论：仅保留默认关闭的实验开关，不进入正式 G5，不运行 E4/E5。下一步回到请求
  产生端寻找结构性方案，不继续 sweep response priority。

继续实现前依次阅读：

1. `src/sst/elements/golem/tests/small/muticore_softmax/PROJECT_HANDOFF.md`
2. 本文件
3. `findings.md`
4. `progress.md`
5. 权威实施计划的 4.5、4.7、Phase C 和 12.1 节
