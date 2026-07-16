# SFU 基于 SimpleNetwork 的 Reduction Transport 设计

## 目标

Phase 4B 当前已经接受 `distributed_reduction_transport=explicit_noc`，但其
typed reduction FIFO 会立即 drain 到进程内 shared reducer。本设计将这个立即
handoff 替换为通过现有 `GlobalMemoryImplement` `SimpleNetwork` 接口发送的
request/response event，同时保持 unified-job softmax 数学路径、Phase 4A
counter 和既有真实 SST smoke contract 不变。

## 范围

本切面只作用于 `distributed_reduction_transport=explicit_noc` 下、带有分布式
列切分的 `SFUJobOp::SOFTMAX_ROW` job。

- `shared` 保持默认 functional mode。
- `modeled_noc` 保持 Phase 4A 直接调用 shared reducer 的行为和 counter 位置。
- 不修改 primitive/batch softmax 路径。
- 不实现 online-softmax recurrence，也不做性能结论。
- 不影响既有 GEMM 路径；尤其是 `GOLEM_SFU_ENABLE=0` 时，不能注册 reduction
  handler、注入 reduction event、占用 NIC send queue，或改变 DMA/NoC 行为。

## 现有拓扑

每个 RoCC 同时拥有 SFU 和 `GlobalMemoryAPI` subcomponent。SFU 通过
`SFU::bindGlobalMemory` 绑定该 API。`GlobalMemoryImplement` 已经拥有本 core
的 `SimpleNetwork` endpoint，连接到 Merlin router，并负责网络 send queue 和
receive callback；SFU 自身没有独立的 network port。

因此，本设计复用既有的 per-core GlobalMemory NIC，而不为 SFU 新增 port 或
拓扑 component。

```text
worker SFU
  -> worker GlobalMemory transport bridge
  -> SimpleNetwork request event
  -> owner GlobalMemory transport bridge
  -> owner SFU reducer
  -> SimpleNetwork response event
  -> worker GlobalMemory transport bridge
  -> worker SFU response inbox
```

## Transport Contract

### Payload

在 GlobalMemory interface 层定义 transport-owned、可序列化的 reduction
payload，避免形成 `globalmemory.h <-> sfu.h` include cycle。payload 包含：

```text
kind: MaxRequest | MaxResponse | SumRequest | SumResponse
job_id, tag, owner_core, row, worker_slot
expected_workers, expected_rows, expected_cols
value
```

request 携带 local max 或 local sum；response 携带对应的 global value。
response 的目标 core 是 `owner_core + worker_slot`，因为现有 distributed
descriptor validation 已经要求该 physical mapping。payload 使用独立 SST
`Event` wrapper 做序列化，不复用 `NetworkDataEvent`，也不把它作为 DMA traffic。

### GlobalMemory Bridge

`GlobalMemoryAPI` 增加三个 reduction 专用方法：

```text
reductionNetworkAvailable() -> bool
sendReductionMessage(destination_core, payload) -> bool
setReductionMessageHandler(handler)
```

`GlobalMemoryImplement` 通过既有 endpoint mapping 解析 `destination_core`，
在配置的 reduction VN 上注入 `SimpleNetwork::Request`，并把收到的 reduction
event 分发给已注册的本地 handler。local-memory fallback 必须报告 unavailable；
`explicit_noc` 必须让 job 失败，不能静默回退为 shared transport。

第一版 transport 使用一个可配置的 VN；如果未设置 dedicated VN，则默认使用既有
request VN。VN 作为 parameter 暴露，后续可在不改变 reduction 语义的前提下选择
隔离 DMA 或刻意与 DMA 竞争。

### SFU State Machine

仅在 `explicit_noc` 下执行以下逻辑：

1. worker 将 typed request 入队、记录 request counter，并要求 GlobalMemory
   向 `owner_core` 发送它。
2. worker job 保持 `Pending`；`wait()` 查询 keyed SFU response inbox，而不再
   直接调用 shared reducer 的 ready function。
3. owner SFU 收到 request callback 后验证 cohort shape，并将 contribution
   提交给 existing shared reducer functional oracle。
4. 某个 row/stage 的最后一个 contribution 到达后，owner 获得 global value，向
   每个 worker slot 精确发送一次 response。
5. worker 收到 response 时记录 response counter，并从 max 推进到 sum，或从
   sum 推进到 normalize；每个 inbox entry 只能消费一次。

shared reducer 仍然提供 functional max/sum result。NoC 模拟的是 transport
ordering、traversal 和 queue effect，不替代浮点 reduction math。

### GEMM 隔离

GlobalMemory reduction bridge 默认处于 inert 状态。只有 SFU 已启用、SFU 已绑定
handler、且 job transport 明确为 `explicit_noc` 时，bridge 才创建或发送
reduction event。普通 GEMM 和 `GOLEM_SFU_ENABLE=0` 的配置不会创建 reduction
payload、不会进入 reduction send queue，也不会改变既有 DMA request VN 或
SimpleNetwork receive path对 memory/DMA event 的处理结果。

## 错误与清理语义

- send rejection 只有在 GlobalMemory send queue 已接管 request 时才让 job 继续
  Pending；NIC unavailable 或 destination 非法时，job 以 `InvalidDescriptor`
  失败，并走现有 distributed abort path。
- 收到的 message 若无法匹配 live job 的
  `(job_id, tag, owner_core, row, worker_slot)`，则作为 stale transport receive
  丢弃并计数；它不能在 abort 后重建 reducer state。
- duplicate request、cohort shape mismatch、duplicate/stale response 均复用现有
  reducer validation 和 terminal cleanup 规则。
- owner 每个 ready row/stage 只生成一次 response fanout；重复 `wait()` 不能再
  发 response 或再次增加 counter。

## 统计

保持 Phase 4A 四个 counter 名称不变：

```text
sfu_reduction_max_requests
sfu_reduction_max_responses
sfu_reduction_sum_requests
sfu_reduction_sum_responses
```

在 `explicit_noc` 下，request 在 SFU 到 GlobalMemory transport 成功接收时
计数；response 在 worker 首次将其投递到 inbox 时计数。新增 transport
observability：sent、received、stale-dropped、send-queued、send-rejected、inbox
high-water mark 和 request-to-response latency。它们用于区分 transport pressure
和 reducer correctness。

## 验证 Contract

首个真实 SST 点保持为：

```text
rows=16, dim=512, chunk=256, worker_cores=4, band_cores=4
transport=explicit_noc
```

必须保持：

- logits golden `checked=8192`、`mismatches=0`；
- 每类 max/sum request/response total 均为 `64`；
- active core0-core3 上每类值均为 `16`；
- DMA read/write issue 和 completion 均为 `64`；
- DMA retry、exhausted 和 write retry 均为 0。

该 smoke 通过后，运行既有 representative worker/band matrix，并单独报告
transport latency 和 queue metric，不将它们和数值、DMA correctness 混为一谈。

每次改动 GlobalMemory transport bridge 后，必须额外运行既有的 SFU-disabled
GEMM real-SST baseline。该回归的验收条件是：GEMM 正确性通过，原有 DMA
issue/completion 生命周期完整，且没有 reduction event、reduction counter 或
reduction queue 活动。任何 GEMM baseline 失败都阻止 Phase 4B transport 合入。

## 非目标

- 不新增直接连接 SFU 的 `SimpleNetwork` port 或 router attachment。
- 不实现 online-softmax `(m,l)` recurrence。
- 不修改 guest ABI 或 `SFUJobDesc` layout。
- 在 actual-event smoke 与 scaling matrix 均通过前，不做性能结论。
