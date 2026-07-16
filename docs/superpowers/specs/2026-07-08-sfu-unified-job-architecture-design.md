# SFU Unified Job Architecture Design

## 背景

当前 SFU 开发已经经历了多轮实验性接口：

- `SFUSoftmaxTileDesc`：早期 fused tile softmax 路径。
- `SFUPrimitiveDesc`：单个 primitive op，例如 `EXP`、`LOG`、`REDUCE_SUM`。
- `SFUPrimitiveBatchDesc`：把多个 primitive child desc 聚合为一次 batch issue。
- guest-side multi-core softmax：用 mailbox、remote write、primitive issue/wait
  在 RISC-V guest 侧拼出跨 core softmax。
- guest-side row pipeline / stage-machine：尝试把 local max、EXP/SUM、normalize
  拆成更多 stage，但真实 SST 结果显示 guest 指令和同步事件开销继续放大。

这些接口帮助验证了 SFU 数学路径、batch 默认路线、多核协同 softmax 和 timeout
瓶颈，但它们不适合作为长期架构继续扩展。继续为每个实验新增 ABI/API，会让
SFU 变成一组零散入口，而不是一个可扩展硬件单元。

因此，从本设计开始，SFU 正式收敛为一种统一提交模型：

```text
guest/RoCC 只提交 SFUJobDesc
SFU 内部根据 op_type 选择 executor
executor 内部管理 queue、stage、reduction、writeback
guest 通过统一 wait 获取 completion/status
```

softmax 是第一个正式 fused job。后续 `exp`、`log`、`gelu`、`layernorm` 等算子
必须复用同一套 job descriptor 和 issue/wait API。

## 设计决策

正式架构采用 **SFU Job Descriptor**。

唯一正式入口：

```text
RoCC:
  sfu_job(desc_addr, tag)
  sfu_wait(tag)

SFU API:
  issueJob(desc_addr, tag)
  wait(tag)
```

旧入口定位：

| 旧接口 | 后续定位 |
| --- | --- |
| `issueSoftmaxTile` / `SFUSoftmaxTileDesc` | legacy fused-softmax prototype |
| `issuePrimitive` / `SFUPrimitiveDesc` | legacy/debug primitive smoke |
| `issuePrimitiveBatch` / `SFUPrimitiveBatchDesc` | legacy/debug HBM streaming and ABI experiment |
| guest-side mailbox softmax | debug/reference implementation |
| guest-side stage-machine | negative experiment; 不作为继续优化方向 |

后续新增算子时，不再新增 RoCC func7 或新的顶层 API。只新增：

- `SFUJobOp` 枚举值；
- 对应 executor；
- 必要的参数结构或 descriptor extension。

## 目标

1. 建立一个长期稳定的 SFU job ABI，支撑 softmax 和后续其他 SFU 算子。
2. softmax 采用 SFU 内部 staged execution，不再依赖 guest 侧逐 stage mailbox。
3. 保持 multi-core cooperative softmax 路线：多个 worker core 分摊同一行的
   column slice，并在 SFU 内部完成跨 core reduction。
4. 减少 guest 可见的控制事件：目标从多次 primitive issue/wait + mailbox
   同步，收敛为一次 job issue + 一次或少量 wait。
5. 保留真实 SST 作为主验证路径；summary/timing-only 不作为主方案。
6. 为后续 `EXP`、`LOG`、`RECIPROCAL`、`GELU`、`LAYERNORM` 等算子提供统一扩展口。

## 非目标

- 不在本阶段实现 RTL 或周期精确数学硬件。
- 不在本阶段删除旧 primitive/batch 源码；先标记为 legacy/debug。
- 不继续优化 guest-side row-block、nbpoll 或 stage-machine 作为主线。
- 不为 softmax 再新增一个只服务 softmax 的临时 RoCC/API。
- 不把 non-batch 重新作为性能基线。

## 统一 ABI

第一版 `SFUJobDesc` 固定为 128 字节，保留扩展字段，避免后续每次新增算子都改
顶层 RoCC/API。

建议结构：

```cpp
enum class SFUJobOp : uint32_t {
    ELEMENTWISE = 0x01,
    REDUCE = 0x02,
    SOFTMAX_ROW = 0x10,
    LAYERNORM = 0x11,
    GELU = 0x12,
};

enum class SFUJobSubOp : uint32_t {
    NONE = 0x00,
    EXP = 0x01,
    LOG = 0x02,
    RECIPROCAL = 0x03,
    RSQRT = 0x04,
    TANH = 0x05,
    SIGMOID = 0x06,
    REDUCE_MAX = 0x20,
    REDUCE_SUM = 0x21,
};

struct SFUJobDesc {
    uint64_t job_id;
    uint64_t input0_addr;
    uint64_t input1_addr;
    uint64_t output_addr;
    uint64_t params_addr;
    uint64_t scratch_addr;
    uint32_t op_type;
    uint32_t sub_op;
    uint32_t dtype;
    uint32_t layout;
    uint32_t rows;
    uint32_t cols;
    uint32_t elem_count;
    uint32_t chunk_elems;
    uint32_t worker_cores;
    uint32_t owner_core;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t reserved1;
    uint64_t reserved2;
    uint64_t reserved3;
    uint64_t reserved4;
};
```

字段语义：

- `op_type`：决定 SFU executor 类型，例如 row-wise softmax、elementwise、reduce。
- `sub_op`：给 elementwise/reduce 类 job 指定具体数学操作。
- `rows/cols`：矩阵型或 row-wise op 的二维形状。
- `elem_count`：一维 elementwise/reduce 的元素数；二维 op 可设为 `rows * cols`。
- `chunk_elems`：SFU 内部流式处理粒度，不再等同于 guest issue 粒度。
- `worker_cores`：参与该 job 的 worker core 数。
- `owner_core`：coordinator / owner core，第一版默认 worker slot 0。
- `params_addr`：复杂算子的参数区，例如 layernorm epsilon、scale/bias 地址。
- `scratch_addr`：可选 scratch 起始地址；若为 0，由 SFU 使用内部/默认 scratch 策略。
- `flags`：控制 in-place、approx mode、debug stats、verification assist 等行为。

## RoCC / SFU API

正式新增一组通用 RoCC func7：

```text
GOLEM_ROCC_FUNC7_SFU_JOB
GOLEM_ROCC_FUNC7_SFU_JOB_WAIT
```

正式新增 SFU API：

```cpp
virtual bool issueJob(uint64_t descAddr, uint64_t tag) = 0;
virtual bool wait(uint64_t tag, uint64_t* status) = 0;
```

guest wrapper：

```cpp
void sfu_job(uint64_t desc_addr, uint64_t tag);
uint64_t sfu_job_wait(uint64_t tag);
```

旧 primitive/batch wrapper 可以保留，但后续新开发不再使用它们作为主路径。

## SFU 内部结构

SFU 内部按 job executor 组织：

```text
SFU
  job queue
  descriptor reader
  executor dispatch
    - ElementwiseExecutor
    - ReduceExecutor
    - SoftmaxRowExecutor
    - LayerNormExecutor
    - GeluExecutor
  reducer / scoreboard
  completion table
  statistics
```

统一执行流程：

```text
issueJob(desc_addr, tag)
  -> read SFUJobDesc
  -> validate common fields
  -> allocate JobState
  -> dispatch to executor by op_type
  -> executor advances internal stages
  -> completion table records status

wait(tag)
  -> if complete: return Success and retire
  -> if not complete: return Pending
  -> if failed: return error status and retire
```

第一版仍可使用 host C++ functional model 计算数学结果，但 stage、queue、status 和
统计要以硬件-like 的结构表达，避免 guest 侧手写复杂控制流。

## Softmax Job 语义

softmax 使用：

```text
op_type = SFUJobOp::SOFTMAX_ROW
sub_op = SFUJobSubOp::NONE
rows = softmax row count
cols = softmax dim
worker_cores = participating cores
chunk_elems = internal stream chunk
```

多核分工：

```text
worker slot i owns columns:
  col_begin = cols * i / worker_cores
  col_end   = cols * (i + 1) / worker_cores
```

SFU 内部 softmax stage：

```text
1. 每个 worker slice 读取 logits
2. 每个 worker 计算 local row max
3. SFU reducer 合并 global row max
4. 每个 worker 计算 exp(x - global_max) 和 local row sum
5. SFU reducer 合并 global row sum
6. SFU 计算 reciprocal / inv_sum
7. 每个 worker normalize 并写回 output
8. job completion
```

关键原则：

- guest 不再显式发布 local max/local sum mailbox。
- guest 不再对每个 row/stage 发 primitive batch。
- cross-core reduction 是 SFU job executor 的内部行为。
- `chunk_elems` 是 SFU 内部流式粒度，不是 guest API 数量。

## 后续算子扩展

### Elementwise

`EXP/LOG/RECIPROCAL/RSQRT/TANH/SIGMOID/GELU` 可作为 elementwise job：

```text
op_type = ELEMENTWISE
sub_op = EXP / LOG / ...
elem_count = N
```

executor 内部按 `chunk_elems` 分块读写，guest 只提交一个 job。

### Reduce

`REDUCE_MAX/REDUCE_SUM` 可作为 reduce job：

```text
op_type = REDUCE
sub_op = REDUCE_MAX / REDUCE_SUM
rows/cols 或 elem_count 描述 reduction 范围
```

该 executor 可被 softmax/layernorm 内部复用，也可暴露为独立 job。

### LayerNorm

LayerNorm 后续可以复用 reduce + elementwise：

```text
mean reduce
variance reduce
rsqrt(var + epsilon)
normalize
optional affine
```

只新增 `op_type = LAYERNORM`，不新增 RoCC API。

## 迁移计划

### Phase 1: ABI 和入口统一

- 新增 `SFUJobDesc`、`SFUJobOp`、`SFUJobSubOp`。
- 新增 `issueJob` API 和 RoCC `sfu_job/sfu_job_wait`。
- 新增 guest wrapper 和静态测试。
- 不删除旧接口。

### Phase 2: Softmax job smoke

- 新增 `GOLEM_SFU_JOB_SOFTMAX=1` 或替换当前正式 softmax path。
- 小点真实 SST correctness：
  `rows=2, dim=512, chunk=256, worker_cores=16, verify=1`。
- PASS 输出必须显示：
  `mode=sfu-job-softmax`、`op_type=SOFTMAX_ROW`、`worker_cores`、
  `chunk_elems`、`max_abs_diff`、`max_row_sum_error`。

### Phase 3: 将旧 softmax primitive 降级为 legacy/debug

- 文档中明确 `GOLEM_SFU_PRIMITIVE_SOFTMAX` 是 legacy/debug。
- 默认 softmax sweep 切到 job path。
- 旧 path 只用于数值对照和故障定位。

### Phase 4: 扩展 elementwise/reduce job

- 用统一 job path 重新实现 HBM streaming primitive benchmark。
- 旧 primitive/batch benchmark 保留为历史数据，不作为主线。

### Phase 4A: modeled reduction transport observability

- 已完成。
- `distributed_reduction_transport=modeled_noc` 仍使用 shared in-SST reducer
  作为 functional reference，但在 distributed softmax max/sum reduction 边界
  记录 message-equivalent request/response counters。
- canonical smoke:
  `tests/artifacts/sweeps/sfu_unified_job_modeled_noc_smoke_20260713_reviewfix`；
  `rows=16, dim=512, chunk=256, worker_cores=4, band_cores=4`；
  golden checked 8192、0 mismatch，四个 active SFU 的四类 reduction counters
  均为 16。

### Phase 4B: explicit NoC reduction transport

- 下一步执行计划：
  `docs/superpowers/plans/2026-07-13-sfu-unified-job-phase4b-explicit-noc-reduction.md`。
- 目标是把 Phase 4A 的 message-equivalent counters 映射到真实 SST NoC
  request/response event path，开始建模 reduction latency、contention 和队列压力。
- functional correctness 仍以 shared reducer golden behavior 为参考，直到 explicit
  event path 完成并通过同一组 real SST smoke。

## 测试计划

### 静态测试

- `SFUJobDesc` size 固定。
- RoCC 声明 `GOLEM_ROCC_FUNC7_SFU_JOB` 和 `GOLEM_ROCC_FUNC7_SFU_JOB_WAIT`。
- `SFUAPI` 声明 `issueJob`。
- wrapper 能导出 job softmax 相关 env。
- 新 softmax path 不调用旧 `sfu_primitive_batch` 作为主调度机制。

### 编译测试

```bash
make clean ARCH=riscv64
make ARCH=riscv64
```

### 真实 SST correctness

第一组：

```text
rows=2
dim=512
chunk=256
worker_cores=16
verify=1
timeout=240s
```

预期：PASS，数值误差与当前 multi-core softmax baseline 同量级。

### 真实 SST timeout probe

第二组：

```text
rows=16
dim=1024
chunk=256
worker_cores=16
verify=0
timeout=360s
```

预期：

- 若 PASS：记录 simulated time、wall time、DMA/SFU stats。
- 若 timeout：必须记录 emergency simulated time、DMA retry/exhausted、active worker
  read/write issue 分布，并判断是否仍为 guest overhead。

## 风险

### 风险 1: 一步实现完整 SFU 内部多核执行过大

处理：第一版只实现 softmax job smoke 的最小功能闭环，先验证统一入口和数值正确。

### 风险 2: 旧接口和新接口并存导致混乱

处理：文档、测试和 wrapper 输出必须明确：

```text
SFU Job path = official architecture path
primitive/batch/stage path = legacy/debug path
```

### 风险 3: SFU 内部仍是 functional model，被误解为 RTL

处理：汇报中明确当前是 SST architecture functional/timing model，不是 RTL。
数学计算可由 host C++ 完成，但 job queue、stage、reduction、status 是架构建模对象。

### 风险 4: 新 job path 小点正确但 rows=16 仍 timeout

处理：这仍有价值，因为它能隔离“统一 API 是否减少 guest overhead”。若仍 timeout，
下一步才进入 SFU 内部 timing model 和 queue/stall 统计，而不是回到 guest mailbox。

## 验收标准

1. 项目文档明确 SFU 未来只走统一 `SFUJobDesc` 正式架构。
2. softmax 第一个使用 `SFUJobDesc` 的 fused job。
3. 旧 primitive/batch/stage API 全部标记为 legacy/debug，不再作为新算子扩展方向。
4. 第一版 job softmax 能通过真实 SST 小点 correctness。
5. 后续新增算子只增加 `op_type/sub_op` 和 executor，不新增顶层 RoCC/API。

## 2026-07-09 Direct Row-Band Streaming Status

- 当前大维度 softmax 主线已经切到 unified SFU job direct row-major HBM：
  `GOLEM_SFU_JOB_SOFTMAX=1`、
  `GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM=1`、
  `GOLEM_SFU_PRIMITIVE_SOFTMAX=0`。
- direct path 不再一次搬完整 row-band；它按
  `GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS` 拆成 sub-job，执行
  `HBM direct load -> SFU_JOB SOFTMAX_ROW -> HBM direct store`。
- 1024 维默认稳定配置已由真实 SST canonical `stable` profile 验证：
  `dim=1024, band_cores=8, job_rows=8, chunk_elems=256,
  worker_cores=16, retry_ticks=320, max_retries=8`。
  对应 artifact：
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable_profile_real`。
- 该 1024 点 manifest 为 `PASS`，DMA summary 为
  `timeout_retry_sum=0, timeout_exhausted_sum=0,
  read/write completion=256/256`，控制台 verifier 为
  `checked=1048576,mismatches=0,max_abs_diff=2.23162767e-10`。
- 负实验边界保持为
  `dim=1024, band_cores=8, job_rows=8, retry_ticks=256, max_retries=8,
  expect=fail`。该点用于验证 DMA load guard 和 wrapper failure gate，而不是
  作为默认正确性回归。

## 2026-07-09 Scaling Direction

- 不再扩展 primitive/batch softmax 主线；后续大维度工作继续沿
  `SFUJobDesc + SFU_JOB SOFTMAX_ROW`。
- 下一阶段从 1024 基线扩展到 2048：
  先固定 `band_cores=8, chunk_elems=256, worker_cores=16`，优先尝试
  `job_rows=4`，再根据 DMA guard 结果决定是否提高到 `job_rows=8` 或放宽
  `retry_ticks`。
- 若 2048 出现 `DMA_LOAD_FAILED`，优先降低 per-sub-job 行数或调整 DMA retry
  window；不要回退到旧 primitive/batch softmax 路径。

### 2048 First Probe Result

- 首个 2048 direct row-major unified SFU job probe 已真实 SST PASS：
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_unified_job_direct_2048_probe_20260709_rt320_jr4_real`。
- 配置：
  `dim=2048, band_cores=8, job_rows=4, chunk_elems=256,
  worker_cores=16, retry_ticks=320, max_retries=8`。
- verifier：
  `checked=4194304,mismatches=0,max_abs_diff=1.13844124e-10`。
- DMA summary：
  `read/write completion=1024/1024, timeout_retry_sum=83,
  timeout_exhausted_sum=0`，simulated time `1.06516 ms`。
- 结论：
  该点证明 unified direct streaming 已扩到 2048；由于已有少量 retry，后续若要
  建立 2048 clean profile，应先比较 `retry_ticks=384/512`，再尝试提高
  `job_rows`。

### 2048 Clean Profile Candidate

- 2048 retry sweep artifact：
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_unified_job_direct_2048_retry_clean_20260709_real`。
- 固定：
  `dim=2048, band_cores=8, job_rows=4, chunk_elems=256,
  worker_cores=16, max_retries=8`。
- `retry_ticks=384` 和 `512` 均真实 SST PASS，均为
  `read/write completion=1024/1024, timeout_retry_sum=0,
  timeout_exhausted_sum=0`，simulated time 均为 `1.01116 ms`。
- 因此当前 2048 clean profile 候选为：
  `band_cores=8, job_rows=4, retry_ticks=384, max_retries=8`。
- 下一步优先测试 throughput-oriented variant：
  `band_cores=8, job_rows=8, retry_ticks=384, max_retries=8`，观察是否仍
  clean PASS。

### 2048 `job_rows=8` Boundary

- `job_rows=8` throughput-oriented variant has been tested at
  `retry_ticks=384` and `512` with `max_retries=8`。
- Both points failed through the direct DMA load guard, not through softmax
  numerical verification:
  - `384/8`: `timeout_retry_sum=541, timeout_exhausted_sum=32`。
  - `512/8`: `timeout_retry_sum=861, timeout_exhausted_sum=32`。
- The failing sub-job load is `sub_job_rows=8, bytes=65536` per executor.
- Current 2048 clean default remains:
  `band_cores=8, job_rows=4, retry_ticks=384, max_retries=8`。
- Any future `job_rows=8` work should be treated as a pressure/throughput
  boundary experiment, not as the default profile, unless a wider retry policy
  such as `retry_ticks=1024` or `max_retries=16` proves clean.

### Stable2048 Profile

- The 2048 clean candidate is now solidified in
  `run_sfu_unified_job_direct_sweep.sh` as
  `GOLEM_SFU_JOB_DIRECT_PROFILE=stable2048`:
  `band_cores=8, job_rows=4, retry_ticks=384, max_retries=8`。
- The default `stable` profile now runs `stable512`, `stable1024`, and
  `stable2048`。
- Canonical real-SST artifact:
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable2048_profile_real`。
- Verification:
  `PASS checked=4194304 mismatches=0 max_abs_diff=1.13844124e-10`。
- DMA evidence:
  `timeout_retry_sum=0, timeout_exhausted_sum=0,
  read/write completion=1024/1024`。
- Timing:
  `simulated_time=1.01116 ms, wall_time_sec=226`。
- This makes 2048 part of the stable unified-job direct row-major HBM baseline.
  The next dimension-scaling step should start from this profile and probe
  4096, while `job_rows=8` remains a separate pressure experiment.

### 4096 First Probe

- The first 4096 unified-job direct row-major probe has passed real SST.
- Configuration:
  `dim=4096, band_cores=8, job_rows=2, chunk_elems=256,
  worker_cores=16, retry_ticks=384, max_retries=8,
  mem_node_size=268435456`。
- `job_rows=2` keeps each direct DMA sub-job load at `32768B`, matching the
  2048 clean profile's `job_rows=4` load size.
- 4096 requires a larger HBM backing than the previous stable profiles:
  row-major input and output are each `64MiB`, so the run uses 256MiB per memory
  node. Without that, layout/capacity failure would precede the SFU job path.
- Canonical artifact:
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_unified_job_direct_4096_probe_20260709_rt384_jr2_mem256_retry1_real`。
- Verification:
  `PASS checked=16777216 mismatches=0 max_abs_diff=5.72476014e-11`。
- DMA evidence:
  `timeout_retry_sum=0, timeout_exhausted_sum=0,
  read/write completion=4096/4096`。
- Timing:
  `simulated_time=3.16213 ms, wall_time_sec=745`。
- This probe promotes 4096 to a correctness anchor for the unified-job direct
  path. A future `stable4096` profile should make the 256MiB mem-node
  requirement explicit or derive it automatically from the dimension.

### Stable4096 Profile

- The 4096 clean point is now solidified in
  `run_sfu_unified_job_direct_sweep.sh` as
  `GOLEM_SFU_JOB_DIRECT_PROFILE=stable4096`。
- Configuration:
  `dim=4096, band_cores=8, job_rows=2, chunk_elems=256,
  worker_cores=16, retry_ticks=384, max_retries=8,
  mem_node_size=268435456`。
- `run_point` accepts an optional per-point `mem_node_size` argument. The
  default remains `134217728`; `stable4096` passes `268435456` explicitly.
- The default `stable` profile remains 512/1024/2048 only. 4096 is a separate
  canonical profile because it is a long run and needs larger HBM backing.
- Canonical real-SST artifact:
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_stable4096_profile_real`。
- Verification:
  `PASS checked=16777216 mismatches=0 max_abs_diff=5.72476014e-11`。
- DMA evidence:
  `timeout_retry_sum=0, timeout_exhausted_sum=0,
  read/write completion=4096/4096`。
- Timing:
  `simulated_time=3.16213 ms, wall_time_sec=721`。
- This makes 4096 part of the verified unified-job direct row-major HBM
  baseline, while keeping routine `stable` regression cost bounded.

### 4096 Job-Rows=4 Pressure Result

- Pressure profiles keep `dim=4096, band_cores=8, job_rows=4,
  chunk_elems=256, worker_cores=16, max_retries=8,
  mem_node_size=268435456` and sweep the DMA read retry window.
- `retry_ticks=384` and `512` both fail before verification through
  `DMA_LOAD_FAILED direct row-major sub-job`: the larger 64KiB sub-job read
  burst exhausts the retry window and the guest guard sees stale local GM.
- `retry_ticks=1024` passes real SST:
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_unified_job_direct_sweep_20260709_pressure4096_jr4_rt1024_real`。
- Verification:
  `PASS checked=16777216 mismatches=0 max_abs_diff=5.72476014e-11`。
- DMA evidence:
  `timeout_retry_sum=0, timeout_exhausted_sum=0,
  read/write issue=4096/4096, read/write completion=4096/4096`。
- Timing:
  `simulated_time=3.05433 ms, wall_time_sec=700`。
- Conclusion:
  `job_rows=4` is viable for 4096 direct row-major unified SFU job softmax,
  and the first clean retry-window point was `retry_ticks=1024`, wider than the
  `stable4096` correctness baseline. The refinement below supersedes that
  initial minimum-known value.

### Retry-Window Refinement at 768

- `pressure4096_jr4_rt768` passes real SST in
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_unified_job_direct_sweep_20260710_pressure4096_jr4_rt768_real`。
- Verification:
  `PASS checked=16777216 mismatches=0 max_abs_diff=5.72476014e-11`。
- DMA evidence:
  `timeout_retry_sum=0, timeout_exhausted_sum=0,
  read/write issue=4096/4096, read/write completion=4096/4096`；the maximum
  observed active-core RTT is `688` ticks.
- Timing:
  `simulated_time=3.05433 ms, wall_time_sec=692`。
- The minimum known clean retry window is therefore reduced from 1024 to 768.
  The next zero-retry boundary probe should be 704 ticks.

### Retry-Window Refinement at 704

- `pressure4096_jr4_rt704` passes real SST in
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_unified_job_direct_sweep_20260713_pressure4096_jr4_rt704_real`。
- Verification:
  `PASS checked=16777216 mismatches=0 max_abs_diff=5.72476014e-11`。
- DMA evidence:
  `timeout_retry_sum=0, timeout_exhausted_sum=0,
  read/write issue=4096/4096, read/write completion=4096/4096`；maximum
  observed RTT remains `688` ticks.
- Timing:
  `simulated_time=3.05433 ms, wall_time_sec=738`。
- 704 is now the minimum known zero-retry clean window. The next probe should
  test 688-tick equal-boundary behavior or 696 ticks with explicit headroom.
