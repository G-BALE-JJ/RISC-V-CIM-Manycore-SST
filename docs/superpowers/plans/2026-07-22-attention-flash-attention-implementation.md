# Attention / FlashAttention Implementation Plan

**Status:** Phase A/B/C/D and Phase E1/E2/E3/E4 are implemented and verified. The
current fused path supports a `B1,H1,S2048,D128` mapping across 4 managers and
16 workers, block-stripes Q/O/K/V over four HBM nodes, streams K/V tiles through
per-Core GlobalMemory, keeps S/P out of HBM, and emits exactly one root tensor
completion after four unique manager bands. Phase E remains open for the
`S4096,D128` scale point.

**Date:** 2026-07-22

**Goal:** Directly establish a correct, measurable FP32 prefill Attention SST
baseline from the existing GEMM and causal Softmax paths, then implement a
tiled fused FlashAttention path that does not materialize the score or
probability matrix in HBM.

**Decision sources:**

- `src/sst/elements/golem/tests/small/muticore_softmax/PROJECT_HANDOFF.md`
- `docs/superpowers/specs/2026-07-16-sfu-softmax-row-engine-noc-architecture-design.md`
- `docs/superpowers/plans/2026-07-17-sfu-softmax-row-engine-noc-implementation.md`

**Approved revisions on 2026-07-22:**

- do not create a standalone CPU Attention development phase; start with the
  real SST workload and retain only a small post-run mathematical verifier;
- add transpose support in the first phase without a standalone SST component
  or full transposed-HBM materialization; Phase A inspection established that
  native K rows already match the existing RoCC vector-loader contract, so a
  new RoCC-local reorder state machine is unnecessary unless later packing
  constraints require local reordering;
- place standard Attention scaling inside the SFU Attention Row Engine: reuse
  the existing RSQRT numerical semantics to derive `1/sqrt(D)` once per job,
  without issuing a generic primitive or creating a GM scalar round trip, then
  fuse scalar multiplication and masking into the score pass before row MAX;
- treat non-causal, causal, partial-tile, and extreme-logit cases as a project
  validation matrix derived from operator semantics, tiling boundaries, and
  numerical stability, not as a formal industry conformance suite.
- retain materialized Attention as the correctness/traffic baseline, but do
  not scale its S/P-to-HBM dataflow as the production path. After the accepted
  S64 pair, move directly to a manager-coordinated, worker-RoCC-executed fused
  kernel that passes score and probability tiles through reserved per-Core
  GlobalMemory windows rather than HBM.
- use the existing per-Core `golem.GlobalMemory` as the only tile-scale local
  storage for fused Attention. Reserve an Attention window in each worker's
  GlobalMemory; do not add a second RoCC-owned 64 KiB byte array or a separate
  scratchpad component in the initial design;
- model SFU arithmetic state as a bounded internal Context Register File plus
  lane-sized FIFO/register buffers. Scalars such as running `(m,l)`, inverse
  sum, scale, mask boundary, and context state remain in SFU registers; full
  score/probability rows or tiles reside in per-Core GlobalMemory;
- use dedicated manager cores as the new Softmax/Attention control plane and
  worker RoCC/SFU/arrays as the data plane. Keep the accepted worker-core-0
  Softmax coordinator path as a regression baseline until the manager path is
  independently accepted;
- implement the new coordinator as a manager-core RoCC control FSM, not as an
  SFU arithmetic context and not as a new standalone SST component initially.
  Worker SFUs execute rows/tiles; the manager SFU datapath is unused;
- replace implicit `band % worker_count == physical_core_id` assumptions with
  an explicit manager-owned worker-slot-to-core-ID map before enabling manager
  coordination;
- treat zero-time C++ vectors, direct component-to-component copies, static
  cross-core maps, and fixed byte-independent transfer delays as model audit
  findings. No fused performance result is accepted until every storage and
  transfer on its critical path has bounded capacity and modeled timing.

## 1. Starting Point

The accepted production baseline is:

- FP32 GEMM through the existing RoCC/MVM/GlobalMemory path;
- FP32 `1024x4096` row-wise Softmax with `4,194,304 / 0` golden
  checked/mismatched values;
- Softmax completion at `66,958` accelerator cycles, using the causal chain
  `input DMA -> MAX -> EXP/SUM -> NORMALIZE -> output DMA ACK -> 16 unique
  band completions`;
- one tensor Softmax job, 16 physical SFUs, four row contexts per SFU, and
  four-node band-striped HBM.

The `66,958` result is the frozen pre-refactor baseline. Adding real Local GM
service time can legitimately change the rerun cycle count. Acceptance then
requires unchanged numerical output and lifecycle plus a complete attribution
of the delta; the historical result must not be silently overwritten or
reported as the post-refactor completion.

The current Softmax is a stable three-stage whole-row implementation. It is
not the online running-max/running-sum algorithm required by FlashAttention.
Phase A has added the required logical `transpose_b` path while keeping K in
its native `[Skv,D]` row-major layout. Phase B has proved the three-stage
materialized composition, including its unnecessary S/P HBM round trips. The
next implementation boundary is therefore local-tile handoff and online
state, not a larger materialized run.

## 2. Scope and Non-Goals

The first delivery targets inference-time prefill:

- FP32 only;
- batch `B=1`, query heads `Hq=1`, KV heads `Hkv=1` initially;
- self-attention with `Sq=Skv` initially;
- both non-causal and causal masks;
- head dimensions `D=64` and `D=128`;
- no dropout, backward pass, training, ALiBi, sparse attention, or paged KV
  cache;
- no BF16/FP16, GQA/MQA, multi-batch, or decode kernel until the fused FP32
  prefill path passes.

RoPE is outside the operator: Q and K are assumed to have already been
projected and position-encoded. Scaling by `1/sqrt(D)` is part of the
Attention numerical contract and is executed by the SFU, not pre-applied to Q
or computed by the host verifier/runtime.

## 3. Acceptance Contract

Every accepted real SST point must satisfy all of the following:

1. The complete SST output tensor passes a small, host-side post-run
   mathematical verifier. The verifier is an oracle only: it does not model
   SST timing or reproduce the fused state machine. Start with the existing
   FP32 tolerance (`atol=1e-5`, `rtol=1e-4`); any change requires recorded
   numerical evidence.
2. Causal mode never includes a key position greater than the global query
   position. Partial query/key tiles must be covered by output verification.
3. Baseline mode executes the visible chain
   `QK^T GEMM -> SFU RSQRT(D) once -> SFU SCALE/MASK+MAX -> EXP/SUM ->
   NORMALIZE -> PV GEMM -> output ACK`.
4. Fused mode executes the causal chain
   `descriptor -> Q/K/V DMA ready -> QK array completion -> score-local-ready
   -> SFU RSQRT/SCALE/MASK+online update -> weight-local-ready -> PV array
   completion -> O-accumulator update -> final normalize -> output DMA ACK ->
   unique query-block completion -> job completion -> guest wait return`.
5. Fused mode writes no score or probability tile to HBM, including temporary
   or debug spill. S/P HBM read and write byte counters must all be zero.
6. DMA issue/completion counts and bytes must match the declared tensor/tile
   layout. Retry exhaustion, stale completion, duplicate completion,
   local-window overflow, and contract failure counts must be zero.
7. Accelerator latency is measured from descriptor acceptance to actual
   completion. Analytical compute estimates remain separately labelled and
   cannot be reported as completion.
8. Baseline and fused runs use identical Q/K/V inputs, shape, mask, clock,
   array/SFU parameters, HBM topology, and verifier tolerance.
9. Existing GEMM and Softmax results remain valid. Shared-component changes
   require focused tests, a `libgolem` rebuild, a Softmax smoke, and the
   canonical GEMM regression.
10. Attention RSQRT/SCALE executes as an internal Attention Row Engine stage.
    It must not call `issuePrimitive()`, create an `SFUPrimitiveDesc`, or move
    the derived scalar through GM. Generic SFU primitives and legacy Softmax
    behavior remain unchanged.
11. Fused tile data uses a controller-reserved window in the worker's existing
    per-Core GlobalMemory. Every local read/write is asynchronous and accounts
    for bytes, latency, ports, queueing, and completion identity. Same-core
    accesses generate no NoC packet; remote-core accesses still use the NoC.
12. SFU Context Register File and lane buffers have explicit capacities. A
    full row/tile cannot remain in an unbounded C++ vector between MAX,
    EXP/SUM, NORMALIZE, or online-update stages.
13. A manager coordinator only dispatches work, tracks dependencies, and
    aggregates unique completions. It does not read, copy, or store worker S/P
    data. Worker-local RoCC owns the array/SFU/GlobalMemory execution sequence.
14. A reported fused completion must include all local-memory and array-buffer
    transfers on the causal path. Fixed delays that do not scale with bytes,
    direct `memcpy` handoffs, and process-global reducer maps are forbidden on
    an accepted performance path.

### 3.1 Validation matrix provenance

There is no claim that the following four cases form a formal industry test
standard. They are the minimum project validation matrix for distinct failure
classes:

| Case | Source | Required execution level |
| --- | --- | --- |
| non-causal | standard Attention operator semantics | small end-to-end SST case |
| causal | autoregressive Attention mask semantics | small end-to-end SST case |
| partial tile | boundary condition created by tiled kernels | non-divisible SST shape plus focused address/mask tests |
| extreme logits | stable and online Softmax numerical requirements | focused SFU/online-state test; add one small SST case when the fused loop exists |

Exact shapes, seeds, and extreme values are project-defined. The categories
are retained because semantic, boundary, and numerical bugs are independent;
they do not all need to be expensive large SST runs.

No speedup target is fixed before the first matched baseline/fused pair.
The initial performance gate is factual: fused HBM traffic for S/P must be
eliminated, fused completion must not regress against the matched materialized
baseline, and the measured bottleneck must be identified from component
statistics.

## 4. Operator and Data Contracts

### 4.1 Mathematical contract

For each batch/head:

```text
S = (Q K^T) / sqrt(D)
P = softmax(S + mask)
O = P V
```

The fused path keeps an unnormalized output accumulator and per-query-row
online state. For each key/value tile:

```text
m_new = max(m_old, row_max(S_tile))
alpha = exp(m_old - m_new)
P_tile = exp(S_tile - m_new)
l_new = alpha * l_old + row_sum(P_tile)
O_acc_new = alpha * O_acc_old + P_tile * V_tile
```

After the final key/value tile, `O = O_acc / l`.

For both paths, the SFU derives `inv_sqrt_d = rsqrt(float(D))` once per
Attention job and applies `S_tile = S_tile_raw * inv_sqrt_d` before masking and
row MAX. It must not evaluate SQRT/RSQRT once per score element.

### 4.2 Initial layout

- Q: `[B, Hq, Sq, D]`, row-major and unscaled. The SFU derives the standard
  Attention scale from descriptor field `D`; generated inputs and the host do
  not pre-scale Q.
- K: `[B, Hkv, Skv, D]`, native row-major. `QK^T` is expressed through the
  shared `transpose_b` layout contract; a full `[B,Hkv,D,Skv]` HBM copy is
  forbidden.
- V: `[B, Hkv, Skv, D]`, row-major/packed for the second GEMM.
- O: `[B, Hq, Sq, D]`.
- Baseline-only S/P: `[B, Hq, Sq, Skv]`.

The layout manifest is the single source used by input generation, guest
addressing, HBM unpacking, post-run verification, and result parsing. Fused
mode may change the internal block packing but must consume the same logical
Q/K/V tensors.

### 4.3 Transpose architecture

The implemented mechanism is a shared `transpose_b` tensor/layout contract.
For `QK^T`, logical `B(k,n)` maps to native `K(n,k)`. Each contiguous native K
row is exactly one B column vector in the existing packed HBM representation,
so the existing RoCC batch vector loader and its completion identity are reused
unchanged. No transpose buffer, new opcode, or additional modeled state is
needed, and a full `K^T` is never written to HBM.

If a later physical packing constraint requires local reordering, add a banked
tile buffer inside RoCC operand loading and model its cycles, stalls, bytes,
and occupancy before considering a separate component.

Reconsider a standalone SST transpose component only if measurements or a
later architecture introduce an independently shared unit with its own queue,
bandwidth, clock, or arbitration across multiple operators/controllers.

### 4.4 Attention scaling architecture

Standard Attention scaling is a dedicated stage of the SFU Attention Row
Engine. On job acceptance it validates `D > 0`, computes
`inv_sqrt_d = RSQRT(float(D))` once, and retains the scalar in job/context
state. The stage reuses the numerical semantics of `SFUPrimitiveOp::RSQRT`,
but it does not call `issuePrimitive()`, create an `SFUPrimitiveDesc`, or
read/write the scalar through GM. The score datapath multiplies each raw QK
result by this scalar before causal masking and row MAX. SCALE/MASK shares the
score-read/MAX pass and must not create another full S read/write pass through
HBM.

Model one configurable RSQRT latency per Attention job and the vector multiply
throughput consumed by valid score elements. Emit separate RSQRT-ready and
SCALE/MASK start/done statistics so neither work nor latency disappears into
functional host code. Focused tests must also prove zero generic primitive
issues and zero GM scalar transfers on the Attention path. The initial standard
mode accepts `D`, not an arbitrary caller-provided scale; custom scaling is
deferred until a real operator needs it. Generic SFU primitives and legacy
Softmax jobs retain their guest-visible ABI and numerical semantics; their
internal execution may be routed through the bounded asynchronous scheduler
required by Section 4.8.

### 4.5 Initial FlashAttention tile and per-Core GlobalMemory budget

Start with `Br=16`, `Bc=32` and parameterize both values. For `D=128`, FP32
Q/K/V/S/O-accumulator plus `(m,l)` storage is 51,328 bytes (50.125 KiB) when
the score buffer is overwritten in place by unnormalized weights. For `D=64`
the corresponding requirement is 26,752 bytes (26.125 KiB). These bytes are
allocated from a controller-reserved Attention window in each worker's
existing per-Core GlobalMemory. They are not a second SFU or RoCC scratchpad.

The window contains aligned Q, K, V, S/P ping-pong, O-accumulator spill, and
descriptor/context-spill regions. Small live `(m,l)`, inverse-sum, scale, mask,
and stage fields remain in the bounded SFU Context Register File and are not
charged as tile SRAM unless explicitly spilled. Descriptor validation must
calculate the exact GlobalMemory reservation, charge metadata/alignment, and
reject overlap with GEMM slots, mailboxes, DMA flags, or another context.

The controller selects the reserved window from the local GlobalMemory layout;
the guest does not supply an arbitrary scratch address. The first prototype
has one context and no double buffering. Do not add more contexts until local
port occupancy and overlap statistics show a need.

### 4.6 ABI direction

Keep the existing 128-byte `SFUJobDesc` and all existing GEMM/SFU opcodes
unchanged. Add a separately versioned `GolemAttentionDescV1`, owned by the
RoCC fused controller, containing at minimum:

- magic, version, and byte size;
- Q/K/V/O base addresses and byte strides;
- `B`, `Hq`, `Hkv`, `Sq`, `Skv`, and `D`;
- `Br`, `Bc`, mask mode, layout identifier, operand transpose flags,
  and logical leading dimensions;
- HBM node mask/stride, manager/group identity, explicit worker-map identity,
  owner/context identity, and completion address.

Reserve distinct Attention issue/wait operation identifiers. The manager
validates the tensor job and dispatch map; each worker-local controller derives
its reserved per-Core GlobalMemory window from core/context identity and the
configured layout. The descriptor never supplies an arbitrary scratch base.
Do not reinterpret legacy Softmax flags or reserved fields. Unknown versions,
layouts, dtypes, masks, shapes, worker maps, and unsafe address ranges must fail
before any DMA or compute is issued.

### 4.7 Fused ownership and local dataflow

The manager-core RoCC control FSM owns tensor-level coordination: descriptor
validation, group selection, explicit worker-slot mapping, query-block
dispatch, phase dependency tracking, and unique completion aggregation. The
manager does not execute QK/PV/SFU arithmetic and never reads or copies S/P
data. The manager SFU datapath is not a worker and receives no row/tile
dispatch.

Each worker-local RoCC is the fused tile executor, not the arithmetic
implementation. It sequences the MVM arrays, SFU, DMA engine, and that worker's
existing GlobalMemory. It owns tile-loop state, callback identity, allocation
inside the reserved local Attention window, and query-block completion. This
two-level split prevents worker 0 from becoming both global coordinator and a
contended datapath while preserving local ownership of all large data.

The SFU gains a narrow asynchronous local-tile API rather than another
guest-visible Softmax descriptor. It reads score chunks from per-Core
GlobalMemory, applies once-per-job RSQRT, scale, mask, row max, exponentiation,
and online `(m,l)` update, then writes weight chunks to the same local window.
Only bounded scalar/context state and the currently serviced lane chunk remain
inside SFU registers. After the arrays compute `weights * V_tile`, the SFU
vector resource models `O_acc = alpha * O_acc + delta_O` and final `O_acc/l`
normalization; O accumulation stays in array accumulators when possible and
spills to the reserved local window otherwise.

```text
manager dispatch
  -> worker Q/K/V DMA into per-Core GlobalMemory
  -> Local GM -> QK array -> Local GM S buffer
  -> Local GM S -> SFU registers/lanes -> Local GM P buffer
  -> Local GM P/V -> PV array -> array O accumulator or Local GM Oacc
  -> final normalize -> Local GM O -> output DMA ACK
  -> worker unique query-block completion -> manager job completion
```

All Local GM and array-buffer handoffs are asynchronous and explicitly modeled
with bytes, cycles, bandwidth/port occupancy, tags, and completion callbacks.
Passing a host pointer or copying a C++ vector is only a functional arithmetic
implementation detail; it must not create storage, zero-time transfer, or
completion. Do not add a standalone scratchpad SST component in the first
prototype. Reconsider internal bank subcomponents only if measured contention
requires a more detailed implementation behind the same GlobalMemory API.

For the existing 64-input/16-output arrays, QK maps each 16-key K panel to an
array matrix and uses two panels per `Bc=32` tile. PV pads the 32-element
weight vector to 64 inputs and uses four 16-output V panels for `D=64` or eight
for `D=128`. All MVM operations and local moves must be counted; no host-side
matrix multiply may stand in for either QK or PV.

### 4.8 SFU row storage and asynchronous execution contract

`TensorWorkerState::Context::values` is not accepted as a hardware row buffer.
The full row or tile must have an address and bounded allocation in the
worker's existing per-Core GlobalMemory. An SFU context contains only bounded
control/register state: job and row identity, stage, Local GM base and length,
chunk cursor, pending request tag, `m`, `l`, inverse sum, scale, mask boundary,
and valid/busy bits. The lane input/output registers or FIFO are also bounded;
one serviced chunk cannot exceed the configured lane capacity. The initial
implementation permits one outstanding chunk per context. More overlap
requires an explicit queue-depth and occupancy model.

Standalone Softmax reuses one Local GM row buffer per active context. For a
4096-element FP32 row this allocation is 16 KiB; four simultaneous contexts
therefore consume 64 KiB of real Local GM capacity. It does not allocate
separate S, E, and P buffers:

| Pass | Local GM read | SFU register/resource work | Local GM write |
| --- | --- | --- | --- |
| MAX | original logits S, one bounded chunk at a time | update scalar `m` | none |
| EXP/SUM | reread S | EXP pipeline computes `E=exp(S-m)` and updates scalar `l` | overwrite the same chunk with E |
| NORMALIZE | read E | vector pipeline multiplies by `1/l` | overwrite the same chunk with P |

Attention uses the same addressable S/P region. SCALE/MASK+MAX reads raw S,
applies scale and mask exactly once, updates the row/tile max, and writes the
transformed logits back to the same chunk. EXP/SUM or the online `(m,l)` update
then reads those transformed logits and overwrites them with unnormalized
weights. A one-key-tile prototype may normalize that buffer in place; the
multi-key-tile algorithm instead feeds the unnormalized weights to PV and
performs final `O_acc/l` only after the last valid key tile. No stage may retain
a complete score or probability row in a C++ container.

Every chunk follows an event-driven state sequence:

```text
issue/dispatch
  -> validate descriptor and reserve bounded context
  -> enqueue asynchronous Local GM read
  -> read callback validates job/row/stage/chunk/request identity
  -> reserve the vector, EXP, or reduction resource
  -> resource-completion event evaluates the chunk result
  -> enqueue asynchronous Local GM write when the pass produces data
  -> write callback advances the chunk or stage
  -> final local-operation completion
```

`issueSoftmaxTile()`, `issuePrimitive()`, and `issuePrimitiveBatch()` must not
read, calculate, and write an arbitrary payload inside the issue call. Their
final implementation is a compatibility frontend to the same bounded
context/chunk/resource scheduler: `issue` only validates, allocates, and
enqueues; a batch admits children subject to context and queue capacity rather
than executing an unbounded loop. Until a legacy frontend has been migrated,
it is explicitly `functional-only`; any Attention or performance runner that
selects it must fail before accelerator timing starts. Existing opcode and
numerical semantics remain unchanged.

Host `std::exp`, `std::sqrt`, and related functions may remain numerical value
evaluators, but results become visible only at the corresponding modeled
resource-completion event. Local GM callbacks move data and update readiness;
they cannot silently perform the arithmetic stage. Backpressure from a full
context file, lane FIFO, Local GM queue, or SFU resource must stall or return a
defined busy status without growing an unbounded host container.

Completion ordering is strict. A local SFU operation completes only after its
last required Local GM write callback. Standalone Softmax band completion also
waits for the address-based output DMA ACK. Fused Attention query-block
completion additionally waits for PV/final normalization, the Local GM O
write, and output DMA ACK before the manager records the unique completion.

Required focused evidence includes exact per-stage Local GM read/write bytes,
chunk counts including a partial final chunk, context/lane high-water marks,
queue and resource stalls, and zero performance-mode legacy immediate issues.
Tests must prove capacity backpressure, no stage advance before its callback,
in-place address bounds, numerical equivalence, callback/poll-frequency
independence, and the full output-ACK completion chain.

## 5. Phase A: SST Workload and Transpose-Aware Operand Loading

New workload root:

`src/sst/elements/golem/tests/small/muticore_attention`

Implemented workload files:

- `Makefile`
- `run_muticore_attention.sh`
- `attention_case.py` for deterministic input generation and small post-run
  mathematical verification
- `test_muticore_attention.py`
- `README.md`

Affected shared files:

- `src/sst/elements/golem/tests/small/mvm_noc_int_array/golem_matmul_runtime.cpp`
  and `gemm_matmul_op.h` for the shared logical transpose contract;
- the base guest, HBM generator, verifier, and runner for propagation of that
  contract. No SFU, GlobalMemory, or RoCC production source was changed.

Tasks:

1. Define one manifest for shapes, logical tensor strides, native K layout,
   transpose mode, mask, HBM placement, dtype, and random seed. Q must remain
   unscaled.
2. Add `transpose_b` to the shared B-operand tensor/layout contract. Map
   logical `B(k,n)` to native `K(n,k)` while reusing the existing RoCC vector
   load and completion path unchanged.
3. Add focused tests for native-K address generation, unsupported transpose
   modes, runner propagation, and padded partial final key tiles.
4. Run a real SST `QK^T` smoke directly. Compare its complete output after the
   run with a small mathematical verifier that has no SST timing or fused-loop
   behavior.
5. Add runner validation for positive shapes, supported D, timeout, artifact
   isolation, native-K layout, and mutually exclusive baseline/fused modes.
6. Keep generated HBM images, tensors, logs, and result JSON outside Git.
7. Do not introduce a standalone transpose SST component or a full `K^T` HBM
   materialization in this phase.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/muticore_attention \
  -p 'test_*.py'

make -C src/sst/elements/golem/tests/small/muticore_attention clean all
make -C build/sst-elements/src/sst/elements/golem -j2
```

Phase A exit completed on 2026-07-22: native-K `QK^T` passed complete output
verification for `S=64,D=64` (4096/4096) and logical `Skv=70` padded to 80
(4480/4480), both with zero mismatches. Focused tests, guest and `libgolem`
builds, canonical GEMM regression, and a `16x4096` Softmax smoke passed. RoCC
was not changed because its existing vector loader already implements the
required physical transfer.

## 6. Phase B: Materialized Standard Attention Baseline

Primary implementation files:

- the new `muticore_attention` guest/runtime/runner;
- the Phase A transpose-aware RoCC operand-loading path;
- existing tensor Softmax runtime and Row Engine;
- `src/sst/elements/golem/sfu/sfu.h` and `.cc` for once-per-job RSQRT state,
  fused SCALE/MASK+MAX execution, latency, and statistics;
- shared HBM generator/unpacker only if the dedicated workload cannot express
  the required Q/K/V/S/P/O layout locally.

Tasks:

1. Execute QK from native row-major K through logical `transpose_b`; no
   transposed K tensor may be generated or written to HBM.
2. Feed raw materialized S through a versioned attention-aware SFU path. The
   SFU computes RSQRT(D) once, then applies SCALE/MASK before row MAX; existing
   Version 1 Softmax jobs remain unscaled and unmasked.
3. Validate SFU-derived `1/sqrt(D)` for `D=64` and `D=128`, reject `D=0`, and
   prove that no generic primitive descriptor or GM scalar transfer is issued;
   then enable and validate causal masking without changing the input tensors.
4. Execute PV through the existing GEMM path and wait for its actual output completion.
5. Parse per-stage descriptor, DMA, compute, ACK, and guest wait timestamps.
6. Record separate HBM byte counts for Q/K/V/O, S, and P.

Required Phase B real SST acceptance, always serial and with fresh artifacts:

```text
B1 H1 S64   D64   non-causal
B1 H1 S64   D64   causal
```

Do not start the fused implementation until both points pass complete post-run
output verification and the baseline result JSON accounts for all S/P HBM
traffic. Materialized `S256,D64` and `S1024,D128` are retained as optional,
matched performance references after the corresponding fused point works;
they are no longer prerequisites for local-fused development.

Implementation result on 2026-07-22: the versioned 64-byte SFU parameter ABI,
once-per-job RSQRT latency, fused SCALE/MASK+MAX pass, V1 compatibility,
materialized three-stage runner, final O verifier, and logical HBM accounting
are implemented. Real `B1,H1,S64,D64` non-causal and causal SST runs both
passed all 4096 output elements with zero mismatches; maximum absolute errors
were `1.2131e-8` and `2.3586e-8`. The causal P upper triangle contained zero
nonzero elements. Each result records 16 KiB S writes/reads, 16 KiB P
writes/reads, and 128 KiB total logical HBM traffic. Phase B is therefore
complete.

## 7. Phase C: Physical Local-Memory Foundation and Fused Prototype

Phase C0 first removes the model shortcuts on the fused critical path. Phase
C1 then proves one complete local producer-consumer Attention output before a
multi-key-tile loop is added. The accepted materialized path remains unchanged
as a regression oracle during this work.

Potentially affected production files:

- `src/sst/elements/golem/rocc/roccAnalog.h`
- `src/sst/elements/golem/globalmemory/globalmemory.h`
- `src/sst/elements/golem/globalmemory/globalmemory.cc`
- `src/sst/elements/golem/sfu/sfu.h`
- `src/sst/elements/golem/sfu/sfu.cc`
- `src/sst/elements/golem/array/computeArray.h`
- `src/sst/elements/golem/array/mvmComputeArray.h`
- `src/sst/elements/golem/workercmdproc/workercmdproc.h` if the fused executor
  reuses WCP paths
- `src/sst/elements/golem/tests/architecture/cpu_builder.py`
- the archive architecture shim only if it remains an active runner dependency

Tasks:

1. Add asynchronous local read/write operations to `GlobalMemoryAPI`. Model
   configured read/write ports, bytes per cycle, base latency, queue depth,
   completion tags, and arbitration among DMA, RoCC/array, SFU, WCP, and
   control clients. Retain synchronous access only for initialization/debug
   paths that are excluded from accelerator timing.
2. Reserve an Attention window in the existing per-Core GlobalMemory layout
   and validate capacity/overlap before any DMA. Do not allocate a RoCC-owned
   byte vector as hardware scratch.
3. Replace the SFU whole-row `context.values` hardware interpretation with a
   bounded Context Register File plus lane FIFO/register buffer. MAX, EXP/SUM,
   NORMALIZE, and online state transitions must issue modeled Local GM
   operations for tile data and follow the in-place protocol in Section 4.8.
4. Route every local transfer used by the fused path through modeled Local GM
   and array-buffer interfaces. Fixed `gm2imat/gm2ivec/ovec2gm` delays may not
   stand in for byte-dependent transfer or shared-port contention.
5. Introduce manager/worker roles and an explicit worker-slot-to-physical-core
   map. A manager dispatches query blocks and aggregates unique completions;
   a manager-core RoCC FSM coordinates while worker-local RoCC executes the
   tile. Keep worker-core-0 coordination only in the legacy Softmax regression
   mode; do not dispatch compute to the manager SFU datapath.
6. Add one versioned Attention descriptor issue/wait path without changing
   existing GEMM or SFU opcode values.
7. Add the narrow asynchronous SFU local-tile operations described in Sections
   4.7 and 4.8. Existing HBM-backed SFU jobs and their ABI remain unchanged.
8. Implement `S=32,D=64,Br=16,Bc=32` non-causal as a two-query-block,
   one-key/value-tile-per-block end-to-end SST case. Process the two query
   blocks sequentially in one reserved Local GM context. Use real Q/K/V DMA
   readiness, QK and PV array completions, local handoff completions, final
   normalize, and output DMA ACK.
9. Require the SFU RSQRT-ready event before SCALE/MASK+MAX and apply the scale
   exactly once to every valid score. Keep the score/weight tile local and
   reject any host-side arithmetic or functional completion shortcut.
10. Emit machine-readable timestamps and byte counts for every causal boundary,
   including local transfers. Record Local GM window high-water mark and prove zero
   S/P HBM reads and writes.
11. Convert legacy primitive/tile entry points into compatibility frontends for
    the bounded asynchronous scheduler. Until each frontend is migrated, mark
    it functional-only and make performance mode reject it before timing.

Phase C0 exit: focused tests prove capacity, byte-scaled latency, port
contention, same-core no-NoC behavior, explicit worker mapping, bounded SFU
state, exact MAX/EXP-SUM/NORMALIZE local bytes, partial-chunk handling,
callback-ordered stage transitions, and rejection of direct critical-path
`memcpy` or legacy immediate-execution shortcuts.

Phase C1 exit: `B1,H1,S32,D64,Br16,Bc32` non-causal fused output passes
complete post-run verification; QK and PV both execute on MVM arrays; S/P HBM
bytes are zero; all S/P local bytes are accounted; and the recorded timeline
is strictly causal through manager completion.

## 8. Phase D: Multi-Tile Online Softmax, Causal, and Boundaries

Potentially affected files:

- the Phase C RoCC/SFU state;
- `src/sst/elements/golem/globalmemory/globalmemory.h` and `.cc` only if the
  existing DMA callbacks cannot carry the required block identity;
- `cpu_builder.py` for Attention block/local-window/resource parameters;
- focused component contract tests beside the current Softmax tests.

Tasks:

1. Extend the Phase C state machine to four query blocks and two key tiles for
   `S=64,Br=16,Bc=32`. Persist each query row's `(m,l)` in the bounded SFU
   Context Register File. Keep O in array accumulators where possible and use
   an explicitly allocated Local GM Oacc spill region otherwise.
2. Implement the exact rescaling recurrence in Section 4.1, including the
   first-tile `m=-inf, l=0` case and changing maxima between tiles.
3. Apply the SFU-derived scale and causal masking before tile row-max. Skip
   wholly future key tiles and mask the diagonal boundary tile element by
   element.
4. Handle partial `Sq`, `Skv`, and `D` tiles without padded values influencing
   max, sum, or output.
5. Normalize only after the last valid key tile, then issue one output DMA per
   completed query block. Job completion requires every unique query-block
   completion after its output DMA ACK.
6. Validate job/block/owner/shape identity on every asynchronous callback and
   reject duplicates or stale completions.

Focused tests must cover:

- two key tiles where the second tile raises the running max;
- RSQRT(D) executes once per job, scales valid scores exactly once for `D=64`
  and `D=128`, and rejects `D=0`;
- numerically extreme logits;
- causal diagonal and wholly masked future tile skipping;
- partial final query/key tiles;
- duplicate/stale completion rejection;
- Local GM window overflow and unsupported descriptor rejection.

Phase exit: fused non-causal and causal `S64,D64` pass complete post-run output
verification with zero S/P HBM bytes. Then pass one partial shape
`Sq=20,Skv=70,D64` and one small extreme-logit SST case. `S256,D64` and
`S1024,D128` move to the scale phase.

Implementation progress on 2026-07-23: D1 non-causal and D2 causal
`B1,H1,S64,D64,Br16,Bc32` both pass complete 4,096-element verification with
zero S/P HBM bytes. In causal mode, query blocks 0/1 issue only key tile 0 and
query blocks 2/3 issue key tiles 0/1. This skips two wholly future tiles and
reduces exact activity from D1 `QK/PV=256/512`, `SFU jobs/rows=8/128` to D2
`192/384`, `6/96`; diagonal tiles mask 992 elements after scale and before
row max. D2 maximum absolute error is `2.3585739111764426e-08`.

D3 `Sq=20,Skv=70,D64` also passes complete 1,280-element verification with
maximum absolute error `7.499891131745873e-09` and zero S/P HBM bytes. The
final query block carries 4 rows and the final key tile carries 6 keys; exact
valid-only activity is `QK/PV=140/240`, `SFU jobs/rows=6/60`, and 1,400 scaled
score elements. At the D3 checkpoint, only the small extreme-logit SST case
remained before Phase D exit.

D4 uses a deterministic `S64,D64` profile whose first and second KV tiles have
scaled logits exactly -100 and +100, forcing the online running max to increase
by 200. The real fused SST passes all 4,096 outputs with zero mismatches and
zero maximum absolute error; exact D1 activity remains `QK/PV=256/512`,
`SFU jobs/rows=8/128`, RSQRT count 1, and zero S/P HBM bytes. A same-run default
D1 regression also passes. Phase D is complete; Phase E is next.

## 9. Phase E: Manager-Coordinated Sixteen-Tile Prefill Mapping

Use the existing dedicated group managers as the hardware control plane. Each
manager coordinates its explicit worker group; worker-local RoCC executes
independent query blocks and retains all tile data locally. Do not introduce a
single global data-moving controller.

Tasks:

1. Define a versioned topology map from manager/group to explicit physical
   worker Core IDs. Assign contiguous query-block bands across 16 workers and
   balance the final partial wave without assuming contiguous IDs.
2. Stripe Q/O bands over the four HBM nodes. Initially block-stripe K/V
   without replication; collect traffic and NoC hotspot data before deciding
   whether per-column K/V replication is justified.
3. Require exactly one unique completion for every issued query block and one
   tensor-level completion after all output DMA ACKs.
4. Preserve per-tile Local GM window isolation and reject overlapping jobs that
   alias contexts or HBM output ranges.
5. Extend transport with distinct Attention dispatch/completion messages;
   never overload Softmax row messages. Manager completion requires the exact
   set of mapped workers/query blocks and all corresponding output DMA ACKs.

Scale ladder:

```text
B1 H1 S64   D64
B1 H1 S256  D64
B1 H1 S1024 D128
B1 H1 S2048 D128
B1 H1 S4096 D128
```

The fused run is required at every point. A matched materialized run is
required at S64 and otherwise run on demand for performance comparison; it is
not allowed to delay fused functional scaling. No larger point starts until
the previous fused shape passes correctness, lifecycle, local-window capacity, and
monotonic-scaling checks.

### Phase E implementation progress (2026-07-23)

Phase E1 passes the real `B1,H1,S256,D64,Br16,Bc32` fused SST with manager
cores 0-3 and explicitly mapped worker cores 4-19. Each manager owns a 64-row
query band and dispatches four 16-row blocks. Q/O and K/V are block-striped
across HBM nodes 1-4; workers stream one 32-row K/V tile at a time through
per-Core GlobalMemory. All 16,384 outputs pass with zero mismatches and maximum
absolute error `4.4967521123807225e-09`; exact per-manager/per-worker activity
passes and S/P HBM bytes remain zero.

This checkpoint produces four manager-level band completions. It does not yet
satisfy task 3's single tensor-level completion. Phase E2 must add a root or
cross-manager aggregation path that waits for all four manager bands (and thus
all 16 workers' output DMA ACKs) before emitting exactly one tensor completion.
The `S1024,D128` scale point starts only after E2 passes.

Phase E2 now passes that condition. Each manager first validates four unique
worker completions. Managers 1-3 then send a distinct
`AttentionManagerComplete` message to root manager core 0; the root validates
manager slot/core identity with a second bitmap. The final S256 run records one
local band completion on each manager, four band completions received at the
root, and exactly one tensor completion on core 0. All 16,384 outputs still pass
with maximum absolute error `4.4967521123807225e-09` and zero S/P HBM bytes.
Phase E3 now passes the real `B1,H1,S1024,D128,Br16,Bc32` fused SST. Each
worker processes four 16-row query blocks and uses eight PV dimension panels,
while retaining only one 32-row K/V tile. Its complete Attention Local GM
window is bounded to 51,328 bytes. All 131,072 outputs pass with zero
mismatches and maximum absolute error `1.5966506535956479e-09`; S/P HBM bytes
remain zero. Exact worker activity, four unique manager bands, four root
receives, and the single root tensor completion all pass. The next Phase E
increment is `B1,H1,S2048,D128`.

Phase E4 now passes the real `B1,H1,S2048,D128,Br16,Bc32` fused SST. Each
worker processes eight 16-row query blocks and 64 KV tiles while retaining the
same 51,328-byte Local GM window. All 262,144 outputs pass with zero mismatches
and maximum absolute error `1.181374809935097e-09`; S/P HBM bytes remain zero.
Exact per-worker activity, four unique manager bands, four root receives, and
the single root tensor completion all pass. The next Phase E increment is
`B1,H1,S4096,D128`, gated by a dry-run and wall-clock/watchdog review.

## 10. Phase F: Performance and LLM-Relevant Extension

For every matched baseline/fused point record:

- descriptor-to-accelerator completion cycles;
- guest issue-to-wait-return cycles and whole SST time separately;
- useful FLOPs and elements/cycle;
- Q/K/V/O and S/P DMA operations/bytes per HBM node, including every K/V
  reload caused by iterating query blocks;
- QK, RSQRT-ready, SCALE/MASK, online Softmax, PV, normalize, and output-ACK
  windows;
- MVM array occupancy, SFU vector/EXP occupancy, Local GM window high-water mark;
- NoC port utilization/stalls and memory queue delay;
- retry, rejected, stale, duplicate, and wait-poll counts;
- host wall time and artifact/signature metadata.

After the single-head FP32 fused path passes, extend in this order:

1. multiple independent query heads;
2. `Hq != Hkv` for GQA/MQA;
3. BF16/FP16 storage with explicitly defined accumulation precision;
4. cross-attention with `Sq != Skv`;
5. decode/KV-cache as a separate kernel and acceptance path.

Do not label the prefill kernel as a decode result. Decode is expected to be
KV-cache bandwidth dominated and requires its own layout, traffic model, and
shape set.

## 11. Regression and Build Boundaries

After any SFU, GlobalMemory, RoCC, array, architecture wiring, or shared runner
change, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu \
  -p 'test_sfu_softmax_*.py'

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/muticore_softmax \
  -p 'test_*.py'

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/muticore_attention \
  -p 'test_*.py'

make -C build/sst-elements/src/sst/elements/golem -j2
make -C src/sst/elements/golem/tests/small/muticore_softmax all
make -C src/sst/elements/golem/tests/small/muticore_attention all
```

Run the canonical GEMM regression serially outside the restricted network
namespace:

```bash
TMPDIR=/data4/jjgong/tmp \
GOLEM_ARTIFACT_ROOT=/data4/jjgong/tmp/attention_gemm_regression \
env -u GOLEM_SFU_ENABLE -u GOLEM_SFU_STANDALONE_SOFTMAX \
  -u GOLEM_SFU_JOB_SOFTMAX -u GOLEM_SFU_PRIMITIVE_SOFTMAX \
  -u GOLEM_SFU_REDUCTION_VN -u GOLEM_DMA_RESPONSE_VN -u GOLEM_ARCH_SCRIPT \
  -u GOLEM_GROUP_MANAGER_ENABLE -u GOLEM_CTRL_LINK_ENABLE \
  -u GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE \
  bash src/sst/elements/golem/tests/run_noc_dma_pipeline.sh \
  --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --dtype fp32 --tensor-source sample --verify-c
```

The GEMM regression requires exit 0, simulation completion, `VERIFY-C PASS`,
complete DMA lifecycle, zero retries, and zero unexpected SFU/Attention
activity.

Real SST Attention and Softmax runs must be serial, use fixed watchdogs, and
write to fresh artifact roots. Final closeout also runs `git diff --check` and
records production library, guest binary, runner, manifest, input, and output
hashes.

## 12. Stop Conditions and Decision Gates

Stop implementation and diagnose before expanding scope when any of these
occurs:

- baseline composition cannot pass complete post-run output verification;
- a completion is released before output DMA ACK;
- fused mode writes S or P to HBM;
- online state changes with callback/poll frequency;
- SFU RSQRT/SCALE is missing, duplicated, or applied after row MAX;
- partial or causal tiles change results outside tolerance;
- the Local GM window capacity is exceeded, aliased, or supplied as an HBM spill address;
- a local score/weight/O transfer completes without modeled bytes and cycles;
- an SFU full row/tile, WCP partial-C tile, or operand payload is retained in
  an unbounded C++ container without a declared hardware capacity;
- RoCC, WCP, or SFU directly reads/writes Local GM or array vectors on the
  performance path without participating in modeled port arbitration;
- manager dispatch treats a worker slot as a physical Core ID or manager code
  reads/copies worker S/P data;
- a shared static reducer/map bypasses explicit NoC transport on a measured
  multi-core path;
- retries, stale callbacks, duplicate completions, or unaccounted bytes are
  non-zero;
- GEMM or accepted `1024x4096` Softmax regression fails;
- the first matched fused run is slower than baseline without statistics that
  identify a specific modeled bottleneck.

### 12.1 Model-realism audit and remediation order

The durable evidence and file references are recorded in
`src/sst/elements/golem/tests/small/muticore_attention/findings.md`. The current
audit classification is:

| Priority | Model boundary | Current shortcut | Required disposition |
| --- | --- | --- | --- |
| P0 | per-Core GlobalMemory | synchronous `std::copy`, no local ports/bandwidth/queue | implement one asynchronous shared local-access scheduler before fused work |
| P0 | SFU Row Engine | full rows in `context.values`; legacy primitive/tile issue performs immediate read/compute/write | use the Section 4.8 bounded async chunk scheduler, keep tiles in Local GM, migrate legacy frontends, and reject unmigrated paths in performance runs |
| P0 | RoCC local moves | fixed byte-independent GM2IMAT/GM2IVEC/OVEC2GM delays and direct vector access | use byte-scaled Local GM and array-buffer operations with contention |
| P0 | manager/worker control | accepted Softmax uses worker 0; dispatch equates slot and physical Core ID | add manager role, explicit mapping, and manager-only control state |
| P1 | WorkerCommandProcessor | panel payloads and partial-C tiles live in host vectors and copy directly to/from arrays | allocate declared Local GM/array storage and model transfer/accumulator capacity before reusing WCP for fused Attention |
| P1 | ComputeArray buffers | arithmetic latency is modeled, but matrix/input/output programming and output moves are immediate | keep functional host arithmetic, but add bounded buffer ports, transfer cycles, occupancy, and stalls before final performance claims |
| P1 | cross-core reduction/control | process-global maps can represent shared reducers; mailbox accesses bypass local-memory timing | require explicit NoC for measured cross-core operations and define mailboxes as modeled registers or Local GM clients |
| Accepted | NoC/HBM/cache | Merlin mesh, memHierarchy, and DRAMSim3 already provide queues, bandwidth, and latency | retain and regress; do not replace with new project-local models |

P0 items are Phase C0 gates. P1 items must be fixed before the affected path is
used for performance conclusions; they do not block unrelated legacy
regressions. Functional C++ arithmetic remains acceptable when capacity,
resource occupancy, transfer timing, and completion are physically modeled;
this project does not require RTL or gate-level floating-point simulation.
P2 calibration remains after functional fused acceptance: validate the MVM
latency formula against the intended CIM datapath, define SFU approximation and
rounding behavior beyond ideal `std::exp/sqrt`, and add area/energy only when
the project begins making those claims.

Native-K `QK^T`, materialized Attention, fused single/multi-KV-tile online
recurrence, causal skipping, partial tiles, extreme-logit acceptance, and the
Phase E1 4-manager/16-worker S256 mapping, Phase E2 cross-manager single
tensor-level completion, Phase E3 `S1024,D128`, and Phase E4 `S2048,D128` fused
scaling are complete. The next milestone is the Phase E `S4096,D128` fused scale point. Larger materialized runs
are optional comparisons. Multi-head, mixed
precision, K/V replication, a single global scheduler, and decode are added
only after the manager-coordinated fused local path is measured and correct.

---

# Attention / FlashAttention 实现计划（中文版）

**状态：** Phase A/B/C/D 和 Phase E1/E2/E3/E4 已实现并验证。当前 fused 路径已通过
`B1,H1,S2048,D128` 的 4 manager/16 worker 映射，将 Q/O/K/V block stripe 到四个
HBM node，并通过 per-Core GlobalMemory 流式处理 K/V tile，S/P 不进入 HBM。
四个唯一 manager band 完成后，root 会产生恰好一次 tensor-level completion。
Phase E 尚未关闭，下一步进入最终 `S4096,D128` 规模点。

**日期：** 2026-07-22

**目标：** 基于现有 GEMM 和 causal Softmax 路径，直接在 SST 中建立正确、
可测量的 FP32 prefill Attention 基线；随后实现分块融合的 FlashAttention
路径，并确保 score 矩阵和 probability 矩阵不会完整写入 HBM。

**决策依据：**

- `src/sst/elements/golem/tests/small/muticore_softmax/PROJECT_HANDOFF.md`
- `docs/superpowers/specs/2026-07-16-sfu-softmax-row-engine-noc-architecture-design.md`
- `docs/superpowers/plans/2026-07-17-sfu-softmax-row-engine-noc-implementation.md`

**2026-07-22 已批准的修订：**

- 不设置独立的 CPU Attention 开发阶段；直接开发真实 SST workload，仅保留
  一个轻量的运行后数学验证器；
- 第一阶段支持 transpose，但不做独立 SST component，也不在 HBM 中物化
  完整转置矩阵。Phase A 检查确认原生 K 行已与现有 RoCC vector loader 契约
  完全匹配，因此除非后续 packing 约束要求局部重排，否则无需新增 RoCC 状态机；
- 标准 Attention scaling 放在 SFU Attention Row Engine 内部：复用现有 RSQRT
  数值语义，每个 job 只计算一次 `1/sqrt(D)`，不发出 generic primitive，也不
  产生 GM 标量往返；随后在 row MAX 前融合完成标量乘法和 mask；
- non-causal、causal、partial tile 和 extreme logits 是依据算子语义、分块
  边界和数值稳定性制定的项目验证矩阵，不宣称它们是正式行业一致性标准。
- materialized Attention 保留为正确性和流量基线，但不再把 S/P 往返 HBM 的
  数据流扩展成生产路径。S64 验收对通过后，直接转入 manager 协调、worker
  RoCC 执行的 fused kernel，让 score/probability tile 经保留的 per-Core
  GlobalMemory window 传递而不经过 HBM。
- 现有每核 `golem.GlobalMemory` 是 fused Attention 唯一的 tile 级本地存储。
  在每个 worker 的 GlobalMemory 中保留 Attention window；首版不新增 RoCC
  私有 64 KiB byte array，也不新增独立 scratchpad component；
- SFU 运算状态建模为容量受限的 Context Register File 加 lane 级 FIFO/寄存器。
  running `(m,l)`、inverse sum、scale、mask boundary 和 context state 等标量
  留在 SFU 寄存器；完整 score/probability row 或 tile 放在 per-Core
  GlobalMemory；
- 新 Softmax/Attention 控制平面使用 dedicated manager core，worker 的
  RoCC/SFU/array 是数据平面。已验收的 worker Core 0 Softmax coordinator 路径
  继续作为回归基线，直到 manager 路径独立验收；
- 新 coordinator 实现在 manager-core RoCC control FSM 中，而不是 SFU 算术
  context；首版也不新增独立 SST component。worker SFU 执行 row/tile，manager
  SFU datapath 不参与计算；
- 启用 manager coordinator 前，必须把隐式的
  `band % worker_count == physical_core_id` 替换为 manager 持有的显式
  worker-slot 到物理 Core ID 映射；
- 零时延 C++ vector、组件间直接复制、跨 core 静态 map 和与传输字节数无关的
  固定延迟均列为模型审计缺口。fused 关键路径上的每项存储和传输在具备明确
  容量和时序模型前，不接受其性能结果。

## 1. 当前起点

已验收的生产基线包括：

- 通过现有 RoCC/MVM/GlobalMemory 路径运行 FP32 GEMM；
- FP32 `1024x4096` 行 Softmax，golden 检查/不匹配数量为
  `4,194,304 / 0`；
- Softmax accelerator completion 为 `66,958` cycles，严格遵循
  `input DMA -> MAX -> EXP/SUM -> NORMALIZE -> output DMA ACK -> 16 unique
  band completions` 因果链；
- 一个 tensor Softmax job、16 个物理 SFU、每个 SFU 四个 row context，
  以及四节点 band-striped HBM。

`66,958` 是重构前冻结基线。加入真实 Local GM service time 后，重新运行得到的
周期数可以发生合理变化；验收要求数值输出和 lifecycle 不变，并完整解释周期
差异。不得静默覆盖历史结果，也不得把它当成重构后的 completion。

当前 Softmax 是稳定的三阶段整行实现，并不是 FlashAttention 所需的 online
running-max/running-sum 算法。Phase A 已加入逻辑 `transpose_b`，同时保持 K
的原生 `[Skv,D]` 行主序布局。Phase B 已证明三段 materialized 组合正确，
也明确暴露了 S/P 的多余 HBM 往返。因此下一个实现边界是 local-tile handoff
和 online state，而不是更大的 materialized 运行。

## 2. 范围和非目标

第一版面向推理阶段的 prefill：

- 仅支持 FP32；
- 初始支持 batch `B=1`、query head `Hq=1`、KV head `Hkv=1`；
- 初始支持 `Sq=Skv` 的 self-attention；
- 同时支持 non-causal 和 causal mask；
- head dimension 支持 `D=64` 和 `D=128`；
- 暂不支持 dropout、反向传播、训练、ALiBi、稀疏 Attention 或 paged KV
  cache；
- FP32 fused prefill 路径通过前，不加入 BF16/FP16、GQA/MQA、多 batch 或
  decode kernel。

RoPE 位于本算子之外：假定 Q 和 K 已完成 projection 和位置编码。
`1/sqrt(D)` 缩放属于 Attention 数值契约的一部分，由 SFU 执行，不在 Q 中
预先应用，也不由 host 验证器或 runtime 代算。

## 3. 验收契约

每一个被接受的真实 SST 测试点必须满足：

1. 完整 SST 输出 tensor 通过轻量的 host 端运行后数学验证器。该验证器只作为
   数值 oracle，不模拟 SST 时序，也不复现 fused 状态机。初始使用现有 FP32
   容差 `atol=1e-5`、`rtol=1e-4`；修改容差必须记录数值证据。
2. causal 模式绝不能包含全局位置大于 query 的 key。partial query/key tile
   必须纳入输出验证。
3. baseline 模式执行可见因果链：
   `QK^T GEMM -> SFU RSQRT(D) once -> SFU SCALE/MASK+MAX -> EXP/SUM ->
   NORMALIZE -> PV GEMM -> output ACK`。
4. fused 模式执行：
   `descriptor -> Q/K/V DMA ready -> QK array completion -> score-local-ready
   -> SFU RSQRT/SCALE/MASK+online update -> weight-local-ready -> PV array
   completion -> O-accumulator update -> final normalize -> output DMA ACK ->
   unique query-block completion -> job completion -> guest wait return`。
5. fused 模式不得把任何 score/probability tile 写入 HBM，包括临时或 debug
   spill；S/P 的 HBM read/write byte counter 必须全部为零。
6. DMA issue/completion 次数和字节数必须与 tensor/tile 布局一致；retry
   exhaustion、stale completion、duplicate completion、Local GM window overflow 和
   contract failure 均必须为零。
7. accelerator latency 从 descriptor acceptance 测量到真实 completion。
   分析计算估算必须单独标记，不能当作 completion 时间。
8. baseline 和 fused 使用完全相同的 Q/K/V 输入、shape、mask、clock、
   array/SFU 参数、HBM 拓扑和验证容差。
9. 现有 GEMM 和 Softmax 结果必须保持有效。修改共享组件后必须运行 focused
   tests、重编 `libgolem`、运行 Softmax smoke 和 canonical GEMM regression。
10. Attention RSQRT/SCALE 必须作为 Attention Row Engine 内部阶段执行，不得
    调用 `issuePrimitive()`、创建 `SFUPrimitiveDesc` 或通过 GM 搬运派生标量。
    generic SFU primitive 和 legacy Softmax 行为保持不变。
11. fused tile 数据使用 controller 在 worker 现有 per-Core GlobalMemory 中
    保留的 window。每次本地读写都必须异步执行，并统计 bytes、latency、port、
    queue 和 completion identity。同核访问不产生 NoC packet，跨核访问仍走 NoC。
12. SFU Context Register File 和 lane buffer 必须有明确容量。完整 row/tile
    不得在 MAX、EXP/SUM、NORMALIZE 或 online update 之间留在无界 C++ vector。
13. manager coordinator 只负责 dispatch、依赖跟踪和 unique completion 聚合，
    不读取、复制或保存 worker 的 S/P 数据。worker-local RoCC 负责本地
    array/SFU/GlobalMemory 执行序列。
14. fused completion 必须包含因果路径中所有 Local GM 和 array buffer 传输。
    与 bytes 无关的固定延迟、直接 `memcpy` handoff 和进程级全局 reducer map
    不得出现在已验收性能路径上。

### 3.1 验证矩阵的来源

以下四类测试不是正式行业标准，而是覆盖不同故障类型的最小项目验证矩阵：

| 测试 | 来源 | 必须执行的层级 |
| --- | --- | --- |
| non-causal | 标准 Attention 算子语义 | 小规模端到端 SST |
| causal | 自回归 Attention mask 语义 | 小规模端到端 SST |
| partial tile | 分块 kernel 产生的边界条件 | 非整除 SST shape，加 focused 地址/mask 测试 |
| extreme logits | stable/online Softmax 数值要求 | focused SFU/online-state 测试；fused loop 完成后增加一个小规模 SST 测试 |

具体 shape、随机种子和极端数值由本项目定义。这些测试分别检查语义、边界和
数值问题，彼此不能替代，但不需要全部运行成昂贵的大规模 SST 测试。

在获得第一组匹配的 baseline/fused 结果前，不预设加速比目标。初始性能门槛
只要求事实成立：fused 必须消除 S/P 的 HBM 流量，completion 不得比匹配的
materialized baseline 更慢，并且必须从组件统计中识别实际瓶颈。

## 4. 算子和数据契约

### 4.1 数学契约

对每个 batch/head：

```text
S = (Q K^T) / sqrt(D)
P = softmax(S + mask)
O = P V
```

fused 路径为每个 query row 保存未归一化输出累加器和 online 状态。每处理一个
key/value tile：

```text
m_new = max(m_old, row_max(S_tile))
alpha = exp(m_old - m_new)
P_tile = exp(S_tile - m_new)
l_new = alpha * l_old + row_sum(P_tile)
O_acc_new = alpha * O_acc_old + P_tile * V_tile
```

最后一个 key/value tile 完成后执行 `O = O_acc / l`。

baseline 和 fused 均由 SFU 为每个 Attention job 计算一次
`inv_sqrt_d = rsqrt(float(D))`，并在 mask 和 row MAX 前执行
`S_tile = S_tile_raw * inv_sqrt_d`。不得对每个 score 元素重复执行
SQRT/RSQRT。

### 4.2 初始布局

- Q：`[B,Hq,Sq,D]`，原生行主序且不预缩放。SFU 根据 descriptor 中的 `D`
  生成标准 Attention scale；输入生成器和 host 不预缩放 Q。
- K：`[B,Hkv,Skv,D]`，原生行主序。`QK^T` 由共享 `transpose_b` layout
  契约表达，禁止生成完整 `[B,Hkv,D,Skv]` HBM 副本。
- V：`[B,Hkv,Skv,D]`，行主序或适用于第二次 GEMM 的 packed 布局。
- O：`[B,Hq,Sq,D]`。
- 仅 baseline 使用的 S/P：`[B,Hq,Sq,Skv]`。

layout manifest 是输入生成、guest 寻址、HBM 解包、运行后验证和结果解析的
唯一数据源。fused 模式可以改变内部 block packing，但必须使用相同的逻辑
Q/K/V tensor。

### 4.3 Transpose 架构

已实现机制是共享 `transpose_b` tensor/layout 契约。对 `QK^T`，逻辑
`B(k,n)` 映射到原生 `K(n,k)`；每个连续的原生 K row 正好对应现有 packed
HBM 中的一条 B column vector，因此直接复用现有 RoCC batch vector loader
及其 completion identity。无需 transpose buffer、新 opcode 或额外建模状态，
也不会把完整 `K^T` 写回 HBM。

若后续物理 packing 确实要求局部重排，再在 RoCC operand loading 内增加
banked tile buffer，并先建模 cycles、stall、bytes 和 occupancy。

只有后续架构或测量表明 transpose 是由多个算子/controller 共享、具有独立
queue、bandwidth、clock 或 arbitration 的资源时，才重新评估是否拆成独立
SST component。

### 4.4 Attention Scaling 架构

标准 Attention scaling 是 SFU Attention Row Engine 的专用内部阶段。SFU 接受
job 时先验证 `D > 0`，计算一次 `inv_sqrt_d = RSQRT(float(D))`，并将结果
保存在 job/context state 中。该阶段复用 `SFUPrimitiveOp::RSQRT` 的数值语义，
但不调用 `issuePrimitive()`、不创建 `SFUPrimitiveDesc`，也不通过 GM 读写该
标量。score datapath 在 causal mask 和 row MAX 前，将每个原始 QK 结果乘以
该标量。SCALE/MASK 与 score-read/MAX pass 融合，不得增加一次完整 S 的 HBM
读写。

时序模型要为每个 Attention job 计入一次可配置 RSQRT latency，并计入有效
score 元素占用的 vector multiply throughput。分别输出 RSQRT-ready 和
SCALE/MASK start/done 统计，禁止把计算或延迟隐藏到 host 功能代码中。focused
tests 还必须证明 Attention 路径的 generic primitive issue 和 GM scalar transfer
均为零。初始标准模式只接受 `D`，不接受调用者自定义 scale；等真实算子需要
时再扩展。generic SFU primitive 和 legacy Softmax job 保持 guest-visible ABI
和数值语义；其内部执行可以并应按第 4.8 节接入有界异步 scheduler。

### 4.5 初始 FlashAttention tile 和 per-Core GlobalMemory 预算

从 `Br=16`、`Bc=32` 开始，并把二者参数化。对 `D=128`，FP32 的
Q/K/V/S/O accumulator 加 `(m,l)` 在 score buffer 被原地覆盖为未归一化
weight 时共需 51,328 bytes（50.125 KiB）；`D=64` 时需要 26,752 bytes
（26.125 KiB）。这些空间从每个 worker 现有 per-Core GlobalMemory 中由
controller 保留的 Attention window 分配，而不是第二块 SFU/RoCC scratchpad。

该 window 包含对齐的 Q、K、V、S/P ping-pong、O-accumulator spill 和
descriptor/context spill 区。活跃的 `(m,l)`、inverse sum、scale、mask 和
stage 字段保存在容量受限的 SFU Context Register File 中，除非显式 spill，
否则不计为 tile SRAM。descriptor validation 必须精确计算 Local GM 需求，
计入 metadata/alignment，并拒绝与 GEMM slot、mailbox、DMA flag 或其他
context 重叠。

controller 根据本地 GlobalMemory layout 选择保留 window；guest 不提供任意
scratch 地址。首个原型仅使用一个 context 且不做 double buffering。在本地
端口 occupancy 和 overlap 统计显示确有需要前，不增加更多 context。

### 4.6 ABI 方向

保持现有 128-byte `SFUJobDesc` 以及所有 GEMM/SFU opcode 不变。新增由 RoCC
fused controller 持有的独立版本化 `GolemAttentionDescV1`，至少包含：

- magic、version、byte size；
- Q/K/V/O 基地址和 byte stride；
- `B`、`Hq`、`Hkv`、`Sq`、`Skv`、`D`；
- `Br`、`Bc`、mask mode、layout identifier、operand transpose flag
  和逻辑 leading dimension；
- HBM node mask/stride、manager/group identity、显式 worker-map identity、
  owner/context identity 和 completion 地址。

为 Attention issue/wait 保留独立 operation identifier。manager 验证 tensor
job 和 dispatch map；每个 worker-local controller 根据 core/context identity
和配置 layout 推导其 per-Core GlobalMemory 保留 window。descriptor 不提供
任意 scratch base。不得重新解释旧 Softmax flag 或 reserved field。未知
version、layout、dtype、mask、shape、worker map 和不安全地址范围必须在任何
DMA/compute 发出前失败。

### 4.7 Fused ownership 和 local dataflow

manager-core RoCC control FSM 负责 tensor 级协调：descriptor validation、group
选择、显式 worker-slot 映射、query-block dispatch、阶段依赖跟踪和 unique
completion 聚合。manager 不执行 QK/PV/SFU 算术，也绝不读取或复制 S/P 数据。
manager SFU datapath 不是 worker，不接收 row/tile dispatch。

每个 worker-local RoCC 是 fused tile executor，而不是算术实现单元。它调度
本核 MVM array、SFU、DMA engine 和现有 GlobalMemory，负责 tile-loop state、
callback identity、在保留 Attention window 内分配空间，以及 query-block
completion。两级控制避免 worker 0 同时承担全局协调和数据计算，同时保持所有
大数据的本地所有权。

SFU 新增窄范围的异步 local-tile API，而不是再增加 guest-visible Softmax
descriptor。它从 per-Core GlobalMemory 分块读取 score，执行每 job 一次的
RSQRT、scale、mask、row max、exp 和 online `(m,l)` update，再把 weight chunk
写入同一 local window。SFU 内只保留容量受限的标量/context state 和当前
lane chunk。array 完成 `weights * V_tile` 后，由 SFU vector resource 建模
`O_acc = alpha * O_acc + delta_O` 和最终 `O_acc/l` normalization；Oacc 优先
留在 array accumulator，容量不足时 spill 到保留的 Local GM 区域。

```text
manager dispatch
  -> worker Q/K/V DMA 到 per-Core GlobalMemory
  -> Local GM -> QK array -> Local GM S buffer
  -> Local GM S -> SFU registers/lanes -> Local GM P buffer
  -> Local GM P/V -> PV array -> array O accumulator 或 Local GM Oacc
  -> final normalize -> Local GM O -> output DMA ACK
  -> worker unique query-block completion -> manager job completion
```

所有 Local GM 和 array-buffer handoff 都必须异步执行，并显式统计 bytes、cycles、
bandwidth/port occupancy、tag 和 completion callback。host pointer 或 C++ vector
复制只能作为功能算术的实现细节，不能隐式提供存储、零时延传输或 completion。
首版不新增独立 scratchpad SST component；只有测得 contention 需要更细粒度
实现时，才在保持同一 GlobalMemory API 的前提下考虑内部 bank subcomponent。

对现有 64-input/16-output array，QK 将每个 16-key K panel 映射为一个 array
matrix，因此 `Bc=32` 需要两个 panel。PV 将 32 元素 weight vector 补齐到 64
input，`D=64` 使用四个 16-output V panel，`D=128` 使用八个。所有 MVM 操作和
local move 都必须计数，QK/PV 均不得由 host-side matrix multiply 代替。

### 4.8 SFU 行存储与异步执行契约

`TensorWorkerState::Context::values` 不再被接受为硬件 Row Buffer。完整 row/tile
必须在 worker 现有的 per-Core GlobalMemory 中拥有地址和有界分配。SFU context
只保存容量受限的控制/寄存器状态：job/row identity、stage、Local GM base/length、
chunk cursor、pending request tag、`m`、`l`、inverse sum、scale、mask boundary、
valid/busy。lane input/output register 或 FIFO 同样必须有上限；单次处理的 chunk
不得超过配置的 lane capacity。首版每个 context 只允许一个 outstanding chunk；
若增加 overlap，必须同时声明 queue depth 并统计 occupancy。

Standalone Softmax 的每个 active context 只复用一个 Local GM row buffer。4096
个 FP32 元素占 16 KiB，四个并发 context 因而真实占用 64 KiB Local GM，而不是
再隐式分配 S、E、P 三份缓冲：

| pass | Local GM read | SFU register/resource work | Local GM write |
| --- | --- | --- | --- |
| MAX | 每次读取一个有界 chunk 的原始 logits S | 更新标量 `m` | 无 |
| EXP/SUM | 重新读取 S | EXP pipeline 计算 `E=exp(S-m)` 并更新标量 `l` | 用 E 原位覆盖同一 chunk |
| NORMALIZE | 读取 E | vector pipeline 乘 `1/l` | 用 P 原位覆盖同一 chunk |

Attention 复用同一个有地址的 S/P 区域。SCALE/MASK+MAX 读取原始 S，对每个有效
score 恰好执行一次 scale/mask，更新 row/tile max，并把变换后的 logits 原位
写回同一 chunk。随后 EXP/SUM 或 online `(m,l)` update 读取变换后的 logits，
并原位覆盖为未归一化 weight。单 key tile 原型可以继续原位 normalize；多 key
tile 算法把未归一化 weight 送入 PV，只在最后一个有效 key tile 后执行最终
`O_acc/l`。任何阶段都不得在 C++ container 中保留完整 score/probability row。

每个 chunk 必须遵循事件驱动的状态序列：

```text
issue/dispatch
  -> 验证 descriptor 并保留有界 context
  -> enqueue 异步 Local GM read
  -> read callback 验证 job/row/stage/chunk/request identity
  -> 预约 vector、EXP 或 reduction resource
  -> resource-completion event 计算 chunk 结果
  -> 若本 pass 产生数据，则 enqueue 异步 Local GM write
  -> write callback 推进 chunk 或 stage
  -> final local-operation completion
```

`issueSoftmaxTile()`、`issuePrimitive()` 和 `issuePrimitiveBatch()` 不得在 issue
调用内对任意长度 payload 完成读、算、写。最终实现应把它们变成同一套有界
context/chunk/resource scheduler 的兼容 frontend：`issue` 只负责验证、分配和
入队；batch 按 context/queue capacity 接纳子操作，不能用无界循环立即执行。
legacy frontend 迁移完成前必须明确标记为 `functional-only`；任何 Attention 或
性能 runner 一旦选中它，必须在 accelerator timing 开始前失败。现有 opcode 和
数值语义保持不变。

host `std::exp`、`std::sqrt` 等函数可以继续作为数值计算器，但结果只能在对应
建模 resource-completion event 到达后可见。Local GM callback 只搬运数据和更新
readiness，不能顺便完成算术阶段。Context Register File、lane FIFO、Local GM
queue 或 SFU resource 满载时，必须 stall 或返回定义明确的 busy 状态，不能通过
增长无界 host container 吸收压力。

completion 顺序必须严格。local SFU operation 只有在最后一个必要 Local GM
write callback 后才能完成；Standalone Softmax 的 band completion 还必须等待
以 Local GM 地址为源的 output DMA ACK。Fused Attention query-block completion
还要等待 PV/final normalization、Local GM O write 和 output DMA ACK，manager
随后才记录 unique completion。

focused evidence 必须包括逐 stage 的精确 Local GM read/write bytes、包含 partial
final chunk 的 chunk 数、context/lane high-water mark、queue/resource stall，以及
性能模式下 legacy immediate issue 为零。测试必须证明 capacity backpressure、
callback 前不推进 stage、原位地址不越界、数值等价、结果不随 callback/poll
频率改变，并保持完整 output-ACK completion 因果链。

## 5. Phase A：SST Workload 和转置感知 Operand Loading

新 workload 根目录：

`src/sst/elements/golem/tests/small/muticore_attention`

已实现 workload 文件：

- `Makefile`
- `run_muticore_attention.sh`
- `attention_case.py`：确定性输入生成和轻量运行后数学验证
- `test_muticore_attention.py`
- `README.md`

已修改的共享文件：

- `src/sst/elements/golem/tests/small/mvm_noc_int_array/golem_matmul_runtime.cpp`：
  与 `gemm_matmul_op.h`：共享逻辑 transpose 契约；
- base guest、HBM generator、verifier 和 runner：传递该契约。没有修改 SFU、
  GlobalMemory 或 RoCC 生产源码。

任务：

1. 定义统一 manifest，包含 shape、逻辑 tensor stride、原生 K 布局、
   transpose mode、mask、HBM placement、dtype 和随机种子。Q 必须保持
   未缩放。
2. 在共享 B-operand tensor/layout 契约中增加 `transpose_b`，将逻辑
   `B(k,n)` 映射到原生 `K(n,k)`，并原样复用现有 RoCC vector load 和
   completion 路径。
3. 增加 focused tests，覆盖原生 K 地址生成、非法 transpose mode、runner
   参数传递和 padded partial final key tile。
4. 直接运行真实 SST `QK^T` smoke；运行结束后使用不包含 SST 时序或 fused
   loop 行为的轻量数学验证器比较完整输出。
5. 增加 runner validation，覆盖正数 shape、受支持 D、timeout、artifact
   isolation、原生 K 布局以及互斥的 baseline/fused mode。
6. 生成的 HBM image、tensor、log 和 result JSON 均保存在 Git 之外。
7. 本阶段不引入独立 transpose SST component，也不在 HBM 中物化完整
   `K^T`。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/muticore_attention \
  -p 'test_*.py'

make -C src/sst/elements/golem/tests/small/muticore_attention clean all
make -C build/sst-elements/src/sst/elements/golem -j2
```

Phase A 已于 2026-07-22 完成：原生 K `QK^T` 在 `S=64,D=64`
（4096/4096）以及逻辑 `Skv=70`、物理补齐到 80（4480/4480）两个用例中
均为 0 mismatch。focused tests、guest 与 `libgolem` 构建、canonical GEMM
regression 和 `16x4096` Softmax smoke 均通过。RoCC 未修改，因为现有 vector
loader 已能完成所需物理传输。

## 6. Phase B：Materialized Standard Attention 基线

主要实现文件：

- 新 `muticore_attention` guest/runtime/runner；
- Phase A 的 transpose-aware RoCC operand-loading 路径；
- 现有 tensor Softmax runtime 和 Row Engine；
- `src/sst/elements/golem/sfu/sfu.h` 和 `.cc`：每 job 一次的 RSQRT state、
  融合 SCALE/MASK+MAX 执行、latency 和 statistics；
- 仅当专用 workload 无法本地表达 Q/K/V/S/P/O 布局时，才修改共享 HBM
  generator/unpacker。

任务：

1. 从原生行主序 K 通过逻辑 `transpose_b` 执行 QK；不得生成转置 K tensor
   或将其写入 HBM。
2. 将原始 materialized S 送入版本化 Attention-aware SFU 路径。SFU 先计算
   一次 RSQRT(D)，再在 row MAX 前执行 SCALE/MASK；现有 Version 1 Softmax
   job 保持不缩放、无 mask。
3. 对 `D=64` 和 `D=128` 验证 SFU 生成的 `1/sqrt(D)`，拒绝 `D=0`，并证明
   没有发出 generic primitive descriptor 或 GM scalar transfer；随后在不修改
   输入 tensor 的情况下启用并验证 causal masking。
4. 通过现有 GEMM 路径执行 PV，并等待真实 output completion。
5. 解析各阶段 descriptor、DMA、compute、ACK 和 guest wait 时间戳。
6. 分别记录 Q/K/V/O、S 和 P 的 HBM 字节数。

Phase B 必须完成的真实 SST 验收需串行执行并使用新 artifact：

```text
B1 H1 S64   D64   non-causal
B1 H1 S64   D64   causal
```

在两个测试点通过完整运行后输出验证、且 baseline result JSON 能解释全部
S/P HBM 流量之前，不进入 fused 实现。materialized `S256,D64` 和
`S1024,D128` 保留为对应 fused 点通过后的可选匹配性能对照，不再作为
local-fused 开发的前置条件。

2026-07-22 实现结果：已完成版本化 64-byte SFU 参数 ABI、每 job 一次的
RSQRT latency、融合 SCALE/MASK+MAX pass、V1 兼容路径、三阶段 materialized
runner、最终 O verifier 与逻辑 HBM 流量统计。真实
`B1,H1,S64,D64` non-causal 和 causal SST 均完成 4096 个输出元素验证，
0 mismatch，最大绝对误差分别为 `1.2131e-8` 和 `2.3586e-8`。causal P 的
上三角非零元素数为 0。每个结果均记录 S 写/读各 16 KiB、P 写/读各
16 KiB，以及 128 KiB 总逻辑 HBM 流量。因此 Phase B 已完成。

## 7. Phase C：物理 Local-Memory 基础与 Fused 原型

Phase C0 先移除 fused 关键路径上的模型 shortcut。Phase C1 再证明一个完整的
local producer-consumer Attention 输出，之后才增加多 key tile 循环。实施期间
保留已验收 materialized 路径作为回归 oracle。

可能修改的生产文件：

- `src/sst/elements/golem/rocc/roccAnalog.h`
- `src/sst/elements/golem/globalmemory/globalmemory.h`
- `src/sst/elements/golem/globalmemory/globalmemory.cc`
- `src/sst/elements/golem/sfu/sfu.h`
- `src/sst/elements/golem/sfu/sfu.cc`
- `src/sst/elements/golem/array/computeArray.h`
- `src/sst/elements/golem/array/mvmComputeArray.h`
- 若 fused executor 复用 WCP 路径，则包括
  `src/sst/elements/golem/workercmdproc/workercmdproc.h`
- `src/sst/elements/golem/tests/architecture/cpu_builder.py`
- 仅当 archive architecture shim 仍是有效 runner 依赖时才修改它

任务：

1. 为 `GlobalMemoryAPI` 增加异步本地读写操作，建模 read/write port、
   bytes/cycle、base latency、queue depth、completion tag，以及 DMA、RoCC/array、
   SFU、WCP 和 control client 之间的 arbitration。同步接口仅保留给不计入
   accelerator timing 的初始化/debug 路径。
2. 在现有 per-Core GlobalMemory layout 中保留 Attention window，并在任何
   DMA 前检查容量和重叠；不得用 RoCC-owned byte vector 充当硬件 scratch。
3. 将 SFU 整行 `context.values` 的硬件含义替换为容量受限的 Context Register
   File 加 lane FIFO/register。MAX、EXP/SUM、NORMALIZE 和 online state transition
   对 tile 数据必须发起建模的 Local GM 操作，并遵守第 4.8 节的原位协议。
4. fused 路径使用的每次本地传输都必须经过建模的 Local GM 和 array-buffer
   接口。固定 `gm2imat/gm2ivec/ovec2gm` 延迟不能替代按字节计时和共享端口竞争。
5. 引入 manager/worker role 和显式 worker-slot 到 physical-core map。manager
   Core 的 RoCC FSM dispatch query block 并聚合 unique completion；worker-local
   RoCC 执行 tile。worker Core 0 coordinator 只保留在 legacy Softmax 回归模式；
   不向 manager SFU datapath dispatch compute。
6. 增加一个版本化 Attention descriptor issue/wait 路径，不改变现有 GEMM
   或 SFU opcode 值。
7. 增加第 4.7 和 4.8 节定义的窄范围异步 SFU local-tile 操作；现有 HBM-backed
   SFU job 及其 ABI 保持不变。
8. 将 `S=32,D=64,Br=16,Bc=32` non-causal 实现为 two-query-block、每个 block
   只有一个 key/value tile 的端到端 SST 用例；两个 query block 在唯一 local
   GM context 中顺序执行。使用真实 Q/K/V DMA readiness、QK/PV array completion、
   local handoff completion、最终 normalize 和 output DMA ACK。
9. SCALE/MASK+MAX 前必须收到 SFU RSQRT-ready，并对每个有效 score 恰好缩放
   一次。score/weight tile 必须留在本地，禁止 host-side arithmetic 或功能性
   completion shortcut。
10. 为所有因果边界输出机器可读时间戳和字节数，包括 local transfer；记录
   Local GM window high-water mark，并证明 S/P HBM read/write 全部为零。
11. 将 legacy primitive/tile entry point 转换为有界异步 scheduler 的兼容
    frontend。每个 frontend 迁移完成前标记为 functional-only，并要求性能模式
    在 timing 开始前拒绝该路径。

Phase C0 退出条件：focused tests 证明容量、按字节扩展的 latency、port
contention、同核不发 NoC、显式 worker mapping、容量受限的 SFU state，以及
MAX/EXP-SUM/NORMALIZE 的精确 local bytes、partial-chunk handling、callback 有序
stage transition，并证明关键路径直接 `memcpy` 或 legacy immediate-execution
shortcut 会被拒绝。

Phase C1 退出条件：`B1,H1,S32,D64,Br16,Bc32` non-causal fused 输出通过
完整运行后验证；QK/PV 均由 MVM array 执行；S/P HBM bytes 为零；所有 S/P
local bytes 均可解释；时间线严格因果并延伸到 manager completion。

## 8. Phase D：多 Tile Online Softmax、Causal 和边界

可能修改的文件：

- Phase C 的 RoCC/SFU state；
- 仅当现有 DMA callback 无法携带所需 block identity 时，修改
  `src/sst/elements/golem/globalmemory/globalmemory.h` 和 `.cc`；
- `cpu_builder.py`：Attention block/local-window/resource 参数；
- 当前 Softmax tests 旁的 focused component contract tests。

任务：

1. 将 Phase C 状态机扩展到 `S=64,Br=16,Bc=32` 的四个 query block、两个
   key tile；每个 query row 的 `(m,l)` 保存在容量受限的 SFU Context Register
   File。O 优先保存在 array accumulator，必要时使用显式分配的 Local GM
   Oacc spill 区。
2. 精确实现第 4.1 节 recurrence，包括首 tile 的 `m=-inf,l=0` 和跨 tile
   最大值变化。
3. 在 tile row-max 前应用 SFU 生成的 scale 和 causal mask；跳过完全位于
   未来的 key tile，并对 diagonal boundary tile 逐元素 mask。
4. 正确处理 partial `Sq`、`Skv`、`D` tile，padding 不得影响 max、sum 或
   output。
5. 仅在最后一个有效 key tile 后归一化，每完成一个 query block 发出一次
   output DMA。每个 unique query-block completion 都必须晚于对应 output DMA
   ACK，全部 query block 完成后才允许 job completion。
6. 在每个异步 callback 上验证 job/block/owner/shape identity，并拒绝 duplicate
   或 stale completion。

focused tests 必须覆盖：

- 第二个 key tile 提高 running max 的两 tile 用例；
- RSQRT(D) 每个 job 只执行一次，对 `D=64` 和 `D=128` 的有效 score 恰好
  缩放一次，并拒绝 `D=0`；
- 数值极端 logits；
- causal diagonal 和完全 masked future tile skipping；
- 最后一个 query/key partial tile；
- duplicate/stale completion rejection；
- Local GM window overflow 和 unsupported descriptor rejection。

阶段退出条件：fused non-causal 和 causal `S64,D64` 通过完整运行后输出验证，
S/P HBM bytes 为零；随后通过 `Sq=20,Skv=70,D64` partial shape 和一个小型
extreme-logit SST 用例。`S256,D64` 和 `S1024,D128` 移到规模阶段。

当前进度（2026-07-23）：D1 non-causal、D2 causal `S64,D64` 和 D3
`Sq=20,Skv=70,D64` partial shape 均已通过完整 SST 验证，S/P HBM bytes 为零，
RSQRT 每 job 一次。D3 尾部 query/key 有效长度为 4/6，QK/PV ops=140/240、
SFU jobs/rows=6/60，1,280 个输出 mismatch=0。D3 检查点当时仅剩小型
extreme-logit fused SST，尚未满足 Phase D 退出条件。

D4 使用确定性的 `S64,D64` 输入，使第一、第二 KV tile 的缩放 logits 分别恰为
-100 和 +100，强制 online running max 增加 200。真实 fused SST 的 4,096 个
输出全部通过，mismatch=0、max abs error=0；精确 D1 活动保持
`QK/PV=256/512`、`SFU jobs/rows=8/128`、RSQRT count=1、S/P HBM bytes=0。
同轮默认 D1 回归也通过。Phase D 已关闭，下一步进入 Phase E。

## 9. Phase E：Manager 协调的 16 Tile Prefill 映射

使用现有 dedicated group manager 作为硬件控制平面。每个 manager 协调其显式
worker group；worker-local RoCC 执行独立 query block，并始终把 tile 数据留在
本地。不增加一个搬运全局数据的单点 controller。

任务：

1. 定义从 manager/group 到显式 physical worker Core ID 的版本化 topology
   map。在 16 个 worker 上分配连续 query-block band 并平衡最后一个 partial
   wave，不假定 Core ID 连续。
2. 在四个 HBM node 上 stripe Q/O band。K/V 初始采用 block stripe、不复制；
   收集流量和 NoC hotspot 后，再决定是否按列复制 K/V。
3. 每个已发出的 query block 必须恰好产生一个 unique completion；所有 output
   DMA ACK 后才产生一次 tensor-level completion。
4. 保持每 tile Local GM window 隔离，拒绝 context 或 HBM output range 重叠的
   job。
5. 扩展独立 Attention dispatch/completion message，不得复用 Softmax row
   message。manager completion 必须等待映射中的精确 worker/query-block 集合，
   并且对应 output DMA ACK 全部完成。

规模阶梯：

```text
B1 H1 S64   D64
B1 H1 S256  D64
B1 H1 S1024 D128
B1 H1 S2048 D128
B1 H1 S4096 D128
```

每个点都必须运行 fused。S64 必须有匹配的 materialized 对照，其他规模仅在
需要性能比较时运行 materialized，不得因此推迟 fused 功能扩展。前一 fused
shape 未通过正确性、生命周期、local-window capacity 和单调 scaling 检查前，不运行
更大 shape。

### Phase E 实施进度（2026-07-23）

Phase E1 的真实 `B1,H1,S256,D64,Br16,Bc32` fused SST 已通过。manager core
0-3 分别显式映射 worker core 4-19；每个 manager 管理 64-row query band，并向
4 个 worker 各分发一个 16-row block。Q/O 和 K/V block stripe 在 HBM node 1-4；
worker 每次只将一个 32-row K/V tile 流入 per-Core GlobalMemory。16,384 个输出
全部通过，mismatch=0、max abs error=`4.4967521123807225e-09`；精确的每 manager/
worker 活动计数通过，S/P HBM bytes=0。

当前检查点产生 4 个 manager-level band completion，尚未满足任务 3 要求的单一
tensor-level completion。Phase E2 必须增加 root/跨 manager aggregation，等待四个
manager band（因此也等待全部 16 个 worker 的 output DMA ACK）后只发出一次 tensor
completion。E2 通过后才进入 `S1024,D128` 规模点。

Phase E2 现已满足该条件。每个 manager 先验证本组四个唯一 worker completion；
manager 1-3 再向 root manager core 0 发送独立的 `AttentionManagerComplete` 消息，
root 使用第二级 bitmap 校验 manager slot/core。最终 S256 运行中，四个 manager
各有一个 local band completion，root 收到四个 band completion，全系统仅 core 0
产生一次 tensor completion。16,384 个输出仍全部通过，max abs error=
`4.4967521123807225e-09`，S/P HBM bytes=0。

Phase E3 的真实 `B1,H1,S1024,D128,Br16,Bc32` fused SST 现已通过。每个 worker
处理四个 16-row query block，并使用 8 个 PV dimension panel，同时只保留一个
32-row K/V tile；完整 Attention Local GM window 有界为 51,328 bytes。131,072 个
输出全部通过，mismatch=0、max abs error=`1.5966506535956479e-09`，S/P HBM
bytes=0。每个 worker 的精确活动计数、四个唯一 manager band、root 的四次接收和
唯一 root tensor completion 均通过。

Phase E4 的真实 `B1,H1,S2048,D128,Br16,Bc32` fused SST 现已通过。每个 worker
顺序处理八个 16-row query block 和 64 个 KV tile，同时保持 51,328-byte Local GM
window 不变。262,144 个输出全部通过，mismatch=0、max abs error=
`1.181374809935097e-09`，S/P HBM bytes=0。精确 worker 活动计数、四个唯一
manager band、root 的四次接收和唯一 tensor completion 均通过。下一 Phase E
增量为 `B1,H1,S4096,D128`，启动前必须先通过 dry-run 和 wall-clock/watchdog 审查。

## 10. Phase F：性能和大模型相关扩展

每一组匹配的 baseline/fused 测试记录：

- descriptor-to-accelerator completion cycles；
- guest issue-to-wait-return cycles 和 whole SST time，二者分开记录；
- useful FLOPs 和 elements/cycle；
- 每个 HBM node 的 Q/K/V/O、S/P DMA 操作数和字节数，包括遍历 query block
  导致的每一次 K/V reload；
- QK、RSQRT-ready、SCALE/MASK、online Softmax、PV、normalize 和
  output-ACK 时间窗口；
- MVM array occupancy、SFU vector/EXP occupancy、Local GM window high-water mark；
- NoC port utilization/stall 和 memory queue delay；
- retry、rejected、stale、duplicate 和 wait-poll 数量；
- host wall time 以及 artifact/signature metadata。

single-head FP32 fused 路径通过后，按以下顺序扩展：

1. 多个独立 query head；
2. `Hq != Hkv` 的 GQA/MQA；
3. BF16/FP16 storage，并明确 accumulation precision；
4. `Sq != Skv` 的 cross-attention；
5. 将 decode/KV-cache 作为独立 kernel 和验收路径。

不得将 prefill kernel 标记为 decode 结果。decode 预计受 KV-cache bandwidth
限制，需要独立布局、流量模型和 shape 集合。

## 11. 回归和编译边界

修改任何 SFU、GlobalMemory、RoCC、array、architecture wiring 或共享 runner
后，运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu \
  -p 'test_sfu_softmax_*.py'

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/muticore_softmax \
  -p 'test_*.py'

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/muticore_attention \
  -p 'test_*.py'

make -C build/sst-elements/src/sst/elements/golem -j2
make -C src/sst/elements/golem/tests/small/muticore_softmax all
make -C src/sst/elements/golem/tests/small/muticore_attention all
```

在 restricted network namespace 之外串行运行 canonical GEMM regression：

```bash
TMPDIR=/data4/jjgong/tmp \
GOLEM_ARTIFACT_ROOT=/data4/jjgong/tmp/attention_gemm_regression \
env -u GOLEM_SFU_ENABLE -u GOLEM_SFU_STANDALONE_SOFTMAX \
  -u GOLEM_SFU_JOB_SOFTMAX -u GOLEM_SFU_PRIMITIVE_SOFTMAX \
  -u GOLEM_SFU_REDUCTION_VN -u GOLEM_DMA_RESPONSE_VN -u GOLEM_ARCH_SCRIPT \
  -u GOLEM_GROUP_MANAGER_ENABLE -u GOLEM_CTRL_LINK_ENABLE \
  -u GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE \
  bash src/sst/elements/golem/tests/run_noc_dma_pipeline.sh \
  --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --dtype fp32 --tensor-source sample --verify-c
```

GEMM regression 必须满足：exit 0、simulation completion、`VERIFY-C PASS`、
完整 DMA lifecycle、零 retry、零非预期 SFU/Attention activity。

真实 SST Attention 和 Softmax 必须串行运行、使用固定 watchdog 并写入新的
artifact root。最终 closeout 还需运行 `git diff --check`，并记录 production
library、guest binary、runner、manifest、input 和 output hash。

## 12. 停止条件和决策门槛

出现以下任一情况时停止扩展并先诊断：

- baseline composition 无法通过完整运行后输出验证；
- output DMA ACK 前释放 completion；
- fused 模式把 S 或 P 写入 HBM；
- online state 随 callback/poll 频率变化；
- SFU RSQRT/SCALE 缺失、重复执行或在 row MAX 之后执行；
- partial 或 causal tile 的结果超出容差；
- Local GM window 容量超限、发生 alias，或被提供为 HBM spill address；
- local score/weight/O transfer 未建模 bytes 和 cycles 即完成；
- SFU 整行/tile、WCP partial-C tile 或 operand payload 保存在没有声明硬件容量的
  C++ container 中；
- RoCC、WCP 或 SFU 在性能路径上直接读写 Local GM/array vector，未参与建模的
  port arbitration；
- manager dispatch 把 worker slot 当作 physical Core ID，或 manager 代码读取/
  复制 worker S/P 数据；
- 已测量的多核路径通过 shared static reducer/map 绕过 explicit NoC transport；
- retry、stale callback、duplicate completion 或无法解释的 bytes 非零；
- GEMM 或已验收的 `1024x4096` Softmax 回归失败；
- 第一组匹配的 fused 运行比 baseline 慢，但统计无法指出具体建模瓶颈。

### 12.1 模型真实性审计与整改顺序

持久化证据和源码位置记录在
`src/sst/elements/golem/tests/small/muticore_attention/findings.md`。当前审计分级：

| 优先级 | 模型边界 | 当前 shortcut | 必须采取的措施 |
| --- | --- | --- | --- |
| P0 | per-Core GlobalMemory | 同步 `std::copy`，没有 local port/bandwidth/queue | fused 开发前实现统一异步本地访问调度器 |
| P0 | SFU Row Engine | 整行留在 `context.values`；legacy primitive/tile issue 立即读算写 | 使用第 4.8 节的有界异步 chunk scheduler，tile 留在 Local GM，迁移 legacy frontend，并在性能运行中拒绝尚未迁移的路径 |
| P0 | RoCC 本地搬运 | GM2IMAT/GM2IVEC/OVEC2GM 使用与 bytes 无关的固定延迟并直接访问 vector | 使用按字节计时且有竞争的 Local GM/array-buffer 操作 |
| P0 | manager/worker control | 已验收 Softmax 使用 worker 0；dispatch 混同 slot 与物理 Core ID | 增加 manager role、显式 mapping 和 manager-only control state |
| P1 | WorkerCommandProcessor | panel payload 和 partial-C tile 保存在 host vector，并直接复制 array | 复用 WCP 前将其放入声明的 Local GM/array storage，并建模 transfer/accumulator capacity |
| P1 | ComputeArray buffer | arithmetic latency 已建模，但 matrix/input/output programming 和 output move 立即完成 | 保留功能算术，增加有界 buffer port、transfer cycle、occupancy 和 stall 后再给最终性能结论 |
| P1 | 跨核 reduction/control | 进程级 static map 可代表共享 reducer；mailbox 绕过本地内存时序 | 已测量跨核操作强制 explicit NoC；mailbox 明确定义为建模寄存器或 Local GM client |
| 已接受 | NoC/HBM/cache | Merlin mesh、memHierarchy 和 DRAMSim3 已提供 queue、bandwidth 和 latency | 保留并回归，不新建项目私有替代模型 |

P0 是 Phase C0 的进入门槛。P1 必须在相关路径被用于性能结论前完成，但不阻塞
无关的 legacy regression。只要容量、resource occupancy、transfer timing 和
completion 已物理建模，功能性 C++ 算术可以保留；本项目不要求 RTL 或门级
浮点仿真。
P2 校准放在 fused 功能验收后：用目标 CIM datapath 校准 MVM latency 公式，
定义 SFU 相对理想 `std::exp/sqrt` 的近似和 rounding 行为，并仅在项目开始报告
area/energy 时加入对应模型。

原生 K `QK^T`、materialized `B1,H1,S64,D64` non-causal/causal、Phase C0
真实性整改、在线 recurrence 四类验收、4 manager/16 worker 映射、单一 tensor
completion、Phase E3 `S1024,D128` 和 Phase E4 `S2048,D128` fused 规模点均已
完成。下一里程碑是 Phase E5 `S4096,D128`。更大的 materialized 运行仅为可选对照；multi-head、mixed
precision、K/V replication、单点全局 scheduler 和 decode 仍放在当前规模阶梯
验收之后。
