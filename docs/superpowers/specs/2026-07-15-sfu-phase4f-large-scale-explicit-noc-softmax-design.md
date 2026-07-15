# SFU Phase 4F Large-Scale Explicit-NoC Softmax Design

## Purpose

Characterize unified-job softmax performance as matrix dimensions, worker
count, and row count scale under the same canonical NoC configuration already
used by the mature GEMM tests. Phase 4F uses only the completed
`explicit_noc` reduction transport and does not sweep bandwidth or VN.

The bandwidth-pressure experiment is deferred as an optional diagnostic. It is
not required to establish normal-operation softmax performance because Phase
4E already validates real reduction request/response traffic under the same
network profile as canonical GEMM.

## Canonical GEMM Network Profile

The source of truth is
`src/sst/elements/golem/tests/configs/30_network.env`, which is automatically
loaded by `run_noc_dma_pipeline.sh` through `configs/default.env` before the
script applies fallback values.

Phase 4F pins the resolved canonical values explicitly:

```text
GOLEM_NOC_LINK_BW=1200GB/s
GOLEM_NOC_XBAR_BW=1200GB/s
GOLEM_DIRCTRL_HIGHLINK_BW=1200GB/s
GOLEM_NOC_INPUT_BUF_SIZE=512KB
GOLEM_NOC_OUTPUT_BUF_SIZE=512KB
GOLEM_NOC_FLIT_SIZE=128B
GOLEM_GM_BUFFER_LENGTH=1024KB
GOLEM_NOC_INTER_ROUTER_NO_CUT=0
GOLEM_NOC_LOCAL_NO_CUT=0
```

These are not the emergency fallback literals inside
`run_noc_dma_pipeline.sh`; those fallbacks are `25GB/s` and `8KB` and apply
only when the preset values are absent. Explicit environment or CLI values can
override both the preset and the fallbacks. The canonical network preset and
completed default GEMM regression resolve link, xbar, and directory highlink
to `1200GB/s`, router input/output buffers to `512KB`, and flit size to `128B`.
The existing 4096-shaped SST artifact records the same resolved network
profile; it is supporting configuration evidence, not a substitute for
identifying the workload mode of a pure GEMM run.

## Softmax Architecture Contract

Every Phase 4F point uses:

```text
distributed_reduction_transport=explicit_noc
num_vns=3
request_vn=0
ordinary_response_vn=1
dma_response_vn=0
reduction_vn=0
chunk_elems=256
staging_rows=4
job_rows=4
cooperative_groups=1
retry_ticks=1024
max_retries=8
```

VN0/VN1/VN2 compatibility is complete in Phase 4E. `modeled_noc` is not part
of the Phase 4F matrix. No bandwidth, xbar, flit, buffer, topology, retry, or
VN parameter may vary across performance points except memory-node capacity
when required to hold a larger tensor shape.

Phase 4F does not modify SFU math, reduction messages, GlobalMemory,
SimpleNetwork, GEMM, guest ABI, or primitive/batch softmax.

## Experiment Matrix

The matrix is staged so only one scale axis changes at a time. Duplicate
anchor identities are executed once.

### Stage A: Dimension Scaling

Fix `rows=16`, `worker_cores=16`, and `band_cores=16`:

```text
16x512
16x1024
16x2048
16x4096
```

This stage measures increasing per-row column work with maximum available
column cooperation.

### Stage B: Worker Scaling at Large Dimension

Fix `rows=16` and `dim=4096`:

```text
worker_cores/band_cores = 4/4
worker_cores/band_cores = 8/8
worker_cores/band_cores = 16/16
```

The `16/16` point is the Stage A `16x4096` anchor and is not rerun when its
signature and artifacts remain valid.

### Stage C: Row Scaling at Large Dimension

Fix `dim=4096`, `worker_cores=16`, and `band_cores=16`:

```text
rows=16
rows=64
rows=256
```

The `rows=16` point is the shared Stage A/B anchor. Stage C runs serially and
stops on the first invalid or timed-out point; it does not silently reduce rows
or change retry/network parameters.

The resulting default matrix has eight unique real SST points:

```text
16:512:16:16
16:1024:16:16
16:2048:16:16
16:4096:16:16
16:4096:4:4
16:4096:8:8
64:4096:16:16
256:4096:16:16
```

## Memory Capacity and Timeout Policy

Memory capacity is a feasibility setting, not a performance-search axis. The
runner records the resolved value and uses:

```text
dim <= 1024: mem_node_size=134217728 bytes
dim >= 2048: mem_node_size=268435456 bytes
```

Timeouts are shape classes and must be recorded:

```text
16x512: 900 seconds
16x1024: 1800 seconds
16x2048: 2400 seconds
16x4096: 3600 seconds
64x4096: 7200 seconds
256x4096: 14400 seconds
```

A timeout is an experiment result with status `TIMEOUT`, not permission to
alter the network or correctness contract. Recovery may resume from completed
markers in the same root only when the complete point signature matches.

## Runner Architecture

Create a dedicated orchestrator:

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  run_sfu_phase4f_large_scale_explicit_noc.sh
```

It invokes the existing
`run_sfu_unified_job_distributed_scaling.sh` one point at a time in child roots.
The generic GEMM runner and architecture files remain unchanged.

The orchestrator must:

- pin the canonical GEMM network profile and explicit-NoC/VN0 contract;
- reject conflicting inherited transport, VN, network, buffer, retry, chunk,
  staging, or job-row values;
- use a collision-safe parent root lock;
- include stage and full point identity in child root names and signatures;
- skip an already valid duplicate anchor instead of rerunning it;
- reject stale manifest schemas, stale markers, hash mismatches, and incomplete
  child artifacts;
- support dry-run, focused point-list override, resume, and stop-on-fail;
- execute real points serially.

Supported controls are:

```text
GOLEM_PHASE4F_LARGE_SCALE_ROOT=<fresh absolute root>
GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN=1
GOLEM_PHASE4F_LARGE_SCALE_STOP_ON_FAIL=1
GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="rows:dim:workers:bands ..."
```

The default point list is exactly the eight-point matrix above. Overrides are
for focused recovery and tests; resolved points remain fully signed.

## Artifact and Correctness Gates

Every usable point requires:

- child manifest `PASS/PASS` with exit code zero;
- independent full-row logits golden with `checked = rows * dim` and zero
  mismatches;
- correct active SFU worker/band counts;
- Max/Sum request and response totals each equal `rows * worker_cores`;
- explicit transport receives equal `4 * rows * worker_cores`;
- GlobalMemory immediate plus queued sends equal the same transport total;
- rejected and stale reduction messages equal zero;
- resolved VN and network profile exactly match the fixed contracts;
- DMA issue/completion and bytes match `rows`, `dim`, and worker partitioning;
- DMA retry, exhaustion, and write-timeout retry equal zero;
- output size, output hash, signature, log, stats, NoC summary, and DMA summary
  are present and mutually consistent.

Reduction queueing is recorded but is not a failure when all other transport
and lifecycle gates pass. The fixed 1200GB/s profile remains unchanged even if
a large-scale point queues.

## Parent Manifest and Metrics

The parent `large_scale_manifest.csv` stores one canonical row per unique point:

```text
run_id,stage,rows,dim,chunk_elems,worker_cores,band_cores,
transport,reduction_vn,num_vns,dma_response_vn,noc_link_bw,noc_xbar_bw,
dirctrl_highlink_bw,noc_input_buffer,noc_output_buffer,gm_buffer,flit_size,
mem_node_size,retry_ticks,max_retries,timeout_sec,status,exit_code,
artifact_validation,golden_checked,golden_mismatches,transport_events,
transport_immediate,transport_queued,transport_rejected,transport_stale,
inbox_high_water,latency_avg_cycles,latency_max_cycles,total_send_packets,
total_send_bits,total_xbar_stalls,simulated_time_us,wall_time_sec,dma_timeout_retry,
dma_timeout_exhausted,dma_write_timeout_retry,output_sha256,child_root
```

Primary performance metrics are:

- simulated time and wall time;
- average and maximum reduction transport latency;
- transport events and queued sends;
- total packets, bits, and xbar stalls;
- normalized time per row and per element;
- worker scaling speedup and efficiency at `dim=4096`;
- DMA issue/completion, bytes, retry, and round-trip metrics.

These are deterministic single SST outcomes. No error bars, confidence
intervals, or statistical significance claims are permitted.

## Analysis and Figure Bundle

Create:

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  plot_sfu_phase4f_large_scale.py
```

It reparses the parent and child evidence, reruns all gates, and generates:

```text
tests/artifacts/sweeps/sfu_phase4f_large_scale_explicit_noc_20260715/report/
  sfu_phase4f_large_scale_source_data.csv
  sfu_phase4f_large_scale.svg
  sfu_phase4f_large_scale.pdf
  sfu_phase4f_large_scale.png
  sfu_phase4f_large_scale_qa.md
```

The English 16:9 figure contains:

- dimension scaling of runtime and reduction latency;
- worker scaling speedup/efficiency at `dim=4096`;
- row scaling of total time and normalized time per row;
- NoC pressure metrics and a compact correctness/lifecycle block.

It uses explicit-NoC only and labels the fixed canonical GEMM network profile.
It does not include modeled-NoC, bandwidth comparisons, future fusion plans,
or inferential statistics.

## Test Strategy

Add:

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  test_sfu_phase4f_large_scale.py
```

Focused tests cover:

- exact canonical GEMM network values and their preset source;
- exact eight-point default matrix and duplicate-anchor elimination;
- dimension-specific memory/timeout resolution;
- explicit-NoC/VN0-only environment and inherited-conflict rejection;
- network profile in signature, manifest, child environment, and runtime log;
- root locking, dry-run, resume, stale schema, stale marker, and corrupted
  output behavior;
- shape-derived golden, transport, DMA, byte, and output-size gates;
- parsing latency, queueing, packet/bit, xbar, runtime, and DMA metrics;
- invalid/timeout stop behavior without parameter mutation;
- deterministic CSV/SVG/PDF/PNG generation and complete source reconstruction;
- editable SVG text, TrueType PDF text, 300 dpi PNG, and visual non-overlap.

Synthetic fixtures and dry-runs cover runner/analyzer behavior. Real SST points
start only after focused tests pass.

## GEMM Isolation

The dedicated Phase 4F runner must not modify
`src/sst/elements/golem/tests/run_noc_dma_pipeline.sh`, GEMM architecture files,
or GEMM guest binaries.

If implementation remains confined to new softmax runner/analyzer/tests, the
existing canonical GEMM artifacts and focused isolation tests are sufficient.
If any shared runner, preset, architecture, or production component changes,
the implementation must rerun the existing default GEMM regression before
Phase 4F is accepted.

## Execution Order

1. Record the canonical GEMM network evidence and Phase 4E explicit-NoC anchor.
2. Implement the dedicated runner contracts under focused TDD.
3. Implement the artifact parser and report generator under focused TDD.
4. Run the complete eight-point dry-run and inspect resolved signatures.
5. Run Stage A serially and stop on the first invalid point.
6. Reuse the valid 16x4096 anchor, then run Stage B serially.
7. Reuse the same anchor, then run Stage C serially.
8. Generate and visually inspect the deterministic report bundle.
9. Run the complete focused softmax suite and the GEMM isolation gate.

## Success Criteria

Phase 4F is complete when:

- all executable points are represented by canonical parent/child artifacts;
- every usable point passes golden, transport, DMA, topology, and artifact
  gates;
- any invalid or timeout point is reported without changing the network or
  correctness contract;
- source CSV and editable deterministic figures reconstruct every reported
  number;
- no modeled-NoC or bandwidth sweep appears in the main matrix;
- the fixed network values match the canonical GEMM preset and runtime logs;
- no GEMM path is changed, or the existing GEMM regression passes if a shared
  file change becomes unavoidable.

## Scope After Phase 4F

After large-scale softmax behavior is characterized, the next decision is
whether to optimize the dominant softmax bottleneck or begin GEMM+softmax
fusion. Bandwidth-pressure experiments remain optional diagnostics and are not
part of the default roadmap.
