# SFU Phase 4F NoC Pressure Experiment Design

> **Status: Deferred optional diagnostic.** This experiment is no longer the
> next softmax mainline. The canonical GEMM and Phase 4E softmax artifacts both
> resolve the same `1200GB/s` NoC profile, so the approved next step is the
> fixed-network large-scale explicit-NoC design in
> `2026-07-15-sfu-phase4f-large-scale-explicit-noc-softmax-design.md`.

## Purpose

Measure where the completed unified-job `explicit_noc` softmax path first
experiences reduction-transport backpressure as NoC link bandwidth decreases.
The experiment keeps the workload, worker placement, VN mapping, endpoint
buffers, retry policy, and softmax implementation fixed so link bandwidth is
the only primary experimental axis.

Phase 4F is an experiment and reporting cut. It does not change SFU math,
reduction message semantics, SimpleNetwork routing, GEMM, guest ABI, or the
primitive/batch softmax path.

## Mode Policy

`explicit_noc` is the performance mainline. Every bandwidth point is an
explicit-NoC run.

`modeled_noc` is retained only as a sparse attribution control because
`GOLEM_NOC_LINK_BW` affects DMA and ordinary memory traffic as well as reduction
traffic. Phase 4F runs modeled-NoC at:

1. the `1200GB/s` high-bandwidth baseline; and
2. the first valid explicit-NoC bandwidth point with nonzero reduction send
   queueing.

If no valid explicit point queues through `25GB/s`, there is no second modeled
control. Later large-scale softmax sweeps use explicit-NoC only; Phase 4E
artifacts remain the completed VN compatibility evidence.

## Fixed Experiment Contract

Every Phase 4F point uses:

```text
rows=16
dim=512
chunk_elems=256
worker_cores=16
band_cores=16
staging_rows=4
job_rows=4
cooperative_groups=1
num_vns=3
reduction_vn=0
dma_response_vn=0
retry_ticks=1024
max_retries=8
NoC input buffer=64KB
NoC output buffer=64KB
GlobalMemory network buffer=64KB
```

The VN mapping is frozen as request VN0, ordinary response VN1, directory DMA
completion VN0, and reduction VN0. VN1/VN2 are not performance variables.

The dedicated runner must explicitly set every fixed value rather than trust
inherited shell state. It must reject inherited values that conflict with this
contract.

## Bandwidth Schedule

Run explicit-NoC serially at:

```text
1200GB/s, 600GB/s, 300GB/s, 150GB/s
```

After those four points:

- if at least one valid explicit point has `transport_queued > 0`, stop the
  explicit schedule and run the sparse modeled control at the first queued
  bandwidth;
- if all four points have `transport_queued == 0`, extend explicit-NoC to
  `75GB/s` and then `25GB/s`;
- after the extension, run the second modeled control only if a valid queued
  point exists;
- if no point queues at `25GB/s`, record `NO_QUEUE_WITHIN_RANGE` and end the
  bandwidth experiment. Endpoint buffer size becomes a separate future axis;
  Phase 4F must not vary bandwidth and buffer size in the same matrix.

"First queued bandwidth" means the first point encountered while descending
from 1200GB/s whose completed explicit artifact has
`gmem_reduction_send_queued > 0`.

## Validity and Classification

Every point must be parsed from completed artifacts. A run is usable only when
all of these gates pass:

- manifest status and artifact validation are `PASS/PASS`;
- the standalone logits golden reports `8192 checked, 0 mismatches`;
- Max/Sum request and response totals each equal `rows * worker_cores = 256`;
- explicit transport receives equal `4 * rows * worker_cores = 1024`;
- `transport_immediate + transport_queued = 1024`;
- reduction rejected and stale totals are zero;
- DMA timeout retry, timeout exhaustion, and write-timeout retry are zero;
- DMA issue/completion and byte totals match the existing scaling-runner
  contract;
- the resolved runtime VN mapping is `num_vns=3`, `reduction_vn=0`, and
  `dma_response_vn=0`;
- the output file hash and point signature match the completion marker.

Classify a point as:

- `VALID_NO_BACKPRESSURE`: all gates pass and `transport_queued == 0`;
- `VALID_BACKPRESSURE`: all gates pass and `transport_queued > 0`;
- `INVALID`: any correctness, lifecycle, transport, topology, timeout, or
  artifact gate fails.

An `INVALID` point is recorded but cannot define the backpressure boundary.
The runner stops on the first invalid explicit point; it must not silently
increase retry limits or change buffers because that would add another axis.

## Runner Architecture

Create a dedicated pressure runner:

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  run_sfu_phase4f_noc_pressure_sweep.sh
```

The pressure runner orchestrates the bandwidth schedule and invokes the
existing `run_sfu_unified_job_distributed_scaling.sh` for one point per child
root. The existing generic GEMM runner remains unchanged.

Each child root includes transport and sanitized bandwidth in its name. The
parent pressure runner owns a collision-safe root lock and a single
`pressure_manifest.csv`. Point identity must include at least:

```text
transport, link_bw, rows, dim, chunk, workers, bands,
reduction_vn, num_vns, dma_response_vn, retry_ticks,
max_retries, buffer sizes, and child runner arguments
```

Changing any identity field invalidates a cached completion marker. A stale or
schema-incompatible parent manifest is rejected rather than appended.

The pressure runner supports:

```text
GOLEM_PHASE4F_BANDWIDTH_LIST="1200GB/s 600GB/s 300GB/s 150GB/s"
GOLEM_PHASE4F_EXTENSION_BANDWIDTH_LIST="75GB/s 25GB/s"
GOLEM_PHASE4F_DRY_RUN=1
GOLEM_PHASE4F_STOP_ON_FAIL=1
GOLEM_PHASE4F_ROOT=<fresh absolute root>
```

The default lists are the approved schedule. Overrides exist for focused
tests and recovery, but each resolved list is recorded verbatim.

## Parent Manifest

`pressure_manifest.csv` contains one canonical row per executed point with
these fields:

```text
run_id,transport,link_bw,rows,dim,chunk_elems,worker_cores,band_cores,
reduction_vn,num_vns,dma_response_vn,noc_input_buffer,noc_output_buffer,
gm_buffer,retry_ticks,max_retries,status,artifact_validation,golden_checked,
golden_mismatches,transport_events,transport_immediate,transport_queued,
transport_rejected,transport_stale,inbox_high_water,latency_avg_cycles,
latency_max_cycles,total_send_packets,total_send_bits,total_xbar_stalls,
simulated_time_us,dma_timeout_retry,dma_timeout_exhausted,
dma_write_timeout_retry,classification,child_root
```

Rows are ordered by execution time. Re-running a valid cached point may append
a `CACHED` audit row, but analysis selects exactly one original `PASS/PASS` row
per transport/bandwidth identity.

## Analysis and Reporting

Create a Python standard-library plus matplotlib analyzer:

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  plot_sfu_phase4f_noc_pressure.py
```

The analyzer reads the parent manifest and child evidence, recomputes all
validity gates, and writes:

```text
tests/artifacts/sweeps/sfu_phase4f_noc_pressure_20260715/report/
  sfu_phase4f_noc_pressure_source_data.csv
  sfu_phase4f_noc_pressure.svg
  sfu_phase4f_noc_pressure.pdf
  sfu_phase4f_noc_pressure.png
  sfu_phase4f_noc_pressure_qa.md
```

The result figure is English and 16:9. It reports deterministic single SST
outcomes without error bars or significance claims. It must show:

- average and maximum explicit reduction latency versus bandwidth;
- reduction queued messages versus bandwidth, with the first valid queued
  point directly marked;
- total xbar stalls and end-to-end simulated time versus bandwidth;
- the two sparse modeled controls only as attribution markers, not as a second
  full performance series;
- a validation block with golden, DMA, rejected/stale, event-total, and fixed
  VN/buffer evidence.

If no valid queued point exists, the figure states
`No reduction queueing observed down to 25GB/s`; it must not invent a threshold.

## Test Strategy

Add focused tests in:

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
  test_sfu_phase4f_noc_pressure.py
```

Tests must cover:

- fixed explicit-NoC/VN0/workload/buffer environment;
- bandwidth in child root, run ID, signature, and parent manifest;
- exact default and extension schedules;
- extension only when the initial schedule has no valid queued point;
- sparse modeled controls at 1200GB/s and the first valid queued bandwidth;
- no second modeled point for `NO_QUEUE_WITHIN_RANGE`;
- rejection of inherited VN, transport, workload, buffer, and retry conflicts;
- root locking, stale manifest schema, stale markers, and corrupted output;
- queue classification and invalid-point stop behavior;
- parser aggregation for latency, queueing, xbar, packets/bits, runtime, DMA,
  and golden evidence;
- failure on missing, duplicate, malformed, non-PASS, or contradictory
  artifacts;
- deterministic CSV and SVG/PDF/PNG generation;
- valid editable SVG text, TrueType PDF text, 300 dpi PNG, and visual layout.

Runner tests use dry-run or synthetic artifact fixtures. Real SST runs occur
only after focused tests pass.

## Execution Order

1. Capture the current explicit-NoC `1200GB/s` anchor and default GEMM evidence.
2. Implement and test pressure-runner identity, schedule, classification, and
   artifact contracts.
3. Run dry-run with the full approved schedule.
4. Run explicit-NoC at 1200/600/300/150GB/s serially.
5. Extend to 75/25GB/s only when required by the queue rule.
6. Run modeled-NoC at 1200GB/s and, if present, the first valid queued
   bandwidth.
7. Generate the source CSV, figure bundle, and QA record.
8. Run the full focused softmax tests.
9. Run the existing default `64x64x64` fp32 GEMM regression through
   `src/sst/elements/golem/tests/run_noc_dma_pipeline.sh --verify-c` with SFU
   and pressure variables explicitly unset.

## Success Criteria

Phase 4F is complete when:

- every scheduled explicit point is either validly classified or the runner
  stops with a recorded invalid point;
- the first valid reduction-queue bandwidth is reported, or
  `NO_QUEUE_WITHIN_RANGE` is reported truthfully;
- sparse modeled controls are limited to the approved baseline and threshold
  identities;
- all usable points pass golden, transport, lifecycle, DMA, topology, and
  artifact gates;
- source CSV and deterministic editable figure exports reconstruct every
  reported number;
- no production SFU/GEMM/NoC component is modified;
- the existing default GEMM regression remains PASS.

## Scope After Phase 4F

After the bandwidth boundary is established, the next plan performs
explicit-NoC-only softmax dimension scaling at fixed high-bandwidth and
near-boundary profiles. GEMM+softmax fusion remains deferred until softmax
large-scale performance is characterized.
