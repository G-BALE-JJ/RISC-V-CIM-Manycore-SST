# MVM NoC Softmax SFU

This directory contains the small-test workload, checker, and local notes for
the `golem.SFU` softmax and standalone primitive paths.

## Current Status

- `golem.SFU` is implemented as an independent SST subcomponent.
- SFU mounting is controlled by `GOLEM_SFU_ENABLE`; the default GEMM path keeps
  SFU disabled.
- The fused path exposes full row-wise softmax over GEMM output tiles.
- Standalone softmax-only mode can read logits from HBM, run SFU softmax, and
  write results back to HBM without running GEMM.
- Standalone primitive mode exposes a generic local-GM fp32 primitive ABI.
- HBM streaming primitive benchmark now covers a minimal `HBM -> local GM ->
  SFU primitive -> HBM` path for `EXP`.
- HBM streaming primitive can optionally batch multiple chunk descriptors into
  one SFU issue with `GOLEM_SFU_PRIMITIVE_HBM_BATCH=1`.
- Current primitive ops are `EXP`, `LOG`, `RECIPROCAL`, `RSQRT`, `TANH`, and
  `SIGMOID`.
- Primitive math is currently implemented as an SST host C++ functional model,
  not RTL or a cycle-accurate math pipeline.

## Modeling Level And SFU Roadmap

The current Golem/Vanadis setup is a C++ SST architecture-level simulation, not
an RTL simulation. Vanadis, memHierarchy, merlin, DRAMSim3, and the Golem
subcomponents are C++ models with event/timing behavior such as queues,
latencies, memory transactions, NoC routing, issue/wait paths, and statistics.
They do not model Verilog/SystemVerilog signals, flip-flops, or combinational
paths.

The current SFU primitive implementation is therefore best described as a
functional SFU model wrapped in SST event/stat infrastructure. Its primitive
math uses host C++ functions such as `std::exp`, `std::log`, and reciprocal
calculation. This is enough for validating the softmax programming model,
descriptor ABI, RoCC command path, HBM/DMA data movement, and event scaling, but
it is not enough to claim RTL-level area, frequency, pipeline hazard, or exact
cycle behavior.

The next SFU modeling target is a hardware-like timing model rather than an RTL
implementation. Planned timing-model extensions include per-primitive latency,
issue bandwidth, queue depth, resource occupancy, backpressure, and per-op
stall/wait statistics. In this model, the math result can still be computed by
host C++ for correctness, while the simulated execution is constrained by SFU
resources in a way that better matches an accelerator pipeline.

## Standalone Primitive Smoke

Enable primitive smoke with:

```bash
GOLEM_SFU_PRIMITIVE_SMOKE=1 \
GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS=1048576 \
./run_noc_dma_softmax_sfu_pipeline.sh --gemm-m 64 --gemm-n 64 --gemm-k 64
```

Important parameters:

- `GOLEM_SFU_PRIMITIVE_SMOKE=1`: enter the standalone primitive smoke path.
- `GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS`: logical element count per primitive op.
  For example, `1048576` means `1024x1024` logical elements per op.
- `GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS`: real local GM working-set size used
  for functional computation and golden checking. The default is `4`.

The smoke intentionally separates real local-GM validation from logical
processed-element accounting:

```text
real checked data       = chunk_elems fp32 values per op
logical processed data  = GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS per op
reported processed_elems = logical_elems * number_of_ops
```

This keeps the smoke practical in SST while still validating the RoCC primitive
ABI, descriptor decoding, local GM input/output, SFU functional model, wait
path, and statistics. It is not an HBM bandwidth benchmark and does not stream a
full `1024x1024` tensor through local GM. A real full-data primitive performance
benchmark should use a separate HBM streaming primitive descriptor and runner.

## HBM Streaming Primitive Benchmark

Enable the minimal HBM streaming primitive path with:

```bash
GOLEM_SFU_PRIMITIVE_HBM_STREAM=1 \
GOLEM_SFU_PRIMITIVE_HBM_OPS=ALL \
GOLEM_SFU_PRIMITIVE_HBM_ELEMS=64 \
GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS=64 \
./run_noc_dma_softmax_sfu_pipeline.sh --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --group-manager-enable 0 --ctrl-link-enable 0
```

This benchmark uses the GEMM C HBM region as the primitive input/output stream.
For each chunk it performs:

```text
HBM C region op slot -> dma_remote_load_to_gm -> local GM input
local GM input -> sfu_primitive(op) -> local GM output
local GM output -> remote_store -> HBM C region op slot
```

`GOLEM_SFU_PRIMITIVE_HBM_OPS` accepts comma-separated ops, or `ALL` for
`EXP,LOG,RECIPROCAL,RSQRT,TANH,SIGMOID`. The HBM generator preloads each op slot
with a positive fp32 input pattern, so `LOG`, `RECIPROCAL`, and `RSQRT` receive
valid inputs without relying on guest-side fire-and-forget `remote_store`
ordering. The guest validates SFU output against the actual values loaded from
HBM.

Reference run:

- `run_20260703_131521_3969240`: `total_elems=64`, `chunk_elems=64`,
  `hbm_read_bytes=256`, `hbm_write_bytes=256`, `sfu_ops_issued=1`,
  `sfu_primitive_elems=64`, DMA summary `read_issue_count=1`,
  `write_issue_count=1`, simulated time `294.77 us`, run-summary wall time
  `54s`.
- `run_20260703_132634_4023666`: `GOLEM_SFU_PRIMITIVE_HBM_OPS=ALL`,
  `total_elems=16`, `chunk_elems=16`, `processed_elems=96`,
  `hbm_init_write_bytes=384`, `hbm_read_bytes=384`, `hbm_write_bytes=384`,
  `sfu_ops_issued=6`, `sfu_primitive_elems=96`, DMA summary
  `read_issue_count=6`, `write_issue_count=6`, simulated time `382.396 us`,
  run-summary wall time `71s`.

Scale sweep and reporting:

- Sweep root:
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_primitive_allops_20260703`.
- Included sizes: `16`, `1024`, and `4096` elements per op, with
  `GOLEM_SFU_PRIMITIVE_HBM_OPS=ALL`.
- Per point, six primitives are run once each because `chunk_elems == total_elems`:
  `EXP`, `LOG`, `RECIPROCAL`, `RSQRT`, `TANH`, and `SIGMOID`.
- Observed HBM stream bytes match the expected
  `total_elems * 6 ops * 4 bytes * 2 directions` exactly:
  `768`, `49152`, and `196608` bytes.
- DMA read/write issue counts and SFU issue counts remain `6` for all included
  points; this is expected for the one-chunk-per-op configuration.
- Generated report files:
  - `figures/sfu_hbm_primitive_sweep_source.csv`
  - `figures/sfu_hbm_primitive_sweep_notes.md`
  - `figures/sfu_hbm_primitive_sweep.svg`
  - `figures/sfu_hbm_primitive_sweep.png`
  - `figures/sfu_hbm_primitive_sweep.pdf`

The attempted `65536` point is excluded from the sweep report. Its log did not
reach guest PASS or `Simulation is complete`, and no stdout PASS directory was
produced before the run was interrupted. The current diagnosis is that the
larger all-op, multi-chunk case should be rerun as a dedicated diagnostic with
reduced statistics or a smaller op subset, instead of mixed into this small
reporting sweep.

Event-scaling sweep:

- Python plotting environment:
  `/data4/jjgong/.venvs/golem-plot` with `matplotlib`, `seaborn`, and `pandas`.
- Sweep root:
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_event_scaling_chunk1024_20260703`.
- Included PASS sizes: `1024` and `2048` elements per op,
  `GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS=1024`,
  `GOLEM_SFU_PRIMITIVE_HBM_OPS=ALL`.
- Results:
  - `1024`: `chunks=6`, DMA read/write issues `6/6`, wait count `12`,
    HBM read/write bytes `24576/24576`, simulated time `2.72756 ms`,
    wall time `570s`.
  - `2048`: `chunks=12`, DMA read/write issues `12/12`, wait count `24`,
    HBM read/write bytes `49152/49152`, simulated time `5.12498 ms`,
    wall time `930s`.
- Generated report files:
  - `figures/sfu_hbm_primitive_sweep_source.csv`
  - `figures/sfu_hbm_primitive_sweep_notes.md`
  - `figures/sfu_hbm_primitive_sweep.svg`
  - `figures/sfu_hbm_primitive_sweep.png`
  - `figures/sfu_hbm_primitive_sweep.pdf`
  - `figures/sfu_hbm_primitive_sweep.tiff`
  - `figures/sfu_hbm_event_scaling_diagnostics.md`

Timeout diagnostics:

- `4096` elements/op, all ops, `chunk_elems=1024` was stopped at `1000s`.
  It did not reach guest PASS. At emergency shutdown, core7 had progressed to
  DMA read/write issues `17/16` out of the expected `24/24`.
- `65536` elements/op, `EXP` only, `chunk_elems=1024`, reduced stats, was
  stopped at `600s`. It did not reach guest PASS. At emergency shutdown, core7
  had progressed to DMA read/write issues `9/8` out of the expected `64/64`.
- Current diagnosis: the long wall time is primarily driven by per-chunk
  guest/SST execution overhead. It is not primarily caused by all-op math,
  HBM preload, or full SST statistics, because the single-op reduced-stats
  diagnostic still advanced only a small number of chunks within the time cap.

Large-chunk `65536` EXP diagnostic:

- Sweep root:
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_largechunk_diag_20260703`.
- `65536` elements/op, `EXP` only, `chunk_elems=4096`: PASS,
  `chunks=16`, HBM read/write bytes `262144/262144`, DMA read/write issues
  `16/16`, wait count `32`, simulated time `2.15259 ms`, wall time `422s`.
- `65536` elements/op, `EXP` only, `chunk_elems=8192`: PASS,
  `chunks=8`, HBM read/write bytes `262144/262144`, DMA read/write issues
  `16/16`, wait count `32`, simulated time `2.09856 ms`, wall time `398s`.
- The identical DMA issue count for `4096` and `8192` is expected:
  `4096 fp32 = 16 KiB` maps to one DMA burst, while
  `8192 fp32 = 32 KiB` maps to two 16 KiB DMA bursts.
- Generated chunk diagnostic report:
  - `figures/sfu_hbm_exp65536_chunk_diag_source.csv`
  - `figures/sfu_hbm_exp65536_chunk_diag_notes.md`
  - `figures/sfu_hbm_exp65536_chunk_diag.svg`
  - `figures/sfu_hbm_exp65536_chunk_diag.png`
  - `figures/sfu_hbm_exp65536_chunk_diag.pdf`
  - `figures/sfu_hbm_exp65536_chunk_diag.tiff`
- Plot script:
  `plot_sfu_hbm_chunk_diag.py`.

Updated interpretation: increasing chunk size from `1024` to `4096/8192`
turns the `65536` single-op diagnostic from timeout into complete PASS runs.
This strengthens the conclusion that the main wall-time amplifier is per-chunk
guest/SST executor overhead. `8192` is currently the practical maximum chunk
size in the primitive path, so the next performance step is a batched primitive
descriptor rather than simply making chunks larger.

Batched primitive smoke:

- Enable with `GOLEM_SFU_PRIMITIVE_HBM_BATCH=1` together with
  `GOLEM_SFU_PRIMITIVE_HBM_STREAM=1`.
- Implementation adds a 32-byte `SFUPrimitiveBatchDesc` pointing to an array of
  existing 64-byte `SFUPrimitiveDesc` records. Existing single-descriptor
  `sfu_primitive(desc, tag)` behavior is unchanged.
- New RoCC wrappers are `sfu_primitive_batch(desc_gm_addr, tag)` and
  `sfu_primitive_batch_wait(tag)`.
- Reference all-stats run:
  `sfu_hbm_batch_exp_elems_1024_chunk256_allstats`, `EXP`, `total_elems=1024`,
  `chunk_elems=256`, `chunks=4`, `processed_elems=1024`,
  HBM read/write bytes `4096/4096`, DMA read/write issues `4/4`,
  wait count `8`, simulated time `663.83 us`, wall time `126s`.
- SFU statistics confirm the batching behavior on executor core7:
  `sfu_ops_issued=1` and `sfu_primitive_elems=1024`, so four child chunk
  descriptors were executed under one SFU issue.
- Batch/non-batch comparison sweep:
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_batch_compare_20260703`.
  Workload is single `EXP`, `chunk_elems=256`, all-stats enabled.
  - `1024` elems: non-batch `sfu_ops_issued=4`, batch `sfu_ops_issued=1`;
    DMA read/write stays `4/4`; wall time `152s -> 142s`; simulated time
    `659.441 us -> 663.83 us`.
  - `4096` elems: non-batch `sfu_ops_issued=16`, batch `sfu_ops_issued=1`;
    DMA read/write stays `16/16`; wall time `346s -> 333s`; simulated time
    `1.72802 ms -> 1.70861 ms`.
  - Generated report files:
    - `figures/sfu_hbm_batch_compare_source.csv`
    - `figures/sfu_hbm_batch_compare_notes.md`
    - `figures/sfu_hbm_batch_compare.svg`
    - `figures/sfu_hbm_batch_compare.png`
    - `figures/sfu_hbm_batch_compare.pdf`
    - `figures/sfu_hbm_batch_compare.tiff`
- Build note: the local SST build tree is a copied tree, not a symlink. After
  editing `src/sst/elements/golem/sfu/*` or `rocc/roccAnalog.h`, sync those
  files into `build/sst-elements/src/sst/elements/golem/` and rebuild both
  `sfu.lo` and `golem.lo`; otherwise old header users can keep an incompatible
  C++ vtable layout.

Recent reference runs:

- `run_20260702_162003_2304427`: `1024x1024` logical primitive smoke for
  `EXP/LOG/RECIPROCAL`, `processed_elems=3145728`, `sfu_ops_issued=3`,
  simulated time `226.537 us`.
- `run_20260703_125355_3868668`: small six-op smoke for
  `EXP/LOG/RECIPROCAL/RSQRT/TANH/SIGMOID`, `processed_elems=96`,
  `sfu_ops_issued=6`, simulated time `276.651 us`.

## SFUPrimitiveDesc

`SFUPrimitiveDesc` is the 64-byte ABI descriptor written by the RISC-V guest
into global memory before it issues `sfu_primitive(desc_gm_addr, tag)`. The
RoCC path passes the descriptor address to the `golem.SFU` component, and the SFU
component decodes it to know which primitive to run, where the input/output
buffers live, how many elements to process, and how to interpret strides and
flags.

Current fields:

| Field | Meaning |
|---|---|
| `job_id` | Logical job/tag identifier carried in the descriptor. |
| `input0_gm_addr` | Global-memory address of the first input buffer. |
| `input1_gm_addr` | Reserved second input address; current repeat-chunk smoke reuses it for logical processed element count. |
| `output_gm_addr` | Global-memory address of the output buffer. |
| `op` | `SFUPrimitiveOp` selector, such as `EXP`, `LOG`, `RECIPROCAL`, `RSQRT`, `TANH`, or `SIGMOID`. |
| `dtype` | Data type. Current implemented primitive path supports fp32. |
| `elem_count` | Real element count in the local-GM chunk processed by this issue. |
| `input0_stride_bytes` | Byte stride between consecutive `input0` elements. |
| `input1_stride_bytes` | Reserved stride for a future second input. |
| `output_stride_bytes` | Byte stride between consecutive output elements. |
| `flags` | Modifier flags. Current smoke uses repeat-chunk mode to distinguish logical processed count from real checked chunk size. |
| `approx_mode` | Reserved exact/approx selector. Current implementation uses exact host-math functional models. |

The descriptor ABI is shared by guest code and the SST component, so its size is
fixed with `static_assert(sizeof(SFUPrimitiveDesc) == 64)`.

## SFUPrimitiveBatchDesc

`SFUPrimitiveBatchDesc` is a 32-byte ABI descriptor used to amortize per-chunk
guest/RoCC overhead. It does not replace `SFUPrimitiveDesc`; it points to an
array of normal primitive descriptors:

| Field | Meaning |
|---|---|
| `job_id` | Logical batch job identifier. |
| `desc_array_gm_addr` | Global-memory address of the first `SFUPrimitiveDesc` in the batch. |
| `desc_count` | Number of child descriptors. Current component cap is 64. |
| `flags` | Reserved for future batch modifiers. |
| `reserved0` | Reserved padding/future extension. |

For v1, every child descriptor still describes one unary fp32 primitive chunk.
The batch path records one `sfu_ops_issued` event for the whole batch and
accumulates all child element counts into `sfu_primitive_elems`.

## Document Map

- `implementation_plan.md`: main roadmap, implementation phases, and next steps.
- `design.md`: architecture and online softmax design.
- `findings.md`: decisions, issues encountered, fixes, and verification log.
- `task_plan.md`: short active checklist.
- `progress.md`: compressed historical progress log.

## Source Map

- `test_noc_dma_softmax_sfu.cpp`: RISC-V workload entry.
- `golem_softmax_sfu_runtime.h/.cpp`: workload-side SFU descriptor/runtime.
- `ex_instr.h`: RoCC instruction wrappers for SFU commands.
- `run_noc_dma_softmax_sfu_pipeline.sh`: build/run/verify wrapper.
- `verify_softmax_sfu_against_golden.py`: `softmax(A @ B)` golden checker.
- `test_*.py`: scaffold and regression tests for ABI, workload, checker, and
  pipeline behavior.

## Generated Files

The following are generated and intentionally ignored:

- `riscv64/`
- `__pycache__/`
- `src/sst/elements/golem/tests/artifacts/`
- temporary tensor files under `src/sst/elements/golem/tests/data/*.bin`

`bin/sst` is intentionally kept as a local SST shim used by the pipeline wrapper.
