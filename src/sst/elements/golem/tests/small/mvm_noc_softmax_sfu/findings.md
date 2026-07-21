# Durable Findings

## Current causal Row Engine result (2026-07-21)

This section supersedes the tensor-controller/model-ready conclusions later in
this file. Those entries remain as dated development history.

- Final artifact:
  `/data4/jjgong/tmp/muticore_softmax_causal_dedupe_r1024_d4096`.
- Successful completion is causally ordered after per-row input DMA, MAX,
  EXP/SUM, NORMALIZE and output DMA ACK.
- The controller requires 16 unique identity-checked band completions.
- Final `1024x4096`: `66,958` accelerator cycles, `73,309` clean guest kernel
  cycles, `640,921` whole-system cycles, full `4,194,304/0` golden PASS.
- Traffic is 1,024 input DMA operations and 1,024 output DMA ACKs, not 64
  coalesced functional callbacks.
- The exact critical path is `11 + 256 + 66,549 + 88 + 54 = 66,958` cycles.
- EXP/SUM is 65,536 of 98,304 aggregate active service cycles. The measured
  max NoC port utilization is 1.257%, but NoC is causal: a 64 GB/s control
  raises `16x4096` latency from 2,076 to 4,294 cycles.
- The historical 66,062-cycle endpoint and 48,450-cycle model gap are removed
  and must not be reported as current end-to-end timing.

## Architecture

- The current algorithm is numerically stable max/sum/normalize Softmax. It
  streams chunks but is not strict online Softmax with merged running max/sum.
- Distributed columns require two global reductions per row. Explicit-NoC sends
  max/sum requests and responses through SST SimpleNetwork; modeled-NoC is only
  historical counter scaffolding.
- The fixed network profile is the same effective profile used by mature GEMM:
  1200GB/s link/xbar/highlink, 512KB NoC buffers, 128B flits and a 1024KB
  GlobalMemory buffer.
- Normal operation uses `num_vns=3`, reduction VN0 and DMA response VN0. VN
  variation is not a performance dimension.

## Performance

- At `16x4096`, 4/8/16 workers took `422.029/423.385/427.053 us`; adding workers
  did not improve time. Transport/synchronization overhead offset column split.
- At `dim=4096, workers=16`, increasing rows from 16 to 256 reduced time per row
  from `26.691` to `7.995 us`, showing useful amortization of fixed overhead.
- `512x4096` and `1024x4096` wall time scaled approximately with simulated time;
  1024 did not exhibit a unique hang or nonlinear slowdown.
- Input/HBM reuse and faster golden checking cannot solve the main problem
  because SST itself consumes about 94.5% of total wall time.
- Sixteen Vanadis cores use the same pipeline-trace path and execute `fprintf`
  on retirement. The visible trace file undercounts actual calls because cores
  overwrite the same path; the 1024 run retired about 41.9 million instructions.
- `adaptive_wait_eq` backoff and guest formatted output are instruction hotspots.
  `GOLEM_BENCH_QUIET_LOGS` is currently shell-only and does not silence the guest.
- RoCC retries an unfinished SFU wait from its clocked command queue. Replacing
  this with completion-driven wakeup may yield larger gains but changes the
  production timing model.
- In the Row Engine v2 `1024x4096` run, all 16 physical SFUs modeled the same
  `66,061`-cycle compute latency, while their issue ticks spanned about `291k`
  accelerator cycles. The `357,385` issue-to-completion result is therefore
  dominated by cross-core input readiness/issue skew rather than SFU compute.
- A controlled 64 KiB DMA-burst run reduced total NoC output-port stalls from
  `175,600,015` to `39,664,429`, but regressed issue-to-completion from
  `357,385` to `365,093` cycles and whole-system time from `1,136,965` to
  `1,174,256` cycles. NoC stall count alone is not the optimization objective;
  256 KiB remains the accepted profile pending a stronger result.
- A 1 MiB boundary run was manually stopped after about 15 minutes without
  completing, over twice the complete 64 KiB run wall time. Its continuously
  growing Vanadis trace showed prolonged wait execution. Bursts larger than the
  512 KiB NoC buffers are rejected for this profile.
- Absolute-completion-tick RoCC deferral reduced Row Engine compatibility wait
  polls from `1,056,100` to zero. The full `1024x4096` result remained exactly
  `357,385` issue-to-completion and `1,136,965` whole-system cycles with the
  same `66,061` modeled compute latency and full golden PASS. Polling was model
  overhead, not the architectural critical path; R3 tensor scheduling is now
  required.
- The archive memory-side DirectoryController MemNIC hard-coded `25GB/s`, so
  the runner's reported `GOLEM_DIRCTRL_HIGHLINK_BW=1200GB/s` never reached the
  model. The observed four-node aggregate lower bound (about `368k` cycles for
  16 MiB) matched the `387,468`-cycle pre-fix result. Wiring the requested
  bandwidth through the softmax shim reduced DMA RTT and made the target
  compute-bound.
- Final tensor-controller `1024x4096`: one job, 16 physical endpoints, four
  row contexts per endpoint, 64 input and 64 output DMA bursts, 16 MiB each,
  32 tensor control messages, zero retry/stale/reduction, `66,062` cycles from
  issue through final output ACK, and full golden PASS.
- The instrumented target rerun is cycle-identical and separates the two
  remaining intervals. Guest start to descriptor acceptance is about `37.6k`
  cycles. After acceptance, all output DMA ACKs arrive by `17,557` cycles and
  all band completions by about `17.6k`, while modeled compute readiness is
  `66,062` cycles. The post-accept critical path is therefore the Row Engine
  compute model; NoC/DMA is hidden. Host functional Softmax runs inside the DMA
  callback with zero simulated duration and is not a hardware latency measure.
- Guest launch instrumentation attributes `33,583` of the target's roughly
  `37.5k` pre-accept cycles to code before the tensor helper. The timer starts
  before a `[SOFTMAX]` `printf/fflush`, so this dominant interval is benchmark
  logging contamination. Validation/descriptor construction costs `3,035`
  cycles, params/descriptor GM writes cost `207/397`, and RoCC issue return
  costs `26`; hardware launch is not the source of the 37.5k interval.
- Moving the guest kernel timer after the diagnostic print reduced the target
  task window from `103,617` to `72,409` cycles without changing the `66,062`
  accelerator path. The clean pre-accept interval is `6,398` cycles, dominated
  by `2,405` cycles of call-site setup and `3,098` cycles of validation and
  descriptor construction; GM writes and RoCC issue total only `629` cycles.
  This is a measurement-boundary correction, not an architecture speedup.
- Fixed-parameter `16/64/256/1024 x 4096` scaling yields accelerator cycles of
  `1,550/4,622/16,910/66,062` and clean guest kernel cycles of
  `7,819/10,954/23,323/72,409`. The steady-state modeled slope is 1,024 cycles
  per additional 16 rows after the initial pipeline fill.
- Independent target stage spans show max `[0,64768]`, exp+sum `[256,65792]`
  and normalize `[1280,66048]`. Across the three stages, 98,304 active cycles
  compress into a 66,048-cycle span through context overlap, saving 32,256
  cycles (32.8%); exp+sum remains the throughput-setting stage.

## Validation

- Phase 4F and capacity acceptance requires golden equality, exact reduction and
  transport totals, complete DMA lifecycle, zero retry/rejected/stale events,
  fixed network/VN signature and immutable output hash.
- A host-only optimization must preserve simulated time, output SHA and all
  counters exactly. Guest quiet mode changes modeled instruction count and must
  use a new result label rather than overwrite historical measurements.
- Every long SST process is wrapped by a fixed watchdog. TIMEOUT is retained as
  evidence and blocks larger points; timeout is not expanded within the run.

## Build and environment

- RISC-V builds use
  `/data/lzq/packages/install/riscv64_musl_toolchain/bin`.
- Real MPI/SST runs need the host network namespace; sandbox OOB failure before
  `MPI_Init` is an environment failure, not a Softmax result.
- Temporary files use `/data4/jjgong/tmp` because the root `/tmp` filesystem is
  space constrained.
- Changes under `src/sst/elements/golem/sfu` require relinking
  `build/sst-elements/src/sst/elements/golem/.libs/libgolem.so` before SST.
- A custom Softmax guest must not trigger the shared pipeline's default GEMM
  build. Default GEMM behavior and its original regression remain mandatory.

## Retention

Keep only four raw evidence roots: Phase 4F, current capacity, the final group
figure and the latest GEMM regression. Other sweeps are reproducible historical
intermediates; their conclusions live in this file, `progress.md` and Git history.

Build-tree test artifacts are also disposable: they consumed about 57 GiB while
the linked Golem library itself is about 60 MiB. Delete artifact subtrees, not
`.libs` or the install tree. Active automated component tests are retained even
when their feature is not the current experiment mainline; deleting them saves
negligible space and weakens shared SFU regression coverage.
