# Durable Findings

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
