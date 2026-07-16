# Unified SFU Job Softmax

## Active scope

The active path is unified SFU job Softmax. Primitive/batch and CPU Softmax
experiments are no longer project mainlines. The implementation performs stable
three-stage Softmax (max, exp/sum, normalize); chunk streaming does not make it
a strict online-Softmax merge algorithm.

The current workflow supports:

- direct row-major HBM input and output;
- row-band and chunk streaming;
- distributed columns across physical worker cores;
- cooperative row-band execution;
- explicit SimpleNetwork reduction request/response traffic;
- full-output golden verification and lifecycle statistics.

## Fixed experiment profile

Capacity and performance experiments use the established GEMM network profile:

```text
transport                 explicit_noc
reduction VN              0 (num_vns=3)
workers / bands            16 / 16
chunk                      256 elements
NoC link/xbar/highlink     1200GB/s
NoC input/output buffer    512KB
GlobalMemory buffer        1024KB
flit                       128B
memory node                256 MiB
```

Do not change these values to bypass a failure or timeout. VN, bandwidth,
worker, chunk and modeled-NoC sweeps are outside the current scope.

## Verified results

Phase 4F completed eight real SST points covering dimensions 512/1024/2048/4096,
workers 4/8/16, and rows 16/64/256. All points passed golden, reduction,
transport, DMA and NoC gates.

The capacity ladder then passed:

| Shape | Simulated time | Child wall time | Golden |
|---|---:|---:|---:|
| `512x4096` | `3773.42 us` | `1462 s` | `2,097,152 / 0` |
| `1024x4096` | `7236.07 us` | `2890 s` | `4,194,304 / 0` |

The largest real-SST verified shape is `1024x4096`. The `2048x4096` and
`4096x4096` runs are deferred while wall-clock cost is optimized; dry-run
markers are not PASS evidence.

## Commands

Focused tests:

```bash
TMPDIR=/data4/jjgong/tmp /data4/jjgong/.venvs/golem-plot/bin/python -m unittest discover \
  -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu -p 'test_*.py'
```

Capacity dry-run:

```bash
TMPDIR=/data4/jjgong/tmp GOLEM_SFU_CAPACITY_DRY_RUN=1 \
  bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh
```

Real SST must run serially outside the restricted network namespace. Watchdogs
are mandatory; timeout status is recorded and no larger point is started.

## Current optimization work

Before resuming the capacity ladder:

1. stop the Softmax custom guest path from rebuilding the unused GEMM guest;
2. disable Vanadis pipeline trace for performance runs;
3. replace per-event NoC/GM/DMA text with one-time configuration and stats;
4. add a separately labelled quiet guest benchmark mode;
5. test selective stats and SST 2/4 host threads;
6. consider event-driven SFU wait only as a separately reviewed model change.

Initial A/B uses `64x4096` with a 600-second watchdog. A successful low-risk
combination is rechecked at `256x4096` before any larger run.

## Canonical evidence

- Phase 4F:
  `tests/artifacts/sweeps/sfu_phase4f_large_scale_explicit_noc_20260715`
- Capacity:
  `tests/artifacts/sweeps/sfu_4096x4096_capacity_explicit_noc_20260716`
- Group figure:
  `tests/artifacts/sweeps/sfu_phase4e_group_report_20260715`
- GEMM regression:
  `tests/artifacts/sweeps/sfu_multi_vn_gemm_regression_20260715`

All other raw sweep artifacts are regenerable and are not retained.

## Source map

- `test_noc_dma_softmax_sfu.cpp`: RISC-V workload.
- `golem_softmax_sfu_runtime.{h,cpp}`: unified job descriptor/runtime.
- `run_noc_dma_softmax_sfu_pipeline.sh`: single-run build/verify wrapper.
- `run_sfu_unified_job_distributed_scaling.sh`: explicit-NoC child runner.
- `run_sfu_phase4f_large_scale_explicit_noc.sh`: Phase 4F parent runner.
- `run_sfu_4096x4096_capacity.sh`: capacity parent runner.
- `plot_sfu_phase4f_large_scale.py`: parser and report generator.
- `sfu_4096x4096_capacity.py`: capacity contracts and report support.

The compatibility CPU directory is not an active test path; it only supplies
the shared request definitions and the current archive architecture shim.
