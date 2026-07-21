# SFU Softmax Row Engine NoC Implementation Plan

**Goal:** Implement the approved row-engine architecture for FP32 row-wise
Softmax, make `1024x4096` run through real SST/Vanadis/NoC, and reduce the
measured issue-to-completion cycle count by at least one order of magnitude
from the existing `16,642,961` CPU-equivalent-cycle baseline.

**Decision source:**
`docs/superpowers/specs/2026-07-16-sfu-softmax-row-engine-noc-architecture-design.md`

**New workload root:**
`src/sst/elements/golem/tests/small/muticore_softmax`

## Acceptance Contract

- Functional result matches a stable CPU Softmax golden within the existing
  FP32 tolerance.
- The `1024x4096` primary mapping uses 16 physical SFUs, one tile per row, and
  distributes exactly 64 rows to each tile.
- The primary mapping emits zero max/sum reduction transport messages.
- It submits at most one row-engine job per tile, instead of 4096 worker jobs.
- Input and output transfers are at most one contiguous DMA operation per row.
- Report both SST timebase timestamps and explicit 2.3 GHz accelerator cycles;
  never label picosecond ticks as cycles.
- First performance gate: issue-to-completion `<= 1,600,000` cycles (10x better
  than baseline). Architecture target: accelerator latency `<= 150,000` cycles
  after four-node HBM row striping is enabled.
- Existing GEMM guest build behavior and canonical GEMM regression remain
  unchanged.

## Measured Result (2026-07-21, causal Row Engine)

The primary `1024x4096` run is retained at
`/data4/jjgong/tmp/muticore_softmax_causal_dedupe_r1024_d4096`.

| Metric | Causal Row Engine | Result |
|---|---:|---:|
| Golden mismatches | `0 / 4,194,304` | PASS |
| Actual descriptor-to-accelerator completion | `66,958 cycles` | PASS |
| Analytical compute estimate | `66,061 cycles` | reference only |
| Guest kernel window | `73,309 cycles` | measured |
| Whole SST equivalent cycles at 2.3 GHz | `640,921 cycles` | measured |
| Tensor job / physical SFUs | `1 / 16` | one controller job, 16 bands |
| Rows dispatched/completed | `1,024 / 1,024` | PASS |
| Input/output DMA ACK events | `1,024 / 1,024` | one transfer per row |
| MAX / EXP-SUM / NORMALIZE events | `1,024 / 1,024 / 1,024` | PASS |
| Reduction request messages | `0` | PASS |
| Max NoC port utilization | `1.257%` | measured |

Completion is no longer released from an independently predicted ready tick.
Each row now follows input DMA, MAX, EXP/SUM, NORMALIZE, output DMA ACK, and
worker completion in order. The controller becomes ready only after all 16
worker bands report completion. The former `66,062` result and its
`48,450`-cycle completion gap describe the removed decoupled model and must not
be used as end-to-end latency.

The NoC sensitivity check confirms that this is a real transport dependency:
for `16x4096`, reducing NoC/DirCtrl bandwidth from `1200 GB/s` to `64 GB/s`
increased accelerator latency from `2,076` to `4,294` cycles while retaining
zero golden mismatches. Reducing EXP lanes from four to two increased the same
point from `2,076` to `3,100` cycles, confirming both transport and compute
resources participate in the causal path.

## Phase A: Build Isolation and Baseline Integrity

Affected files:

- `src/sst/elements/golem/tests/run_noc_dma_pipeline.sh`
- `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_noc_dma_softmax_sfu_pipeline.sh`
- `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/Makefile`
- focused runner tests beside those scripts

Tasks:

1. Add an explicit `GOLEM_SKIP_DEFAULT_GUEST_BUILD=1` contract for custom
   guests. Validate that `VANADIS_EXE` exists and is executable.
2. Ensure custom Softmax runs neither rebuild nor rewrite the unused GEMM ELF
   or its build metadata, including when `GOLEM_SKIP_BUILD=0`.
3. Add the omitted CPU Softmax runtime files to custom guest freshness checks.
4. Run default-GEMM dry-run controls and compare GEMM ELF/metadata hash and
   mtime across a custom workload smoke test.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  src.sst.elements.golem.tests.small.mvm_noc_softmax_sfu.test_run_noc_dma_softmax_sfu_pipeline
```

## Phase B: Row-Engine ABI, Timing, and Statistics

Affected files:

- `src/sst/elements/golem/sfu/sfu.h`
- `src/sst/elements/golem/sfu/sfu.cc`
- `src/sst/elements/golem/tests/architecture/cpu_builder.py`
- production/component contract tests

Tasks:

1. Add a distinct row-local job flag without changing the 128-byte
   `SFUJobDesc` layout or legacy flag values.
2. Add component parameters for accelerator clock, 16 vector lanes, 4 EXP
   lanes, pipeline latencies, scratchpad capacity, and row contexts.
3. Model row-local readiness from resource throughput:
   `ceil(cols/16) + ceil(cols/4) + ceil(cols/16)` per row, plus explicit
   pipeline drain. Functional data are computed in batches; modeled progress
   is independent of `sfu_job_wait()` poll frequency.
4. Add row-engine job/row/phase-cycle/wait-poll counters and expose timestamps
   with unambiguous `_ticks`, `_ps`, or `_cycles` units.
5. Preserve the legacy distributed-column path byte-for-byte unless a focused
   regression requires an adaptation.

Focused verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu \
  -p 'test_sfu_softmax_*.py'
```

## Phase C: Dedicated 16-Tile Workload

New files under `src/sst/elements/golem/tests/small/muticore_softmax`:

- `Makefile`
- `test_muticore_softmax.cpp`
- `run_muticore_softmax.sh`
- `parse_muticore_softmax.py`
- `test_muticore_softmax.py`
- `README.md`

Tasks:

1. Assign one contiguous 64-row band to each of 16 tiles; stage each tile's
   rows contiguously in its local GlobalMemory region.
2. Use one input DMA and one output DMA per row, one row-engine descriptor per
   tile, and one completion wait per tile.
3. Emit machine-readable per-tile issue/completion `rdcycle`, assigned rows,
   DMA operation counts, lifecycle counters, and final golden result.
4. Start with the legacy single-HBM-node layout for a small functional smoke,
   then switch the performance profile to four-node row striping.
5. Keep generated inputs/results and run artifacts outside the source tree's
   executable inputs and give each run a production/guest SHA signature.

Verification boundary:

```bash
make -C src/sst/elements/golem/tests/small/muticore_softmax clean all
bash src/sst/elements/golem/tests/small/muticore_softmax/run_muticore_softmax.sh \
  --rows 16 --cols 4096 --timeout 600
```

The smoke must PASS, report 16 jobs, 16 input and 16 output row transfers, and
zero reduction messages.

## Phase D: Four-Node HBM Row Striping

Affected files:

- workload generator/parser under `muticore_softmax`
- the shared HBM initialization helper only if the dedicated workload cannot
  express the layout without changing it
- architecture control only where four data-node image paths are wired

Tasks:

1. Define one structured mapping used by generator, guest address arithmetic,
   golden reconstruction, and parser:
   `band = row / rows_per_tile`, `node = 1 + (band % 4)`, with bands assigned
   contiguously within each node.
2. Generate four non-replicated input/output images and keep a single-node
   regression layout.
3. Verify each HBM node receives equal useful bytes for `1024x4096` and that
   every compute tile accesses the HBM node in its mesh column.
4. Measure single-node versus row-striped accelerator and issue-to-completion
   cycles with identical component timing parameters.

## Phase E: Scale and Performance Gate

Run real SST points in order: `16x4096`, `64x4096`, `256x4096`, then
`1024x4096`. Every run has a watchdog and a fresh artifact directory.

For each point record:

- functional PASS/failure and max error;
- SST simulated time and exact clock conversion;
- accelerator and issue-to-completion cycles;
- rows/tile, rows/cycle, elements/cycle;
- DMA operations/bytes per HBM node;
- row-engine phase active cycles and wait polls;
- NoC reduction messages, retries, xbar/output stalls;
- Vanadis retired instructions and simulator wall time.

Stop and diagnose at the first scale where correctness, lifecycle, or monotonic
scaling fails. The final report must compare the new `1024x4096` point to the
existing `7236.07 us / 16,642,961` equivalent-cycle baseline and state whether
the 10x gate and `150k` architecture target were met.

## Phase F: Compatibility and Closeout

1. Run the canonical GEMM regression and the focused Softmax suite.
2. Confirm a custom Softmax run leaves the GEMM ELF and metadata unchanged.
3. Run source-format/syntax checks and relink `libgolem.so` from the modified
   production source.
4. Update the design status and durable findings with measured facts only.

The `(m,l)` pair collective and 2/4-tile mappings are the next implementation
phase for rows wider than 4096. They are not on the critical path for the
`1024x4096` row-local acceptance run, where the correct reduction-message count
is zero.

## Final Verification

The 2026-07-21 causal implementation passed real SST runs at `16x4096`,
`64x4096`, `256x4096`, and `1024x4096`. The largest point completed in
`66,958` accelerator cycles and passed the full FP32 golden comparison with
maximum absolute error `5.72476014e-11`. Its timeline contains exactly 1,024
input DMA-ready events, 1,024 events for every compute stage, 1,024 output DMA
ACKs, 16 worker completions, and one guest wait observation.

- `libgolem` relinks successfully with the new self-link stage events.
- All 19 focused Row Engine tests pass.
- All 12 dedicated workload, parser, runner, and plotting tests pass.
- The `1024x4096` causal contract passes with zero reduction requests and zero
  wait polling.
- The split PNG/SVG/PDF/TIFF figures and CSV source data are regenerated from
  the retained causal result artifact.
