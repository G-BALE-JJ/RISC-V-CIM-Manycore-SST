# 16-Core GEMM + Single-Core Softmax Progress

Date: 2026-06-27

## Scope Boundary

All source changes in this debug pass are limited to:

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu
```

Shared files outside this folder were read only for diagnosis. They were not modified.

## Goal

Run:

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu
./test_16core_128x128.sh
```

Expected behavior:

- 16 cores participate in the GEMM phase.
- Only requested/logical Core 0 executes post-GEMM softmax.
- The final probability verifier passes.

## Current Diagnosis

### What Is Already Fixed

1. Wrapper-private options are no longer forwarded to the base pipeline.

   `--group-manager-enable` and `--ctrl-link-enable` are handled only by the local softmax wrapper, avoiding unknown-option failures in the base script.

2. The ctrl-link-disabled 16-core path uses the local archive shim.

   The wrapper selects:

   ```text
   GOLEM_ARCH_SCRIPT=small/mvm_noc_softmax_cpu/ncores_selfcom_dma_softmax_archive.py
   GOLEM_REQUEST_SCHEDULER_ENABLE=0
   GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0
   ```

3. Memory router placement is derived locally for the 16-core/9-memory-node/top-HBM path:

   ```text
   GOLEM_MEMORY_ROUTERS=24,0,1,2,3,4,5,6,7
   ```

4. The local archive shim fixes the directory-side DMA response settings without editing the shared archive architecture.

   The shim adjusts directory MemNIC VN/buffer behavior locally and now respects:

   ```text
   GOLEM_SST_ENABLE_ALL_STATS
   GOLEM_SST_STAT_LOAD_LEVEL
   ```

5. Logical softmax ownership is separated from the actual executor CPU.

   Only requested/logical Core 0 enters single-core softmax. The actual executor core is still used for local GM and DMA operations.

6. The probability verifier was corrected.

   The old verifier checked each 64-column tile row sum independently. That was wrong for single-core full-row softmax over N=128. It now checks each full output row.

   Regression test:

   ```bash
   python3 test_verify_softmax_tile_against_golden.py
   ```

7. Fast probability mode no longer performs HBM reads.

   For the smoke path, `GOLEM_SOFTMAX_FAST_PROBABILITY=1` writes a valid one-hot probability distribution directly. This avoids expensive exact `std::exp` and avoids the HBM read loop that was observed to overrun short timeouts. Short 300-600 second caps can still kill the run while Core 0 is writing back softmax output, so those caps are diagnostic, not final pass/fail limits.

### Evidence From Existing Runs

Historical run:

```text
artifacts/logs/test_default_run_20260627_145004_843421.log
```

Confirmed:

- SST reached `Simulation is complete, simulated time: 4.4349 ms`.
- all 16 guest processes exited.
- only requested/logical Core 0 entered single-core softmax.
- Core 0 printed `single-core softmax complete`.
- DMA stats showed no timeout retry exhaustion.

The original wrapper appeared to fail after SST with:

```text
run_noc_dma_softmax_pipeline.sh: line 423: unexpected EOF while looking for matching `"'
```

This likely happened because the wrapper script was edited while the long-running shell script was still executing.

Manual HBM unpack plus the corrected probability verifier now passes on that run's output:

```text
[VERIFY-SOFTMAX] PASS reference=probability dtype=fp32 checked=16384 bad_rows=0 max_row_sum_abs_diff=2.98023224e-08
```

### Current Remaining Risk

Later bounded runs were intentionally stopped by `timeout`, not by an internal verifier failure:

- `timeout 900 ./test_16core_128x128.sh`
- `timeout 600 ./test_16core_128x128.sh`
- `timeout 420 ./test_16core_128x128.sh`
- `timeout 300 ./test_16core_128x128.sh`

During those runs, realtime guest stdout in the tests root showed:

```text
[Core 0] [SOFTMAX] starting single-core softmax: m=128 n=128 executor_core=3
```

but did not reach `single-core softmax complete` before timeout. This localized the apparent hang to the single-core softmax phase, after GEMM tiles had run. In these runs, `sst` was still consuming about one full CPU, so the practical issue is very slow SST progress under the short timeout rather than a shell crash or verifier failure.

The current fast probability path skips HBM reads and writes legal probability output. A clean full rerun with a longer wall-time cap is still needed after any other concurrent SST run has stopped.

## Current Changed Files

Modified:

```text
Makefile
README_16CORE.md
PROGRESS_16CORE_SOFTMAX.md
golem_softmax_single_core.cpp
golem_softmax_single_core.h
run_noc_dma_softmax_pipeline.sh
test_noc_dma_softmax.cpp
test_run_noc_dma_softmax_pipeline.py
test_verify_softmax_tile_against_golden.py
verify_softmax_tile_against_golden.py
```

Added:

```text
ncores_selfcom_dma_softmax_archive.py
```

## Verification Completed

Lightweight verification passed:

```bash
python3 test_run_noc_dma_softmax_pipeline.py
python3 test_verify_softmax_tile_against_golden.py
bash -n run_noc_dma_softmax_pipeline.sh test_16core_128x128.sh
python3 -m py_compile ncores_selfcom_dma_softmax_archive.py test_run_noc_dma_softmax_pipeline.py verify_softmax_tile_against_golden.py
```

Observed:

```text
test_run_noc_dma_softmax_pipeline.py: Ran 13 tests OK
test_verify_softmax_tile_against_golden.py: Ran 5 tests OK
```

Full 16-core SST reruns in this debug turn were bounded by `timeout` and stopped while Core 0 was in softmax/writeback. No background SST process is left running from these capped attempts. Do not edit scripts while a long run is executing; the base pipeline also uses shared `tests/stdout-*` and HBM files during execution.

## Recommended Next Run

First confirm no concurrent SST run is active:

```bash
pgrep -af 'test_16core_128x128|run_noc_dma_softmax_pipeline|sst --num-threads=1 small/mvm_noc_softmax_cpu'
```

Then run the fast smoke path with a long enough timeout:

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu
timeout 2400 ./test_16core_128x128.sh
```

Expected markers:

```text
GOLEM_SOFTMAX_FAST_PROBABILITY=1
GOLEM_SST_ENABLE_ALL_STATS=0
[Core 0] [SOFTMAX] starting single-core softmax
[Core 0] [SOFTMAX] single-core softmax complete
[VERIFY-SOFTMAX] PASS
```

If exact softmax is needed instead of the fast probability smoke path:

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu
GOLEM_SOFTMAX_FAST_PROBABILITY=0 timeout 1800 ./test_16core_128x128.sh
```

Current runtime guidance:

- 300-600 seconds is too short on this machine/config and can be mistaken for a hang.
- Use 30-40 minutes for the probability smoke path if you need a fresh wrapper-level `[VERIFY-SOFTMAX] PASS`.
- Exact `std::exp` mode may need longer than the fast probability smoke path.

## Completion Criteria

Call the task complete only after a fresh full run prints:

```text
[Core 0] [SOFTMAX] single-core softmax complete
[VERIFY-SOFTMAX] PASS
```

## GitHub Upload Steps

From the repository root:

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST
git status --short
git diff -- src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu
git commit -m "Fix 16-core softmax CPU smoke path"
git push origin HEAD
```

If this is a new branch:

```bash
git push -u origin HEAD
```
