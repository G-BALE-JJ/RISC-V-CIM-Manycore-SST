# SFU Primitive Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal batched SFU primitive descriptor path so one RoCC issue can execute multiple unary fp32 primitive descriptors.

**Architecture:** Keep the existing 64-byte `SFUPrimitiveDesc` ABI unchanged and add a separate 32-byte `SFUPrimitiveBatchDesc` that points to an array of primitive descriptors in global memory. Add new RoCC func7 values for batch issue/wait, dispatch them to `SFUAPI::issuePrimitiveBatch`, and make the HBM primitive benchmark optionally use batch mode with `GOLEM_SFU_PRIMITIVE_HBM_BATCH=1`.

**Tech Stack:** SST/golem C++, RISC-V guest C++, Python unittest scaffold tests, existing shell pipeline.

## Global Constraints

- Preserve existing `sfu_primitive(desc, tag)` behavior and all current smoke tests.
- Batch v1 supports only existing unary fp32 primitives through existing `SFUPrimitiveDesc`.
- Batch issue counts as one SFU op in `sfu_ops_issued`; `sfu_primitive_elems` accumulates all batch item elements.
- Avoid reduction, fused softmax, and new math semantics in this task.
- Follow existing `mvm_noc_softmax_sfu` static-test patterns.

---

### Task 1: Batch ABI And RoCC Dispatch Tests

**Files:**
- Modify: `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_descriptor_scaffold.py`
- Modify: `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_rocc_sfu_integration.py`
- Modify: `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_workload_scaffold.py`

**Interfaces:**
- Produces expected names: `SFUPrimitiveBatchDesc`, `issuePrimitiveBatch`, `sfu_primitive_batch`, `GOLEM_SFU_PRIMITIVE_HBM_BATCH`.

- [x] **Step 1: Add failing descriptor/API tests**
- [x] **Step 2: Add failing RoCC dispatch tests**
- [x] **Step 3: Add failing workload env tests**
- [x] **Step 4: Run targeted tests and confirm RED**

### Task 2: Batch ABI And SFU Component Implementation

**Files:**
- Modify: `src/sst/elements/golem/sfu/sfu.h`
- Modify: `src/sst/elements/golem/sfu/sfu.cc`

**Interfaces:**
- Consumes existing `SFUPrimitiveDesc`.
- Produces `SFUAPI::issuePrimitiveBatch(uint64_t descAddr, uint64_t tag)`.

- [x] **Step 1: Define 32-byte `SFUPrimitiveBatchDesc`**
- [x] **Step 2: Add `PrimitiveBatchOpState` and helper declarations**
- [x] **Step 3: Implement batch descriptor read/validate**
- [x] **Step 4: Implement `issuePrimitiveBatch` by executing each child descriptor**
- [x] **Step 5: Make wait retire batch state**
- [x] **Step 6: Run targeted tests and confirm GREEN**

### Task 3: RoCC And Guest Wrapper Implementation

**Files:**
- Modify: `src/sst/elements/golem/rocc/roccAnalog.h`
- Modify: `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/ex_instr.h`
- Modify: `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_noc_dma_softmax_sfu.cpp`
- Modify: `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_noc_dma_softmax_sfu_pipeline.sh`
- Modify: `src/sst/elements/golem/tests/architecture/ncores_selfcom_dma_ctrl.py`
- Modify: `src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/ncores_selfcom_dma_softmax_archive.py`

**Interfaces:**
- Adds func7 `0x1b/0x1c`.
- Adds env `GOLEM_SFU_PRIMITIVE_HBM_BATCH`.

- [x] **Step 1: Add func7 constants and wrappers**
- [x] **Step 2: Dispatch batch issue/wait to SFU component**
- [x] **Step 3: Forward env through wrappers and architecture scripts**
- [x] **Step 4: Add HBM stream batch path that groups chunk descriptors per op**
- [x] **Step 5: Run targeted tests and rebuild RISC-V workload**

### Task 4: Verification And Documentation

**Files:**
- Modify: `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/README.md`
- Modify: `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/progress.md`
- Modify: `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/task_plan.md`

**Interfaces:**
- Documents expected batch behavior and verification command.

- [x] **Step 1: Run Python unittest discovery**
- [x] **Step 2: Build golem component**
- [x] **Step 3: Run a small real SST batch smoke if build succeeds**
- [x] **Step 4: Update docs with exact results**

## Verification Record

- `python3 -m unittest discover -s . -p 'test_*.py'`: 75 tests OK.
- `make -C build/sst-elements/src/sst/elements/golem -j16 V=1`: rebuilt and
  relinked `libgolem.so` after syncing the copied build tree.
- `make -B ARCH=riscv64`: rebuilt the RISC-V guest workload.
- Real SST all-stats batch smoke:
  `sfu_hbm_batch_exp_elems_1024_chunk256_allstats`, PASS,
  `total_elems=1024`, `chunk_elems=256`, `chunks=4`,
  HBM read/write bytes `4096/4096`, DMA read/write issues `4/4`,
  `sfu_ops_issued=1`, `sfu_primitive_elems=1024`,
  simulated time `663.83 us`, wall time `126s`.

## Follow-Up

- Batch/non-batch sweep completed under
  `src/sst/elements/golem/tests/artifacts/sweeps/sfu_hbm_batch_compare_20260703`.
  The generated `sfu_hbm_batch_compare` figure shows issue count, DMA count,
  wall time, and simulated time at fixed `chunk_elems=256`.
- Consider exporting SFU stats into a compact CSV alongside `dma_summary.csv` so
  later sweeps do not need manual `rg` parsing of `stats_selfcom.txt`.
- Next SFU model target: build a hardware-like SST C++ timing model, not an RTL
  implementation. Keep host C++ math for functional correctness, but add
  per-primitive latency, issue bandwidth, queue depth, pipeline occupancy,
  backpressure, and per-op stall/wait/latency statistics. Validate this first on
  standalone primitive/HBM streaming benchmarks before folding it back into
  fused softmax.
