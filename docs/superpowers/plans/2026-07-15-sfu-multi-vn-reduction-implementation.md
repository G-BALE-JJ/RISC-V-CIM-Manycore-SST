# SFU Multi-VN Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Restore standalone unified-job softmax to three VNs, make DMA response VN explicit and backward-compatible, and run auditable explicit-NoC VN0/VN1/VN2 experiments without regressing GEMM.

**Architecture:** Keep the existing per-core GlobalMemory SimpleNetwork bridge. Parameterize only the directory MemNIC Golem DMA completion VN; preserve GlobalMemory ordinary response VN1 and request VN0. Add VN-aware runner identity, resolved-topology validation, root locking, and explicit-NoC-only VN sweeps.

**Tech Stack:** C++ SST memHierarchy/Golem components, Python SST architecture scripts, Bash sweep runners, Python unittest, Merlin SimpleNetwork, RISC-V musl guest toolchain.

## Global Constraints

- Do not modify SFUJobDesc ABI, softmax math, or primitive/batch softmax paths.
- Preserve default ctrl-link GEMM behavior and add a separate archive/no-ctrl compatibility regression.
- Use num_vns=3: GlobalMemory request VN0; ordinary READ response VN1; softmax directory DMA completion VN0; SFU reduction VN 0, 1, or 2.
- VN sweeps must use GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc.
- Use /data4/jjgong/.tmp for compiler temporary files.
- Build/install memHierarchy before rebuilding/linking Golem.
- Do not reuse or concurrently write existing VN0 scaling roots.

---

### Task 1: Capture Baselines

**Files:** `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/progress.md`,
`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/findings.md`, generated root
`src/sst/elements/golem/tests/artifacts/sweeps/sfu_multi_vn_baseline_20260715/`.

- [ ] Run the clean default GEMM before source changes:

~~~bash
TMPDIR=/data4/jjgong/.tmp \
GOLEM_ARTIFACT_ROOT=/data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_multi_vn_baseline_20260715 \
env -u GOLEM_SFU_ENABLE -u GOLEM_SFU_STANDALONE_SOFTMAX \
  -u GOLEM_SFU_JOB_SOFTMAX -u GOLEM_SFU_PRIMITIVE_SOFTMAX \
  -u GOLEM_SFU_REDUCTION_VN -u GOLEM_DMA_RESPONSE_VN \
  -u GOLEM_ARCH_SCRIPT -u GOLEM_GROUP_MANAGER_ENABLE \
  -u GOLEM_CTRL_LINK_ENABLE -u GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE \
  -u GOLEM_WCP_PREFETCH_WINDOWS -u GOLEM_WCP_RESIDENT_K_TILES \
  bash src/sst/elements/golem/tests/run_noc_dma_pipeline.sh \
  --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --dtype fp32 --tensor-source sample --verify-c
~~~

Expected: exit 0, VERIFY-C PASS, simulation completion, complete DMA lifecycle, zero retry/exhaustion, and no nonzero reduction activity.

- [ ] Record run ID, output/log SHA-256, guest ELF SHA-256, both SST library SHA-256 values, simulated time, and dma_summary.csv in progress.md.
- [ ] Run the existing focused Python suite and record any pre-existing failure:

~~~bash
python3 -m unittest discover -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu -p 'test_*.py' -v
~~~

- [ ] If baseline notes changed, commit only those two planning files:

~~~bash
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/progress.md \
        src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/findings.md
git commit -m "Record pre-change multi-VN baselines"
~~~

### Task 2: Parameterize MemNIC DMA Response VN

**Files:** `src/sst/elements/memHierarchy/memNICBase.h`, create
`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_memnic_dma_response_vn.py`.

**Interface:** absent golem_dma_response_vn derives num_vns >= 2 ? 1 : 0; explicit value is used; value >= num_vns is fatal.

- [ ] Add failing tests requiring the ELI parameter, explicit Params lookup with a
  found/not-found distinction, default derivation, range check, and trace diagnostic.
- [ ] Run the new test and confirm RED.
- [ ] Add the ELI parameter and implement the minimal Params lookup/range check in MemNICBase::build. Keep ordinary MemNIC response behavior unchanged.
- [ ] Run the focused test and git diff --check; expect PASS.
- [ ] Run a minimal real SST initialization with num_vns=3 and
  GOLEM_DMA_RESPONSE_VN=3; expect MemNIC initialization fatal before guest execution:

~~~bash
TMPDIR=/data4/jjgong/.tmp GOLEM_DMA_RESPONSE_VN=3 \
GOLEM_ARCH_SCRIPT=architecture/archive/ncores_selfcom_dma.py \
bash src/sst/elements/golem/tests/run_noc_dma_pipeline.sh \
  --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --dtype fp32 --tensor-source sample
~~~
- [ ] Commit only memNICBase.h and its focused test.

### Task 3: Preserve Archive Legacy VN and Restore Softmax to Three VNs

**Files:** `src/sst/elements/golem/tests/architecture/archive/ncores_selfcom_dma.py`,
`src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/ncores_selfcom_dma_softmax_archive.py`,
`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_run_noc_dma_softmax_sfu_pipeline.py`,
`src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/test_run_noc_dma_softmax_pipeline.py`.

- [ ] Add failing tests requiring archive default response VN1, softmax num_vns=3, explicit DMA VN0, and exactly-once shim replacement.
- [ ] Run the two architecture test files and confirm RED:

~~~bash
python3 -m unittest \
  src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_run_noc_dma_softmax_sfu_pipeline.py \
  src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/test_run_noc_dma_softmax_pipeline.py -v
~~~
- [ ] Change the base archive default to os.getenv("GOLEM_DMA_RESPONSE_VN", "1"). Change the softmax shim replacement to retain num_vns=3, inject the network buffer/drain parameters, and inject golem_dma_response_vn="0". Raise if the source fragment count is not exactly one.
- [ ] Run both architecture test files and expect PASS.
- [ ] Commit only the archive/shim changes and tests.

### Task 4: Make the Runner VN-Aware and Collision-Safe

**Files:** `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_unified_job_distributed_scaling.sh`,
`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_workload_scaffold.py`.

**Interface:** GOLEM_SFU_VN_SWEEP=1 requires exact explicit_noc; reduction VN is 0, 1, or 2; run IDs include `_vn${REDUCTION_VN}`; manifest includes reduction_vn,num_vns,dma_response_vn.

- [ ] Add failing tests for unique VN IDs/signatures, invalid VN, modeled-NoC rejection, explicit child environment, stale manifest rejection, root locking, and resolved-VN mismatch.
- [ ] Run the test file and confirm RED:

~~~bash
python3 -m unittest src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_workload_scaffold.py -v
~~~
- [ ] Parse REDUCTION_VN, require explicit_noc for VN sweep, reject inherited nonzero DMA response VN, and include reduction VN, num_vns, and DMA VN in signature/manifest.
- [ ] Add a nonblocking flock on SWEEP_ROOT, exact manifest-header validation, and explicit child variables:

~~~bash
GOLEM_SFU_REDUCTION_VN="$REDUCTION_VN"
GOLEM_DMA_RESPONSE_VN=0
GOLEM_GM_VERBOSE=2
GOLEM_DMA_TRACE=1
~~~

- [ ] Add resolved-topology validation requiring GlobalMemory mapping request_vn=0, response_vn=1, reduction_vn=$REDUCTION_VN, num_vns=3; require DMA trace evidence with vn=0 and aggregate immediate/queued/rejected/received diagnostics.
- [ ] Run the runner tests and commit only runner/test changes.

### Task 5: Rebuild and Verify Shared Libraries

**Files:** build outputs `build/sst-elements/src/sst/elements/memHierarchy/.libs/libmemHierarchy.so` and `build/sst-elements/src/sst/elements/golem/.libs/libgolem.so`.

- [ ] Build/install memHierarchy using data-volume temporaries:

~~~bash
TMPDIR=/data4/jjgong/.tmp make -C build/sst-elements/src/sst/elements/memHierarchy -j4
TMPDIR=/data4/jjgong/.tmp make -C build/sst-elements/src/sst/elements/memHierarchy install
~~~

- [ ] Rebuild Golem:

~~~bash
TMPDIR=/data4/jjgong/.tmp make -C build/sst-elements/src/sst/elements/golem -j4
~~~

- [ ] Verify ELI and symbols:

~~~bash
sst-info memHierarchy.MemNIC | rg 'golem_dma_response_vn'
strings build/sst-elements/src/sst/elements/memHierarchy/.libs/libmemHierarchy.so | rg 'golem_dma_response_vn'
strings build/sst-elements/src/sst/elements/golem/.libs/libgolem.so | rg 'sfu_reduction_transport_received'
~~~

- [ ] Record library hashes and runtime LD_LIBRARY_PATH; do not commit generated libraries.

### Task 6: Run VN0/VN1/VN2 Real SST Anchors

**Files:** fresh roots
`src/sst/elements/golem/tests/artifacts/sweeps/sfu_multi_vn_vn0_20260715/`,
`src/sst/elements/golem/tests/artifacts/sweeps/sfu_multi_vn_vn1_20260715/`, and
`src/sst/elements/golem/tests/artifacts/sweeps/sfu_multi_vn_vn2_20260715/`.

- [ ] Run VN0 anchor with rows=16, dim=512, workers=4, band=4, explicit_noc, VN sweep enabled, and a fresh root:

~~~bash
TMPDIR=/data4/jjgong/.tmp GOLEM_SFU_VN_SWEEP=1 GOLEM_SFU_REDUCTION_VN=0 \
GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc \
GOLEM_SFU_DISTRIBUTED_POINT_LIST='16:512:4:4' GOLEM_STOP_ON_FAIL=1 \
GOLEM_SWEEP_ROOT=/data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_multi_vn_vn0_20260715 \
bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_unified_job_distributed_scaling.sh
~~~
- [ ] Repeat the anchor serially for VN1 and VN2, changing only VN and root; do not run them concurrently:

~~~bash
for vn in 1 2; do
  root=/data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_multi_vn_vn${vn}_20260715
  TMPDIR=/data4/jjgong/.tmp GOLEM_SFU_VN_SWEEP=1 GOLEM_SFU_REDUCTION_VN=$vn \
    GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc \
    GOLEM_SFU_DISTRIBUTED_POINT_LIST='16:512:4:4' GOLEM_STOP_ON_FAIL=1 \
    GOLEM_SWEEP_ROOT=$root \
    bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_unified_job_distributed_scaling.sh || exit $?
done
~~~
- [ ] Require each point: golden 8192/0; each request/response counter 64; transport receive 256; DMA issue/completion 64; zero retries/exhaustion; resolved topology `num_vns=3,reduction_vn=<requested VN>,dma_response_vn=0,globalmemory_response_vn=1`.
- [ ] Record input/HBM SHA-256, guest SHA-256, library SHA-256, simulated time, transport latency, inbox high-water, queued-send totals, and runtime mapping evidence.
- [ ] Re-enter each root through cache validation; expect skip validated PASS and no SST process.

### Task 7: GEMM Non-Regression

**Files:** `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/progress.md`,
`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/findings.md`, generated regression root
`src/sst/elements/golem/tests/artifacts/sweeps/sfu_multi_vn_gemm_regression_20260715/`.

- [ ] Re-run the clean default ctrl-link GEMM from Task 1. Require VERIFY-C PASS, simulation completion, unchanged DMA lifecycle, and zero reduction activity.
- [ ] Run an explicit archive/no-ctrl GEMM smoke with GOLEM_DMA_RESPONSE_VN unset. Require legacy response VN1, successful verification, complete DMA lifecycle, and zero retries. Label it archive compatibility, not default GEMM:

~~~bash
TMPDIR=/data4/jjgong/.tmp \
GOLEM_ARTIFACT_ROOT=/data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_multi_vn_archive_gemm_20260715 \
GOLEM_ARCH_SCRIPT=architecture/archive/ncores_selfcom_dma.py \
GOLEM_GM_VERBOSE=2 GOLEM_DMA_TRACE=1 \
env -u GOLEM_SFU_ENABLE -u GOLEM_SFU_STANDALONE_SOFTMAX \
  -u GOLEM_SFU_JOB_SOFTMAX -u GOLEM_SFU_PRIMITIVE_SOFTMAX \
  -u GOLEM_SFU_REDUCTION_VN -u GOLEM_DMA_RESPONSE_VN \
  bash src/sst/elements/golem/tests/run_noc_dma_pipeline.sh \
  --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --dtype fp32 --tensor-source sample --verify-c
~~~
- [ ] Compare correctness, DMA counts/bytes/retries, simulated time, guest hash, and library identities against Task 1. Any nonzero reduction event blocks completion.
- [ ] Record both regression outcomes.

### Task 8: Final Verification and Documentation

**Files:** `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/task_plan.md`,
`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/progress.md`,
`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/findings.md`.

- [ ] Run the complete focused Python suite again; expect zero failures.
- [ ] Run git diff --check and inspect git status; do not stage unrelated worktree changes or generated artifacts.
- [ ] Update Phase 4D with exact commit IDs, artifact roots, VN results, library hashes, and both GEMM regression outcomes.
- [ ] Mark the plan complete only after all correctness, runtime-VN, transport, and GEMM gates pass.
