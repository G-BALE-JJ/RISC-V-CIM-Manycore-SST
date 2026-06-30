# Softmax Migration Design

Date: 2026-06-29

## Goal

Migrate all softmax work from `G-BALE-JJ/RISC-V-CIM-Manycore-SST` into the current
`wt-huti-v0-full` worktree while preserving the existing GEMM-only baseline.

The migration includes both layers:

- CPU fallback softmax test flow, documentation, wrappers, and checkers.
- SFU/RoCC scaffold in `libgolem`, including the `golem.SFU` subcomponent and
  RoCC softmax hook instructions.

The implementation will follow option B: migrate and verify the isolated CPU
softmax flow first, then migrate the more invasive SFU/RoCC layer.

## Source

Source repository:

```text
https://github.com/G-BALE-JJ/RISC-V-CIM-Manycore-SST.git
branch: softmax-update
commit observed: ef8de7a Add SFU scaffold and RoCC softmax hooks
```

The local current branch is `wt-huti-v0-full`. It currently has no softmax files
or `softmax` references under `src/sst/elements/golem`.

## Migration Scope

### Phase 1: CPU Softmax Flow

Copy and adapt:

```text
src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/
src/sst/elements/golem/tests/small/golem_operator_api.h
```

The CPU softmax directory must remain isolated beside the GEMM-only test:

```text
tests/small/mvm_noc_int_array/        existing GEMM-only baseline
tests/small/mvm_noc_softmax_cpu/      migrated GEMM + CPU softmax flow
```

The softmax directory may include and link the GEMM runtime files from
`../mvm_noc_int_array`, but it must not rewrite the GEMM-only runtime as part of
Phase 1.

Expected CPU flow capabilities:

- Build `riscv64/test_noc_dma_softmax`.
- Reuse the base `tests/run_noc_dma_pipeline.sh` through a wrapper.
- Set `VANADIS_EXE` to the softmax RISC-V binary.
- Default to disabling GEMM-only C verification because softmax overwrites C.
- Support softmax probability verification and `a_b` verification modes.
- Preserve the included design, usage, progress, and delivery documentation.

### Phase 2: SFU/RoCC Scaffold

Copy and adapt:

```text
src/sst/elements/golem/sfu/sfu.h
src/sst/elements/golem/sfu/sfu.cc
src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/
```

Merge the source changes from the old `rocc/roccAnalog.h` into the current file:

- Include `sst/elements/golem/sfu/sfu.h`.
- Add func7 values for SFU softmax tile and SFU wait.
- Add optional `sfuEnable` construction via `loadUserSubComponent`.
- Bind the SFU to GlobalMemory and core metadata.
- Forward `init`, `setup`, `complete`, and `finish` lifecycle calls.
- Dispatch SFU softmax tile and wait commands before the generic RoCC path.
- Add private `SFUAPI* sfu` and `bool sfuEnable` members.

Update:

```text
src/sst/elements/golem/Makefile.am
```

so `sfu/sfu.h` and `sfu/sfu.cc` are part of `libgolem.la`.

## Local Path Adaptation

The old softmax wrapper contains hard-coded paths from earlier work. In this
worktree they must be adapted to the current local build layout:

```text
SST core:       /data4/jjgong/local/sstcore
Elements lib:   /data4/jjgong/RISC-V-CIM-Manycore-SST/install/lib/sst-elements-library
Python lib:     /data4/jjgong/miniconda3/lib
Build tests:    /data4/jjgong/RISC-V-CIM-Manycore-SST/build/sst-elements/src/sst/elements/golem/tests
```

The wrapper should prefer environment overrides over hard-coded defaults. It
should not depend on `/data4/lishun/pkg/sst_install` or
`/data4/jjgong/local/sstelements`.

## Build Tree Handling

The source of truth is the root worktree under:

```text
/data4/jjgong/RISC-V-CIM-Manycore-SST
```

After source files are migrated, use the existing local build flow to refresh the
copied build tree:

```bash
scripts/build_and_install_local.sh --reconfigure --jobs 16
```

Manual edits to `build/sst-elements` should be avoided except for diagnostics,
because that directory is regenerated from the root worktree.

## Verification Plan

Phase 1 checks:

```bash
bash -n src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/run_noc_dma_softmax_pipeline.sh
bash -n src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/test_16core_128x128.sh
python3 src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/test_run_noc_dma_softmax_pipeline.py
python3 src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/test_verify_softmax_tile_against_golden.py
python3 -m py_compile src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/*.py
```

Then build the RISC-V softmax binary from the refreshed build tree:

```bash
cd build/sst-elements/src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu
make ARCH=riscv64
./run_noc_dma_softmax_pipeline.sh --dry-run
```

Phase 2 checks:

```bash
scripts/build_and_install_local.sh --reconfigure --jobs 16
python3 src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_component_scaffold.py
python3 src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_rocc_sfu_integration.py
python3 src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_verify_softmax_sfu_against_golden.py
python3 -m py_compile src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/*.py
```

End-to-end SST smoke should be run from a normal user shell, not the Codex
sandbox, because MPI/socket initialization is blocked in the sandbox.

Recommended smoke:

```bash
cd /data4/jjgong/RISC-V-CIM-Manycore-SST
source scripts/env_local_install.sh
cd build/sst-elements/src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu
./run_noc_dma_softmax_pipeline.sh \
  --groups 1 --num-cores 1 --gemm-cores 1 \
  --num-mem-nodes 2 --mesh-dim-x 1 \
  --num-arrays 64 --array-in 64 --array-out 64 \
  --gemm-m 64 --gemm-n 64 --gemm-k 64 \
  --gemm-block-m 64 --gemm-block-n 64 --gemm-block-k 64 \
  --group-manager-enable 0 --ctrl-link-enable 0 \
  --verify-softmax \
  --softmax-reference probability \
  --log softmax_smoke_1core.log
```

Success evidence:

```text
[SOFTMAX]
single-core softmax complete or tile-local softmax complete
[VERIFY-SOFTMAX] PASS
```

## Risks and Guardrails

- Do not overwrite current local script fixes for SST core and Python runtime
  paths.
- Keep GEMM-only `mvm_noc_int_array` runnable as a baseline.
- The old softmax docs mention several historical paths and commits; preserve
  the useful content but make current usage clear in a new top-level note if
  needed.
- SFU/RoCC hooks must compile with the current `roccAnalog.h`, which has evolved
  after the old branch. Merge by intent, not by blindly replacing the file.
- Build artifacts under `build/` and `install/` are local outputs and should not
  be treated as source.

## Acceptance Criteria

- CPU softmax files and docs exist in the current source tree.
- SFU scaffold files and docs exist in the current source tree.
- `libgolem` includes `golem.SFU` sources.
- `roccAnalog.h` contains optional SFU softmax dispatch without changing default
  behavior when `sfuEnable=0`.
- Static shell and Python checks pass.
- `scripts/build_and_install_local.sh --reconfigure --jobs 16` completes.
- A normal-shell softmax smoke can be started from the documented command and
  produces the expected softmax PASS evidence.
