# MVM NoC Softmax SFU

This directory contains the small-test workload, checker, and local notes for
the first `golem.SFU` softmax path.

## Current Status

- `golem.SFU` is implemented as an independent SST subcomponent.
- SFU mounting is controlled by `GOLEM_SFU_ENABLE`; the default GEMM path keeps
  SFU disabled.
- The current exposed operation is fused full row-wise softmax over GEMM output
  tiles.
- The fused path has passed `64x64`, `128x128`, and `256x256` golden checks.
- `512x512` remains the next stress/performance check.
- The next architecture direction is a generic standalone SFU primitive ABI for
  `EXP`, `LOG`, `RECIPROCAL`, then `RSQRT`, `TANH`, `SIGMOID`, and reductions.

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
