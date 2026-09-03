# FlashAttention

This directory contains the active local FlashAttention scale/archive
workload. The default E3 profile is `B1,H1,S1024,D128`.

Run the verified baseline:

```bash
./run_flash_attention.sh
```

Run the frozen multi-rank query-block MPI regressions from the repository root:

```bash
scripts/test_flash_attention.sh --mpi-ranks 2
scripts/test_flash_attention.sh --mpi-ranks 4
```

Optional pressure profiles:

```bash
./run_flash_attention.sh --profile e4
./run_flash_attention.sh --profile e5 --allow-expensive
```

The active path uses the locally built SST/Golem library and the RISC-V guest
binary built by the local `Makefile`. MPI supports 2 or 4 ranks. Each manager
and its four workers remain colocated, so the four query bands are assigned
two-per-rank or one-per-rank.

Key files:

- `run_flash_attention.sh`: canonical entry point;
- `run_fused_attention_scale.sh`: SST execution and verification pipeline;
- `golem_attention_runtime.{h,cpp}`: RISC-V guest runtime;
- `attention_case.py`: deterministic Q/K/V generation;
- `verify_fused_attention_scale_output.py`: numerical verification;
- `verify_fused_attention_scale_stats.py`: lifecycle/statistics verification.
- `verify_attention_mpi_partition.py`: query-block rank placement verification.
