# FlashAttention

This directory contains the active local FlashAttention scale/archive
workload. The default E3 profile is `B1,H1,S1024,D128`.

Run the verified baseline:

```bash
./run_flash_attention.sh
```

Optional pressure profiles:

```bash
./run_flash_attention.sh --profile e4
./run_flash_attention.sh --profile e5 --allow-expensive
```

The active path uses the locally built SST/Golem library and the RISC-V guest
binary built by the local `Makefile`. It is intentionally single-rank; the
scale/archive architecture rejects `GOLEM_MPI_RANKS > 1`.

Key files:

- `run_flash_attention.sh`: canonical entry point;
- `run_fused_attention_scale.sh`: SST execution and verification pipeline;
- `golem_attention_runtime.{h,cpp}`: RISC-V guest runtime;
- `attention_case.py`: deterministic Q/K/V generation;
- `verify_fused_attention_scale_output.py`: numerical verification;
- `verify_fused_attention_scale_stats.py`: lifecycle/statistics verification.
