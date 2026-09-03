# FlashAttention Baseline

`run_flash_attention.sh` is the canonical entry point for the locally built
FlashAttention workload. It runs the fused scale/archive architecture and
defaults to the verified E3 profile:

```bash
./run_flash_attention.sh
```

E3 is `B1,H1,S1024,D128` and is the recommended development/regression case.
Use E4 for a larger pressure run and E5 only when an expensive run is
intentional:

```bash
./run_flash_attention.sh --profile e4
./run_flash_attention.sh --profile e5 --allow-expensive
```

The scale/archive architecture supports query-block MPI with 2 or 4 ranks:

```bash
GOLEM_MPI_RANKS=2 ./run_flash_attention.sh
GOLEM_MPI_RANKS=4 ./run_flash_attention.sh
```

It uses `sst.self` so the explicit manager/worker placement in the archive
configuration is retained. The post-run verifier rejects missing, duplicated,
or misplaced cores.

The wrapper still accepts scale-runner options, for example `--dry-run`,
`--artifact-root`, and the optimization flags documented by
`run_fused_attention_scale.sh`.
