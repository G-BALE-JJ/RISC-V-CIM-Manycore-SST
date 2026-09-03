# FlashAttention E3 Baseline

The active correctness and performance baseline is the fused scale/archive
FlashAttention profile:

```text
B1,H1,S1024,D128,FP32
4 manager cores, 16 worker cores, 1, 2, or 4 MPI ranks
```

Run it through the unified regression command:

```bash
scripts/build_and_install_local.sh --reconfigure --jobs 16
scripts/test_flash_attention.sh
scripts/test_flash_attention.sh --mpi-ranks 2
scripts/test_flash_attention.sh --mpi-ranks 4
```

The build command installs SST elements and builds all FlashAttention RISC-V
guests. The test command never invokes a compiler. Repeat only the test command
when the local install and guest binaries are unchanged.
Generated logs, HBM images, tensors, and detailed statistics stay under the
artifact directory and are not part of this baseline record.

The numerical gate checks all 131072 output values. The lifecycle gate checks
manager dispatch, worker QK/Softmax/PV completion ordering, output DMA ACK, and
the single tensor completion.

The single-rank result is recorded in `result.json`; the query-block MPI result
is recorded in `mpi2/result.json`, and the one-group-per-rank result is in
`mpi4/result.json`. Rerunning the corresponding command
must match its numerical and lifecycle gates. The unified runner also forces
the current worktree `install/` library so a stale build-tree library cannot
silently replace the tested artifact.
