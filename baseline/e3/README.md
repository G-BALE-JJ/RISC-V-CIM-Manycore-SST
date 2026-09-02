# FlashAttention E3 Baseline

The active correctness and performance baseline is the fused scale/archive
FlashAttention profile:

```text
B1,H1,S1024,D128,FP32
4 manager cores, 16 worker cores, single MPI rank
```

Run it through the unified regression command:

```bash
scripts/build_and_install_local.sh --reconfigure --jobs 16
scripts/test_flash_attention.sh
```

The build and test commands are intentionally separate. Repeat only the test
command when the local install is unchanged.
Generated logs, HBM images, tensors, and detailed statistics stay under the
artifact directory and are not part of this baseline record.

The numerical gate checks all 131072 output values. The lifecycle gate checks
manager dispatch, worker QK/Softmax/PV completion ordering, output DMA ACK, and
the single tensor completion.

The last verified result is recorded in `result.json`; rerunning the command
must match its numerical and lifecycle gates. The unified runner also forces
the current worktree `install/` library so a stale build-tree library cannot
silently replace the tested artifact.
