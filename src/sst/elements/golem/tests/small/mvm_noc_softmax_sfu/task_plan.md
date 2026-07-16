# Active Task Plan

## Current state

- [x] Unified direct row-major HBM job path.
- [x] Distributed columns and cooperative row bands.
- [x] Real explicit-NoC reduction transport.
- [x] Phase 4F eight-point real SST matrix.
- [x] Capacity `512x4096` and `1024x4096` real SST PASS.
- [x] Obsolete generated artifacts and superseded documents cleaned: project
  `115 GiB -> 24 GiB`, docs `27 -> 6`, raw sweep roots `88 -> 4`.
- [x] Post-cleanup verification: `280/280` focused tests, syntax/compile/diff
  checks and RISC-V Softmax guest rebuild PASS.
- [ ] `2048x4096` and `4096x4096` deferred.

## Wall-time optimization

- [ ] W0: prevent custom Softmax runs from rebuilding/overwriting the unused
  GEMM guest; preserve default GEMM behavior and run its canonical regression.
- [ ] W1: disable Vanadis pipeline trace; A/B at `64x4096`, watchdog `600s`.
- [ ] W2: disable per-event NoC/GM/DMA/band text while retaining configuration
  evidence and required statistics.
- [ ] W3: add a separately labelled quiet guest benchmark mode; retain errors
  and final PASS output.
- [ ] W4: evaluate selective statistics and SST `--num-threads=2/4`, one variable
  at a time.
- [ ] W5: verify the best low-risk combination at `256x4096`.
- [ ] Decide whether event-driven SFU wait warrants a separate model design.

## Constraints

- No bandwidth, VN, worker, chunk, retry or memory DSE.
- No change to Softmax mathematics for wall-time optimization.
- Every SST run has a fixed watchdog; timeout stops the experiment.
- Do not resume 1024/2048/4096 until the smaller A/B is accepted.
- Any shared runner or RoCC change requires the original GEMM regression.

## Current documents

- `docs/superpowers/specs/2026-07-08-sfu-unified-job-architecture-design.md`
- `docs/superpowers/specs/2026-07-14-sfu-simple-network-reduction-design.md`
- `docs/superpowers/specs/2026-07-15-sfu-phase4f-large-scale-explicit-noc-softmax-design.md`
- `docs/superpowers/plans/2026-07-15-sfu-phase4f-large-scale-explicit-noc-softmax-implementation.md`
- `docs/superpowers/specs/2026-07-16-sfu-4096x4096-explicit-noc-capacity-validation-design.md`
- `docs/superpowers/plans/2026-07-16-sfu-4096x4096-explicit-noc-capacity-validation-implementation.md`
