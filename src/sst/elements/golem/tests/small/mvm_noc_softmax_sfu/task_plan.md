# Active Task Plan

## Current state

> 2026-07-21 closeout: the maintained tensor-controller result is now the
> causal Row Engine implementation documented under `muticore_softmax`.
> Historical v2/model-ready measurements below are retained as development
> history and are not the current completion contract.

- [x] Unified direct row-major HBM job path.
- [x] Distributed columns and cooperative row bands.
- [x] Real explicit-NoC reduction transport.
- [x] Phase 4F eight-point real SST matrix.
- [x] Capacity `512x4096` and `1024x4096` real SST PASS.
- [x] Obsolete generated artifacts and superseded documents cleaned: project
  `115 GiB -> 24 GiB`, docs `27 -> 6`, raw sweep roots `88 -> 4`.
- [x] Post-cleanup verification: `280/280` focused tests, syntax/compile/diff
  checks and RISC-V Softmax guest rebuild PASS.
- [x] Row Engine v2 `1024x4096` final-source real SST PASS: `494.332 us`,
  `1,136,965` whole-architecture cycles, `1,136,396` Vanadis critical cycles,
  `66,061` modeled compute cycles, `357,385` issue-to-SFU-completion cycles,
  `566,048` guest-through-output-DMA cycles, zero reduction
  messages and golden mismatch 0.
- [x] Canonical FP32 `64x64x64` GEMM regression PASS with output SHA identical
  to the retained pre-change regression.
- [ ] `2048x4096` and `4096x4096` deferred.

## Approved architecture redesign

- [x] Design direction approved: use one tile per row as the `dim=4096`
  primary path, with 16 independent row groups.
- [x] Preserve the current 16-worker distributed-column path as a legacy
  correctness/regression mode.
- [x] R0: accelerator/Vanadis/system-cycle accounting and production/guest/
  runner/input/output SHA signatures are persisted per run.
- [ ] R1: run the fixed-total-tile `1x16, 2x8, 4x4, 8x2, 16x1`
  (`tiles_per_row x groups`) mapping matrix.
- [x] R2: implement absolute-ready-tick Row Engine timing, scratchpad capacity,
  and four-context row pipeline overlap.
- [x] R3: one versioned tensor job, NoC hardware row-band scheduling across 16
  physical SFU/GlobalMemory endpoints, and output-ACK completion aggregation.
- [x] R4: stripe contiguous tile row bands over four data HBM nodes and
  coalesce the Row Engine profile to 256 KiB DMA bursts. Hardware-owned
  DMA/compute/store overlap remains open.
- [x] R4a: reject 64 KiB bursts as a performance fix: despite reducing NoC
  output-port stalls, the real `1024x4096` issue-to-completion and whole-system
  cycles regressed to `365,093` and `1,174,256` respectively.
- [x] Reject a 1 MiB burst boundary test: it did not complete after roughly
  15 minutes (more than twice the 64 KiB run wall time) and remained in the
  Vanadis wait path. Keep 256 KiB frozen; no further burst sweep.
- [x] R4b implementation and real validation: absolute-completion-tick deferral
  produced zero wait polls at both `16x4096` and `1024x4096`. The full run is
  cycle-identical to v2 (`357,385` issue-to-completion, `1,136,965` whole
  system) with full golden PASS, proving wait polling was not the cycle cause.
- [ ] R5: add `(m,l)` pair collective only for rows requiring multiple tiles.
- [x] R6: pass both gates at `1024x4096`: `66,958` actual
  descriptor-to-accelerator completion cycles (`<=200k`) and `640,921` whole
  simulated cycles. The final path uses 1,024 input DMA operations, 1,024
  output DMA ACKs and 16 unique band completions.

Decision source:
`docs/superpowers/specs/2026-07-16-sfu-softmax-row-engine-noc-architecture-design.md`.

Active implementation plan:
`docs/superpowers/plans/2026-07-17-sfu-softmax-row-engine-noc-implementation.md`.

Implementation started on 2026-07-17. The first delivery gate is the dedicated
`tests/small/muticore_softmax` 16-tile row-local workload, followed by four-node
HBM row striping and a fresh `1024x4096` SST measurement.

The `120k--150k` accelerator-cycle range is a design target, not a measured
result. The redesign starts with R0 instrumentation and uses new artifact roots;
historical Phase 4F/capacity PASS markers are never reused after production or
guest changes.

## Wall-time optimization

- [x] W0: prevent custom Softmax runs from rebuilding/overwriting the unused
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
- [x] Absolute-completion-tick wait deferral selected because the final v2 run
  performed `1,056,100` compatibility polls for only 16 jobs.

## Constraints

- No bandwidth, VN, worker, chunk, retry or memory DSE.
- No change to Softmax mathematics for wall-time optimization.
- Every SST run has a fixed watchdog; timeout stops the experiment.
- Do not resume 1024/2048/4096 until the smaller A/B is accepted.
- Any shared runner or RoCC change requires the original GEMM regression.

## Current documents

- `docs/superpowers/specs/2026-07-16-sfu-softmax-row-engine-noc-architecture-design.md`
- `docs/superpowers/specs/2026-07-08-sfu-unified-job-architecture-design.md`
- `docs/superpowers/specs/2026-07-14-sfu-simple-network-reduction-design.md`
- `docs/superpowers/specs/2026-07-15-sfu-phase4f-large-scale-explicit-noc-softmax-design.md`
- `docs/superpowers/plans/2026-07-15-sfu-phase4f-large-scale-explicit-noc-softmax-implementation.md`
- `docs/superpowers/specs/2026-07-16-sfu-4096x4096-explicit-noc-capacity-validation-design.md`
- `docs/superpowers/plans/2026-07-16-sfu-4096x4096-explicit-noc-capacity-validation-implementation.md`
