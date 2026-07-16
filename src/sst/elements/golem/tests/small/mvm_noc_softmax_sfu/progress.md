# Progress

## Milestones

- Unified SFU job descriptor and direct row-major HBM execution replaced the
  primitive/batch Softmax development path.
- Column slices were distributed across physical workers, followed by
  cooperative row-band execution and strict reduction/DMA observability.
- Phase 3B representative scaling passed real SST.
- Modeled-NoC established the original counter contract; explicit-NoC then
  moved reduction requests/responses onto real SimpleNetwork events.
- VN compatibility was verified once and the normal profile was frozen at
  `num_vns=3`, reduction VN0 and DMA response VN0.
- Phase 4F completed eight real SST points and generated deterministic source
  CSV plus SVG/PDF/PNG reports.
- Capacity validation passed `512x4096` and `1024x4096`; larger rows were paused
  by user decision, not by failure or timeout.

## Capacity evidence

`512x4096`:

- child wall `1462s`, simulated `3773.42 us`;
- golden `2,097,152/0`;
- four reduction classes `8,192` each, transport `32,768`;
- zero retry, rejected and stale events.

`1024x4096`:

- child wall `2890s`, simulated `7236.07 us`;
- golden `4,194,304/0`;
- four reduction classes `16,384` each, transport `65,536`;
- DMA read/write `16,384` operations and `16,777,216` bytes each;
- zero retry, rejected and stale events.

## Wall-time investigation

The 1024 point spent about `113s` in preparation, `2755s` in SST and `26s` in
post-processing. SST therefore accounts for roughly 94.5% of wall time.

Across 16 cores the run accumulated:

- `115,777,120` core cycles;
- `41,858,634` retired instructions;
- about `301,434` RoCC commands.

The main low-risk overheads are per-retire Vanadis pipeline trace and per-event
NoC/GlobalMemory/DMA text. Mailbox adaptive waits and cycle-polled RoCC SFU wait
are model-level event costs and need separate treatment.

The Softmax wrapper also rebuilt the unused GEMM guest because the shared
pipeline saw `GOLEM_SKIP_BUILD=0`. This changed the ignored GEMM binary and its
metadata without changing GEMM source. Build isolation is the first blocker.

## Cleanup

On 2026-07-16 the project occupied about 115 GiB. Cleanup removed build-tree
artifact copies, non-canonical raw sweeps, scattered HBM/log/stats outputs,
generated logits, caches and superseded CPU Softmax prototypes/documents.
Canonical Phase 4F, capacity, group-report and GEMM-regression roots were kept.

After cleanup the project occupies about 24 GiB, releasing roughly 91 GiB.
`docs/superpowers` was reduced from 27 files to 6, raw sweep roots from 88 to 4,
and the legacy CPU Softmax directory to a 48 KiB compatibility layer. Ten tests
that only asserted deleted historical sweep wrappers were removed; production
primitive and unified component coverage remains. The final focused suite is
`280/280 PASS`; shell syntax, Python compilation, `git diff --check`, and a real
RISC-V Softmax guest rebuild with the fixed musl toolchain all pass.

The retained build library is
`build/sst-elements/src/sst/elements/golem/.libs/libgolem.so` with SHA-256
`00bb13483978a77f282e78458a61b26f627ce1ea23b60ef04714cce4031b63a4`.
