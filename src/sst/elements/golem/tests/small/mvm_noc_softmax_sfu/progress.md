# Progress

## Final causal integration (2026-07-21)

The tensor-controller path now performs functional computation on data returned
by real NoC DMA and schedules MAX, EXP/SUM and NORMALIZE with SST self-link
events. A worker sends success only after its output DMA ACK; the coordinator
becomes ready only after 16 unique band completions.

The latest `1024x4096` run passed all 4,194,304 values with zero mismatches and
reported `66,958` accelerator cycles, `73,309` clean guest kernel cycles and
`640,921` whole-system cycles. Event counts are 1,024 input DMA, 1,024 for each
compute stage, 1,024 output ACK and 16 completion messages. Duplicate/stale
completion, unsafe scratch/band mappings and overlapping tensor jobs are
rejected. The maintained result and handoff documents live in
`tests/small/muticore_softmax`; later sections below are chronological history.

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

## Row Engine v2 follow-up

The retained `1024x4096` Row Engine run completes each physical 64-row job in
`66,061` modeled cycles, but the 16 issue ticks are spread across roughly
`291k` cycles. A 64 KiB burst control run passed correctness but regressed the
primary timings (`365,093` issue-to-completion, `1,174,256` whole-system
cycles), so that profile was rejected. Absolute-completion-tick RoCC wait
deferral is under implementation to remove the observed `1,056,100` no-op SFU
polls without changing the ready tick or functional result.

The first production-library smoke after implementing completion-tick deferral
passed at `16x4096`: all `65,536` values matched, 16 jobs completed, reduction
traffic remained zero, issue-to-completion was `29,734` cycles and Row Engine
wait polls fell to zero. A full `1024x4096` run is the next verification gate.

The full `1024x4096` verification also passed and was cycle-identical to the
retained v2 run: `357,385` issue-to-completion, `1,136,965` whole-system cycles,
`66,061` modeled compute cycles, zero reductions and zero wait polls. This
closes the wait-poll hypothesis and activates the single tensor-job hardware
row scheduler phase.

## Tensor controller completion

R3 now uses a versioned 64-byte parameter ABI and one coordinator job. The
controller sends 16 row-band dispatches over the explicit NoC, each physical
SFU runs four 16-row DMA/compute/store contexts, and 16 completion messages are
aggregated only after output DMA ACKs. Fixing the archive DirectoryController
MemNIC bandwidth wiring made the configured 1200GB/s value effective.

The final `1024x4096` run passed all `4,194,304` golden values with zero
mismatches. It measured `66,062` issue-to-final-output cycles, `103,617` guest
task cycles, `598,221` whole simulated cycles, 64 read and 64 write bursts,
16 MiB each, 32 tensor control messages and zero retry/stale/reduction events.
The strict `<=200k` accelerator target is complete.

Cycle-breakdown instrumentation now records descriptor acceptance, band and
worker dispatch, input DMA readiness, functional-compute callback completion,
output DMA ACK, coordinator completion receipt and SFU wait return. A fresh
`1024x4096` run preserved `66,062/103,617/598,221` accelerator/guest/whole
cycles and full golden PASS. It measured `37,591` cycles before descriptor
acceptance, `17,557` cycles from acceptance through final output ACK, and
`66,062` cycles from acceptance through modeled compute readiness.

A launch-timeline pass at `16x4096` and `1024x4096` added guest `rdcycle`
boundaries without changing accelerator parameters. The target run attributes
`33,583` pre-helper cycles to benchmark setup/logging, `3,035` to validation and
descriptor construction, `207/397` to parameter/descriptor GM writes and `26`
to RoCC issue. This closes the RoCC-launch hypothesis: the dominant pre-accept
cost is the diagnostic print inside the old task timing window.

The timer boundary was then moved after the diagnostic print. Fresh
`16x4096` and `1024x4096` runs passed golden and lifecycle contracts. The target
guest kernel window is now `72,409` cycles, with `6,398` cycles before descriptor
acceptance and the same `66,062` accelerator critical path. The historical
`103,617` result remains documented as the contaminated task-window baseline.

Fixed-parameter scaling then passed at `16/64/256/1024 x 4096`. Accelerator
issue-to-completion cycles were `1,550/4,622/16,910/66,062`; canonical clean
guest kernel windows were `7,819/10,954/23,323/72,409`. The 64 and 256 profiles
also exposed and fixed a runner capacity bug: tensor-controller scratch, not
only worker double buffering, must be included when deriving per-core global
memory stride.

Row Engine stage-span statistics are now independent. At the target point,
max, exp+sum and normalize contribute `16,384/65,536/16,384` active cycles.
Their scheduled temporal span is `66,048` rather than the `98,304`-cycle
sequential sum, so four-context pipelining overlaps `32,256` cycles (32.8%).
The modeled critical path is span plus a 13-cycle pipeline drain.

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
