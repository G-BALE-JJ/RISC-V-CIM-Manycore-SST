# Muticore Softmax Row Engine

This directory is the dedicated 16-tile row-local Softmax profile. It reuses
the shared, regression-tested SFU guest and runner instead of maintaining a
second copy of the workload.

For the current implementation status, verification evidence, repository
boundaries, and next-session entry point, see [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md).

Run the functional smoke:

```bash
./run_muticore_softmax.sh --rows 16 --cols 4096 --timeout 600
```

Run the target point:

```bash
./run_muticore_softmax.sh --rows 1024 --cols 4096 --timeout 3600
```

The default path uses one coordinator-issued tensor job. Its hardware scheduler
dispatches 16 row bands over the NoC to 16 physical SFU/GlobalMemory endpoints,
then aggregates completion after every output DMA ACK.

`row_engine_result.json` reports critical cycles. The global sum of
all 16 SFU modeled cycles is retained under `modeled_global_sum_not_latency`
and must not be used as accelerator latency.

The final `1024x4096` evidence is under
`/data4/jjgong/tmp/muticore_softmax_causal_dedupe_r1024_d4096`.
Output completion is callback/ACK driven; no fixed post-store delay or
independent modeled-ready gate is used.

## Verified target result

The real `1024x4096` run on 2026-07-21 passed the full logits golden with
`4,194,304` values checked and zero mismatches. It used one tensor job, 16
physical SFUs, four row contexts per SFU, four-node band-striped HBM, 1,024
input DMAs, 1,024 output DMAs, 16 band dispatch/completion pairs and zero
reduction requests.

- Actual descriptor-to-accelerator completion: `66,958` cycles.
- Analytical compute reference: `66,061` cycles; this is not completion.
- Clean guest kernel window: `73,309` cycles.
- Vanadis critical core: `640,600` cycles.
- Entire SST interval: `278.661 us`, or `640,921` cycles at 2.3 GHz.
- Maximum NoC port utilization: `1.257%`.

The strict `200k` accelerator target passes. Completion requires 16 distinct
band identities; duplicate, stale or shape-mismatched completion messages are
rejected and cannot advance the completed-row count.

## Verified causal timeline

The critical path at 2.3 GHz is composed from one descriptor origin so that
rounding preserves the measured total:

- descriptor to first worker: `11` cycles;
- first worker to first input DMA ready: `256` cycles;
- DMA-fed MAX/EXP-SUM/NORMALIZE row pipeline: `66,549` cycles;
- last Normalize to final output DMA ACK: `88` cycles;
- final output ACK to actual accelerator completion: `54` cycles.

These five non-overlapping segments sum to `66,958` cycles. The final causal
ordering is last Normalize, final output DMA ACK, last unique band completion,
accelerator ready, then guest wait return. The earlier independently modeled
ready endpoint has been removed.

## Fixed-parameter shape scaling

The 2026-07-21 scaling run holds the architecture fixed at 16 physical SFUs,
four row contexts per SFU, four HBM data nodes, 256 KiB DMA bursts, 1200 GB/s
NoC links and 2.3 GHz. Every point passed the full logits golden and lifecycle
contract.

| Shape | Actual accelerator completion | Analytical reference | Clean guest kernel |
| --- | ---: | ---: | ---: |
| 16x4096 | 2,076 | 1,549 | 8,346 |
| 64x4096 | 5,651 | 4,621 | 11,984 |
| 256x4096 | 17,790 | 16,909 | 24,201 |
| 1024x4096 | 66,958 | 66,061 | 73,309 |

`kernel_window_cycles` is the canonical guest metric: the guest `rdcycle`
window starts after benchmark diagnostics and ends when the accelerator wait
returns. `guest_core_critical_cycles` remains as a compatibility alias.

At `1024x4096`, aggregate active service is `16,384` MAX, `65,536` EXP/SUM
and `16,384` Normalize cycles. The actual first-start-to-last-done windows are
`62,973`, `66,032` and `65,252` cycles, respectively; the full causal stage
pipeline spans `66,549` cycles. Reducing NoC/DirCtrl bandwidth from `1200 GB/s`
to `64 GB/s` increases the `16x4096` completion latency from `2,076` to `4,294`
cycles, confirming that the DMA path is part of the timing chain.
