# Causal Row Engine Findings

## Final architecture

The maintained tensor-controller path is:

`descriptor -> 16 NoC band dispatches -> per-row input DMA -> MAX -> EXP/SUM -> NORMALIZE -> per-row output DMA ACK -> 16 unique band completions -> accelerator ready -> guest wait return`

Functional arithmetic consumes data returned by DMA. Stage latency is modeled
with SST self-link events and shared vector/EXP resource availability. This is
causal functional simulation, not gate-level or per-instruction floating-point
simulation.

## Final `1024x4096` result

| Metric | Value |
| --- | ---: |
| Actual descriptor-to-accelerator completion | 66,958 cycles |
| Analytical compute reference | 66,061 cycles |
| Clean guest kernel window | 73,309 cycles |
| Whole SST interval at 2.3 GHz | 640,921 cycles |
| FP32 values checked / mismatches | 4,194,304 / 0 |
| Input DMA / output DMA ACK | 1,024 / 1,024 |
| MAX / EXP-SUM / NORMALIZE events | 1,024 / 1,024 / 1,024 |
| Unique band completions | 16 |
| Reduction requests / wait polls | 0 / 0 |
| Max NoC port utilization | 1.257% |

The exact non-overlapping accelerator path is `11 + 256 + 66,549 + 88 + 54
= 66,958` cycles: control dispatch, first input DMA, DMA-fed row pipeline,
final output DMA, and completion delivery.

## Bottleneck conclusion

EXP/SUM consumes 65,536 of 98,304 aggregate active service cycles (66.7%),
four times either vector stage. The measured NoC is lightly utilized at the
default 1200 GB/s, so the near-term compute target is higher EXP throughput.
NoC remains causal: reducing NoC/DirCtrl bandwidth to 64 GB/s increases the
`16x4096` accelerator latency from 2,076 to 4,294 cycles.

## Completion safety

- A success completion is emitted only after the corresponding output DMA ACK.
- The coordinator validates job, row, worker, shape, and band identity.
- Duplicate or stale band completions cannot advance the row count.
- A tensor job is rejected unless explicit NoC, scratch capacity, and the
  one-band-per-worker mapping are valid.
- Concurrent tensor jobs cannot alias a physical SFU's contexts or scratch.
- Invalid worker dispatches return a failure completion instead of hanging.

## Superseded evidence

The historical `66,062` cycles, `64 read + 64 write` burst count, and
`48,450`-cycle model-completion gap belong to the removed decoupled timing
model. They may be mentioned only as superseded history, never as the current
end-to-end result.
