# Causal Row Engine Closeout Plan

## Goal

Leave the `1024x4096` Softmax Row Engine work in a reproducible, documented,
and GitHub-ready state before starting the next task in a new session.

## Source of truth

- Final artifact: `/data4/jjgong/tmp/muticore_softmax_causal_dedupe_r1024_d4096`
- Actual accelerator completion: `66,958` cycles at 2.3 GHz
- Analytical compute reference: `66,061` cycles, not completion
- Functional check: `4,194,304` FP32 values, zero mismatches
- Causal events: 1,024 input DMA, 1,024 events per compute stage,
  1,024 output DMA ACK, 16 unique band completions

## Phases

- [x] Integrate DMA-fed MAX -> EXP/SUM -> NORMALIZE into the SST event path.
- [x] Gate successful completion on output DMA ACK and all unique band completions.
- [x] Add stale/duplicate completion rejection and tensor resource admission checks.
- [x] Pass real 16/64/256/1024 scale runs and full `1024x4096` golden.
- [x] Demonstrate NoC sensitivity with the 64 GB/s control run.
- [x] Mark obsolete 66,062-cycle/model-ready claims as superseded in maintained engineering documents.
- [x] Retain regenerated causal cycle figures and their CSV source data.
- [x] Run final documentation, build, and focused-test verification.
- [x] Prepare GitHub commands and a new-session handoff prompt.

## Guardrails

- Do not report the removed 66,062-cycle decoupled result as end-to-end latency.
- Do not sum overlapping stage windows or aggregate SFU active cycles as latency.
- Do not commit `/data4/jjgong/tmp` artifacts, HBM images, generated tensors,
  build trees, or local environments.
- Do not modify or stage group-meeting reports, PPT sources, Draw.io files, or
  presentation outputs as part of this engineering closeout.
- Do not push, merge, force-push, or delete branches without explicit approval.
