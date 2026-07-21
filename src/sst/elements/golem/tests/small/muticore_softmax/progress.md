# Causal Row Engine Progress

## 2026-07-21

- Replaced the independent tensor ready tick with DMA-driven stage events.
- Added one-row physical contexts and shared vector/EXP resource scheduling.
- Delayed successful worker completion until output DMA ACK.
- Required 16 unique, identity-checked band completions before accelerator ready.
- Added rejection for unsafe transport, scratch, band, and concurrent-job cases.
- Added actual stage timestamps, event counts, causal parser checks, and NoC
  bandwidth controls.
- Passed real `16x4096`, `64x4096`, `256x4096`, and `1024x4096` runs.
- Passed the final `1024x4096` golden with 4,194,304 values and zero mismatches.
- Measured final accelerator completion at 66,958 cycles.
- Regenerated three cycle/bottleneck figures and their CSV source data.
- Completed two rounds of independent code review with no remaining Critical
  or Important findings.

## Engineering closeout

- Updated maintained engineering documentation to use the causal result and
  retained group-meeting/PPT/Draw.io materials unchanged.
- Added `PROJECT_HANDOFF.md` with repository boundaries, reproducible evidence,
  GitHub-safe staging guidance, and a new-session prompt.
- Final verification passed: 19 focused Row Engine tests, 12 dedicated
  workload/parser/layout/plot tests, incremental `libgolem` and guest builds,
  retained-result contract parsing, and `git diff --check`.
