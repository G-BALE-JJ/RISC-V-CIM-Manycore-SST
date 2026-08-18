# Multicore Attention

Architecture and implementation references:

- `docs/superpowers/plans/2026-07-22-attention-flash-attention-implementation.md`:
  authoritative bilingual implementation plan;
- `task_plan.md`: current Phase D/Phase E status and locked decisions;
- `findings.md`: component-level model-realism audit and remediation order;
- `progress.md`: accepted results and the latest handoff state.

This directory validates the first Attention dataflow step:

```text
Q[M,D] x K[N,D]^T -> scores[M,N]
```

`K` stays in its native key-major `[N,D]` layout. The matmul descriptor sets
`transpose_b=1`, so each native K row is consumed as one logical B column. This
maps directly onto the existing RoCC batch vector loader and does not require a
materialized transpose buffer or a new transpose state machine.

Run the default smoke case:

```bash
./run_muticore_attention.sh
```

Exercise a partial final key tile. The runner pads physical K storage and the
GEMM N extent to a multiple of 16, then crops back to the logical key length:

```bash
./run_muticore_attention.sh --queries 64 --keys 70 --head-dim 64
```

The runner generates deterministic fp32 Q/K inputs, runs the real SST
GEMM/RoCC path, dumps the complete score matrix, and verifies every output
element against `Q x K^T`. Artifacts are written outside the repository under
`/data4/jjgong/tmp` by default.

Run the accepted Phase C1 fused case:

```bash
./run_fused_attention.sh
```

This fixed case is `B1,H1,S32,D64,Br16,Bc32`, non-causal. It uses manager core
0 and worker core 1, keeps S/P only in the worker's per-Core GlobalMemory, and
verifies both all 2,048 output values and the exact manager/QK/PV/SFU/HBM
activity counters. Multi-KV-tile online recurrence starts in Phase D.

Run the accepted Phase D1 online fused case:

```bash
./run_fused_attention_online.sh
```

This fixed case is `B1,H1,S64,D64,Br16,Bc32`, non-causal. It traverses two
KV tiles with bounded SFU `(m,l)` state, restores and rescales Oacc through the
Array output buffer, keeps S/P out of HBM, and verifies all 4,096 outputs plus
the exact QK/PV/SFU/RSQRT counters.

Run the accepted Phase D2 causal fused case:

```bash
./run_fused_attention_online.sh --causal 1
```

The causal run uses the same `B1,H1,S64,D64,Br16,Bc32` layout. Fully future
KV tiles are skipped before QK/PV/SFU issue, while diagonal tiles are masked
element by element after scale and before row max. The runner verifies all
4,096 outputs and exact activity (`QK/PV=192/384`, `SFU jobs/rows=6/96`,
`masked=992`) with zero S/P HBM bytes.

Run the accepted Phase D3 partial-tile case:

```bash
./run_fused_attention_online.sh --partial
```

This case is `B1,H1,Sq20,Skv70,D64,Br16,Bc32`, non-causal. The final query
block contains 4 rows and the final key tile contains 6 keys. HBM DMA and
Array issue use only those valid extents; zero padding exists only in fixed
Array operand buffers. The runner verifies all 1,280 outputs and exact
activity (`QK/PV=140/240`, `SFU jobs/rows=6/60`, `scaled=1400`) with zero S/P
HBM bytes.

Run the accepted Phase D4 extreme-logit case:

```bash
./run_fused_attention_online.sh --extreme-logits
```

This uses the D1 shape and hardware path, but constructs two deterministic KV
tiles whose scaled logits are exactly -100 and +100. It therefore forces an
online running-max jump of 200 while retaining the exact D1 activity gate. The
runner verifies all 4,096 finite outputs with zero S/P HBM bytes. Phase D is
complete.

Run the accepted Phase E4 scale case (the default):

```bash
./run_fused_attention_scale.sh
```

This case is `B1,H1,S2048,D128,Br16,Bc32`, with manager cores 0-3 and explicitly
mapped worker cores 4-19. Q/O and K/V are block-striped across HBM nodes 1-4;
each worker processes eight 16-row query blocks and streams one 32-row K/V tile
at a time through its local GlobalMemory. The runner verifies all 262,144
outputs and exact per-manager and per-worker activity with zero S/P HBM bytes.

Phase E2 now uses the same command and scale dataflow. Manager cores 1-3 send
an explicit manager-band completion to root manager core 0 after their local
worker bitmaps complete. The root accepts four unique manager slots and emits
exactly one tensor completion. The exact stats gate checks local bands
`1/1/1/1`, root manager completions `4`, and tensor completions `1/0/0/0`.
The accepted E2/E3 cases remain available with:

```bash
./run_fused_attention_scale.sh --scale-point e2
./run_fused_attention_scale.sh --scale-point e3
```

The next scale point is `B1,H1,S4096,D128`.
