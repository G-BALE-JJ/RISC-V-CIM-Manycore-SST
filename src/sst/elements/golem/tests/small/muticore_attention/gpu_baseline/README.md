# GPU Attention Baseline

This directory measures end-to-end GPU latency for the same operator shape as
the formal SST E3 workload: `B1,H1,S1024,D128,FP32`, non-causal scaled
dot-product attention.
The default `project` input profile uses the same deterministic Q/K/V formulas
as `attention_case.py`.

Run:

```bash
python3 benchmark_attention.py \
  --output results/v100_e3_fp32.json
```

The timed region is one complete task path: host-to-device copies for Q/K/V,
PyTorch `scaled_dot_product_attention(Q, K, V)`, device-to-host output copy,
and a host synchronization after the output copy. Input construction, one-time
buffer allocation, warmup, and the explicit correctness reference are excluded.
The reported latency uses host wall-clock time; the CUDA-event kernel-only time
is retained as a diagnostic. GPU cycles are estimates based on one sampled SM
clock; latency is the primary comparison metric because GPU clocks are dynamic.

The SST reference is the formal G5 result: `699,750 cycles @ 1 GHz = 0.69975 ms`.

## V100 E3 result

Result file: `results/v100_e3_fp32.json`

The checked-in JSON currently contains the earlier kernel-only run and must be
regenerated with the command above after CUDA access is available. The table
below is intentionally left pending until the end-to-end run completes.

| Metric | Value |
|---|---:|
| GPU | Tesla V100-SXM2-32GB |
| PyTorch / CUDA build | 2.10.0+cu128 / 12.8 |
| Median / p95 end-to-end latency | rerun required |
| Max error vs explicit GPU reference | 1.91e-9 |
| Sampled SM clock | 1290 MHz |
| Estimated GPU cycles | rerun required |
| Formal SST latency | 0.69975 ms |
| SST / GPU latency | rerun required |

The GPU and SST cycle counts are both in the `10^5` range. The GPU cycle value
is not a fixed architectural count because the SM clock is dynamic; use kernel
latency as the primary result.
