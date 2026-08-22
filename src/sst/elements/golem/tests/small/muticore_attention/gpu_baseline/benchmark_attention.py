#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def sm_clock_mhz(device):
    try:
        value = subprocess.check_output(
            ["nvidia-smi", f"--id={device}", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        return int(value)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    shape = (args.batch, args.heads, args.sequence, args.head_dim)
    if args.input_profile == "project":
        row = torch.arange(args.sequence)[:, None]
        dim = torch.arange(args.head_dim)[None, :]
        q = (((row * 17 + dim * 5) % 29) - 14).float() / 32.0
        k = (((row * 11 + dim * 7 + 3) % 31) - 15).float() / 32.0
        v = (((row * 13 + dim * 3 + 5) % 37) - 18).float() / 32.0
        q, k, v = (tensor.expand(shape).contiguous() for tensor in (q, k, v))
    else:
        q = torch.randn(shape, dtype=torch.float32)
        k = torch.randn(shape, dtype=torch.float32)
        v = torch.randn(shape, dtype=torch.float32)

    try:
        q_host, k_host, v_host = (tensor.pin_memory() for tensor in (q, k, v))
    except RuntimeError:
        q_host, k_host, v_host = q, k, v
    q_device = torch.empty(shape, device=device, dtype=torch.float32)
    k_device = torch.empty(shape, device=device, dtype=torch.float32)
    v_device = torch.empty(shape, device=device, dtype=torch.float32)

    def attention(q_value, k_value, v_value):
        return F.scaled_dot_product_attention(q_value, k_value, v_value, is_causal=args.causal)

    q_device.copy_(q_host)
    k_device.copy_(k_host)
    v_device.copy_(v_host)
    torch.cuda.synchronize(device)
    actual = attention(q_device, k_device, v_device)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(args.head_dim)
    if args.causal:
        scores.masked_fill_(torch.ones_like(scores, dtype=torch.bool).triu(1), -math.inf)
    reference = torch.softmax(scores, dim=-1) @ v
    max_abs_error = (actual.cpu() - reference).abs().max().item()
    if max_abs_error > 1.0e-4:
        raise RuntimeError(f"SDPA correctness check failed: max_abs_error={max_abs_error}")
    for _ in range(args.warmup):
        q_device.copy_(q_host, non_blocking=True)
        k_device.copy_(k_host, non_blocking=True)
        v_device.copy_(v_host, non_blocking=True)
        output_host = attention(q_device, k_device, v_device).to("cpu", non_blocking=True)
        torch.cuda.synchronize(device)
    torch.cuda.synchronize(device)

    samples_ms = []
    kernel_samples_ms = []
    for _ in range(args.iterations):
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()
        q_device.copy_(q_host, non_blocking=True)
        k_device.copy_(k_host, non_blocking=True)
        v_device.copy_(v_host, non_blocking=True)
        output_device = attention(q_device, k_device, v_device)
        output_host = output_device.to("cpu", non_blocking=True)
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - wall_start) * 1000.0)
        end_event.record()
        end_event.synchronize()
        kernel_samples_ms.append(start_event.elapsed_time(end_event))

    samples_ms.sort()
    median_ms = statistics.median(samples_ms)
    p95_ms = samples_ms[math.ceil(0.95 * len(samples_ms)) - 1]
    kernel_median_ms = statistics.median(kernel_samples_ms)
    clock_mhz = sm_clock_mhz(args.device)
    props = torch.cuda.get_device_properties(device)
    result = {
        "status": "PASS",
        "operator": "H2D -> torch.nn.functional.scaled_dot_product_attention -> D2H",
        "timing_scope": "end_to_end_host_wall_clock",
        "backend_policy": "pytorch_default",
        "shape": {"batch": args.batch, "heads": args.heads, "sequence": args.sequence, "head_dim": args.head_dim},
        "dtype": "float32",
        "input_profile": args.input_profile,
        "causal": args.causal,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency_ms": {"median": median_ms, "mean": statistics.mean(samples_ms), "min": samples_ms[0], "p95": p95_ms,
                       "kernel_event_median": kernel_median_ms},
        "correctness": {"max_abs_error_vs_explicit_gpu": max_abs_error},
        "gpu": {"index": args.device, "name": props.name, "compute_capability": list(torch.cuda.get_device_capability(device)), "sm_clock_mhz_sample": clock_mhz},
        "software": {"torch": torch.__version__, "torch_cuda": torch.version.cuda},
        "estimated_gpu_cycles_at_sampled_sm_clock": None if clock_mhz is None else round(median_ms * clock_mhz * 1000),
        "sst_reference": {"clock_ghz": 1.0, "cycles": args.sst_cycles, "latency_ms": args.sst_cycles / 1_000_000.0},
        "sst_to_gpu_latency_ratio": (args.sst_cycles / 1_000_000.0) / median_ms,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="PyTorch GPU baseline for the fused Attention SST workload")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-profile", choices=("project", "random"), default="project")
    parser.add_argument("--sst-cycles", type=int, default=699750)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be non-negative")
    result = benchmark(args)
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
