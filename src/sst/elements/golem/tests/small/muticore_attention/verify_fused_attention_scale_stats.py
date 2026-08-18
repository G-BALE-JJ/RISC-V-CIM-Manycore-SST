#!/usr/bin/env python3
"""Verify exact Phase E four-manager/sixteen-worker activity."""

import argparse
import csv
import json
import re
from decimal import Decimal
from pathlib import Path


PROFILES = {
    "e2": {"qk": 256, "pv": 512, "jobs": 8, "rows": 128, "scaled": 4096},
    "e3": {
        "qk": 4096,
        "pv": 16384,
        "jobs": 128,
        "rows": 2048,
        "scaled": 65536,
    },
    "e4": {
        "qk": 16384,
        "pv": 65536,
        "jobs": 512,
        "rows": 8192,
        "scaled": 262144,
    },
}


def parse_frequency_hz(value):
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMgGtT]?)Hz\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(f"unsupported frequency: {value}")
    scale = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
    hz = Decimal(match.group(1)) * scale[match.group(2).lower()]
    if hz != hz.to_integral_value() or hz <= 0:
        raise argparse.ArgumentTypeError(f"frequency must be a positive integer Hz: {value}")
    return int(hz)


def ticks_to_cycles(ticks, clock_hz, timebase_ticks_per_second):
    return (ticks * clock_hz + timebase_ticks_per_second - 1) // timebase_ticks_per_second


def verify(path, profile, accelerator_clock_hz=1_000_000_000,
           timebase_ticks_per_second=10**12):
    activity = PROFILES[profile]
    observed = {}
    counts = {}
    minima = {}
    maxima = {}
    with Path(path).open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream):
            key = (row["ComponentName"], row["StatisticName"])
            observed[key] = int(row["Sum.u64"])
            counts[key] = int(row["Count.u64"])
            minima[key] = int(row["Min.u64"])
            maxima[key] = int(row["Max.u64"])
    expected = {}
    expected_counts = {}
    for core in range(4):
        component = f"core{core}:rocc"
        expected[(component, "attention_manager_jobs_issued")] = 1
        expected[(component, "attention_manager_jobs_completed")] = 1
        expected[(component, "attention_manager_bands_completed")] = 1
        expected[(component, "attention_manager_band_completions_received")] = (
            4 if core == 0 else 0
        )
        expected[(component, "attention_tensor_jobs_completed")] = (
            1 if core == 0 else 0
        )
        expected[(f"core{core}:rocc:sfu", "sfu_attention_jobs")] = 0
        for stat in (
            "attention_manager_descriptor_accept_tick",
            "attention_manager_dispatch_tick",
            "attention_manager_local_complete_tick",
            "attention_manager_wait_observed_tick",
        ):
            expected_counts[(component, stat)] = 1
        expected_counts[(component, "attention_manager_band_completion_received_tick")] = (
            4 if core == 0 else 0
        )
        expected_counts[(component, "attention_tensor_complete_tick")] = (
            1 if core == 0 else 0
        )
    for core in range(4, 20):
        expected[(f"core{core}:rocc", "attention_qk_array_ops")] = activity["qk"]
        expected[(f"core{core}:rocc", "attention_pv_array_ops")] = activity["pv"]
        expected[(f"core{core}:rocc", "attention_sp_hbm_bytes")] = 0
        expected[(f"core{core}:rocc:sfu", "sfu_attention_jobs")] = activity["jobs"]
        expected[(f"core{core}:rocc:sfu", "sfu_softmax_rows")] = activity["rows"]
        expected[(f"core{core}:rocc:sfu", "sfu_attention_scaled_elements")] = activity["scaled"]
    mismatches = {
        f"{component}/{stat}": {"expected": value, "actual": observed.get((component, stat))}
        for (component, stat), value in expected.items()
        if observed.get((component, stat)) != value
    }
    mismatches.update({
        f"{component}/{stat}.Count": {
            "expected": value,
            "actual": counts.get((component, stat)),
        }
        for (component, stat), value in expected_counts.items()
        if counts.get((component, stat)) != value
    })
    for core in range(4, 20):
        key = (f"core{core}:rocc:sfu", "sfu_attention_rsqrt_ready_tick")
        if counts.get(key) != 1:
            mismatches[f"{key[0]}/{key[1]}.Count"] = {
                "expected": 1, "actual": counts.get(key)
            }
    lifecycle_stats = (
        "attention_manager_descriptor_accept_tick",
        "attention_manager_dispatch_tick",
        "attention_manager_local_complete_tick",
        "attention_manager_wait_observed_tick",
    )
    manager_ticks = {
        stat: [observed.get((f"core{core}:rocc", stat)) for core in range(4)]
        for stat in lifecycle_stats
    }
    for core in range(4):
        ticks = [manager_ticks[stat][core] for stat in lifecycle_stats]
        if all(tick is not None for tick in ticks) and ticks != sorted(ticks):
            mismatches[f"core{core}:rocc/attention_lifecycle_order"] = {
                "expected": "accept <= dispatch <= local_complete <= wait",
                "actual": ticks,
            }

    root = "core0:rocc"
    root_accept = observed.get((root, "attention_manager_descriptor_accept_tick"))
    tensor_complete = observed.get((root, "attention_tensor_complete_tick"))
    root_wait = observed.get((root, "attention_manager_wait_observed_tick"))
    received_key = (root, "attention_manager_band_completion_received_tick")
    lifecycle = {}
    if all(value is not None for value in (root_accept, tensor_complete, root_wait)):
        if not root_accept <= tensor_complete <= root_wait:
            mismatches[f"{root}/attention_tensor_lifecycle_order"] = {
                "expected": "accept <= tensor_complete <= wait",
                "actual": [root_accept, tensor_complete, root_wait],
            }
        accept_skew_ticks = max(manager_ticks[lifecycle_stats[0]]) - min(
            manager_ticks[lifecycle_stats[0]]
        )
        local_complete_skew_ticks = max(manager_ticks[lifecycle_stats[2]]) - min(
            manager_ticks[lifecycle_stats[2]]
        )
        accelerator_completion_ticks = tensor_complete - root_accept
        wait_return_ticks = root_wait - root_accept
        lifecycle = {
            "accelerator_clock_hz": accelerator_clock_hz,
            "sst_timebase_ticks_per_second": timebase_ticks_per_second,
            "root_descriptor_accept_tick": root_accept,
            "root_tensor_complete_tick": tensor_complete,
            "root_wait_observed_tick": root_wait,
            "accelerator_completion_ticks": accelerator_completion_ticks,
            "wait_return_ticks": wait_return_ticks,
            "accelerator_completion_cycles": ticks_to_cycles(
                accelerator_completion_ticks, accelerator_clock_hz,
                timebase_ticks_per_second
            ),
            "wait_return_cycles": ticks_to_cycles(
                wait_return_ticks, accelerator_clock_hz,
                timebase_ticks_per_second
            ),
            "manager_descriptor_accept_skew_ticks": accept_skew_ticks,
            "manager_descriptor_accept_skew_cycles": ticks_to_cycles(
                accept_skew_ticks, accelerator_clock_hz, timebase_ticks_per_second
            ),
            "manager_local_complete_skew_ticks": local_complete_skew_ticks,
            "manager_local_complete_skew_cycles": ticks_to_cycles(
                local_complete_skew_ticks, accelerator_clock_hz,
                timebase_ticks_per_second
            ),
            "root_band_receive_first_tick": minima.get(received_key),
            "root_band_receive_last_tick": maxima.get(received_key),
        }
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "lifecycle": lifecycle,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--accelerator-clock", type=parse_frequency_hz,
                        default=parse_frequency_hz("1.0GHz"))
    parser.add_argument("--timebase-ticks-per-second", type=int, default=10**12)
    parser.add_argument("--result-json")
    parser.add_argument("stats_file")
    args = parser.parse_args()
    result = verify(args.stats_file, args.profile, args.accelerator_clock,
                    args.timebase_ticks_per_second)
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
