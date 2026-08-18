#!/usr/bin/env python3
"""Verify exact Phase E four-manager/sixteen-worker activity."""

import argparse
import csv
import json
import re
from decimal import Decimal
from pathlib import Path


PROFILES = {
    "e2": {
        "qk": 256, "pv": 512, "jobs": 8, "qblocks": 1,
        "rows": 128, "scaled": 4096,
    },
    "e3": {
        "qk": 4096,
        "pv": 16384,
        "jobs": 128,
        "qblocks": 4,
        "rows": 2048,
        "scaled": 65536,
    },
    "e4": {
        "qk": 16384,
        "pv": 65536,
        "jobs": 512,
        "qblocks": 8,
        "rows": 8192,
        "scaled": 262144,
    },
    "e5": {
        "qk": 65536,
        "pv": 262144,
        "jobs": 2048,
        "qblocks": 16,
        "rows": 32768,
        "scaled": 1048576,
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


def summarize_worker_critical_path(observed, minima, maxima, worker_cores,
                                   accelerator_clock_hz,
                                   timebase_ticks_per_second):
    suffixes = (
        ("dispatch_accept", "attention_worker_dispatch_accept_tick", observed),
        ("final_qk_tile_complete", "attention_worker_qk_tile_complete_tick", maxima),
        ("final_softmax_tile_complete",
         "attention_worker_softmax_tile_complete_tick", maxima),
        ("final_pv_tile_complete", "attention_worker_pv_tile_complete_tick", maxima),
        ("final_output_dma_ack", "attention_worker_output_dma_ack_tick", maxima),
    )
    paths = {}
    for core in worker_cores:
        component = f"core{core}:rocc"
        milestones = {
            label: values.get((component, statistic))
            for label, statistic, values in suffixes
        }
        if all(value is not None for value in milestones.values()):
            paths[core] = milestones
    if not paths:
        return {}

    slowest_core = max(paths, key=lambda core: paths[core]["final_output_dma_ack"])
    milestones = paths[slowest_core]
    ordered = list(milestones.values())
    if ordered != sorted(ordered):
        return {
            "slowest_worker_core": slowest_core,
            "milestone_ticks": milestones,
            "order_valid": False,
        }

    stage_pairs = (
        ("dispatch_to_final_qk", "dispatch_accept", "final_qk_tile_complete"),
        ("final_qk_to_final_softmax", "final_qk_tile_complete",
         "final_softmax_tile_complete"),
        ("final_softmax_to_final_pv", "final_softmax_tile_complete",
         "final_pv_tile_complete"),
        ("final_pv_to_output_dma_ack", "final_pv_tile_complete",
         "final_output_dma_ack"),
    )
    stage_ticks = {
        label: milestones[end] - milestones[start]
        for label, start, end in stage_pairs
    }
    component = f"core{slowest_core}:rocc"
    qk_sum = observed[(component, "attention_worker_qk_tile_complete_tick")]
    softmax_sum = observed[(component, "attention_worker_softmax_tile_complete_tick")]
    pv_sum = observed[(component, "attention_worker_pv_tile_complete_tick")]
    aggregate_ticks = {
        "dispatch_to_first_qk": minima[
            (component, "attention_worker_qk_tile_complete_tick")
        ] - milestones["dispatch_accept"],
        "all_qk_to_softmax": softmax_sum - qk_sum,
        "all_softmax_to_pv": pv_sum - softmax_sum,
        "inter_tile_pv_to_next_qk": (
            qk_sum - minima[(component, "attention_worker_qk_tile_complete_tick")]
            - pv_sum + milestones["final_pv_tile_complete"]
        ),
        "final_pv_to_output_dma_ack": (
            milestones["final_output_dma_ack"] -
            milestones["final_pv_tile_complete"]
        ),
    }
    return {
        "slowest_worker_core": slowest_core,
        "milestone_ticks": milestones,
        "stage_ticks": stage_ticks,
        "stage_cycles": {
            label: ticks_to_cycles(
                ticks, accelerator_clock_hz, timebase_ticks_per_second
            )
            for label, ticks in stage_ticks.items()
        },
        "aggregate_online_pipeline_cycles": {
            label: ticks_to_cycles(
                ticks, accelerator_clock_hz, timebase_ticks_per_second
            )
            for label, ticks in aggregate_ticks.items()
        },
        "aggregate_interpretation": (
            "PV-to-next-QK includes next-tile KV/QK preparation and any "
            "intermediate query-block output DMA."
        ),
        "order_valid": True,
    }


def summarize_system_frontier(observed, maxima, accelerator_clock_hz,
                              timebase_ticks_per_second):
    root = "core0:rocc"
    milestones = {
        "root_descriptor_accept": observed.get(
            (root, "attention_manager_descriptor_accept_tick")
        ),
        "manager_dispatch_complete": max(
            observed.get((f"core{core}:rocc", "attention_manager_dispatch_tick"), 0)
            for core in range(4)
        ),
        "worker_dispatch_accept_complete": max(
            observed.get((f"core{core}:rocc", "attention_worker_dispatch_accept_tick"), 0)
            for core in range(4, 20)
        ),
        "final_qk_tile_complete": max(
            maxima.get((f"core{core}:rocc", "attention_worker_qk_tile_complete_tick"), 0)
            for core in range(4, 20)
        ),
        "final_softmax_tile_complete": max(
            maxima.get((f"core{core}:rocc", "attention_worker_softmax_tile_complete_tick"), 0)
            for core in range(4, 20)
        ),
        "final_pv_tile_complete": max(
            maxima.get((f"core{core}:rocc", "attention_worker_pv_tile_complete_tick"), 0)
            for core in range(4, 20)
        ),
        "final_output_dma_ack": max(
            maxima.get((f"core{core}:rocc", "attention_worker_output_dma_ack_tick"), 0)
            for core in range(4, 20)
        ),
        "manager_local_complete": max(
            observed.get((f"core{core}:rocc", "attention_manager_local_complete_tick"), 0)
            for core in range(4)
        ),
        "root_tensor_complete": observed.get((root, "attention_tensor_complete_tick")),
        "software_wait_observed": observed.get(
            (root, "attention_manager_wait_observed_tick")
        ),
    }
    if any(value in (None, 0) for value in milestones.values()):
        return {}
    ordered = list(milestones.values())
    if ordered != sorted(ordered):
        return {"milestone_ticks": milestones, "order_valid": False}

    labels = list(milestones)
    stage_ticks = {
        f"{labels[index]}_to_{labels[index + 1]}":
            milestones[labels[index + 1]] - milestones[labels[index]]
        for index in range(len(labels) - 1)
    }
    return {
        "milestone_ticks": milestones,
        "milestone_cycles_from_root_accept": {
            label: ticks_to_cycles(
                tick - milestones["root_descriptor_accept"],
                accelerator_clock_hz, timebase_ticks_per_second,
            )
            for label, tick in milestones.items()
        },
        "stage_cycles": {
            label: ticks_to_cycles(
                ticks, accelerator_clock_hz, timebase_ticks_per_second
            )
            for label, ticks in stage_ticks.items()
        },
        "order_valid": True,
        "interpretation": (
            "Frontier deltas describe when all parallel workers cross each milestone; "
            "dispatch-to-final-QK includes earlier online QK-Softmax-PV tiles."
        ),
    }


def verify(path, profile, accelerator_clock_hz=1_000_000_000,
           timebase_ticks_per_second=10**12, pv_matrix_broadcast=False):
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
        component = f"core{core}:rocc"
        expected[(component, "attention_qk_array_ops")] = activity["qk"]
        expected[(component, "attention_pv_array_ops")] = activity["pv"]
        expected[(component, "attention_sp_hbm_bytes")] = 0
        expected[(component, "attention_pv_matrix_broadcasts")] = (
            activity["pv"] // 16 if pv_matrix_broadcast else 0
        )
        expected_counts[(component, "attention_worker_dispatch_accept_tick")] = 1
        for stat in (
            "attention_worker_qk_tile_complete_tick",
            "attention_worker_softmax_tile_complete_tick",
            "attention_worker_pv_tile_complete_tick",
        ):
            expected_counts[(component, stat)] = activity["jobs"]
        expected_counts[(component, "attention_worker_output_dma_ack_tick")] = (
            activity["qblocks"]
        )
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
        worker = f"core{core}:rocc"
        worker_ticks = [
            observed.get((worker, "attention_worker_dispatch_accept_tick")),
            maxima.get((worker, "attention_worker_qk_tile_complete_tick")),
            maxima.get((worker, "attention_worker_softmax_tile_complete_tick")),
            maxima.get((worker, "attention_worker_pv_tile_complete_tick")),
            maxima.get((worker, "attention_worker_output_dma_ack_tick")),
        ]
        if all(tick is not None for tick in worker_ticks) and worker_ticks != sorted(worker_ticks):
            mismatches[f"{worker}/attention_worker_lifecycle_order"] = {
                "expected": "dispatch <= final_qk <= final_softmax <= final_pv <= output_ack",
                "actual": worker_ticks,
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
            "worker_critical_path": summarize_worker_critical_path(
                observed, minima, maxima, range(4, 20), accelerator_clock_hz,
                timebase_ticks_per_second,
            ),
            "system_frontier": summarize_system_frontier(
                observed, maxima, accelerator_clock_hz,
                timebase_ticks_per_second,
            ),
        }
        critical_path = lifecycle["worker_critical_path"]
        if critical_path and not critical_path.get("order_valid", False):
            mismatches["attention_worker_critical_path_order"] = {
                "expected": "dispatch <= final_qk <= final_softmax <= final_pv <= output_ack",
                "actual": critical_path["milestone_ticks"],
            }
        system_frontier = lifecycle["system_frontier"]
        if system_frontier and not system_frontier.get("order_valid", False):
            mismatches["attention_system_frontier_order"] = {
                "expected": "monotonic system milestone frontiers",
                "actual": system_frontier["milestone_ticks"],
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
    parser.add_argument("--pv-matrix-broadcast", action="store_true")
    parser.add_argument("--result-json")
    parser.add_argument("stats_file")
    args = parser.parse_args()
    result = verify(args.stats_file, args.profile, args.accelerator_clock,
                    args.timebase_ticks_per_second, args.pv_matrix_broadcast)
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
