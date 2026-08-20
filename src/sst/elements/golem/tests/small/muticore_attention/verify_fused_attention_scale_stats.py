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

INTER_TILE_PHASE_STATS = (
    ("output_dma", "attention_worker_intertile_output_dma_ticks"),
    ("query_load", "attention_worker_intertile_query_load_ticks"),
    ("kv_load", "attention_worker_intertile_kv_load_ticks"),
    ("q_local_read", "attention_worker_intertile_q_local_read_ticks"),
    ("qk_matrix_program", "attention_worker_intertile_qk_matrix_program_ticks"),
    ("qk_input_program", "attention_worker_intertile_qk_input_program_ticks"),
    ("qk_compute_readout", "attention_worker_intertile_qk_compute_readout_ticks"),
)

TILE_PIPELINE_PHASE_STATS = (
    ("kv_load", "attention_worker_tile_kv_load_ticks"),
    ("q_local_read", "attention_worker_tile_q_local_read_ticks"),
    ("qk_matrix_program", "attention_worker_tile_qk_matrix_program_ticks"),
    ("qk_input_program", "attention_worker_tile_qk_input_program_ticks"),
    ("qk_compute_readout", "attention_worker_tile_qk_compute_readout_ticks"),
    ("softmax", "attention_worker_tile_softmax_ticks"),
    ("pv_matrix_program", "attention_worker_tile_pv_matrix_program_ticks"),
    ("pv_input_program", "attention_worker_tile_pv_input_program_ticks"),
    ("pv_restore_output", "attention_worker_tile_pv_restore_output_ticks"),
    ("pv_compute", "attention_worker_tile_pv_compute_ticks"),
    ("pv_output_readwrite", "attention_worker_tile_pv_output_readwrite_ticks"),
)


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


def summarize_intertile_breakdown(observed, counts, worker_core,
                                  accelerator_clock_hz,
                                  timebase_ticks_per_second):
    component = f"core{worker_core}:rocc"
    total_stat = "attention_worker_intertile_total_ticks"
    total_ticks = observed.get((component, total_stat))
    if total_ticks is None:
        return {}
    phase_ticks = {
        label: observed.get((component, statistic), 0)
        for label, statistic in INTER_TILE_PHASE_STATS
    }
    attributed_ticks = sum(phase_ticks.values())
    unattributed_ticks = total_ticks - attributed_ticks
    return {
        "worker_core": worker_core,
        "transition_count": counts.get((component, total_stat), 0),
        "total_ticks": total_ticks,
        "total_cycles": ticks_to_cycles(
            total_ticks, accelerator_clock_hz, timebase_ticks_per_second
        ),
        "phase_ticks": phase_ticks,
        "phase_cycles": {
            label: ticks_to_cycles(
                ticks, accelerator_clock_hz, timebase_ticks_per_second
            )
            for label, ticks in phase_ticks.items()
        },
        "phase_counts": {
            label: counts.get((component, statistic), 0)
            for label, statistic in INTER_TILE_PHASE_STATS
        },
        "attributed_ticks": attributed_ticks,
        "unattributed_ticks": unattributed_ticks,
        "conservation_valid": unattributed_ticks == 0,
    }


def summarize_tile_pipeline_breakdown(observed, counts, worker_core,
                                      accelerator_clock_hz,
                                      timebase_ticks_per_second):
    component = f"core{worker_core}:rocc"
    total_stat = "attention_worker_tile_total_ticks"
    total_ticks = observed.get((component, total_stat))
    if total_ticks is None:
        return {}
    phase_ticks = {
        label: observed.get((component, statistic), 0)
        for label, statistic in TILE_PIPELINE_PHASE_STATS
    }
    attributed_ticks = sum(phase_ticks.values())
    unattributed_ticks = total_ticks - attributed_ticks
    return {
        "worker_core": worker_core,
        "tile_count": counts.get((component, total_stat), 0),
        "total_ticks": total_ticks,
        "total_cycles": ticks_to_cycles(
            total_ticks, accelerator_clock_hz, timebase_ticks_per_second
        ),
        "phase_ticks": phase_ticks,
        "phase_cycles": {
            label: ticks_to_cycles(
                ticks, accelerator_clock_hz, timebase_ticks_per_second
            )
            for label, ticks in phase_ticks.items()
        },
        "phase_counts": {
            label: counts.get((component, statistic), 0)
            for label, statistic in TILE_PIPELINE_PHASE_STATS
        },
        "attributed_ticks": attributed_ticks,
        "unattributed_ticks": unattributed_ticks,
        "conservation_valid": unattributed_ticks == 0,
    }


def summarize_kv_second_lookahead(observed, counts, maxima, worker_cores,
                                  max_candidates, accelerator_clock_hz,
                                  timebase_ticks_per_second):
    components = [f"core{core}:rocc" for core in worker_cores]
    timing_stats = {
        "k_release": "attention_kv_k_release_ticks",
        "v_release": "attention_kv_v_release_ticks",
        "available_lead": "attention_kv_second_lookahead_lead_ticks",
    }
    timing_ticks = {
        label: sum(observed.get((component, statistic), 0)
                   for component in components)
        for label, statistic in timing_stats.items()
    }
    timing_counts = {
        label: sum(counts.get((component, statistic), 0)
                   for component in components)
        for label, statistic in timing_stats.items()
    }
    candidates = sum(observed.get(
        (component, "attention_kv_second_lookahead_candidates"), 0
    ) for component in components)
    ready_at_release = sum(observed.get(
        (component, "attention_kv_next_ready_at_release_tiles"), 0
    ) for component in components)
    max_lead_ticks = max((maxima.get(
        (component, "attention_kv_second_lookahead_lead_ticks"), 0
    ) for component in components), default=0)
    return {
        "max_candidates": max_candidates,
        "candidates": candidates,
        "candidate_rate": candidates / max_candidates if max_candidates else 0.0,
        "ready_at_release": ready_at_release,
        "ready_after_release_before_boundary": candidates - ready_at_release,
        "timing_ticks": timing_ticks,
        "timing_counts": timing_counts,
        "mean_cycles": {
            label: (
                ticks_to_cycles(timing_ticks[label], accelerator_clock_hz,
                                timebase_ticks_per_second) / count
                if count else 0.0
            )
            for label, count in timing_counts.items()
        },
        "max_available_lead_cycles": ticks_to_cycles(
            max_lead_ticks, accelerator_clock_hz, timebase_ticks_per_second
        ),
    }


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
           timebase_ticks_per_second=10**12, pv_matrix_broadcast=False,
           qk_matrix_broadcast=False, qk_dataflow_transpose=False,
           kv_double_buffer=False, pv_input_pipeline=False,
           pv_matrix_softmax_overlap=False, pv_restore_pipeline=False,
           pv_output_pipeline=False, pv_early_compute=False,
           pv_v_tile_reuse=False):
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
        expected_prefetches = activity["jobs"] - activity["qblocks"]
        expected[(component, "attention_kv_prefetch_tiles")] = (
            expected_prefetches if kv_double_buffer else 0
        )
        expected_counts[(component, "attention_kv_prefetch_dma_ticks")] = (
            expected_prefetches if kv_double_buffer else 0
        )
        expected_counts[(component, "attention_kv_k_release_ticks")] = (
            activity["jobs"] if kv_double_buffer and qk_dataflow_transpose else 0
        )
        expected_counts[(component, "attention_kv_v_release_ticks")] = (
            activity["jobs"] if kv_double_buffer and pv_v_tile_reuse else 0
        )
        expected[(component, "attention_pv_input_pipeline_rows")] = (
            activity["pv"] if pv_input_pipeline else 0
        )
        restore_rows = activity["pv"] * (
            activity["jobs"] - activity["qblocks"]
        ) // activity["jobs"]
        expected[(component, "attention_pv_restore_pipeline_rows")] = (
            restore_rows if pv_restore_pipeline else 0
        )
        expected[(component, "attention_pv_output_pipeline_rows")] = (
            activity["pv"] if pv_output_pipeline else 0
        )
        expected[(component, "attention_pv_early_compute_arrays")] = (
            activity["pv"] if pv_early_compute else 0
        )
        expected[(component, "attention_pv_matrix_overlap_tiles")] = (
            activity["jobs"] if pv_matrix_softmax_overlap else 0
        )
        expected[(component, "attention_qk_matrix_broadcasts")] = (
            (activity["qk"] // 16 if qk_dataflow_transpose else activity["jobs"])
            if qk_matrix_broadcast else 0
        )
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
        for stat in (
            "attention_worker_intertile_total_ticks",
            *(statistic for _, statistic in INTER_TILE_PHASE_STATS),
        ):
            expected_counts[(component, stat)] = activity["jobs"] - 1
        for stat in (
            "attention_worker_tile_total_ticks",
            *(statistic for _, statistic in TILE_PIPELINE_PHASE_STATS),
        ):
            expected_counts[(component, stat)] = activity["jobs"]
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
        component = f"core{core}:rocc"
        expected_consumed = (
            activity["jobs"] - activity["qblocks"] if kv_double_buffer else 0
        )
        hits = observed.get((component, "attention_kv_prefetch_hits"), 0)
        waits = observed.get((component, "attention_kv_prefetch_waits"), 0)
        if hits + waits != expected_consumed:
            mismatches[f"{component}/attention_kv_prefetch_consumed"] = {
                "expected": expected_consumed,
                "actual": hits + waits,
            }
        timing_counts = (
            ("attention_kv_prefetch_ready_lead_ticks", hits),
            ("attention_kv_prefetch_wait_ticks", waits),
        )
        for statistic, expected_count in timing_counts:
            if counts.get((component, statistic)) != expected_count:
                mismatches[f"{component}/{statistic}.Count"] = {
                    "expected": expected_count,
                    "actual": counts.get((component, statistic)),
                }
        candidates = observed.get(
            (component, "attention_kv_second_lookahead_candidates"), 0
        )
        ready_at_release = observed.get(
            (component, "attention_kv_next_ready_at_release_tiles"), 0
        )
        candidate_count = counts.get(
            (component, "attention_kv_second_lookahead_lead_ticks"), 0
        )
        max_candidates = max(activity["jobs"] - 2 * activity["qblocks"], 0)
        observation_enabled = (
            kv_double_buffer and qk_dataflow_transpose and pv_v_tile_reuse
        )
        if not observation_enabled:
            max_candidates = 0
        if candidates > max_candidates or ready_at_release > candidates or \
                candidate_count != candidates:
            mismatches[f"{component}/attention_kv_second_lookahead_window"] = {
                "expected": {
                    "candidates_at_most": max_candidates,
                    "ready_at_release_at_most": candidates,
                    "lead_count": candidates,
                },
                "actual": {
                    "candidates": candidates,
                    "ready_at_release": ready_at_release,
                    "lead_count": candidate_count,
                },
            }
        overlap_hits = observed.get(
            (component, "attention_pv_matrix_overlap_hits"), 0
        )
        overlap_waits = observed.get(
            (component, "attention_pv_matrix_overlap_waits"), 0
        )
        expected_overlaps = activity["jobs"] if pv_matrix_softmax_overlap else 0
        if overlap_hits + overlap_waits != expected_overlaps:
            mismatches[f"{component}/attention_pv_matrix_overlap_consumed"] = {
                "expected": expected_overlaps,
                "actual": overlap_hits + overlap_waits,
            }
    for core in range(4, 20):
        breakdown = summarize_intertile_breakdown(
            observed, counts, core, accelerator_clock_hz,
            timebase_ticks_per_second,
        )
        if breakdown and not breakdown["conservation_valid"]:
            mismatches[f"core{core}:rocc/attention_worker_intertile_conservation"] = {
                "expected": 0,
                "actual": breakdown["unattributed_ticks"],
            }
        tile_breakdown = summarize_tile_pipeline_breakdown(
            observed, counts, core, accelerator_clock_hz,
            timebase_ticks_per_second,
        )
        if tile_breakdown and not tile_breakdown["conservation_valid"]:
            mismatches[f"core{core}:rocc/attention_worker_tile_conservation"] = {
                "expected": 0,
                "actual": tile_breakdown["unattributed_ticks"],
            }
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
            "kv_second_lookahead_window": summarize_kv_second_lookahead(
                observed, counts, maxima, range(4, 20),
                16 * max(activity["jobs"] - 2 * activity["qblocks"], 0)
                if kv_double_buffer and qk_dataflow_transpose and pv_v_tile_reuse
                else 0,
                accelerator_clock_hz, timebase_ticks_per_second,
            ),
        }
        critical_path = lifecycle["worker_critical_path"]
        if critical_path:
            critical_path["inter_tile_breakdown"] = summarize_intertile_breakdown(
                observed, counts, critical_path["slowest_worker_core"],
                accelerator_clock_hz, timebase_ticks_per_second,
            )
            critical_path["tile_pipeline_breakdown"] = \
                summarize_tile_pipeline_breakdown(
                    observed, counts, critical_path["slowest_worker_core"],
                    accelerator_clock_hz, timebase_ticks_per_second,
                )
            component = f"core{critical_path['slowest_worker_core']}:rocc"
            prefetch_stats = {
                "dma": "attention_kv_prefetch_dma_ticks",
                "ready_lead": "attention_kv_prefetch_ready_lead_ticks",
                "wait": "attention_kv_prefetch_wait_ticks",
            }
            critical_path["kv_prefetch_timing"] = {
                "ticks": {
                    label: observed.get((component, statistic), 0)
                    for label, statistic in prefetch_stats.items()
                },
                "cycles": {
                    label: ticks_to_cycles(
                        observed.get((component, statistic), 0),
                        accelerator_clock_hz, timebase_ticks_per_second,
                    )
                    for label, statistic in prefetch_stats.items()
                },
                "counts": {
                    label: counts.get((component, statistic), 0)
                    for label, statistic in prefetch_stats.items()
                },
            }
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
    parser.add_argument("--qk-matrix-broadcast", action="store_true")
    parser.add_argument("--qk-dataflow-transpose", action="store_true")
    parser.add_argument("--kv-double-buffer", action="store_true")
    parser.add_argument("--pv-v-tile-reuse", action="store_true")
    parser.add_argument("--pv-input-pipeline", action="store_true")
    parser.add_argument("--pv-restore-pipeline", action="store_true")
    parser.add_argument("--pv-output-pipeline", action="store_true")
    parser.add_argument("--pv-early-compute", action="store_true")
    parser.add_argument("--pv-matrix-softmax-overlap", action="store_true")
    parser.add_argument("--result-json")
    parser.add_argument("stats_file")
    args = parser.parse_args()
    result = verify(args.stats_file, args.profile, args.accelerator_clock,
                    args.timebase_ticks_per_second, args.pv_matrix_broadcast,
                    args.qk_matrix_broadcast, args.qk_dataflow_transpose,
                    args.kv_double_buffer, args.pv_input_pipeline,
                    args.pv_matrix_softmax_overlap, args.pv_restore_pipeline,
                    args.pv_output_pipeline, args.pv_early_compute,
                    args.pv_v_tile_reuse)
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
