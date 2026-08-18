#!/usr/bin/env python3

import argparse
import csv
import json
import math
import pathlib
import re
import sys


ROW_ENGINE_LINE = re.compile(
    r"\[SOFTMAX-ROW-ENGINE\] core=(?P<core>\d+) rows=(?P<rows>\d+) "
    r"start_cycle=(?P<start>\d+) end_cycle=(?P<end>\d+) cycles=(?P<cycles>\d+)"
)
LAUNCH_TIMELINE = re.compile(
    r"launch_start_cycle=(?P<launch_start>\d+) "
    r"descriptors_ready_cycle=(?P<descriptors_ready>\d+) "
    r"params_write_done_cycle=(?P<params_write_done>\d+) "
    r"desc_write_done_cycle=(?P<desc_write_done>\d+) "
    r"issue_return_cycle=(?P<issue_return>\d+) "
    r"wait_start_cycle=(?P<wait_start>\d+) "
    r"wait_return_cycle=(?P<wait_return>\d+)"
)


def load_component_stats(path, marker):
    components = {}
    with open(path, newline="", encoding="utf-8") as stats_file:
        for row in csv.reader(stats_file):
            if len(row) < 11 or marker not in row[0]:
                continue
            try:
                value_sum = int(float(row[6]))
                value_count = int(float(row[8]))
                value_min = int(float(row[9]))
                value_max = int(float(row[10]))
            except ValueError:
                continue
            components.setdefault(row[0], {})[row[1]] = {
                "sum": value_sum,
                "count": value_count,
                "min": value_min,
                "max": value_max,
            }
    return components


def load_sfu_stats(path):
    return load_component_stats(path, ":sfu")


def load_system_envelope(path):
    simulated_time_ps = 0
    vanadis_cycles = []
    with open(path, newline="", encoding="utf-8") as stats_file:
        for row in csv.reader(stats_file):
            if len(row) < 11:
                continue
            try:
                simulated_time_ps = max(simulated_time_ps, int(float(row[4])))
            except ValueError:
                pass
            if re.fullmatch(r"core\d+", row[0]) and row[1] == "cycles":
                try:
                    vanadis_cycles.append(int(float(row[6])))
                except ValueError:
                    pass
    return simulated_time_ps, max(vanadis_cycles, default=0)


def stat_sum(components, name):
    return sum(stats.get(name, {}).get("sum", 0) for stats in components.values())


def stat_critical_max(components, name):
    return max((stats.get(name, {}).get("sum", 0) for stats in components.values()), default=0)


def stat_min(components, name):
    values = [
        stats[name]["min"]
        for stats in components.values()
        if name in stats and stats[name]["sum"] > 0
    ]
    return min(values) if values else 0


def stat_max(components, name):
    values = [stats[name]["max"] for stats in components.values() if name in stats]
    return max(values) if values else 0


def stat_count(components, name):
    return sum(stats.get(name, {}).get("count", 0) for stats in components.values())


def ticks_to_cycles(ticks, clock_hz, ticks_per_second):
    return math.ceil(max(0, ticks) * clock_hz / ticks_per_second)


def completed_rows(components):
    total = 0
    for stats in components.values():
        jobs = stats.get("sfu_row_engine_jobs", {}).get("sum", 0)
        completed = stats.get("sfu_row_engine_completed_jobs", {}).get("sum", 0)
        if jobs > 0 and completed == jobs:
            total += stats.get("sfu_row_engine_rows", {}).get("sum", 0)
    return total


def tile_contract_failures(components, rows, tensor_controller=False,
                           expected_sfus=16, worker_count=16,
                           manager_coordinator=False):
    failures = []
    if len(components) != expected_sfus:
        failures.append(f"physical_sfus={len(components)} expected={expected_sfus}")
    if rows % worker_count != 0:
        failures.append(f"rows={rows} is not divisible by workers={worker_count}")
        return failures
    if manager_coordinator:
        return failures
    if tensor_controller:
        jobs = stat_sum(components, "sfu_row_engine_jobs")
        assigned = stat_sum(components, "sfu_row_engine_rows")
        completed = stat_sum(components, "sfu_row_engine_completed_jobs")
        if (jobs, assigned, completed) != (1, rows, 1):
            failures.append(
                f"tensor jobs/rows/completed={jobs}/{assigned}/{completed} "
                f"expected=1/{rows}/1"
            )
        return failures
    expected_rows = rows // 16
    for component, stats in sorted(components.items()):
        jobs = stats.get("sfu_row_engine_jobs", {}).get("sum", 0)
        assigned = stats.get("sfu_row_engine_rows", {}).get("sum", 0)
        completed = stats.get("sfu_row_engine_completed_jobs", {}).get("sum", 0)
        modeled = stats.get("sfu_row_engine_modeled_cycles", {}).get("sum", 0)
        if (jobs, assigned, completed) != (1, expected_rows, 1):
            failures.append(
                f"{component}: jobs/rows/completed={jobs}/{assigned}/{completed} "
                f"expected=1/{expected_rows}/1"
            )
        if modeled <= 0:
            failures.append(f"{component}: modeled_cycles={modeled} expected>0")
    return failures


def load_guest_cycles(stdout_dir):
    records = []
    if not stdout_dir:
        return records
    for path in pathlib.Path(stdout_dir).rglob("stdout-*"):
        text = path.read_text(errors="ignore")
        launch = LAUNCH_TIMELINE.search(text)
        launch_values = (
            {key: int(value) for key, value in launch.groupdict().items()} if launch else {}
        )
        for match in ROW_ENGINE_LINE.finditer(text):
            record = {key: int(value) for key, value in match.groupdict().items()}
            record.update(launch_values)
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--stdout-dir", default="")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--clock-hz", type=int, default=2_300_000_000)
    parser.add_argument("--timebase-ticks-per-second", type=int, default=1_000_000_000_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--require-contract", action="store_true")
    parser.add_argument("--tensor-controller", action="store_true")
    parser.add_argument("--manager-coordinator", action="store_true")
    parser.add_argument("--worker-count", type=int, default=16)
    parser.add_argument("--manager-count", type=int, default=0)
    args = parser.parse_args()

    components = load_sfu_stats(args.stats)
    manager_components = load_component_stats(args.stats, ":rocc")
    simulated_time_ps, vanadis_critical_cycles = load_system_envelope(args.stats)
    issue_tick = stat_min(
        manager_components if args.manager_coordinator else components,
        "tensor_manager_descriptor_accept_tick" if args.manager_coordinator
        else "sfu_row_engine_issue_tick",
    )
    ready_tick = stat_max(
        manager_components if args.manager_coordinator else components,
        "tensor_manager_complete_tick" if args.manager_coordinator
        else "sfu_row_engine_ready_tick",
    )
    observed_tick = stat_max(
        manager_components if args.manager_coordinator else components,
        "tensor_manager_wait_observed_tick" if args.manager_coordinator
        else "sfu_row_engine_completion_observed_tick",
    )
    issue_to_ready_ticks = max(0, ready_tick - issue_tick)
    issue_to_observed_ticks = max(0, observed_tick - issue_tick)
    timeline_ticks = {
        "descriptor_accept": issue_tick,
        "accelerator_complete": ready_tick,
        "first_band_dispatch": stat_min(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_band_dispatch_tick" if args.manager_coordinator
            else "sfu_tensor_band_dispatch_tick",
        ),
        "last_band_dispatch": stat_max(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_band_dispatch_tick" if args.manager_coordinator
            else "sfu_tensor_band_dispatch_tick",
        ),
        "first_worker_dispatch": stat_min(components, "sfu_tensor_worker_dispatch_tick"),
        "last_worker_dispatch": stat_max(components, "sfu_tensor_worker_dispatch_tick"),
        "first_input_dma_ready": stat_min(components, "sfu_tensor_input_dma_ready_tick"),
        "last_input_dma_ready": stat_max(components, "sfu_tensor_input_dma_ready_tick"),
        "first_max_start": stat_min(components, "sfu_tensor_max_start_tick"),
        "last_max_done": stat_max(components, "sfu_tensor_max_done_tick"),
        "first_exp_sum_start": stat_min(components, "sfu_tensor_exp_sum_start_tick"),
        "last_exp_sum_done": stat_max(components, "sfu_tensor_exp_sum_done_tick"),
        "first_normalize_start": stat_min(components, "sfu_tensor_normalize_start_tick"),
        "last_normalize_done": stat_max(components, "sfu_tensor_normalize_done_tick"),
        "first_compute_done": stat_min(components, "sfu_tensor_compute_done_tick"),
        "last_compute_done": stat_max(components, "sfu_tensor_compute_done_tick"),
        "first_output_dma_ack": stat_min(components, "sfu_tensor_output_dma_ack_tick"),
        "final_output_dma_ack": stat_max(components, "sfu_tensor_output_dma_ack_tick"),
        "first_completion_received": stat_min(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_completion_received_tick" if args.manager_coordinator
            else "sfu_tensor_completion_received_tick",
        ),
        "last_completion_received": stat_max(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_completion_received_tick" if args.manager_coordinator
            else "sfu_tensor_completion_received_tick",
        ),
        "guest_wait_observed": stat_max(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_wait_observed_tick" if args.manager_coordinator
            else "sfu_tensor_guest_wait_observed_tick",
        ),
    }
    timeline_pairs = {
        "descriptor_to_first_band_dispatch": ("descriptor_accept", "first_band_dispatch"),
        "band_dispatch_span": ("first_band_dispatch", "last_band_dispatch"),
        "first_band_dispatch_to_first_worker_dispatch": ("first_band_dispatch", "first_worker_dispatch"),
        "first_worker_dispatch_to_first_input_dma_ready": ("first_worker_dispatch", "first_input_dma_ready"),
        "input_dma_ready_span": ("first_input_dma_ready", "last_input_dma_ready"),
        "max_stage_window": ("first_max_start", "last_max_done"),
        "exp_sum_stage_window": ("first_exp_sum_start", "last_exp_sum_done"),
        "normalize_stage_window": ("first_normalize_start", "last_normalize_done"),
        "row_engine_pipeline_window": ("first_max_start", "last_normalize_done"),
        "first_input_dma_ready_to_last_compute_done": ("first_input_dma_ready", "last_compute_done"),
        "compute_done_span": ("first_compute_done", "last_compute_done"),
        "last_compute_done_to_final_output_ack": ("last_compute_done", "final_output_dma_ack"),
        "output_dma_ack_span": ("first_output_dma_ack", "final_output_dma_ack"),
        "descriptor_to_final_output_ack": ("descriptor_accept", "final_output_dma_ack"),
        "descriptor_to_accelerator_complete": ("descriptor_accept", "accelerator_complete"),
        "final_output_ack_to_completion_received": ("final_output_dma_ack", "last_completion_received"),
        "completion_received_to_guest_wait_observed": ("last_completion_received", "guest_wait_observed"),
    }
    timeline_cycles = {
        name: ticks_to_cycles(
            timeline_ticks[end] - timeline_ticks[start],
            args.clock_hz,
            args.timebase_ticks_per_second,
        )
        for name, (start, end) in timeline_pairs.items()
        if timeline_ticks[start] and timeline_ticks[end]
    }
    guest_records = load_guest_cycles(args.stdout_dir)
    if guest_records and issue_tick:
        descriptor_accept_cycle = ticks_to_cycles(
            issue_tick, args.clock_hz, args.timebase_ticks_per_second
        )
        timeline_cycles["guest_start_to_descriptor_accept"] = max(
            0, descriptor_accept_cycle - min(record["start"] for record in guest_records)
        )
    if guest_records and timeline_ticks["guest_wait_observed"]:
        guest_wait_observed_cycle = ticks_to_cycles(
            timeline_ticks["guest_wait_observed"],
            args.clock_hz,
            args.timebase_ticks_per_second,
        )
        timeline_cycles["guest_wait_observed_to_guest_end"] = max(
            0, max(record["end"] for record in guest_records) - guest_wait_observed_cycle
        )
    timeline_event_counts = {
        "band_dispatch": stat_count(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_band_dispatch_tick" if args.manager_coordinator
            else "sfu_tensor_band_dispatch_tick",
        ),
        "worker_dispatch": stat_count(components, "sfu_tensor_worker_dispatch_tick"),
        "input_dma_ready": stat_count(components, "sfu_tensor_input_dma_ready_tick"),
        "max_start": stat_count(components, "sfu_tensor_max_start_tick"),
        "max_done": stat_count(components, "sfu_tensor_max_done_tick"),
        "exp_sum_start": stat_count(components, "sfu_tensor_exp_sum_start_tick"),
        "exp_sum_done": stat_count(components, "sfu_tensor_exp_sum_done_tick"),
        "normalize_start": stat_count(components, "sfu_tensor_normalize_start_tick"),
        "normalize_done": stat_count(components, "sfu_tensor_normalize_done_tick"),
        "compute_done": stat_count(components, "sfu_tensor_compute_done_tick"),
        "output_dma_ack": stat_count(components, "sfu_tensor_output_dma_ack_tick"),
        "completion_received": stat_count(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_completion_received_tick" if args.manager_coordinator
            else "sfu_tensor_completion_received_tick",
        ),
        "guest_wait_observed": stat_count(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_wait_observed_tick" if args.manager_coordinator
            else "sfu_tensor_guest_wait_observed_tick",
        ),
    }
    actual_stage_windows_cycles = {
        "max": timeline_cycles.get("max_stage_window", 0),
        "exp_sum": timeline_cycles.get("exp_sum_stage_window", 0),
        "normalize": timeline_cycles.get("normalize_stage_window", 0),
        "pipeline": timeline_cycles.get("row_engine_pipeline_window", 0),
    }
    guest_launch_cycles = {}
    launch_records = [record for record in guest_records if "launch_start" in record]
    if launch_records:
        launch = max(launch_records, key=lambda record: record["cycles"])
        guest_launch_cycles = {
            "task_start_to_launch": max(0, launch["launch_start"] - launch["start"]),
            "descriptor_construction": max(0, launch["descriptors_ready"] - launch["launch_start"]),
            "params_write": max(0, launch["params_write_done"] - launch["descriptors_ready"]),
            "descriptor_write": max(0, launch["desc_write_done"] - launch["params_write_done"]),
            "issue_return": max(0, launch["issue_return"] - launch["desc_write_done"]),
            "issue_return_to_wait_entry": max(0, launch["wait_start"] - launch["issue_return"]),
            "wait_entry_to_return": max(0, launch["wait_return"] - launch["wait_start"]),
        }
    whole_architecture_cycles = math.ceil(
        simulated_time_ps * args.clock_hz / args.timebase_ticks_per_second
    )
    kernel_window_cycles = max((record["cycles"] for record in guest_records), default=0)
    stage_cycles = {
        "max_active": stat_critical_max(components, "sfu_row_engine_max_cycles"),
        "max_start": stat_critical_max(components, "sfu_row_engine_max_start_cycles"),
        "max_end": stat_critical_max(components, "sfu_row_engine_max_end_cycles"),
        "exp_sum_active": stat_critical_max(components, "sfu_row_engine_exp_sum_cycles"),
        "exp_sum_start": stat_critical_max(components, "sfu_row_engine_exp_sum_start_cycles"),
        "exp_sum_end": stat_critical_max(components, "sfu_row_engine_exp_sum_end_cycles"),
        "normalize_active": stat_critical_max(components, "sfu_row_engine_normalize_cycles"),
        "normalize_start": stat_critical_max(components, "sfu_row_engine_normalize_start_cycles"),
        "normalize_end": stat_critical_max(components, "sfu_row_engine_normalize_end_cycles"),
    }
    stage_cycles["sequential_active"] = sum(
        stage_cycles[name] for name in ("max_active", "exp_sum_active", "normalize_active")
    )
    stage_cycles["temporal_span"] = max(
        stage_cycles[name] for name in ("max_end", "exp_sum_end", "normalize_end")
    )
    stage_cycles["overlap_cycles"] = max(
        0, stage_cycles["sequential_active"] - stage_cycles["temporal_span"]
    )
    stage_cycles["overlap_fraction"] = (
        stage_cycles["overlap_cycles"] / stage_cycles["sequential_active"]
        if stage_cycles["sequential_active"] else 0.0
    )

    result = {
        "attempt_id": args.attempt_id,
        "rows": args.rows,
        "cols": args.cols,
        "physical_sfus_with_stats": len(components),
        "row_engine_jobs": stat_sum(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_jobs_issued" if args.manager_coordinator else "sfu_row_engine_jobs",
        ),
        "rows_dispatched": stat_sum(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_rows_dispatched" if args.manager_coordinator else "sfu_row_engine_rows",
        ),
        "rows_completed": stat_sum(manager_components, "tensor_manager_rows_completed")
            if args.manager_coordinator else completed_rows(components),
        "completed_jobs": stat_sum(
            manager_components if args.manager_coordinator else components,
            "tensor_manager_jobs_completed" if args.manager_coordinator
            else "sfu_row_engine_completed_jobs",
        ),
        "vector_max_active_critical_cycles": stat_critical_max(components, "sfu_row_engine_max_cycles"),
        "exp_sum_active_critical_cycles": stat_critical_max(components, "sfu_row_engine_exp_sum_cycles"),
        "normalize_active_critical_cycles": stat_critical_max(components, "sfu_row_engine_normalize_cycles"),
        "analytical_compute_cycles": stat_critical_max(components, "sfu_row_engine_modeled_cycles"),
        "modeled_critical_cycles": stat_critical_max(components, "sfu_row_engine_modeled_cycles"),
        "stage_cycles": stage_cycles,
        "modeled_global_sum_not_latency": stat_sum(components, "sfu_row_engine_modeled_cycles"),
        "issue_to_ready_ticks": issue_to_ready_ticks,
        "issue_to_ready_cycles": math.ceil(issue_to_ready_ticks * args.clock_hz / args.timebase_ticks_per_second),
        "accelerator_latency_cycles": math.ceil(
            issue_to_ready_ticks * args.clock_hz / args.timebase_ticks_per_second
        ),
        "issue_to_completion_observed_cycles": math.ceil(issue_to_observed_ticks * args.clock_hz / args.timebase_ticks_per_second),
        "timeline_ticks": timeline_ticks,
        "timeline_cycles": timeline_cycles,
        "timeline_event_counts": timeline_event_counts,
        "actual_stage_windows_cycles": actual_stage_windows_cycles,
        "guest_launch_cycles": guest_launch_cycles,
        "vanadis_critical_cycles": vanadis_critical_cycles,
        "sst_simulated_time_ps": simulated_time_ps,
        "whole_architecture_cycles_at_clock": whole_architecture_cycles,
        "noc_simulated_window_cycles_at_clock": whole_architecture_cycles,
        "kernel_window_cycles": kernel_window_cycles,
        "kernel_window_definition": "guest rdcycle after diagnostics through accelerator wait return",
        "guest_core_critical_cycles": kernel_window_cycles,
        "guest_global_first_start_to_last_end_cycles": (
            max(record["end"] for record in guest_records) - min(record["start"] for record in guest_records)
            if guest_records else 0
        ),
        "reduction_request_messages": (
            stat_sum(components, "sfu_reduction_max_requests")
            + stat_sum(components, "sfu_reduction_sum_requests")
        ),
        "wait_polls": stat_sum(components, "sfu_row_engine_wait_polls"),
    }
    failures = tile_contract_failures(
        components, args.rows, args.tensor_controller,
        expected_sfus=args.worker_count + args.manager_count,
        worker_count=args.worker_count,
        manager_coordinator=args.manager_coordinator,
    )
    if args.manager_coordinator:
        if stat_sum(manager_components, "tensor_manager_workers_mapped") != args.worker_count:
            failures.append("manager topology map worker count is inconsistent")
        for manager_core in range(args.manager_count):
            manager_sfu = components.get(f"core{manager_core}:sfu", {})
            if any(manager_sfu.get(name, {}).get("sum", 0) != 0 for name in (
                    "sfu_row_engine_jobs", "sfu_row_engine_rows",
                    "sfu_tensor_worker_dispatch_tick")):
                failures.append(f"manager core{manager_core} SFU datapath was used")
    if args.tensor_controller:
        expected_control_events = min(args.rows, args.worker_count)
        required_timeline = [
            "descriptor_accept", "first_band_dispatch", "first_worker_dispatch",
            "first_input_dma_ready", "first_max_start", "last_max_done",
            "first_exp_sum_start", "last_exp_sum_done", "first_normalize_start",
            "last_normalize_done", "first_compute_done", "first_output_dma_ack",
            "final_output_dma_ack", "last_completion_received", "guest_wait_observed",
            "accelerator_complete",
        ]
        if any(timeline_ticks[name] == 0 for name in required_timeline):
            failures.append("tensor timeline is incomplete")
        control_events_match = (
            timeline_event_counts["band_dispatch"] == expected_control_events
            and timeline_event_counts["worker_dispatch"] == expected_control_events
            and timeline_event_counts["completion_received"] == expected_control_events
        )
        row_events_match = all(
            timeline_event_counts[name] == args.rows
            for name in (
                "input_dma_ready", "max_start", "max_done", "exp_sum_start",
                "exp_sum_done", "normalize_start", "normalize_done",
                "compute_done", "output_dma_ack",
            )
        )
        if not control_events_match or not row_events_match or \
                timeline_event_counts["guest_wait_observed"] != 1:
            failures.append("tensor timeline event counts are inconsistent")
        first_row_causal = [
            timeline_ticks["first_input_dma_ready"],
            timeline_ticks["first_max_start"],
            stat_min(components, "sfu_tensor_max_done_tick"),
            timeline_ticks["first_exp_sum_start"],
            stat_min(components, "sfu_tensor_exp_sum_done_tick"),
            timeline_ticks["first_normalize_start"],
            stat_min(components, "sfu_tensor_normalize_done_tick"),
        ]
        final_causal = [
            timeline_ticks["last_normalize_done"],
            timeline_ticks["final_output_dma_ack"],
            timeline_ticks["last_completion_received"],
            timeline_ticks["accelerator_complete"],
            timeline_ticks["guest_wait_observed"],
        ]
        if first_row_causal != sorted(first_row_causal) or \
                final_causal != sorted(final_causal):
            failures.append("tensor timeline violates DMA/compute/completion causality")
    if result["rows_completed"] != args.rows:
        failures.append(
            f"rows_completed={result['rows_completed']} expected={args.rows}"
        )
    if result["modeled_critical_cycles"] > 150000:
        failures.append(
            f"modeled_critical_cycles={result['modeled_critical_cycles']} limit=150000"
        )
    if (args.rows, args.cols) == (1024, 4096) and result["modeled_critical_cycles"] > 70000:
        failures.append(
            f"modeled_critical_cycles={result['modeled_critical_cycles']} v2_limit=70000"
        )
    if result["issue_to_completion_observed_cycles"] > 1600000:
        failures.append(
            "issue_to_completion_observed_cycles="
            f"{result['issue_to_completion_observed_cycles']} limit=1600000"
        )
    if (args.rows, args.cols) == (1024, 4096) and args.tensor_controller and \
            result["issue_to_completion_observed_cycles"] > 200000:
        failures.append(
            "issue_to_completion_observed_cycles="
            f"{result['issue_to_completion_observed_cycles']} tensor_limit=200000"
        )
    expected_jobs = 1 if args.tensor_controller else 16
    aggregate_contract = (
        result["row_engine_jobs"] == expected_jobs
        and result["completed_jobs"] == expected_jobs
        and result["rows_dispatched"] == args.rows
        and result["reduction_request_messages"] == 0
    )
    if not aggregate_contract:
        failures.append("aggregate job/row/completion/reduction contract failed")
    result["contract_failures"] = failures
    result["contract_pass"] = not failures
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.require_contract and not result["contract_pass"]:
        print("row-engine contract failed", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
