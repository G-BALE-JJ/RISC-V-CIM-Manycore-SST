#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import hashlib
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile


@dataclasses.dataclass(frozen=True)
class PointSpec:
    stage: str
    rows: int
    dim: int
    worker_cores: int
    band_cores: int
    mem_node_size: int
    timeout_sec: int


@dataclasses.dataclass(frozen=True)
class PointRecord:
    spec: PointSpec
    run_id: str
    chunk_elems: int
    cooperative_groups: int
    transport: str
    reduction_vn: int
    num_vns: int
    dma_response_vn: int
    noc_link_bw: str
    noc_xbar_bw: str
    dirctrl_highlink_bw: str
    noc_input_buffer: str
    noc_output_buffer: str
    gm_buffer: str
    flit_size: str
    retry_ticks: int
    max_retries: int
    status: str
    exit_code: int
    artifact_validation: str
    golden_checked: int | None
    golden_mismatches: int | None
    transport_events: int | None
    transport_immediate: int | None
    transport_queued: int | None
    transport_rejected: int | None
    transport_stale: int | None
    inbox_high_water: int | None
    latency_avg_cycles: float | None
    latency_max_cycles: int | None
    total_send_packets: int | None
    total_send_bits: int | None
    total_xbar_stalls: int | None
    simulated_time_us: float | None
    wall_time_sec: float | None
    dma_timeout_retry: int | None
    dma_timeout_exhausted: int | None
    dma_write_timeout_retry: int | None
    output_sha256: str | None
    child_root: str


CANONICAL_NETWORK: dict[str, str] = {
    "GOLEM_NOC_LINK_BW": "1200GB/s",
    "GOLEM_NOC_XBAR_BW": "1200GB/s",
    "GOLEM_DIRCTRL_HIGHLINK_BW": "1200GB/s",
    "GOLEM_NOC_INPUT_BUF_SIZE": "512KB",
    "GOLEM_NOC_OUTPUT_BUF_SIZE": "512KB",
    "GOLEM_NOC_FLIT_SIZE": "128B",
    "GOLEM_GM_BUFFER_LENGTH": "1024KB",
    "GOLEM_NOC_INTER_ROUTER_NO_CUT": "0",
    "GOLEM_NOC_LOCAL_NO_CUT": "0",
}

TRANSPORT = "explicit_noc"
NUM_VNS = 3
REDUCTION_VN = 0
DMA_RESPONSE_VN = 0

DEFAULT_POINTS: tuple[PointSpec, ...] = (
    PointSpec("A", 16, 512, 16, 16, 134217728, 900),
    PointSpec("A", 16, 1024, 16, 16, 134217728, 1800),
    PointSpec("A", 16, 2048, 16, 16, 268435456, 2400),
    PointSpec("A", 16, 4096, 16, 16, 268435456, 3600),
    PointSpec("B", 16, 4096, 4, 4, 268435456, 3600),
    PointSpec("B", 16, 4096, 8, 8, 268435456, 3600),
    PointSpec("C", 64, 4096, 16, 16, 268435456, 7200),
    PointSpec("C", 256, 4096, 16, 16, 268435456, 14400),
)

_POINTS_BY_IDENTITY = {
    (point.rows, point.dim, point.worker_cores, point.band_cores): point
    for point in DEFAULT_POINTS
}


def resolve_point(rows: int, dim: int, workers: int, bands: int) -> PointSpec:
    values = (rows, dim, workers, bands)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("point fields must be positive integers")
    if workers == 1 or bands == 1:
        raise ValueError("single-worker points are not part of Phase 4F")
    if workers != bands:
        raise ValueError("worker and band core counts must match")

    try:
        return _POINTS_BY_IDENTITY[values]
    except KeyError as exc:
        raise ValueError(
            f"point is outside the Phase 4F matrix: {rows}:{dim}:{workers}:{bands}"
        ) from exc


def parse_point_list(value: str | None) -> tuple[PointSpec, ...]:
    if value is None:
        return DEFAULT_POINTS
    tokens = value.split()
    if not tokens:
        raise ValueError("point list must not be empty")

    points: list[PointSpec] = []
    seen: set[tuple[int, int, int, int]] = set()
    for token in tokens:
        fields = token.split(":")
        if len(fields) != 4:
            raise ValueError(f"invalid point syntax: {token!r}")
        try:
            values = tuple(int(field) for field in fields)
        except ValueError as exc:
            raise ValueError(f"point fields must be integers: {token!r}") from exc

        identity = (values[0], values[1], values[2], values[3])
        if identity in seen:
            raise ValueError(f"duplicate point: {token}")
        points.append(resolve_point(*identity))
        seen.add(identity)

    return tuple(points)


CHILD_MANIFEST_FIELDS = [
    "run_id", "rows", "dim", "chunk_elems", "worker_cores", "band_cores",
    "cooperative_groups", "reduction_vn", "num_vns", "dma_response_vn",
    "staging_rows", "job_rows", "retry_ticks", "max_retries", "status",
    "exit_code", "timeout_sec", "artifact_validation",
]

PARENT_MANIFEST_FIELDS = [
    "run_id", "stage", "rows", "dim", "chunk_elems", "worker_cores",
    "band_cores", "transport", "reduction_vn", "num_vns", "dma_response_vn",
    "noc_link_bw", "noc_xbar_bw", "dirctrl_highlink_bw", "noc_input_buffer",
    "noc_output_buffer", "gm_buffer", "flit_size", "mem_node_size",
    "retry_ticks", "max_retries", "timeout_sec", "status", "exit_code",
    "artifact_validation", "golden_checked", "golden_mismatches",
    "transport_events", "transport_immediate", "transport_queued",
    "transport_rejected", "transport_stale", "inbox_high_water",
    "latency_avg_cycles", "latency_max_cycles", "total_send_packets",
    "total_send_bits", "total_xbar_stalls", "simulated_time_us",
    "wall_time_sec", "dma_timeout_retry", "dma_timeout_exhausted",
    "dma_write_timeout_retry", "output_sha256", "child_root",
]

POINT_STATUS_FIELDS = [
    "stage", "rows", "dim", "worker_cores", "band_cores", "status",
    "exit_code", "transport", "reduction_vn", "num_vns", "dma_response_vn",
    "noc_link_bw", "noc_xbar_bw", "dirctrl_highlink_bw", "noc_input_buffer",
    "noc_output_buffer", "gm_buffer", "flit_size", "inter_router_no_cut",
    "local_no_cut", "mem_node_size", "retry_ticks", "max_retries",
    "timeout_sec", "child_root",
]

SOURCE_DATA_FIELDS = PARENT_MANIFEST_FIELDS + [
    "time_per_row_us", "time_per_element_us",
]

_STAT_FIELDS = ("StatisticName", "Sum.u64", "Count.u64", "Max.u64")
_POINT_ID_FIELDS = (
    "stage", "rows", "dim", "worker_cores", "band_cores", "mem_node_size",
    "timeout_sec",
)


def _error(root: pathlib.Path, run_id: str, field: str, detail: str) -> ValueError:
    return ValueError(
        f"child_root={root} run_id={run_id or '<manifest>'} field={field}: {detail}"
    )


def _read_csv(
    path: pathlib.Path,
    root: pathlib.Path,
    run_id: str,
    field: str,
    required_fields: tuple[str, ...] | list[str],
    *,
    exact_header: bool = False,
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _error(root, run_id, field, str(exc)) from exc
    names = reader.fieldnames
    if names is None:
        raise _error(root, run_id, field, "CSV header is missing")
    inferred_run_id = rows[0].get("run_id", run_id) if rows else run_id
    if exact_header and names != list(required_fields):
        raise _error(root, inferred_run_id, field, f"schema mismatch: {names!r}")
    missing = [name for name in required_fields if name not in names]
    if missing:
        raise _error(root, inferred_run_id, field, f"missing columns {missing!r}")
    if not rows:
        raise _error(root, run_id, field, "CSV contains no data rows")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise _error(root, inferred_run_id, field, "malformed CSV structure")
    return rows


def _integer(root: pathlib.Path, run_id: str, field: str, value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise _error(root, run_id, field, f"expected integer, got {value!r}") from exc


def _nonnegative_integer(
    root: pathlib.Path, run_id: str, field: str, value: str
) -> int:
    result = _integer(root, run_id, field, value)
    if result < 0:
        raise _error(root, run_id, field, f"expected nonnegative integer, got {result}")
    return result


def _require_equal(root, run_id, field, actual, expected):
    if actual != expected:
        raise _error(root, run_id, field, f"expected {expected!r}, got {actual!r}")


def _canonical_run_id(spec: PointSpec) -> str:
    return (
        f"sfu_job_dist_r{spec.rows}_d{spec.dim}_w{spec.worker_cores}"
        f"_bc{spec.band_cores}_g1_vn0"
    )


def select_child_manifest_row(
    child_root: pathlib.Path, spec: PointSpec
) -> dict[str, str]:
    child_root = pathlib.Path(child_root)
    rows = _read_csv(
        child_root / "sweep_manifest.csv",
        child_root,
        "",
        "sweep_manifest.csv",
        CHILD_MANIFEST_FIELDS,
        exact_header=True,
    )
    expected_ints = {
        "rows": spec.rows,
        "dim": spec.dim,
        "chunk_elems": 256,
        "worker_cores": spec.worker_cores,
        "band_cores": spec.band_cores,
        "cooperative_groups": 1,
        "reduction_vn": REDUCTION_VN,
        "num_vns": NUM_VNS,
        "dma_response_vn": DMA_RESPONSE_VN,
        "staging_rows": 4,
        "job_rows": 4,
        "retry_ticks": 1024,
        "max_retries": 8,
        "timeout_sec": spec.timeout_sec,
    }
    run_ids = {row.get("run_id", "") for row in rows}
    if "" in run_ids or len(run_ids) != 1:
        raise _error(child_root, next(iter(run_ids), ""), "run_id", "expected one run ID")
    run_id = next(iter(run_ids))
    expected_run_id = _canonical_run_id(spec)
    if run_id != expected_run_id:
        raise _error(
            child_root, run_id, "run_id",
            f"expected canonical {expected_run_id!r}, got {run_id!r}",
        )
    canonical = []
    for row in rows:
        for field, expected in expected_ints.items():
            _require_equal(
                child_root,
                run_id,
                field,
                _integer(child_root, run_id, field, row[field]),
                expected,
            )
        _require_equal(child_root, run_id, "status", row["status"], "PASS")
        _require_equal(
            child_root,
            run_id,
            "exit_code",
            _integer(child_root, run_id, "exit_code", row["exit_code"]),
            0,
        )
        artifact = row["artifact_validation"]
        if artifact == "PASS":
            canonical.append(row)
        elif artifact != "CACHED":
            raise _error(child_root, run_id, "artifact_validation", f"unexpected {artifact!r}")
    if len(canonical) != 1:
        raise _error(
            child_root,
            run_id,
            "artifact_validation",
            f"expected one canonical PASS/PASS row, got {len(canonical)}",
        )
    return canonical[0]


def _stat_values(root, run_id, rows, statistic, field="Sum.u64") -> list[int]:
    matches = [row for row in rows if row["StatisticName"] == statistic]
    if not matches:
        raise _error(root, run_id, statistic, "statistic is missing")
    return [
        _nonnegative_integer(root, run_id, f"{statistic}.{field}", row[field])
        for row in matches
    ]


def _stat_sum(root, run_id, rows, statistic) -> int:
    return sum(_stat_values(root, run_id, rows, statistic))


def _metric(root, run_id, path, metric, value_field) -> int:
    rows = _read_csv(path, root, run_id, path.name, ("metric", value_field))
    matches = [row for row in rows if row["metric"] == metric]
    if len(matches) != 1:
        raise _error(root, run_id, metric, f"expected one row, got {len(matches)}")
    return _nonnegative_integer(root, run_id, metric, matches[0][value_field])


def _check_log(root: pathlib.Path, run_id: str, log: pathlib.Path) -> float:
    try:
        text = log.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _error(root, run_id, "sst_log", str(exc)) from exc
    required = (
        "[NoC] input_buf_size=512KB, output_buf_size=512KB, link_bw=1200GB/s, "
        "xbar_bw=1200GB/s, flit_size=128B",
        "[NoC] inter_router_no_cut=0, local_no_cut=0",
        "[GOLEM] GlobalMemory link buffer_length=1024KB",
        "GlobalMemory VN mapping: request_vn=0 response_vn=1 reduction_vn=0 (num_vns=3)",
        "resolved golem_dma_response_vn=0 num_vns=3 explicit=1",
    )
    lines = text.splitlines()
    missing = [
        evidence
        for evidence in required
        if not any(
            re.search(rf"{re.escape(evidence)}(?=$|[\s,;])", line)
            for line in lines
        )
    ]
    if missing:
        raise _error(root, run_id, "noc_profile", f"missing exact evidence: {missing!r}")
    matches = re.findall(
        r"Simulation is complete, simulated time:\s*([0-9]+(?:\.[0-9]+)?)\s*(us|ms|s)\b",
        text,
    )
    if len(matches) != 1:
        raise _error(root, run_id, "simulated_time_us", f"expected one completion line, got {len(matches)}")
    value, unit = matches[0]
    return float(value) * {"us": 1.0, "ms": 1_000.0, "s": 1_000_000.0}[unit]


def _golden(root, run_id, spec, output, verifier) -> tuple[int, int]:
    logits = root / "inputs" / f"softmax_logits_{spec.rows}x{spec.dim}.bin"
    required = (verifier, root / "inputs" / "a.bin", root / "inputs" / "b.bin", logits, output)
    for path in required:
        if not pathlib.Path(path).is_file():
            raise _error(root, run_id, "golden_input", f"missing {path}")
    try:
        result = subprocess.run([
            sys.executable, str(verifier), "--a-file", str(root / "inputs" / "a.bin"),
            "--b-file", str(root / "inputs" / "b.bin"), "--c-file", str(output),
            "--m", str(spec.rows), "--n", str(spec.dim), "--k", str(spec.dim),
            "--block-m", "4", "--block-n", "64", "--dtype", "fp32",
            "--reference", "logits", "--logits-file", str(logits),
        ], check=False, capture_output=True, text=True)
    except (OSError, UnicodeError) as exc:
        raise _error(root, run_id, "verifier", str(exc)) from exc
    if result.returncode != 0:
        raise _error(root, run_id, "verifier", f"exit={result.returncode}: {result.stdout.strip()}")
    evidence = re.findall(r"^\[VERIFY-SFU-SOFTMAX\] PASS.*\bchecked=(\d+)\b.*\bmismatches=(\d+)\b", result.stdout, re.MULTILINE)
    if len(evidence) != 1:
        raise _error(root, run_id, "verifier_output", f"expected one PASS evidence line, got {len(evidence)}")
    checked, mismatches = map(int, evidence[0])
    _require_equal(root, run_id, "golden_checked", checked, spec.rows * spec.dim)
    _require_equal(root, run_id, "golden_mismatches", mismatches, 0)
    return checked, mismatches


def parse_child_point(
    child_root: pathlib.Path, spec: PointSpec, verifier: pathlib.Path
) -> PointRecord:
    root = pathlib.Path(child_root)
    row = select_child_manifest_row(root, spec)
    run_id = row["run_id"]
    logs = sorted((root / "logs").glob(f"*{run_id}*.log"))
    if len(logs) != 1:
        raise _error(root, run_id, "sst_log", f"expected one matching log, got {len(logs)}")
    simulated_time_us = _check_log(root, run_id, logs[0])
    output = root / "outputs" / f"{run_id}.bin"
    try:
        output_size = output.stat().st_size
    except OSError as exc:
        raise _error(root, run_id, "output_size", str(exc)) from exc
    _require_equal(root, run_id, "output_size", output_size, spec.rows * spec.dim * 4)
    try:
        output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    except OSError as exc:
        raise _error(root, run_id, "output_sha256", str(exc)) from exc

    stdout_dir = root / "stdout" / "overlap0" / run_id
    pass_pattern = re.compile(
        rf"mode=sfu-standalone-job-softmax.*rows={spec.rows} dim={spec.dim}.*"
        rf"worker_cores={spec.worker_cores}.*staging_rows=4 job_rows=4 "
        rf"band_cores={spec.band_cores}.*distributed_columns=1 PASS"
    )
    pass_count = 0
    try:
        stdout_files = list(stdout_dir.glob("stdout-*"))
        for path in stdout_files:
            if pass_pattern.search(path.read_text(encoding="utf-8")):
                pass_count += 1
    except (OSError, UnicodeError) as exc:
        raise _error(root, run_id, "physical_pass_cores", str(exc)) from exc
    _require_equal(root, run_id, "physical_pass_cores", pass_count, spec.band_cores)

    stats_dir = root / "stats" / "overlap0" / run_id
    stats = _read_csv(stats_dir / "stats_selfcom.txt", root, run_id, "stats_selfcom.txt", _STAT_FIELDS)
    for row_value in stats:
        for field in _STAT_FIELDS[1:]:
            _nonnegative_integer(
                root, run_id,
                f"{row_value['StatisticName']}.{field}", row_value[field],
            )
    for statistic in (
        "sfu_ops_issued", "sfu_job_softmax_max_chunks",
        "sfu_job_softmax_sum_chunks", "sfu_job_softmax_norm_chunks",
    ):
        active = sum(value > 0 for value in _stat_values(root, run_id, stats, statistic))
        _require_equal(root, run_id, f"active_{statistic}", active, spec.band_cores)
    worker_rows = spec.rows * spec.worker_cores
    for statistic in (
        "sfu_reduction_max_requests", "sfu_reduction_max_responses",
        "sfu_reduction_sum_requests", "sfu_reduction_sum_responses",
    ):
        _require_equal(root, run_id, statistic, _stat_sum(root, run_id, stats, statistic), worker_rows)
    transport_events = _stat_sum(root, run_id, stats, "sfu_reduction_transport_received")
    expected_transport = 4 * worker_rows
    _require_equal(root, run_id, "sfu_reduction_transport_received", transport_events, expected_transport)
    transport_immediate = _stat_sum(root, run_id, stats, "gmem_reduction_send_immediate")
    transport_queued = _stat_sum(root, run_id, stats, "gmem_reduction_send_queued")
    transport_rejected = _stat_sum(root, run_id, stats, "gmem_reduction_send_rejected")
    transport_received = _stat_sum(root, run_id, stats, "gmem_reduction_received")
    transport_stale = _stat_sum(root, run_id, stats, "sfu_reduction_transport_stale_dropped")
    _require_equal(root, run_id, "gmem_reduction_send_total", transport_immediate + transport_queued, expected_transport)
    _require_equal(root, run_id, "gmem_reduction_received", transport_received, expected_transport)
    _require_equal(root, run_id, "gmem_reduction_send_rejected", transport_rejected, 0)
    _require_equal(root, run_id, "sfu_reduction_transport_stale_dropped", transport_stale, 0)
    latency_sum = sum(_stat_values(root, run_id, stats, "sfu_reduction_transport_latency_cycles", "Sum.u64"))
    latency_count = sum(_stat_values(root, run_id, stats, "sfu_reduction_transport_latency_cycles", "Count.u64"))
    if latency_count <= 0:
        raise _error(root, run_id, "latency_avg_cycles", "sample count must be positive")
    latency_avg = latency_sum / latency_count
    latency_max = max(_stat_values(root, run_id, stats, "sfu_reduction_transport_latency_cycles", "Max.u64"))
    inbox_high_water = max(_stat_values(root, run_id, stats, "sfu_reduction_transport_inbox_high_water", "Sum.u64"))

    dma_path = stats_dir / "dma_summary.csv"
    dma_values = {name: _metric(root, run_id, dma_path, name, "sum") for name in (
        "read_issue_count", "write_issue_count", "completion", "write_completion",
        "read_bytes_total", "write_bytes_total", "timeout_retry",
        "timeout_exhausted", "write_timeout_retry",
    )}
    for name in ("read_issue_count", "write_issue_count", "completion", "write_completion"):
        _require_equal(root, run_id, name, dma_values[name], worker_rows)
    expected_bytes = spec.rows * spec.dim * 4
    for name in ("read_bytes_total", "write_bytes_total"):
        _require_equal(root, run_id, name, dma_values[name], expected_bytes)
    for name in ("timeout_retry", "timeout_exhausted", "write_timeout_retry"):
        _require_equal(root, run_id, name, dma_values[name], 0)

    noc_path = stats_dir / "noc_summary.csv"
    noc_values = {
        name: _metric(root, run_id, noc_path, name, "value")
        for name in ("total_send_packets", "total_send_bits", "total_xbar_stalls")
    }
    summaries = _read_csv(
        root / "stats" / "run_summary.csv", root, run_id, "run_summary.csv",
        ("run_id", "noc_link_bw", "noc_xbar_bw", "noc_flit_size",
         "dirctrl_highlink_bw", "mem_node_size_bytes", "wall_time_sec"),
    )
    summary_matches = [summary for summary in summaries if summary["run_id"] == run_id]
    if len(summary_matches) != 1:
        raise _error(root, run_id, "run_summary.csv", f"expected one matching row, got {len(summary_matches)}")
    summary = summary_matches[0]
    for field, expected in (
        ("noc_link_bw", CANONICAL_NETWORK["GOLEM_NOC_LINK_BW"]),
        ("noc_xbar_bw", CANONICAL_NETWORK["GOLEM_NOC_XBAR_BW"]),
        ("noc_flit_size", CANONICAL_NETWORK["GOLEM_NOC_FLIT_SIZE"]),
        ("dirctrl_highlink_bw", CANONICAL_NETWORK["GOLEM_DIRCTRL_HIGHLINK_BW"]),
    ):
        _require_equal(root, run_id, field, summary[field], expected)
    _require_equal(root, run_id, "mem_node_size_bytes", _integer(root, run_id, "mem_node_size_bytes", summary["mem_node_size_bytes"]), spec.mem_node_size)
    try:
        wall_time_sec = float(summary["wall_time_sec"])
    except ValueError as exc:
        raise _error(root, run_id, "wall_time_sec", f"invalid {summary['wall_time_sec']!r}") from exc
    if not math.isfinite(wall_time_sec):
        raise _error(root, run_id, "wall_time_sec", f"non-finite {summary['wall_time_sec']!r}")
    if wall_time_sec < 0:
        raise _error(root, run_id, "wall_time_sec", f"negative {wall_time_sec}")
    golden_checked, golden_mismatches = _golden(root, run_id, spec, output, pathlib.Path(verifier))
    return PointRecord(
        spec=spec, run_id=run_id, chunk_elems=256,
        cooperative_groups=1, transport=TRANSPORT,
        reduction_vn=REDUCTION_VN, num_vns=NUM_VNS,
        dma_response_vn=DMA_RESPONSE_VN,
        noc_link_bw=CANONICAL_NETWORK["GOLEM_NOC_LINK_BW"],
        noc_xbar_bw=CANONICAL_NETWORK["GOLEM_NOC_XBAR_BW"],
        dirctrl_highlink_bw=CANONICAL_NETWORK["GOLEM_DIRCTRL_HIGHLINK_BW"],
        noc_input_buffer=CANONICAL_NETWORK["GOLEM_NOC_INPUT_BUF_SIZE"],
        noc_output_buffer=CANONICAL_NETWORK["GOLEM_NOC_OUTPUT_BUF_SIZE"],
        gm_buffer=CANONICAL_NETWORK["GOLEM_GM_BUFFER_LENGTH"],
        flit_size=CANONICAL_NETWORK["GOLEM_NOC_FLIT_SIZE"], retry_ticks=1024,
        max_retries=8, status="PASS", exit_code=0, artifact_validation="PASS",
        golden_checked=golden_checked, golden_mismatches=golden_mismatches,
        transport_events=transport_events, transport_immediate=transport_immediate,
        transport_queued=transport_queued, transport_rejected=transport_rejected,
        transport_stale=transport_stale, inbox_high_water=inbox_high_water,
        latency_avg_cycles=latency_avg, latency_max_cycles=latency_max,
        total_send_packets=noc_values["total_send_packets"],
        total_send_bits=noc_values["total_send_bits"],
        total_xbar_stalls=noc_values["total_xbar_stalls"],
        simulated_time_us=simulated_time_us, wall_time_sec=wall_time_sec,
        dma_timeout_retry=dma_values["timeout_retry"],
        dma_timeout_exhausted=dma_values["timeout_exhausted"],
        dma_write_timeout_retry=dma_values["write_timeout_retry"],
        output_sha256=output_sha256, child_root=str(root),
    )


def _record_to_row(record: PointRecord) -> dict[str, str]:
    values = {
        "run_id": record.run_id,
        "stage": record.spec.stage,
        "rows": record.spec.rows,
        "dim": record.spec.dim,
        "chunk_elems": record.chunk_elems,
        "worker_cores": record.spec.worker_cores,
        "band_cores": record.spec.band_cores,
        "transport": record.transport,
        "reduction_vn": record.reduction_vn,
        "num_vns": record.num_vns,
        "dma_response_vn": record.dma_response_vn,
        "noc_link_bw": record.noc_link_bw,
        "noc_xbar_bw": record.noc_xbar_bw,
        "dirctrl_highlink_bw": record.dirctrl_highlink_bw,
        "noc_input_buffer": record.noc_input_buffer,
        "noc_output_buffer": record.noc_output_buffer,
        "gm_buffer": record.gm_buffer,
        "flit_size": record.flit_size,
        "mem_node_size": record.spec.mem_node_size,
        "retry_ticks": record.retry_ticks,
        "max_retries": record.max_retries,
        "timeout_sec": record.spec.timeout_sec,
        "status": record.status,
        "exit_code": record.exit_code,
        "artifact_validation": record.artifact_validation,
        "golden_checked": record.golden_checked,
        "golden_mismatches": record.golden_mismatches,
        "transport_events": record.transport_events,
        "transport_immediate": record.transport_immediate,
        "transport_queued": record.transport_queued,
        "transport_rejected": record.transport_rejected,
        "transport_stale": record.transport_stale,
        "inbox_high_water": record.inbox_high_water,
        "latency_avg_cycles": record.latency_avg_cycles,
        "latency_max_cycles": record.latency_max_cycles,
        "total_send_packets": record.total_send_packets,
        "total_send_bits": record.total_send_bits,
        "total_xbar_stalls": record.total_xbar_stalls,
        "simulated_time_us": record.simulated_time_us,
        "wall_time_sec": record.wall_time_sec,
        "dma_timeout_retry": record.dma_timeout_retry,
        "dma_timeout_exhausted": record.dma_timeout_exhausted,
        "dma_write_timeout_retry": record.dma_write_timeout_retry,
        "output_sha256": record.output_sha256,
        "child_root": record.child_root,
    }
    return {
        field: "" if values[field] is None else str(values[field])
        for field in PARENT_MANIFEST_FIELDS
    }


def _manifest_int(path, row, field, optional=False):
    value = row[field]
    if optional and value == "":
        return None
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"manifest={path} field={field}: invalid integer {value!r}") from exc
    if result < 0:
        raise ValueError(f"manifest={path} field={field}: negative integer {result}")
    return result


def _manifest_float(path, row, field, optional=False):
    value = row[field]
    if optional and value == "":
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"manifest={path} field={field}: invalid float {value!r}") from exc
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"manifest={path} field={field}: non-finite float {value!r}")
    if result < 0:
        raise ValueError(f"manifest={path} field={field}: negative float {result}")
    return result


def _row_to_record(path: pathlib.Path, row: dict[str, str]) -> PointRecord:
    spec = PointSpec(
        stage=row["stage"], rows=_manifest_int(path, row, "rows"),
        dim=_manifest_int(path, row, "dim"),
        worker_cores=_manifest_int(path, row, "worker_cores"),
        band_cores=_manifest_int(path, row, "band_cores"),
        mem_node_size=_manifest_int(path, row, "mem_node_size"),
        timeout_sec=_manifest_int(path, row, "timeout_sec"),
    )
    optional_ints = {
        field: _manifest_int(path, row, field, optional=True)
        for field in (
            "golden_checked", "golden_mismatches", "transport_events",
            "transport_immediate", "transport_queued", "transport_rejected",
            "transport_stale", "inbox_high_water", "latency_max_cycles",
            "total_send_packets", "total_send_bits", "total_xbar_stalls",
            "dma_timeout_retry", "dma_timeout_exhausted", "dma_write_timeout_retry",
        )
    }
    return PointRecord(
        spec=spec, run_id=row["run_id"],
        chunk_elems=_manifest_int(path, row, "chunk_elems"),
        cooperative_groups=1, transport=row["transport"],
        reduction_vn=_manifest_int(path, row, "reduction_vn"),
        num_vns=_manifest_int(path, row, "num_vns"),
        dma_response_vn=_manifest_int(path, row, "dma_response_vn"),
        noc_link_bw=row["noc_link_bw"], noc_xbar_bw=row["noc_xbar_bw"],
        dirctrl_highlink_bw=row["dirctrl_highlink_bw"],
        noc_input_buffer=row["noc_input_buffer"],
        noc_output_buffer=row["noc_output_buffer"], gm_buffer=row["gm_buffer"],
        flit_size=row["flit_size"], retry_ticks=_manifest_int(path, row, "retry_ticks"),
        max_retries=_manifest_int(path, row, "max_retries"), status=row["status"],
        exit_code=_manifest_int(path, row, "exit_code"),
        artifact_validation=row["artifact_validation"],
        golden_checked=optional_ints["golden_checked"],
        golden_mismatches=optional_ints["golden_mismatches"],
        transport_events=optional_ints["transport_events"],
        transport_immediate=optional_ints["transport_immediate"],
        transport_queued=optional_ints["transport_queued"],
        transport_rejected=optional_ints["transport_rejected"],
        transport_stale=optional_ints["transport_stale"],
        inbox_high_water=optional_ints["inbox_high_water"],
        latency_avg_cycles=_manifest_float(path, row, "latency_avg_cycles", optional=True),
        latency_max_cycles=optional_ints["latency_max_cycles"],
        total_send_packets=optional_ints["total_send_packets"],
        total_send_bits=optional_ints["total_send_bits"],
        total_xbar_stalls=optional_ints["total_xbar_stalls"],
        simulated_time_us=_manifest_float(path, row, "simulated_time_us", optional=True),
        wall_time_sec=_manifest_float(path, row, "wall_time_sec", optional=True),
        dma_timeout_retry=optional_ints["dma_timeout_retry"],
        dma_timeout_exhausted=optional_ints["dma_timeout_exhausted"],
        dma_write_timeout_retry=optional_ints["dma_write_timeout_retry"],
        output_sha256=row["output_sha256"] or None, child_root=row["child_root"],
    )


def _identity(record: PointRecord) -> tuple[object, ...]:
    return tuple(getattr(record.spec, field) for field in _POINT_ID_FIELDS)


def load_parent_manifest(manifest: pathlib.Path) -> list[PointRecord]:
    path = pathlib.Path(manifest)
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"manifest={path} field=manifest: {exc}") from exc
    if reader.fieldnames != PARENT_MANIFEST_FIELDS:
        raise ValueError(f"manifest={path} field=header: schema mismatch")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"manifest={path} field=manifest: malformed CSV structure")
    records = [_row_to_record(path, row) for row in rows]
    seen = set()
    for record in records:
        identity = _identity(record)
        if identity in seen:
            raise ValueError(f"manifest={path} field=identity: duplicate {identity!r}")
        seen.add(identity)
    return records


def upsert_parent_manifest(manifest: pathlib.Path, record: PointRecord) -> None:
    path = pathlib.Path(manifest)
    records = load_parent_manifest(path) if path.exists() else []
    identity = _identity(record)
    replaced = False
    for index, existing in enumerate(records):
        if _identity(existing) == identity:
            records[index] = record
            replaced = True
            break
    if not replaced:
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=PARENT_MANIFEST_FIELDS)
            writer.writeheader()
            for existing in records:
                writer.writerow(_record_to_row(existing))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _matrix_identity(record: PointRecord) -> tuple[int, int, int, int]:
    return (
        record.spec.rows,
        record.spec.dim,
        record.spec.worker_cores,
        record.spec.band_cores,
    )


def _sort_records(records: list[PointRecord]) -> list[PointRecord]:
    order = {
        (point.rows, point.dim, point.worker_cores, point.band_cores): index
        for index, point in enumerate(DEFAULT_POINTS)
    }
    return sorted(records, key=lambda record: order.get(_matrix_identity(record), 999))


def _validate_profile(record: PointRecord) -> None:
    expected = {
        "transport": TRANSPORT,
        "reduction_vn": REDUCTION_VN,
        "num_vns": NUM_VNS,
        "dma_response_vn": DMA_RESPONSE_VN,
        "noc_link_bw": CANONICAL_NETWORK["GOLEM_NOC_LINK_BW"],
        "noc_xbar_bw": CANONICAL_NETWORK["GOLEM_NOC_XBAR_BW"],
        "dirctrl_highlink_bw": CANONICAL_NETWORK["GOLEM_DIRCTRL_HIGHLINK_BW"],
        "noc_input_buffer": CANONICAL_NETWORK["GOLEM_NOC_INPUT_BUF_SIZE"],
        "noc_output_buffer": CANONICAL_NETWORK["GOLEM_NOC_OUTPUT_BUF_SIZE"],
        "gm_buffer": CANONICAL_NETWORK["GOLEM_GM_BUFFER_LENGTH"],
        "flit_size": CANONICAL_NETWORK["GOLEM_NOC_FLIT_SIZE"],
        "chunk_elems": 256,
        "cooperative_groups": 1,
        "retry_ticks": 1024,
        "max_retries": 8,
    }
    for field, value in expected.items():
        if getattr(record, field) != value:
            raise ValueError(
                f"point={_matrix_identity(record)} field={field}: "
                f"expected {value!r}, got {getattr(record, field)!r}"
            )


def validate_complete_matrix(records: list[PointRecord]) -> None:
    expected = {
        (point.rows, point.dim, point.worker_cores, point.band_cores): point
        for point in DEFAULT_POINTS
    }
    seen: set[tuple[int, int, int, int]] = set()
    for record in records:
        identity = _matrix_identity(record)
        if identity not in expected:
            raise ValueError(f"point={identity}: outside the Phase 4F matrix")
        if identity in seen:
            raise ValueError(f"point={identity}: duplicate outcome")
        seen.add(identity)
        if record.spec != expected[identity]:
            raise ValueError(f"point={identity} field=spec: noncanonical point metadata")
        if not record.child_root:
            raise ValueError(f"point={identity} field=child_root: missing attempt root")
        _validate_profile(record)
        if record.status not in {"PASS", "TIMEOUT", "FAIL", "ARTIFACT_FAIL"}:
            raise ValueError(f"point={identity} field=status: unsupported {record.status!r}")
        optional = (
            "golden_checked", "golden_mismatches", "transport_events",
            "transport_immediate", "transport_queued", "transport_rejected",
            "transport_stale", "inbox_high_water", "latency_avg_cycles",
            "latency_max_cycles", "total_send_packets", "total_send_bits",
            "total_xbar_stalls", "simulated_time_us", "wall_time_sec",
            "dma_timeout_retry", "dma_timeout_exhausted",
            "dma_write_timeout_retry", "output_sha256",
        )
        if record.status != "PASS":
            if record.exit_code == 0:
                raise ValueError(f"point={identity} field=exit_code: failure cannot exit 0")
            if record.artifact_validation != record.status:
                raise ValueError(
                    f"point={identity} field=artifact_validation: status mismatch"
                )
            populated = [field for field in optional if getattr(record, field) is not None]
            if populated:
                raise ValueError(
                    f"point={identity} field=performance: unavailable outcome has {populated!r}"
                )
            continue

        expected_transport = 4 * record.spec.rows * record.spec.worker_cores
        pass_expected = {
            "exit_code": 0,
            "artifact_validation": "PASS",
            "golden_checked": record.spec.rows * record.spec.dim,
            "golden_mismatches": 0,
            "transport_events": expected_transport,
            "transport_rejected": 0,
            "transport_stale": 0,
            "dma_timeout_retry": 0,
            "dma_timeout_exhausted": 0,
            "dma_write_timeout_retry": 0,
        }
        for field, value in pass_expected.items():
            if getattr(record, field) != value:
                raise ValueError(
                    f"point={identity} field={field}: expected {value!r}, "
                    f"got {getattr(record, field)!r}"
                )
        if record.transport_immediate is None or record.transport_queued is None:
            raise ValueError(f"point={identity} field=transport_lifecycle: missing counters")
        if record.transport_immediate + record.transport_queued != expected_transport:
            raise ValueError(f"point={identity} field=transport_lifecycle: total mismatch")
        required_metrics = (
            "inbox_high_water", "latency_avg_cycles", "latency_max_cycles",
            "total_send_packets", "total_send_bits", "total_xbar_stalls",
            "simulated_time_us", "wall_time_sec",
        )
        if any(getattr(record, field) is None for field in required_metrics):
            raise ValueError(f"point={identity} field=measurement: missing PASS metric")
        if record.simulated_time_us <= 0 or record.wall_time_sec <= 0:
            raise ValueError(f"point={identity} field=measurement: time must be positive")
        if not re.fullmatch(r"[0-9a-f]{64}", record.output_sha256 or ""):
            raise ValueError(f"point={identity} field=output_sha256: invalid hash")
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"matrix is incomplete; missing outcomes={sorted(missing)!r}")


def derive_metrics(records: list[PointRecord]) -> dict[str, float]:
    validate_complete_matrix(records)
    metrics: dict[str, float] = {}
    passed = { _matrix_identity(record): record for record in records if record.status == "PASS" }
    for identity, record in passed.items():
        prefix = f"r{identity[0]}_d{identity[1]}_w{identity[2]}"
        runtime = float(record.simulated_time_us)
        metrics[f"{prefix}_time_per_row_us"] = runtime / record.spec.rows
        metrics[f"{prefix}_time_per_element_us"] = runtime / (
            record.spec.rows * record.spec.dim
        )

    worker_records = {
        workers: passed.get((16, 4096, workers, workers))
        for workers in (4, 8, 16)
    }
    if all(worker_records.values()):
        baseline = float(worker_records[4].simulated_time_us)
        previous_time = None
        for workers in (4, 8, 16):
            runtime = float(worker_records[workers].simulated_time_us)
            speedup = baseline / runtime
            prefix = f"r16_d4096_w{workers}"
            metrics[f"{prefix}_speedup"] = speedup
            metrics[f"{prefix}_efficiency"] = speedup / (workers / 4)
            if previous_time is not None:
                metrics[f"{prefix}_marginal_gain"] = (
                    previous_time - runtime
                ) / previous_time
            previous_time = runtime
    return metrics


def _atomic_csv(path: pathlib.Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            pathlib.Path(temp_name).unlink(missing_ok=True)


def write_source_csv(records: list[PointRecord], path: pathlib.Path) -> None:
    validate_complete_matrix(records)
    rows = []
    for record in _sort_records(records):
        row = _record_to_row(record)
        if record.status == "PASS":
            runtime = float(record.simulated_time_us)
            row["time_per_row_us"] = format(runtime / record.spec.rows, ".17g")
            row["time_per_element_us"] = format(
                runtime / (record.spec.rows * record.spec.dim), ".17g"
            )
        else:
            row["time_per_row_us"] = ""
            row["time_per_element_us"] = ""
        rows.append(row)
    _atomic_csv(pathlib.Path(path), SOURCE_DATA_FIELDS, rows)


def load_source_csv(path: pathlib.Path) -> list[PointRecord]:
    path = pathlib.Path(path)
    rows = _read_csv(
        path, path.parent, "", "source_data", SOURCE_DATA_FIELDS, exact_header=True
    )
    records = [_row_to_record(path, row) for row in rows]
    validate_complete_matrix(records)
    for record, row in zip(records, rows):
        if record.status == "PASS":
            expected_row = float(record.simulated_time_us) / record.spec.rows
            expected_element = float(record.simulated_time_us) / (
                record.spec.rows * record.spec.dim
            )
            for field, expected in (
                ("time_per_row_us", expected_row),
                ("time_per_element_us", expected_element),
            ):
                try:
                    actual = float(row[field])
                except ValueError as exc:
                    raise ValueError(f"source={path} field={field}: invalid value") from exc
                if actual != expected:
                    raise ValueError(f"source={path} field={field}: derived value mismatch")
        elif row["time_per_row_us"] or row["time_per_element_us"]:
            raise ValueError(f"source={path} field=derived: unavailable outcome has metrics")
    return records


def _plot_series(axis, x, y, *, marker="o", color="#176B87", label=None):
    axis.plot(x, y, marker=marker, linewidth=2.0, markersize=5, color=color, label=label)
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.7)


def render_figure(records: list[PointRecord], output_prefix: pathlib.Path) -> None:
    validate_complete_matrix(records)
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "svg.fonttype": "none",
        "svg.hashsalt": "sfu-phase4f-large-scale",
        "pdf.fonttype": 42,
        "pdf.compression": 0,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    passed = [record for record in records if record.status == "PASS"]
    metrics = derive_metrics(records)
    fig, axes = plt.subplots(2, 2, figsize=(13.333, 7.5), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
    fig.suptitle(
        "SFU Softmax Large-Scale Explicit-NoC Results",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.952,
        "Fixed GEMM network profile: 1200 GB/s link/xbar/highlink, 512 KB I/O buffers, 128 B flit",
        ha="center", color="#475467", fontsize=9,
    )

    dimension = sorted(
        (record for record in passed if record.spec.rows == 16 and record.spec.worker_cores == 16),
        key=lambda record: record.spec.dim,
    )
    ax = axes[0, 0]
    ax.set_title("Dimension scaling")
    _plot_series(ax, [r.spec.dim for r in dimension], [r.simulated_time_us for r in dimension],
                 color="#176B87", label="Runtime")
    for index, record in enumerate(dimension):
        time_per_element = metrics[
            f"r16_d{record.spec.dim}_w16_time_per_element_us"
        ]
        high_point = record.simulated_time_us == max(
            item.simulated_time_us for item in dimension
        )
        ax.annotate(
            f"{time_per_element:.3g} us/element",
            (record.spec.dim, record.simulated_time_us),
            xytext=(0, -11 if high_point else 7), textcoords="offset points",
            ha="left" if index == 0 else "right" if index == len(dimension) - 1 else "center",
            fontsize=6,
        )
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Simulated time (us)", color="#176B87")
    latency_ax = ax.twinx()
    latency_ax.plot([r.spec.dim for r in dimension], [r.latency_avg_cycles for r in dimension],
                    marker="s", linewidth=1.8, color="#C2410C", label="Avg. reduction latency")
    latency_ax.set_ylabel("Reduction latency (cycles)", color="#C2410C")

    ax = axes[0, 1]
    ax.set_title("Worker scaling (4-worker baseline)")
    workers = [4, 8, 16]
    available = [w for w in workers if f"r16_d4096_w{w}_speedup" in metrics]
    _plot_series(ax, available, [metrics[f"r16_d4096_w{w}_speedup"] for w in available],
                 color="#176B87", label="Speedup")
    _plot_series(ax, available, [metrics[f"r16_d4096_w{w}_efficiency"] for w in available],
                 marker="s", color="#2E7D32", label="Parallel efficiency")
    marginal_x = [w for w in (8, 16) if f"r16_d4096_w{w}_marginal_gain" in metrics]
    if marginal_x:
        ax.scatter(marginal_x, [metrics[f"r16_d4096_w{w}_marginal_gain"] for w in marginal_x],
                   marker="D", color="#C2410C", label="Marginal gain")
    ax.set_xticks(workers)
    ax.set_xlabel("Workers / bands")
    ax.set_ylabel("Relative metric")
    ax.legend(frameon=False, ncols=3, fontsize=7, loc="upper left")

    ax = axes[1, 0]
    ax.set_title("Row scaling")
    row_data = sorted(
        (record for record in passed if record.spec.dim == 4096 and record.spec.worker_cores == 16),
        key=lambda record: record.spec.rows,
    )
    _plot_series(ax, [r.spec.rows for r in row_data], [r.simulated_time_us for r in row_data],
                 color="#176B87", label="Total time")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Simulated time (us)", color="#176B87")
    row_ax = ax.twinx()
    row_ax.plot(
        [r.spec.rows for r in row_data],
        [metrics[f"r{r.spec.rows}_d4096_w16_time_per_row_us"] for r in row_data],
        marker="s", linewidth=1.8, color="#C2410C",
    )
    row_ax.set_ylabel("Time per row (us)", color="#C2410C")
    for index, record in enumerate(row_data):
        per_element = metrics[
            f"r{record.spec.rows}_d4096_w16_time_per_element_us"
        ]
        high_point = record.simulated_time_us == max(
            item.simulated_time_us for item in row_data
        )
        ax.annotate(
            f"{per_element:.3g} us/element",
            (record.spec.rows, record.simulated_time_us),
            xytext=(0, -11 if high_point else 7), textcoords="offset points",
            ha="left" if index == 0 else "right" if index == len(row_data) - 1 else "center",
            fontsize=6,
        )

    ax = axes[1, 1]
    ax.set_title("NoC and correctness")
    labels = [f"{r.spec.rows}x{r.spec.dim}\nw{r.spec.worker_cores}" for r in passed]
    x = list(range(len(passed)))
    ax.bar([value - 0.22 for value in x], [r.total_send_packets / 1000 for r in passed],
           width=0.22, color="#176B87", label="Packets (k)")
    ax.bar(x, [r.total_send_bits / 1_000_000 for r in passed], width=0.22,
           color="#7A9E45", label="Bits (M)")
    ax.bar([value + 0.22 for value in x], [r.total_xbar_stalls for r in passed],
           width=0.22, color="#C2410C", label="Xbar stalls")
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=6)
    ax.set_ylabel("Observed NoC metrics")
    max_metric = max(
        [r.total_send_packets / 1000 for r in passed]
        + [r.total_send_bits / 1_000_000 for r in passed]
        + [r.total_xbar_stalls for r in passed]
    )
    ax.set_ylim(0, max_metric * 1.3)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False, ncols=3, fontsize=7, loc="upper left")
    pass_count = len(passed)
    ax.text(
        0.99, 0.96,
        f"{pass_count}/8 PASS measurements\nGolden, transport, DMA: gated\nRejected/stale/retry: 0",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#D0D5DD", "pad": 4},
    )

    output_prefix = pathlib.Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_prefix.parent) as temporary:
        temp_prefix = pathlib.Path(temporary) / output_prefix.name
        metadata = {"Date": None, "Creator": "Phase 4F report"}
        fig.savefig(temp_prefix.with_suffix(".svg"), metadata=metadata)
        fig.savefig(temp_prefix.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(temp_prefix.with_suffix(".png"), dpi=300, metadata={"Software": "Phase 4F report"})
        plt.close(fig)
        for suffix in (".svg", ".pdf", ".png"):
            os.replace(temp_prefix.with_suffix(suffix), output_prefix.with_suffix(suffix))


def _atomic_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            pathlib.Path(temp_name).unlink(missing_ok=True)


def write_qa(records: list[PointRecord], output_path: pathlib.Path) -> None:
    validate_complete_matrix(records)
    metrics = derive_metrics(records)
    lines = [
        "# SFU Phase 4F Large-Scale Figure QA",
        "",
        "Fixed network: link/xbar/highlink 1200GB/s; input/output buffers 512KB; "
        "flit 128B; GlobalMemory buffer 1024KB; explicit NoC, VN0 reduction.",
        "",
        "## Measurements",
        "",
        "| Identity | Status | Golden | Transport | DMA | Child root | Output hash |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    unavailable = []
    for record in _sort_records(records):
        identity = ":".join(str(value) for value in _matrix_identity(record))
        if record.status == "PASS":
            golden = f"{record.golden_checked} checked / {record.golden_mismatches} mismatch"
            transport = f"{record.transport_events} events; rejected/stale=0"
            dma = "retry/exhausted/write-retry=0"
            output_hash = record.output_sha256
        else:
            golden = transport = dma = "unavailable"
            output_hash = "unavailable"
            unavailable.append((identity, record.status, record.exit_code, record.child_root))
        lines.append(
            f"| {identity} | {record.status} | {golden} | {transport} | {dma} | "
            f"`{record.child_root}` | `{output_hash}` |"
        )
    lines.extend(["", "## Derived metrics", ""])
    if metrics:
        for name in sorted(metrics):
            lines.append(f"- `{name}` = {metrics[name]:.9g}")
    else:
        lines.append("- No derived performance metrics are available.")
    lines.extend(["", "## Unavailable outcomes", ""])
    if unavailable:
        for identity, status, exit_code, child_root in unavailable:
            lines.append(
                f"- `{identity}`: {status}, exit code {exit_code}; stopped at `{child_root}`. "
                "Performance fields are unavailable and excluded from trends."
            )
    else:
        lines.append("- None; all eight outcomes are validated PASS measurements.")
    _atomic_text(pathlib.Path(output_path), "\n".join(lines) + "\n")


def _status_row(record: PointRecord) -> dict[str, str]:
    return {
        "stage": record.spec.stage,
        "rows": str(record.spec.rows),
        "dim": str(record.spec.dim),
        "worker_cores": str(record.spec.worker_cores),
        "band_cores": str(record.spec.band_cores),
        "status": record.status,
        "exit_code": str(record.exit_code),
        "transport": record.transport,
        "reduction_vn": str(record.reduction_vn),
        "num_vns": str(record.num_vns),
        "dma_response_vn": str(record.dma_response_vn),
        "noc_link_bw": record.noc_link_bw,
        "noc_xbar_bw": record.noc_xbar_bw,
        "dirctrl_highlink_bw": record.dirctrl_highlink_bw,
        "noc_input_buffer": record.noc_input_buffer,
        "noc_output_buffer": record.noc_output_buffer,
        "gm_buffer": record.gm_buffer,
        "flit_size": record.flit_size,
        "inter_router_no_cut": "0",
        "local_no_cut": "0",
        "mem_node_size": str(record.spec.mem_node_size),
        "retry_ticks": str(record.retry_ticks),
        "max_retries": str(record.max_retries),
        "timeout_sec": str(record.spec.timeout_sec),
        "child_root": record.child_root,
    }


def _marker_name(spec: PointSpec) -> str:
    return (
        f"stage_{spec.stage}_r{spec.rows}_d{spec.dim}_"
        f"w{spec.worker_cores}_b{spec.band_cores}.marker"
    )


def _read_marker(path: pathlib.Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"marker={path}: {exc}") from exc
    values: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            raise ValueError(f"marker={path}: malformed line")
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"marker={path}: duplicate key {key!r}")
        values[key] = value
    required = {
        "schema", "state", "signature_sha256", "signature",
        "child_runner_sha256", "pipeline_args_sha256", "child_root",
        "output_sha256",
    }
    if set(values) != required or values["schema"] != "phase4f-parent-v1":
        raise ValueError(f"marker={path}: schema mismatch")
    return values


def _validate_marker_signature(
    path: pathlib.Path, marker: dict[str, str], spec: PointSpec
) -> None:
    signature = marker["signature"]
    if hashlib.sha256(signature.encode("utf-8")).hexdigest() != marker["signature_sha256"]:
        raise ValueError(f"marker={path}: signature hash mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", marker["child_runner_sha256"]):
        raise ValueError(f"marker={path}: invalid child runner hash")
    if not re.fullmatch(r"[0-9a-f]{64}", marker["pipeline_args_sha256"]):
        raise ValueError(f"marker={path}: invalid pipeline hash")
    fields: dict[str, str] = {}
    for token in signature.split(";"):
        if "=" not in token:
            raise ValueError(f"marker={path}: malformed signature")
        key, value = token.split("=", 1)
        if key in fields:
            raise ValueError(f"marker={path}: duplicate signature key {key!r}")
        fields[key] = value
    expected = {
        "schema": "phase4f-parent-v1",
        "stage": spec.stage,
        "rows": str(spec.rows),
        "dim": str(spec.dim),
        "workers": str(spec.worker_cores),
        "bands": str(spec.band_cores),
        "cooperative_groups": "1",
        "transport": TRANSPORT,
        "request_vn": "0",
        "ordinary_response_vn": "1",
        "reduction_vn": "0",
        "num_vns": "3",
        "dma_response_vn": "0",
        "noc_link_bw": "1200GB/s",
        "noc_xbar_bw": "1200GB/s",
        "dirctrl_highlink_bw": "1200GB/s",
        "noc_input_buffer": "512KB",
        "noc_output_buffer": "512KB",
        "gm_buffer": "1024KB",
        "flit_size": "128B",
        "inter_router_no_cut": "0",
        "local_no_cut": "0",
        "mem_node_size": str(spec.mem_node_size),
        "timeout_sec": str(spec.timeout_sec),
        "chunk": "256",
        "staging_rows": "4",
        "job_rows": "4",
        "retry_ticks": "1024",
        "max_retries": "8",
        "child_runner_sha256": marker["child_runner_sha256"],
        "pipeline_args_sha256": marker["pipeline_args_sha256"],
    }
    if fields != expected:
        raise ValueError(f"marker={path}: noncanonical signature")


def _failure_record(row: dict[str, str], spec: PointSpec) -> PointRecord:
    status = row["status"]
    return PointRecord(
        spec=spec, run_id=_canonical_run_id(spec), chunk_elems=256,
        cooperative_groups=1, transport=TRANSPORT, reduction_vn=REDUCTION_VN,
        num_vns=NUM_VNS, dma_response_vn=DMA_RESPONSE_VN,
        noc_link_bw=CANONICAL_NETWORK["GOLEM_NOC_LINK_BW"],
        noc_xbar_bw=CANONICAL_NETWORK["GOLEM_NOC_XBAR_BW"],
        dirctrl_highlink_bw=CANONICAL_NETWORK["GOLEM_DIRCTRL_HIGHLINK_BW"],
        noc_input_buffer=CANONICAL_NETWORK["GOLEM_NOC_INPUT_BUF_SIZE"],
        noc_output_buffer=CANONICAL_NETWORK["GOLEM_NOC_OUTPUT_BUF_SIZE"],
        gm_buffer=CANONICAL_NETWORK["GOLEM_GM_BUFFER_LENGTH"],
        flit_size=CANONICAL_NETWORK["GOLEM_NOC_FLIT_SIZE"], retry_ticks=1024,
        max_retries=8, status=status, exit_code=int(row["exit_code"]),
        artifact_validation=status, golden_checked=None, golden_mismatches=None,
        transport_events=None, transport_immediate=None, transport_queued=None,
        transport_rejected=None, transport_stale=None, inbox_high_water=None,
        latency_avg_cycles=None, latency_max_cycles=None, total_send_packets=None,
        total_send_bits=None, total_xbar_stalls=None, simulated_time_us=None,
        wall_time_sec=None, dma_timeout_retry=None, dma_timeout_exhausted=None,
        dma_write_timeout_retry=None, output_sha256=None, child_root=row["child_root"],
    )


def _load_report_records(root: pathlib.Path, verifier: pathlib.Path) -> list[PointRecord]:
    status_path = root / "point_status.csv"
    status_rows = _read_csv(
        status_path, root, "", "point_status.csv", POINT_STATUS_FIELDS,
        exact_header=True,
    )
    if len(status_rows) != len(DEFAULT_POINTS):
        raise ValueError(f"status={status_path}: expected 8 outcomes, got {len(status_rows)}")
    parent_path = root / "large_scale_manifest.csv"
    parent_records = load_parent_manifest(parent_path) if parent_path.exists() else []
    parent_by_identity = {}
    for record in parent_records:
        identity = _matrix_identity(record)
        if identity in parent_by_identity:
            raise ValueError(f"manifest={parent_path}: duplicate matrix identity {identity!r}")
        if identity not in _POINTS_BY_IDENTITY or record.spec != _POINTS_BY_IDENTITY[identity]:
            raise ValueError(f"manifest={parent_path}: noncanonical identity {identity!r}")
        _validate_profile(record)
        if record.status != "PASS":
            raise ValueError(f"manifest={parent_path}: non-PASS canonical record")
        parent_by_identity[identity] = record
    records = []
    seen = set()
    for row in status_rows:
        try:
            spec = resolve_point(
                int(row["rows"]), int(row["dim"]),
                int(row["worker_cores"]), int(row["band_cores"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"status={status_path}: invalid point identity") from exc
        identity = (spec.rows, spec.dim, spec.worker_cores, spec.band_cores)
        if identity in seen:
            raise ValueError(f"status={status_path}: duplicate identity {identity!r}")
        seen.add(identity)
        if row["stage"] != spec.stage:
            raise ValueError(f"status={status_path} point={identity}: stage mismatch")
        expected_status = _status_row(_failure_record(row, spec))
        for field in POINT_STATUS_FIELDS:
            if field in {"status", "exit_code", "child_root"}:
                continue
            if row[field] != expected_status[field]:
                raise ValueError(
                    f"status={status_path} point={identity} field={field}: profile drift"
                )
        marker_path = root / "completed" / _marker_name(spec)
        marker = _read_marker(marker_path)
        _validate_marker_signature(marker_path, marker, spec)
        if marker["state"] != row["status"] or marker["child_root"] != row["child_root"]:
            raise ValueError(f"status={status_path} point={identity}: marker mismatch")
        if row["status"] == "PASS":
            if identity not in parent_by_identity:
                raise ValueError(f"manifest={parent_path} point={identity}: PASS is missing")
            parsed = parse_child_point(pathlib.Path(row["child_root"]), spec, verifier)
            if parsed != parent_by_identity[identity]:
                raise ValueError(f"manifest={parent_path} point={identity}: child evidence drift")
            if marker["output_sha256"] != parsed.output_sha256:
                raise ValueError(f"marker point={identity}: output hash mismatch")
            records.append(parsed)
        elif row["status"] in {"TIMEOUT", "FAIL", "ARTIFACT_FAIL"}:
            if identity in parent_by_identity:
                raise ValueError(f"manifest={parent_path} point={identity}: failed outcome has PASS record")
            if marker["output_sha256"]:
                raise ValueError(f"marker point={identity}: failed outcome has output hash")
            records.append(_failure_record(row, spec))
        else:
            raise ValueError(f"status={status_path} point={identity}: unpublished status")
    validate_complete_matrix(records)
    return _sort_records(records)


def _report(root: pathlib.Path, output_dir: pathlib.Path, verifier: pathlib.Path) -> None:
    records = _load_report_records(root, verifier)
    output_dir = pathlib.Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = pathlib.Path(temporary)
        source = staging / "sfu_phase4f_large_scale_source_data.csv"
        prefix = staging / "sfu_phase4f_large_scale"
        qa = staging / "sfu_phase4f_large_scale_qa.md"
        write_source_csv(records, source)
        reconstructed = load_source_csv(source)
        render_figure(reconstructed, prefix)
        write_qa(reconstructed, qa)
        expected = [
            source, prefix.with_suffix(".svg"), prefix.with_suffix(".pdf"),
            prefix.with_suffix(".png"), qa,
        ]
        if not all(path.is_file() and path.stat().st_size > 0 for path in expected):
            raise ValueError("report export set is incomplete")
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in expected:
            os.replace(path, output_dir / path.name)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "collect":
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=pathlib.Path, required=True)
        parser.add_argument("--output-dir", type=pathlib.Path, required=True)
        parser.add_argument(
            "--verifier", type=pathlib.Path,
            default=pathlib.Path(__file__).with_name("verify_softmax_sfu_against_golden.py"),
        )
        args = parser.parse_args(argv)
        _report(args.root, args.output_dir, args.verifier)
        return 0

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--child-root", type=pathlib.Path, required=True)
    collect.add_argument("--stage", required=True)
    collect.add_argument("--rows", type=int, required=True)
    collect.add_argument("--dim", type=int, required=True)
    collect.add_argument("--workers", type=int, required=True)
    collect.add_argument("--bands", type=int, required=True)
    collect.add_argument("--parent-manifest", type=pathlib.Path, required=True)
    collect.add_argument("--verifier", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    spec = resolve_point(args.rows, args.dim, args.workers, args.bands)
    if args.stage != spec.stage:
        parser.error(f"stage mismatch: expected {spec.stage}, got {args.stage}")
    record = parse_child_point(args.child_root, spec, args.verifier)
    upsert_parent_manifest(args.parent_manifest, record)
    print(f"run_id={record.run_id} output_sha256={record.output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
