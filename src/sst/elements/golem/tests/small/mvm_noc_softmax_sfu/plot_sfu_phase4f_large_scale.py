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
        f"sfu_unified_job_r{spec.rows}_d{spec.dim}_c256_w{spec.worker_cores}"
        f"_b{spec.band_cores}_g1_vn0"
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
    missing = [evidence for evidence in required if evidence not in text]
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


def main(argv=None) -> int:
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
