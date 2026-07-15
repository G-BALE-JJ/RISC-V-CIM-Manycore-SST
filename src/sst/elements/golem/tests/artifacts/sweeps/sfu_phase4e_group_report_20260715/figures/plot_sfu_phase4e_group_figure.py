#!/usr/bin/env python3

import csv
import dataclasses
import pathlib
import re
import subprocess
import sys


ROWS = 16
DIM = 512
EXPECTED_WORKERS = (4, 8, 16)
ROOT_SPECS = (
    ("sfu_phase4e_modeled_control_matrix_20260715", "modeled-NoC", None),
    ("sfu_phase4e_explicit_vn0_matrix_20260715", "explicit-NoC", 0),
    ("sfu_phase4e_explicit_vn1_matrix_20260715", "explicit-NoC", 1),
    ("sfu_phase4e_explicit_vn2_matrix_20260715", "explicit-NoC", 2),
)


@dataclasses.dataclass(frozen=True)
class PointRecord:
    transport: str
    reduction_vn: int | None
    workers: int
    band_cores: int
    simulated_time_us: float
    transport_events: int
    latency_average_cycles: float
    latency_max_cycles: int
    inbox_high_water: int
    queued: int
    rejected: int
    stale: int
    total_send_packets: int
    total_send_bits: int
    total_xbar_stalls: int
    dma_timeout_retry: int
    dma_timeout_exhausted: int
    dma_write_timeout_retry: int
    golden_checked: int
    golden_mismatches: int
    source_root: str
    run_id: str


def _error(root: pathlib.Path, run_id: str, field: str, detail: str) -> ValueError:
    return ValueError(f"root={root} run_id={run_id or '<manifest>'} field={field}: {detail}")


def _required_int(root: pathlib.Path, row: dict[str, str], field: str) -> int:
    run_id = row.get("run_id", "")
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise _error(root, run_id, field, f"expected integer, got {row.get(field)!r}") from exc


def select_manifest_rows(
    root: pathlib.Path, expected_transport: str, expected_vn: int | None
) -> list[dict[str, str]]:
    expected_names = {
        (transport, vn): name for name, transport, vn in ROOT_SPECS
    }
    expected_name = expected_names.get((expected_transport, expected_vn))
    if expected_name is None:
        raise _error(root, "", "transport", f"unsupported transport/VN {expected_transport}/{expected_vn}")
    if root.name != expected_name:
        raise _error(
            root,
            "",
            "root",
            f"VN/root mismatch: expected {expected_name} for {expected_transport} VN={expected_vn}",
        )

    manifest = root / "sweep_manifest.csv"
    try:
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _error(root, "", "sweep_manifest.csv", str(exc)) from exc
    if not rows:
        raise _error(root, "", "sweep_manifest.csv", "manifest is empty")
    if reader.fieldnames is None or any(
        None in row or any(value is None for value in row.values()) for row in rows
    ):
        raise _error(root, "", "sweep_manifest.csv", "malformed CSV structure")

    observed_workers: set[int] = set()
    by_run_id: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id:
            raise _error(root, "", "run_id", "missing value")
        workers = _required_int(root, row, "worker_cores")
        observed_workers.add(workers)
        if workers not in EXPECTED_WORKERS:
            raise _error(root, run_id, "worker_cores", f"unexpected worker count {workers}")
        if _required_int(root, row, "rows") != ROWS:
            raise _error(root, run_id, "rows", f"expected {ROWS}")
        if _required_int(root, row, "dim") != DIM:
            raise _error(root, run_id, "dim", f"expected {DIM}")
        if _required_int(root, row, "band_cores") != workers:
            raise _error(root, run_id, "band_cores", f"expected {workers}")
        if expected_transport == "explicit-NoC":
            row_vn = _required_int(root, row, "reduction_vn")
            if row_vn != expected_vn or not run_id.endswith(f"_vn{expected_vn}"):
                raise _error(
                    root,
                    run_id,
                    "reduction_vn",
                    f"VN/root mismatch: expected VN {expected_vn}, got {row_vn}",
                )
        by_run_id.setdefault(run_id, []).append(row)

    if observed_workers != set(EXPECTED_WORKERS):
        raise _error(
            root,
            "",
            "worker_cores",
            f"expected worker points {list(EXPECTED_WORKERS)}, got {sorted(observed_workers)}",
        )

    selected: list[dict[str, str]] = []
    for run_id, run_rows in by_run_id.items():
        canonical = []
        for row in run_rows:
            status = row.get("status")
            artifact = row.get("artifact_validation")
            if status != "PASS":
                raise _error(root, run_id, "status", f"expected PASS, got {status!r}")
            if artifact == "PASS":
                canonical.append(row)
            elif artifact != "CACHED":
                raise _error(
                    root,
                    run_id,
                    "artifact_validation",
                    f"expected PASS or ignorable CACHED, got {artifact!r}",
                )
        if len(canonical) != 1:
            detail = "duplicate PASS/PASS rows" if len(canonical) > 1 else "missing PASS/PASS row"
            raise _error(root, run_id, "artifact_validation", detail)
        row = canonical[0]
        if row.get("exit_code") != "0":
            raise _error(root, run_id, "exit_code", f"expected 0, got {row.get('exit_code')!r}")
        selected.append(row)

    selected.sort(key=lambda row: _required_int(root, row, "worker_cores"))
    if len(selected) != len(EXPECTED_WORKERS):
        raise _error(root, "", "run_id", f"expected 3 canonical rows, got {len(selected)}")
    return selected


def _read_stats(
    root: pathlib.Path,
    run_id: str,
    path: pathlib.Path,
    required_fields: tuple[str, ...],
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _error(root, run_id, path.name, str(exc)) from exc
    if not rows:
        raise _error(root, run_id, path.name, "CSV evidence is empty")
    fieldnames = reader.fieldnames or []
    missing = [field for field in required_fields if field not in fieldnames]
    if missing:
        raise _error(root, run_id, path.name, f"missing columns {missing}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise _error(root, run_id, path.name, "malformed CSV structure")
    return rows


def _stat_values(
    root: pathlib.Path, run_id: str, rows: list[dict[str, str]], statistic: str, field: str
) -> list[int]:
    matching = [row for row in rows if row.get("StatisticName") == statistic]
    if not matching:
        raise _error(root, run_id, statistic, "statistic is missing")
    values = []
    for row in matching:
        try:
            values.append(int(row[field]))
        except (KeyError, TypeError, ValueError) as exc:
            raise _error(root, run_id, f"{statistic}.{field}", f"invalid value {row.get(field)!r}") from exc
    return values


def _metric_value(
    root: pathlib.Path,
    run_id: str,
    path: pathlib.Path,
    metric: str,
    value_field: str,
) -> int:
    rows = _read_stats(root, run_id, path, ("metric", value_field))
    matches = [row for row in rows if row.get("metric") == metric]
    if len(matches) != 1:
        raise _error(root, run_id, metric, f"expected one row in {path.name}, got {len(matches)}")
    try:
        return int(matches[0][value_field])
    except (KeyError, TypeError, ValueError) as exc:
        raise _error(
            root, run_id, f"{metric}.{value_field}", f"invalid value {matches[0].get(value_field)!r}"
        ) from exc


def _simulated_time_us(root: pathlib.Path, run_id: str, log: pathlib.Path) -> float:
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise _error(root, run_id, "simulated_time", str(exc)) from exc
    match = re.search(
        r"Simulation is complete, simulated time:\s*([0-9]+(?:\.[0-9]+)?)\s*(us|ms|s)\b",
        text,
    )
    if match is None:
        raise _error(root, run_id, "simulated_time", "completion line is missing or malformed")
    multipliers = {"us": 1.0, "ms": 1_000.0, "s": 1_000_000.0}
    return float(match.group(1)) * multipliers[match.group(2)]


def _golden_evidence(
    root: pathlib.Path, run_id: str, output_file: pathlib.Path, verifier: pathlib.Path
) -> tuple[int, int]:
    required = [
        verifier,
        root / "inputs" / "a.bin",
        root / "inputs" / "b.bin",
        output_file,
        root / "inputs" / "softmax_logits_16x512.bin",
    ]
    for path in required:
        if not path.is_file():
            raise _error(root, run_id, "golden_input", f"missing {path}")
    command = [
        sys.executable,
        str(verifier),
        "--a-file",
        str(root / "inputs" / "a.bin"),
        "--b-file",
        str(root / "inputs" / "b.bin"),
        "--c-file",
        str(output_file),
        "--m",
        "16",
        "--n",
        "512",
        "--k",
        "512",
        "--block-m",
        "4",
        "--block-n",
        "64",
        "--dtype",
        "fp32",
        "--reference",
        "logits",
        "--logits-file",
        str(root / "inputs" / "softmax_logits_16x512.bin"),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise _error(root, run_id, "verifier", f"exit {result.returncode}: {result.stdout.strip()}")
    evidence = re.search(
        r"^\[VERIFY-SFU-SOFTMAX\]\s+(PASS|FAIL).*\bchecked=(\d+)\b.*\bmismatches=(\d+)\b",
        result.stdout,
        flags=re.MULTILINE,
    )
    if evidence is None:
        raise _error(root, run_id, "verifier_output", "successful PASS evidence line is missing")
    checked = int(evidence.group(2))
    mismatches = int(evidence.group(3))
    if checked != ROWS * DIM:
        raise _error(root, run_id, "checked", f"expected {ROWS * DIM}, got {checked}")
    if mismatches != 0:
        raise _error(root, run_id, "mismatches", f"expected 0, got {mismatches}")
    if evidence.group(1) != "PASS":
        raise _error(root, run_id, "verifier_output", "verifier did not report PASS")
    return checked, mismatches


def parse_point(
    root: pathlib.Path,
    manifest_row: dict[str, str],
    transport: str,
    reduction_vn: int | None,
    verifier: pathlib.Path,
) -> PointRecord:
    run_id = manifest_row.get("run_id", "")
    if not run_id:
        raise _error(root, "", "run_id", "missing value")
    workers = _required_int(root, manifest_row, "worker_cores")
    band_cores = _required_int(root, manifest_row, "band_cores")
    stats_dir = root / "stats" / "overlap0" / run_id
    stats_file = stats_dir / "stats_selfcom.txt"
    try:
        sst_log = next((root / "logs").glob(f"*{run_id}*.log"))
    except StopIteration as exc:
        raise _error(root, run_id, "sst_log", "matching log is missing") from exc
    output_file = root / "outputs" / f"{run_id}.bin"

    stats_rows = _read_stats(
        root,
        run_id,
        stats_file,
        ("StatisticName", "Sum.u64", "Count.u64", "Max.u64"),
    )
    for stats_row in stats_rows:
        for field in ("Sum.u64", "Count.u64", "Max.u64"):
            try:
                int(stats_row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise _error(
                    root,
                    run_id,
                    f"{stats_file.name}.{field}",
                    f"invalid value {stats_row.get(field)!r}",
                ) from exc
    noc_file = stats_dir / "noc_summary.csv"
    dma_file = stats_dir / "dma_summary.csv"
    total_send_packets = _metric_value(root, run_id, noc_file, "total_send_packets", "value")
    total_send_bits = _metric_value(root, run_id, noc_file, "total_send_bits", "value")
    total_xbar_stalls = _metric_value(root, run_id, noc_file, "total_xbar_stalls", "value")
    dma_timeout_retry = _metric_value(root, run_id, dma_file, "timeout_retry", "sum")
    dma_timeout_exhausted = _metric_value(root, run_id, dma_file, "timeout_exhausted", "sum")
    dma_write_timeout_retry = _metric_value(root, run_id, dma_file, "write_timeout_retry", "sum")

    if transport == "explicit-NoC":
        transport_events = sum(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_received", "Sum.u64")
        )
        latency_sum = sum(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_latency_cycles", "Sum.u64")
        )
        latency_count = sum(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_latency_cycles", "Count.u64")
        )
        if latency_count == 0:
            raise _error(root, run_id, "latency_average_cycles", "zero sample count")
        latency_average_cycles = latency_sum / latency_count
        latency_max_cycles = max(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_latency_cycles", "Max.u64")
        )
        inbox_high_water = max(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_inbox_high_water", "Sum.u64")
        )
        stale = sum(
            _stat_values(root, run_id, stats_rows, "sfu_reduction_transport_stale_dropped", "Sum.u64")
        )
        queued = sum(
            _stat_values(root, run_id, stats_rows, "gmem_reduction_send_queued", "Sum.u64")
        )
        rejected = sum(
            _stat_values(root, run_id, stats_rows, "gmem_reduction_send_rejected", "Sum.u64")
        )
    elif transport == "modeled-NoC":
        transport_events = 0
        latency_average_cycles = 0.0
        latency_max_cycles = 0
        inbox_high_water = 0
        stale = 0
        queued = 0
        rejected = 0
    else:
        raise _error(root, run_id, "transport", f"unexpected value {transport!r}")

    golden_checked, golden_mismatches = _golden_evidence(
        root, run_id, output_file, verifier
    )
    return PointRecord(
        transport=transport,
        reduction_vn=reduction_vn,
        workers=workers,
        band_cores=band_cores,
        simulated_time_us=_simulated_time_us(root, run_id, sst_log),
        transport_events=transport_events,
        latency_average_cycles=latency_average_cycles,
        latency_max_cycles=latency_max_cycles,
        inbox_high_water=inbox_high_water,
        queued=queued,
        rejected=rejected,
        stale=stale,
        total_send_packets=total_send_packets,
        total_send_bits=total_send_bits,
        total_xbar_stalls=total_xbar_stalls,
        dma_timeout_retry=dma_timeout_retry,
        dma_timeout_exhausted=dma_timeout_exhausted,
        dma_write_timeout_retry=dma_write_timeout_retry,
        golden_checked=golden_checked,
        golden_mismatches=golden_mismatches,
        source_root=str(root),
        run_id=run_id,
    )


def load_and_validate_matrix(
    sweeps_root: pathlib.Path, verifier: pathlib.Path
) -> list[PointRecord]:
    records = []
    for root_name, transport, vn in ROOT_SPECS:
        root = sweeps_root / root_name
        selected = select_manifest_rows(root, transport, vn)
        records.extend(parse_point(root, row, transport, vn, verifier) for row in selected)
    if len(records) != 12:
        raise _error(sweeps_root, "", "record_count", f"expected 12, got {len(records)}")

    combinations = {(record.transport, record.reduction_vn, record.workers) for record in records}
    expected = {
        (transport, vn, workers)
        for _, transport, vn in ROOT_SPECS
        for workers in EXPECTED_WORKERS
    }
    if combinations != expected:
        raise _error(sweeps_root, "", "matrix", f"unexpected combinations {sorted(combinations, key=str)}")
    for record in records:
        if record.transport == "explicit-NoC":
            expected_events = 4 * ROWS * record.workers
            if record.transport_events != expected_events:
                raise _error(
                    pathlib.Path(record.source_root),
                    record.run_id,
                    "transport_events",
                    f"expected {expected_events}, got {record.transport_events}",
                )
        zero_fields = {
            "dma_timeout_retry": record.dma_timeout_retry,
            "dma_timeout_exhausted": record.dma_timeout_exhausted,
            "dma_write_timeout_retry": record.dma_write_timeout_retry,
            "queued": record.queued,
            "rejected": record.rejected,
            "stale": record.stale,
        }
        for field, value in zero_fields.items():
            if value != 0:
                raise _error(pathlib.Path(record.source_root), record.run_id, field, f"expected 0, got {value}")
    return records


def write_source_csv(records: list[PointRecord], output_csv: pathlib.Path) -> None:
    raise NotImplementedError("source CSV generation is implemented in Task 2")


def render_figure(records: list[PointRecord], output_prefix: pathlib.Path) -> None:
    raise NotImplementedError("figure rendering is implemented in Task 3")
