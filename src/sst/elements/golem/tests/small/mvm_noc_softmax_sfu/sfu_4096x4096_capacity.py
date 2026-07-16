#!/usr/bin/env python3

import argparse
import csv
import dataclasses
import os
import pathlib
import shutil
import sys
import tempfile

import plot_sfu_phase4f_large_scale as phase4f


MEM_NODE_SIZE = 268_435_456
BIAS_STRIDE = 16_384
BIAS_BASE = MEM_NODE_SIZE - BIAS_STRIDE
MIN_ARTIFACT_FREE_BYTES = 16 * 1024**3
MIN_AVAILABLE_MEMORY_BYTES = 8 * 1024**3
CANONICAL_TMPDIR = pathlib.Path("/data4/jjgong/tmp")


@dataclasses.dataclass(frozen=True)
class CapacityPoint:
    rows: int
    dim: int
    worker_cores: int
    band_cores: int
    mem_node_size: int
    timeout_sec: int
    rowmajor_region_end: int


@dataclasses.dataclass(frozen=True)
class CapacityEvidence:
    point: CapacityPoint
    elements: int
    tensor_bytes: int
    expected_reduction_each: int
    expected_transport_total: int
    expected_dma_ops: int
    expected_dma_bytes: int
    bias_base: int
    layout_margin_bytes: int


@dataclasses.dataclass(frozen=True)
class ResourceSnapshot:
    artifact_free_bytes: int
    available_memory_bytes: int
    tmpdir: pathlib.Path
    tmpdir_writable: bool


DEFAULT_POINTS = (
    CapacityPoint(512, 4096, 16, 16, MEM_NODE_SIZE, 3_600, 37_748_736),
    CapacityPoint(1024, 4096, 16, 16, MEM_NODE_SIZE, 7_200, 58_720_256),
    CapacityPoint(2048, 4096, 16, 16, MEM_NODE_SIZE, 10_800, 100_663_296),
    CapacityPoint(4096, 4096, 16, 16, MEM_NODE_SIZE, 14_400, 184_549_376),
)
_POINTS_BY_IDENTITY = {
    (point.rows, point.dim, point.worker_cores, point.band_cores): point
    for point in DEFAULT_POINTS
}


def resolve_point(rows: int, dim: int, workers: int, bands: int) -> CapacityPoint:
    identity = (rows, dim, workers, bands)
    if any(type(value) is not int or value <= 0 for value in identity):
        raise ValueError("point fields must be positive integers")
    try:
        return _POINTS_BY_IDENTITY[identity]
    except KeyError as exc:
        raise ValueError(
            f"point is outside the capacity ladder: {rows}:{dim}:{workers}:{bands}"
        ) from exc


def parse_point_list(value: str | None) -> tuple[CapacityPoint, ...]:
    if value is None:
        return DEFAULT_POINTS
    tokens = value.split()
    if not tokens:
        raise ValueError("point list must not be empty")

    points = []
    for token in tokens:
        fields = token.split(":")
        if len(fields) != 4:
            raise ValueError(f"invalid point syntax: {token!r}")
        try:
            point = resolve_point(*(int(field) for field in fields))
        except ValueError as exc:
            raise ValueError(f"invalid capacity point: {token!r}") from exc
        points.append(point)

    result = tuple(points)
    if result != DEFAULT_POINTS[: len(result)]:
        raise ValueError("point list must be a nonempty ordered prefix of the capacity ladder")
    return result


def derive_capacity(point: CapacityPoint) -> CapacityEvidence:
    elements = point.rows * point.dim
    tensor_bytes = elements * 4
    reduction_each = point.rows * point.worker_cores
    return CapacityEvidence(
        point=point,
        elements=elements,
        tensor_bytes=tensor_bytes,
        expected_reduction_each=reduction_each,
        expected_transport_total=4 * reduction_each,
        expected_dma_ops=reduction_each,
        expected_dma_bytes=tensor_bytes,
        bias_base=BIAS_BASE,
        layout_margin_bytes=BIAS_BASE - point.rowmajor_region_end,
    )


def read_resource_snapshot(
    root: pathlib.Path, tmpdir: pathlib.Path = CANONICAL_TMPDIR
) -> ResourceSnapshot:
    root = pathlib.Path(root)
    tmpdir = pathlib.Path(tmpdir)
    artifact_free = shutil.disk_usage(root).free
    available_memory = None
    with pathlib.Path("/proc/meminfo").open(encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                available_memory = int(line.split()[1]) * 1024
                break
    if available_memory is None:
        raise ValueError("/proc/meminfo does not contain MemAvailable")
    return ResourceSnapshot(
        artifact_free_bytes=artifact_free,
        available_memory_bytes=available_memory,
        tmpdir=tmpdir,
        tmpdir_writable=tmpdir.is_dir() and os.access(tmpdir, os.W_OK | os.X_OK),
    )


def validate_resource_snapshot(snapshot: ResourceSnapshot) -> None:
    if snapshot.artifact_free_bytes < MIN_ARTIFACT_FREE_BYTES:
        raise ValueError(
            "artifact filesystem free space is below 16 GiB: "
            f"{snapshot.artifact_free_bytes} bytes"
        )
    if snapshot.available_memory_bytes < MIN_AVAILABLE_MEMORY_BYTES:
        raise ValueError(
            "available host memory is below 8 GiB: "
            f"{snapshot.available_memory_bytes} bytes"
        )
    if snapshot.tmpdir != CANONICAL_TMPDIR:
        raise ValueError(
            f"TMPDIR must be {CANONICAL_TMPDIR}, got {snapshot.tmpdir}"
        )
    if not snapshot.tmpdir_writable:
        raise ValueError(f"TMPDIR is not writable: {snapshot.tmpdir}")


PREFLIGHT_FIELDS = [
    "rows", "dim", "worker_cores", "band_cores", "mem_node_size",
    "timeout_sec", "elements", "tensor_bytes", "rowmajor_region_end",
    "bias_base", "layout_margin_bytes", "expected_reduction_each",
    "expected_transport_total", "expected_dma_ops", "expected_dma_bytes",
    "artifact_free_bytes", "available_memory_bytes", "tmpdir", "status",
]


def _atomic_csv(path: pathlib.Path, fields: list[str], rows: list[dict]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = pathlib.Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_preflight_csv(
    points: tuple[CapacityPoint, ...],
    snapshot: ResourceSnapshot,
    path: pathlib.Path,
) -> None:
    validate_resource_snapshot(snapshot)
    rows = []
    for point in points:
        evidence = derive_capacity(point)
        rows.append(
            {
                "rows": point.rows,
                "dim": point.dim,
                "worker_cores": point.worker_cores,
                "band_cores": point.band_cores,
                "mem_node_size": point.mem_node_size,
                "timeout_sec": point.timeout_sec,
                "elements": evidence.elements,
                "tensor_bytes": evidence.tensor_bytes,
                "rowmajor_region_end": point.rowmajor_region_end,
                "bias_base": evidence.bias_base,
                "layout_margin_bytes": evidence.layout_margin_bytes,
                "expected_reduction_each": evidence.expected_reduction_each,
                "expected_transport_total": evidence.expected_transport_total,
                "expected_dma_ops": evidence.expected_dma_ops,
                "expected_dma_bytes": evidence.expected_dma_bytes,
                "artifact_free_bytes": snapshot.artifact_free_bytes,
                "available_memory_bytes": snapshot.available_memory_bytes,
                "tmpdir": str(snapshot.tmpdir),
                "status": "PASS",
            }
        )
    _atomic_csv(pathlib.Path(path), PREFLIGHT_FIELDS, rows)


def to_phase4f_spec(point: CapacityPoint) -> phase4f.PointSpec:
    return phase4f.PointSpec(
        "CAP",
        point.rows,
        point.dim,
        point.worker_cores,
        point.band_cores,
        point.mem_node_size,
        point.timeout_sec,
    )


def collect_child(
    child_root: pathlib.Path,
    point: CapacityPoint,
    verifier: pathlib.Path,
    parent_manifest: pathlib.Path,
) -> phase4f.PointRecord:
    record = phase4f.parse_child_point(
        pathlib.Path(child_root), to_phase4f_spec(point), pathlib.Path(verifier)
    )
    phase4f.upsert_parent_manifest(pathlib.Path(parent_manifest), record)
    return record


SOURCE_FIELDS = [
    "rows", "dim", "worker_cores", "band_cores", "mem_node_size",
    "timeout_sec", "status", "golden_checked", "golden_mismatches",
    "expected_reduction_each", "expected_transport_total", "transport_events",
    "transport_immediate", "transport_queued", "transport_rejected",
    "transport_stale", "expected_dma_ops", "expected_dma_bytes",
    "dma_timeout_retry", "dma_timeout_exhausted", "dma_write_timeout_retry",
    "simulated_time_us", "wall_time_sec", "output_sha256", "child_root",
]


def _validate_record(point: CapacityPoint, record: phase4f.PointRecord) -> None:
    evidence = derive_capacity(point)
    expected_spec = to_phase4f_spec(point)
    if record.spec != expected_spec:
        raise ValueError(f"point rows={point.rows}: noncanonical spec")
    expected = {
        "status": "PASS",
        "golden_checked": evidence.elements,
        "golden_mismatches": 0,
        "transport_events": evidence.expected_transport_total,
        "transport_rejected": 0,
        "transport_stale": 0,
        "dma_timeout_retry": 0,
        "dma_timeout_exhausted": 0,
        "dma_write_timeout_retry": 0,
    }
    for field, value in expected.items():
        if getattr(record, field) != value:
            raise ValueError(
                f"point rows={point.rows} field={field}: "
                f"expected {value!r}, got {getattr(record, field)!r}"
            )
    if record.transport_immediate + record.transport_queued != evidence.expected_transport_total:
        raise ValueError(f"point rows={point.rows}: transport send total mismatch")


def _source_row(point: CapacityPoint, record: phase4f.PointRecord) -> dict:
    evidence = derive_capacity(point)
    return {
        "rows": point.rows,
        "dim": point.dim,
        "worker_cores": point.worker_cores,
        "band_cores": point.band_cores,
        "mem_node_size": point.mem_node_size,
        "timeout_sec": point.timeout_sec,
        "status": record.status,
        "golden_checked": record.golden_checked,
        "golden_mismatches": record.golden_mismatches,
        "expected_reduction_each": evidence.expected_reduction_each,
        "expected_transport_total": evidence.expected_transport_total,
        "transport_events": record.transport_events,
        "transport_immediate": record.transport_immediate,
        "transport_queued": record.transport_queued,
        "transport_rejected": record.transport_rejected,
        "transport_stale": record.transport_stale,
        "expected_dma_ops": evidence.expected_dma_ops,
        "expected_dma_bytes": evidence.expected_dma_bytes,
        "dma_timeout_retry": record.dma_timeout_retry,
        "dma_timeout_exhausted": record.dma_timeout_exhausted,
        "dma_write_timeout_retry": record.dma_write_timeout_retry,
        "simulated_time_us": record.simulated_time_us,
        "wall_time_sec": record.wall_time_sec,
        "output_sha256": record.output_sha256,
        "child_root": record.child_root,
    }


def _summary(records: list[phase4f.PointRecord]) -> str:
    lines = [
        "# SFU 4096x4096 Explicit-NoC Softmax 容量验证结果",
        "",
        "| Rows | Dim | Status | Golden | Reduction each | Transport | DMA bytes | Wall time (s) |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for point, record in zip(DEFAULT_POINTS, records):
        evidence = derive_capacity(point)
        lines.append(
            f"| {point.rows:,} | {point.dim:,} | {record.status} | "
            f"{record.golden_checked:,}/{record.golden_mismatches} | "
            f"{evidence.expected_reduction_each:,} | "
            f"{evidence.expected_transport_total:,} | "
            f"{evidence.expected_dma_bytes:,} | {record.wall_time_sec:g} |"
        )
    final = derive_capacity(DEFAULT_POINTS[-1])
    lines.extend(
        [
            "",
            "## 4096x4096 Final Gate",
            "",
            f"- Golden: {final.elements:,} checked / 0 mismatches",
            f"- Max/Sum request/response: {final.expected_reduction_each:,} each",
            f"- Explicit-NoC transport: {final.expected_transport_total:,}",
            f"- DMA read/write bytes: {final.expected_dma_bytes:,} each",
            "- Retry/rejected/stale: 0",
            "",
        ]
    )
    return "\n".join(lines)


def write_capacity_report(
    root: pathlib.Path, output_dir: pathlib.Path, verifier: pathlib.Path
) -> None:
    root = pathlib.Path(root)
    manifest = root / "capacity_manifest.csv"
    records = phase4f.load_parent_manifest(manifest)
    by_identity = {
        (record.spec.rows, record.spec.dim, record.spec.worker_cores, record.spec.band_cores): record
        for record in records
    }
    expected_identities = [
        (point.rows, point.dim, point.worker_cores, point.band_cores)
        for point in DEFAULT_POINTS
    ]
    if set(by_identity) != set(expected_identities) or len(records) != len(DEFAULT_POINTS):
        raise ValueError("capacity manifest must contain exactly four canonical PASS points")

    reparsed = []
    for point, identity in zip(DEFAULT_POINTS, expected_identities):
        record = by_identity[identity]
        parsed = phase4f.parse_child_point(
            pathlib.Path(record.child_root), to_phase4f_spec(point), pathlib.Path(verifier)
        )
        if parsed != record:
            raise ValueError(f"point rows={point.rows}: child evidence drift")
        _validate_record(point, parsed)
        reparsed.append(parsed)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(
        output_dir / "sfu_4096x4096_capacity_source_data.csv",
        SOURCE_FIELDS,
        [_source_row(point, record) for point, record in zip(DEFAULT_POINTS, reparsed)],
    )
    summary_path = output_dir / "sfu_4096x4096_capacity_summary.md"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(_summary(reparsed), encoding="utf-8", newline="\n")
        os.replace(temporary, summary_path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--root", type=pathlib.Path, required=True)
    preflight.add_argument("--tmpdir", type=pathlib.Path, required=True)
    preflight.add_argument("--output", type=pathlib.Path, required=True)
    preflight.add_argument("--point-list")

    collect = commands.add_parser("collect")
    collect.add_argument("--child-root", type=pathlib.Path, required=True)
    collect.add_argument("--rows", type=int, required=True)
    collect.add_argument("--parent-manifest", type=pathlib.Path, required=True)
    collect.add_argument("--verifier", type=pathlib.Path, required=True)

    report = commands.add_parser("report")
    report.add_argument("--root", type=pathlib.Path, required=True)
    report.add_argument("--output-dir", type=pathlib.Path, required=True)
    report.add_argument("--verifier", type=pathlib.Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "preflight":
        points = parse_point_list(args.point_list)
        snapshot = read_resource_snapshot(args.root, args.tmpdir)
        write_preflight_csv(points, snapshot, args.output)
    elif args.command == "collect":
        point = resolve_point(args.rows, 4096, 16, 16)
        record = collect_child(
            args.child_root, point, args.verifier, args.parent_manifest
        )
        print(f"run_id={record.run_id} output_sha256={record.output_sha256}")
    else:
        write_capacity_report(args.root, args.output_dir, args.verifier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
