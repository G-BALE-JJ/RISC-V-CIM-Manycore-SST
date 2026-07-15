#!/usr/bin/env python3

import dataclasses


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
