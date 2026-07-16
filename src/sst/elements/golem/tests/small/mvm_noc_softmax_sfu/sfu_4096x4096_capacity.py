#!/usr/bin/env python3

import dataclasses


MEM_NODE_SIZE = 268_435_456
BIAS_STRIDE = 16_384
BIAS_BASE = MEM_NODE_SIZE - BIAS_STRIDE


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
