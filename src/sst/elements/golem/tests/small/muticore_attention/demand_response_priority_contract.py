"""Small, implementation-independent contract model for DMA response ordering."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Response:
    request_id: int
    kind: str
    tile_distance: int
    issue_seq: int


@dataclass(frozen=True)
class Completion:
    response: Response
    arrival_tick: int
    completion_tick: int

    @property
    def wait_ticks(self):
        return self.completion_tick - self.arrival_tick


@dataclass(frozen=True)
class ReplayResult:
    completed: tuple
    max_queue_depth: int


KIND_ORDER = {"consumer": 0, "query": 1, "output": 2, "prefetch": 3}


def priority_key(response):
    """Return the deterministic order required by the G13 contract."""
    if response.kind not in KIND_ORDER:
        raise ValueError(f"unknown response kind: {response.kind}")
    # Current-tile consumer reads outrank all future-tile work.
    distance = max(response.tile_distance, 0) if response.kind == "prefetch" else 0
    return (KIND_ORDER[response.kind], distance, response.issue_seq,
            response.request_id)


def ordered_responses(responses):
    return sorted(responses, key=priority_key)


def validate_completion_order(issued, completed):
    """Validate deterministic priority and completion of a finite trace."""
    if {response.request_id for response in issued} != {
        response.request_id for response in completed
    }:
        raise AssertionError("every issued response must complete exactly once")
    expected = ordered_responses(issued)
    if [response.request_id for response in completed] != [
        response.request_id for response in expected
    ]:
        raise AssertionError("response order violates G13 priority contract")


def replay_responses(arrivals):
    """Drain a finite arrival trace at one response per tick."""
    pending = []
    completed = []
    max_depth = 0
    arrivals = sorted(arrivals, key=lambda item: (item[0], item[1].issue_seq))
    cursor = 0
    tick = arrivals[0][0] if arrivals else 0
    while cursor < len(arrivals) or pending:
        if not pending and cursor < len(arrivals) and tick < arrivals[cursor][0]:
            tick = arrivals[cursor][0]
        while cursor < len(arrivals) and arrivals[cursor][0] <= tick:
            pending.append(arrivals[cursor])
            cursor += 1
        max_depth = max(max_depth, len(pending))
        arrival_tick, response = min(pending, key=lambda item: priority_key(item[1]))
        pending.remove((arrival_tick, response))
        completed.append(Completion(response, arrival_tick, tick))
        tick += 1
    if len({item.response.request_id for item in completed}) != len(completed):
        raise AssertionError("every replay request must complete exactly once")
    return ReplayResult(tuple(completed), max_depth)
