#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from demand_response_priority_contract import Response, replay_responses
from dma_kind_trace import parse_events


CONTRACT_KIND = {
    "attention_kv": "consumer",
    "attention_query": "query",
    "attention_output": "output",
    "attention_kv_prefetch": "prefetch",
}


def replay_trace_events(events, arrival_quantum):
    if arrival_quantum <= 0:
        raise ValueError("arrival_quantum must be positive")
    known = [(cycle, kind) for cycle, kind in events if kind in CONTRACT_KIND]
    first_cycle = min((cycle for cycle, _ in known), default=0)
    arrivals = []
    for issue_seq, (cycle, semantic_kind) in enumerate(known):
        kind = CONTRACT_KIND[semantic_kind]
        arrivals.append(
            ((cycle - first_cycle) // arrival_quantum,
             Response(issue_seq + 1, kind, 1 if kind == "prefetch" else 0, issue_seq))
        )
    result = replay_responses(arrivals)
    waits = defaultdict(list)
    for completion in result.completed:
        waits[completion.response.kind].append(completion.wait_ticks)
    completed_ids = [item.response.request_id for item in result.completed]
    counts = Counter(item.response.kind for item in result.completed)
    return {
        "issued": len(arrivals),
        "completed": len(result.completed),
        "exactly_once": len(completed_ids) == len(set(completed_ids)) == len(arrivals),
        "drained": len(result.completed) == len(arrivals),
        "max_queue_depth": result.max_queue_depth,
        "completion_counts": dict(sorted(counts.items())),
        "max_wait_ticks": {kind: max(values) for kind, values in sorted(waits.items())},
        "completion_order": [item.response.kind for item in result.completed],
    }


def main():
    parser = argparse.ArgumentParser(description="Replay semantic DMA trace under compressed contention")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--arrival-quantum", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = replay_trace_events(parse_events(args.trace), args.arrival_quantum)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
