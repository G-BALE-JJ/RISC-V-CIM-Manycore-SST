#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


KIND_NAMES = {
    0: "unknown",
    1: "attention_query",
    2: "attention_kv",
    3: "attention_kv_prefetch",
    4: "attention_output",
}
EVENT_RE = re.compile(r"send (?:READ_RESP|WRITE_COMPLETE)\s+cycle=(\d+).*?\bkind=(\d+)\b")


def parse_events(path: Path):
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVENT_RE.search(line)
        if match:
            events.append((int(match.group(1)), KIND_NAMES.get(int(match.group(2)), "invalid")))
    return events


def parse_trace(path: Path):
    counts = Counter()
    for _, kind in parse_events(path):
        counts[kind] += 1
    return {
        "counts": {name: counts[name] for name in KIND_NAMES.values()},
        "unknown_events": counts["unknown"],
        "total_events": sum(counts.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize semantic DMA kinds in an SST trace")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = parse_trace(args.trace)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
