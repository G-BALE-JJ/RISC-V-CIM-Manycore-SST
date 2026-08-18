#!/usr/bin/env python3
"""Verify the exact Phase C1 fused Attention activity counters."""

import argparse
import csv
import json
from pathlib import Path


EXPECTED = {
    ("core0:rocc", "attention_manager_jobs_issued"): 1,
    ("core0:rocc", "attention_manager_jobs_completed"): 1,
    ("core0:rocc:sfu", "sfu_attention_jobs"): 0,
    ("core1:rocc", "attention_qk_array_ops"): 64,
    ("core1:rocc", "attention_pv_array_ops"): 128,
    ("core1:rocc", "attention_sp_hbm_bytes"): 0,
    ("core1:rocc:sfu", "sfu_attention_jobs"): 2,
    ("core1:rocc:sfu", "sfu_softmax_rows"): 32,
}


def verify_stats(path):
    observed = {}
    with Path(path).open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream):
            key = (row["ComponentName"], row["StatisticName"])
            if key in EXPECTED:
                observed[key] = int(row["Sum.u64"])

    missing = [f"{component}/{stat}" for component, stat in EXPECTED if (component, stat) not in observed]
    mismatches = {
        f"{component}/{stat}": {"expected": expected, "actual": observed.get((component, stat))}
        for (component, stat), expected in EXPECTED.items()
        if observed.get((component, stat)) != expected
    }
    return {
        "status": "PASS" if not missing and not mismatches else "FAIL",
        "missing": missing,
        "mismatches": mismatches,
        "checked": len(EXPECTED),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_file")
    args = parser.parse_args()
    result = verify_stats(args.stats_file)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
