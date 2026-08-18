#!/usr/bin/env python3
"""Verify exact Phase D1/D2/D3 fused Attention activity."""

import argparse
import csv
import json
from pathlib import Path


BASE_EXPECTED_SUMS = {
    ("core0:rocc", "attention_manager_jobs_issued"): 1,
    ("core0:rocc", "attention_manager_jobs_completed"): 1,
    ("core0:rocc:sfu", "sfu_attention_jobs"): 0,
    ("core1:rocc", "attention_qk_array_ops"): 256,
    ("core1:rocc", "attention_pv_array_ops"): 512,
    ("core1:rocc", "attention_sp_hbm_bytes"): 0,
    ("core1:rocc:sfu", "sfu_attention_jobs"): 8,
    ("core1:rocc:sfu", "sfu_softmax_rows"): 128,
}
RSQRT_KEY = ("core1:rocc:sfu", "sfu_attention_rsqrt_ready_tick")


def expected_sums(causal=False, partial=False):
    expected = dict(BASE_EXPECTED_SUMS)
    if partial:
        expected.update({
            ("core1:rocc", "attention_qk_array_ops"): 140,
            ("core1:rocc", "attention_pv_array_ops"): 240,
            ("core1:rocc:sfu", "sfu_attention_jobs"): 6,
            ("core1:rocc:sfu", "sfu_softmax_rows"): 60,
            ("core1:rocc:sfu", "sfu_attention_scaled_elements"): 1400,
        })
    elif causal:
        expected.update({
            ("core1:rocc", "attention_qk_array_ops"): 192,
            ("core1:rocc", "attention_pv_array_ops"): 384,
            ("core1:rocc:sfu", "sfu_attention_jobs"): 6,
            ("core1:rocc:sfu", "sfu_softmax_rows"): 96,
            ("core1:rocc:sfu", "sfu_attention_scaled_elements"): 3072,
            ("core1:rocc:sfu", "sfu_attention_masked_elements"): 992,
        })
    return expected


def verify_stats(path, causal=False, partial=False):
    expected = expected_sums(causal, partial)
    observed = {}
    rsqrt_count = None
    with Path(path).open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream):
            key = (row["ComponentName"], row["StatisticName"])
            if key in expected:
                observed[key] = int(row["Sum.u64"])
            if key == RSQRT_KEY:
                rsqrt_count = int(row["Count.u64"])
    mismatches = {
        f"{component}/{stat}": {
            "expected": expected_value,
            "actual": observed.get((component, stat)),
        }
        for (component, stat), expected_value in expected.items()
        if observed.get((component, stat)) != expected_value
    }
    if rsqrt_count != 1:
        mismatches["core1:rocc:sfu/sfu_attention_rsqrt_ready_tick.Count"] = {
            "expected": 1, "actual": rsqrt_count,
        }
    return {"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_file")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = verify_stats(args.stats_file, args.causal, args.partial)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
