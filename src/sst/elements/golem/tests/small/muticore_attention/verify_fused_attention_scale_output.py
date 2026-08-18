#!/usr/bin/env python3
"""Verify a four-node striped S256 fused Attention output."""

import argparse
import json
import math
from pathlib import Path

import attention_case


def verify(q_file, k_file, v_file, hbm_dir, output_offset,
           queries, keys, head_dim, band_rows):
    q = attention_case._read_f32(q_file, queries * head_dim)
    k = attention_case._read_f32(k_file, keys * head_dim)
    v = attention_case._read_f32(v_file, keys * head_dim)
    expected = attention_case.compute_attention(
        q, k, v, queries, keys, head_dim, False
    )
    actual = []
    band_values = band_rows * head_dim
    for node in range(1, 5):
        actual.extend(attention_case._read_f32(
            Path(hbm_dir) / f"hbm_out_node{node}.bin", band_values, output_offset
        ))
    mismatches = 0
    max_abs_error = 0.0
    first_mismatch = None
    for index, (got, want) in enumerate(zip(actual, expected)):
        error = abs(got - want)
        max_abs_error = max(max_abs_error, error)
        if not math.isclose(got, want, rel_tol=2.0e-4, abs_tol=2.0e-4):
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "query": index // head_dim,
                    "dim": index % head_dim,
                    "actual": got,
                    "expected": want,
                    "abs_error": error,
                }
    return {
        "status": "PASS" if mismatches == 0 else "FAIL",
        "checked": len(expected),
        "mismatches": mismatches,
        "max_abs_error": max_abs_error,
        "first_mismatch": first_mismatch,
        "shape": {"queries": queries, "keys": keys, "head_dim": head_dim},
        "hbm_output_nodes": [1, 2, 3, 4],
        "score_probability_hbm_bytes": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-file", required=True)
    parser.add_argument("--k-file", required=True)
    parser.add_argument("--v-file", required=True)
    parser.add_argument("--hbm-dir", required=True)
    parser.add_argument("--output-offset", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--keys", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--band-rows", type=int, required=True)
    parser.add_argument("--result-json")
    args = parser.parse_args()
    result = verify(args.q_file, args.k_file, args.v_file, args.hbm_dir,
                    args.output_offset, args.queries, args.keys,
                    args.head_dim, args.band_rows)
    print(json.dumps(result, indent=2))
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
