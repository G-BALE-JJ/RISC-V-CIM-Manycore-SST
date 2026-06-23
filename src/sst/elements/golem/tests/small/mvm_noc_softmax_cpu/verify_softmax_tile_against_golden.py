#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys

if __package__ in {None, ""}:
    _tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)

from golem_dtype import (  # noqa: E402
    default_tolerance,
    elem_nbytes,
    normalize_dtype,
    numpy_dtype_name,
    unpack_values,
    values_close,
)


def load_matrix(path: str, rows: int, cols: int, name: str, dtype: str):
    if path.endswith(".csv"):
        vals = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                for item in row:
                    item = item.strip()
                    if item:
                        vals.append(float(item))
    elif path.endswith(".npy"):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                f"{name} uses .npy but numpy is unavailable: {path}"
            ) from exc
        arr = np.load(path)
        if arr.shape != (rows, cols):
            raise ValueError(
                f"{name} shape mismatch, expected ({rows},{cols}), got {arr.shape}"
            )
        return arr.astype(numpy_dtype_name(dtype), copy=False).tolist()
    else:
        with open(path, "rb") as f:
            data = f.read()
        expected = rows * cols * elem_nbytes(dtype)
        if len(data) != expected:
            raise ValueError(
                f"{name} binary size mismatch, expected {expected} bytes, got {len(data)}"
            )
        vals = unpack_values(dtype, data)

    if len(vals) != rows * cols:
        raise ValueError(
            f"{name} element count mismatch, expected {rows * cols}, got {len(vals)}"
        )
    return [vals[r * cols : (r + 1) * cols] for r in range(rows)]


def matmul(a, b, m: int, n: int, k: int, bias_enable: int, bias_value: float):
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc += float(a[i][kk]) * float(b[kk][j])
            if bias_enable != 0:
                acc += bias_value
            out[i][j] = acc
    return out


def softmax_row(row):
    max_v = max(row)
    exps = [math.exp(float(v) - max_v) for v in row]
    denom = sum(exps)
    return [v / denom for v in exps]


def tile_local_softmax(logits, m: int, n: int, block_m: int, block_n: int):
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(
            f"M/N must be divisible by block_m/block_n, got m={m}, n={n}, block_m={block_m}, block_n={block_n}"
        )
    ref = [[0.0 for _ in range(n)] for _ in range(m)]
    for m0 in range(0, m, block_m):
        for n0 in range(0, n, block_n):
            for r in range(block_m):
                row = logits[m0 + r][n0 : n0 + block_n]
                sm = softmax_row(row)
                for c, value in enumerate(sm):
                    ref[m0 + r][n0 + c] = value
    return ref


def verify_probability_rows(c, m: int, n: int, block_m: int, block_n: int, atol: float, rtol: float, max_mismatches: int):
    mismatches = 0
    max_abs = 0.0
    printed = 0
    for m0 in range(0, m, block_m):
        for n0 in range(0, n, block_n):
            for r in range(block_m):
                row_sum = 0.0
                row_has_bad_value = False
                row_idx = m0 + r
                for cc in range(block_n):
                    col_idx = n0 + cc
                    value = float(c[row_idx][col_idx])
                    if not math.isfinite(value) or value < -atol or value > 1.0 + atol:
                        row_has_bad_value = True
                        if printed < max_mismatches:
                            print(
                                f"[VERIFY-SOFTMAX] invalid probability at ({row_idx},{col_idx}): C={value:.9g}"
                            )
                            printed += 1
                    row_sum += value
                diff = row_sum - 1.0
                abs_diff = abs(diff)
                max_abs = max(max_abs, abs_diff)
                if row_has_bad_value or abs_diff > (atol + rtol):
                    mismatches += 1
                    if printed < max_mismatches:
                        print(
                            f"[VERIFY-SOFTMAX] invalid probability row tile_origin=({m0},{n0}) row={row_idx}: sum={row_sum:.9g}, diff={diff:.9g}"
                        )
                        printed += 1
    return mismatches, max_abs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify C tensor against tile-local softmax(A@B) golden"
    )
    parser.add_argument("--a-file", required=True)
    parser.add_argument("--b-file", required=True)
    parser.add_argument("--c-file", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--dtype", default=os.getenv("GOLEM_MATMUL_DTYPE", "fp32"))
    parser.add_argument(
        "--reference",
        choices=["a_b", "probability"],
        default="a_b",
        help="Reference mode: a_b compares against tile-local softmax(A@B); probability only checks each tile-local row is a finite probability distribution",
    )
    parser.add_argument("--bias-enable", type=int, default=0)
    parser.add_argument("--bias-value", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=8,
        help="Maximum mismatches to print before summarizing",
    )
    args = parser.parse_args(argv)

    dtype = normalize_dtype(args.dtype)
    if dtype != "fp32":
        raise ValueError(f"softmax checker only supports fp32, got {dtype}")

    block_m = args.block_m if args.block_m is not None else args.m
    block_n = args.block_n if args.block_n is not None else args.n
    if block_m <= 0 or block_n <= 0:
        raise ValueError(f"block shape must be positive, got ({block_m},{block_n})")

    c = load_matrix(args.c_file, args.m, args.n, "C", dtype)

    atol, rtol = default_tolerance(dtype)
    if args.atol is not None:
        atol = args.atol
    if args.rtol is not None:
        rtol = args.rtol

    total = args.m * args.n
    if args.reference == "probability":
        mismatches, max_abs = verify_probability_rows(
            c, args.m, args.n, block_m, block_n, atol, rtol, args.max_mismatches
        )
        if mismatches:
            print(
                f"[VERIFY-SOFTMAX] FAIL reference=probability dtype={dtype} checked={total} bad_rows={mismatches} max_row_sum_abs_diff={max_abs:.9g} atol={atol} rtol={rtol} block=({block_m},{block_n})"
            )
            return 1
        print(
            f"[VERIFY-SOFTMAX] PASS reference=probability dtype={dtype} checked={total} bad_rows=0 max_row_sum_abs_diff={max_abs:.9g} atol={atol} rtol={rtol} block=({block_m},{block_n})"
        )
        return 0

    a = load_matrix(args.a_file, args.m, args.k, "A", dtype)
    b = load_matrix(args.b_file, args.k, args.n, "B", dtype)
    logits = matmul(a, b, args.m, args.n, args.k, args.bias_enable, args.bias_value)
    ref = tile_local_softmax(logits, args.m, args.n, block_m, block_n)

    mismatches = 0
    max_abs = 0.0
    printed = 0
    for i in range(args.m):
        for j in range(args.n):
            got = float(c[i][j])
            exp = float(ref[i][j])
            diff = got - exp
            abs_diff = abs(diff)
            max_abs = max(max_abs, abs_diff)
            if not values_close(dtype, got, exp, atol=atol, rtol=rtol):
                mismatches += 1
                if printed < args.max_mismatches:
                    print(
                        f"[VERIFY-SOFTMAX] mismatch at ({i},{j}): C={got:.9g}, REF={exp:.9g}, diff={diff:.9g}"
                    )
                    printed += 1

    if mismatches:
        print(
            f"[VERIFY-SOFTMAX] FAIL reference=a_b dtype={dtype} checked={total} mismatches={mismatches} max_abs_diff={max_abs:.9g} atol={atol} rtol={rtol} block=({block_m},{block_n})"
        )
        return 1

    print(
        f"[VERIFY-SOFTMAX] PASS reference=a_b dtype={dtype} checked={total} mismatches=0 max_abs_diff={max_abs:.9g} atol={atol} rtol={rtol} block=({block_m},{block_n})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
