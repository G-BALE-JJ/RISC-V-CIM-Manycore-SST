#!/usr/bin/env python3

import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(SCRIPT_DIR, "verify_softmax_sfu_against_golden.py")


def pack_fp32(values):
    return struct.pack(f"<{len(values)}f", *[float(v) for v in values])


def matmul(a, b):
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    out = []
    for i in range(m):
        row = []
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc += float(a[i][kk]) * float(b[kk][j])
            row.append(acc)
        out.append(row)
    return out


def softmax_rows(rows):
    out = []
    for row in rows:
        max_v = max(row)
        exps = [math.exp(float(v) - max_v) for v in row]
        denom = sum(exps)
        out.append([v / denom for v in exps])
    return out


def tile_local_softmax(rows, block_n):
    out = []
    for row in rows:
        merged = []
        for n0 in range(0, len(row), block_n):
            tile = row[n0 : n0 + block_n]
            max_v = max(tile)
            exps = [math.exp(float(v) - max_v) for v in tile]
            denom = sum(exps)
            merged.extend([v / denom for v in exps])
        out.append(merged)
    return out


def flatten(rows):
    return [value for row in rows for value in row]


class SoftmaxSfuGoldenCheckerTest(unittest.TestCase):
    def write_matrix_case(self, tmpdir, a_rows, b_rows, c_rows):
        a_path = os.path.join(tmpdir, "a.bin")
        b_path = os.path.join(tmpdir, "b.bin")
        c_path = os.path.join(tmpdir, "c.bin")
        with open(a_path, "wb") as f:
            f.write(pack_fp32(flatten(a_rows)))
        with open(b_path, "wb") as f:
            f.write(pack_fp32(flatten(b_rows)))
        with open(c_path, "wb") as f:
            f.write(pack_fp32(flatten(c_rows)))
        return a_path, b_path, c_path

    def write_logits_case(self, tmpdir, logits_rows, c_rows):
        logits_path = os.path.join(tmpdir, "logits.bin")
        c_path = os.path.join(tmpdir, "c.bin")
        with open(logits_path, "wb") as f:
            f.write(pack_fp32(flatten(logits_rows)))
        with open(c_path, "wb") as f:
            f.write(pack_fp32(flatten(c_rows)))
        return logits_path, c_path

    def run_checker(self, tmpdir, a_rows, b_rows, c_rows, extra_args=None):
        a_path, b_path, c_path = self.write_matrix_case(tmpdir, a_rows, b_rows, c_rows)
        return subprocess.run(
            [
                sys.executable,
                CHECKER,
                "--a-file",
                a_path,
                "--b-file",
                b_path,
                "--c-file",
                c_path,
                "--m",
                str(len(a_rows)),
                "--n",
                str(len(b_rows[0])),
                "--k",
                str(len(b_rows)),
                "--block-m",
                str(len(a_rows)),
                "--block-n",
                "2",
            ]
            + list(extra_args or []),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_passes_for_full_rowwise_softmax_of_a_times_b(self):
        a = [[1, 0], [0, 1]]
        b = [[1, 2, 3, 4], [4, 3, 2, 1]]
        c = softmax_rows(matmul(a, b))
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_checker(tmpdir, a, b, c)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[VERIFY-SFU-SOFTMAX] PASS", result.stdout)
        self.assertIn("reference=a_b", result.stdout)

    def test_fails_for_tile_local_softmax_when_row_crosses_tiles(self):
        a = [[1, 0], [0, 1]]
        b = [[1, 2, 3, 4], [4, 3, 2, 1]]
        logits = matmul(a, b)
        c = tile_local_softmax(logits, block_n=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_checker(tmpdir, a, b, c)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[VERIFY-SFU-SOFTMAX] FAIL", result.stdout)

    def test_probability_reference_checks_full_rows(self):
        a = [[1, 0], [0, 1]]
        b = [[1, 2, 3, 4], [4, 3, 2, 1]]
        c = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_checker(tmpdir, a, b, c, ["--reference", "probability"])

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[VERIFY-SFU-SOFTMAX] PASS", result.stdout)
        self.assertIn("reference=probability", result.stdout)

    def test_probability_reference_rejects_tile_local_row_sums(self):
        a = [[1, 0], [0, 1]]
        b = [[1, 2, 3, 4], [4, 3, 2, 1]]
        c = [[0.25, 0.75, 0.25, 0.75], [0.75, 0.25, 0.75, 0.25]]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_checker(tmpdir, a, b, c, ["--reference", "probability"])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[VERIFY-SFU-SOFTMAX] FAIL", result.stdout)

    def test_handles_numerically_large_logits(self):
        a = [[1, 0]]
        b = [[1000, 1001, 1002, 1003], [0, 0, 0, 0]]
        c = softmax_rows(matmul(a, b))
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_checker(tmpdir, a, b, c)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_logits_reference_passes_for_standalone_softmax_input(self):
        logits = [[1, 2, 3, 4], [4, 3, 2, 1]]
        c = softmax_rows(logits)
        with tempfile.TemporaryDirectory() as tmpdir:
            logits_path, c_path = self.write_logits_case(tmpdir, logits, c)
            result = subprocess.run(
                [
                    sys.executable,
                    CHECKER,
                    "--a-file",
                    os.path.join(tmpdir, "unused_a.bin"),
                    "--b-file",
                    os.path.join(tmpdir, "unused_b.bin"),
                    "--c-file",
                    c_path,
                    "--logits-file",
                    logits_path,
                    "--reference",
                    "logits",
                    "--m",
                    "2",
                    "--n",
                    "4",
                    "--k",
                    "1",
                    "--block-m",
                    "2",
                    "--block-n",
                    "2",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[VERIFY-SFU-SOFTMAX] PASS", result.stdout)
        self.assertIn("reference=logits", result.stdout)

    def test_logits_reference_requires_logits_file(self):
        a = [[1, 0]]
        b = [[1, 2, 3, 4], [0, 0, 0, 0]]
        c = softmax_rows(matmul(a, b))
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_checker(tmpdir, a, b, c, ["--reference", "logits"])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--logits-file is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
