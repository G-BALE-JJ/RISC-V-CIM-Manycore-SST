#!/usr/bin/env python3

import os
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(SCRIPT_DIR, "verify_softmax_tile_against_golden.py")


def pack_fp32(values):
    return struct.pack(f"<{len(values)}f", *[float(v) for v in values])


def softmax_rows(rows):
    out = []
    for row in rows:
        max_v = max(row)
        exps = [pow(2.718281828459045, v - max_v) for v in row]
        denom = sum(exps)
        out.append([v / denom for v in exps])
    return out


class SoftmaxTileGoldenCheckerTest(unittest.TestCase):
    def run_checker(self, tmpdir, c_values, extra_args=None):
        a_path = os.path.join(tmpdir, "a.bin")
        b_path = os.path.join(tmpdir, "b.bin")
        c_path = os.path.join(tmpdir, "c.bin")

        # A is 2x2, B is 2x3, so A@B is:
        # row0 = [1, 2, 3]
        # row1 = [4, 5, 6]
        with open(a_path, "wb") as f:
            f.write(pack_fp32([1, 0, 0, 1]))
        with open(b_path, "wb") as f:
            f.write(pack_fp32([1, 2, 3, 4, 5, 6]))
        with open(c_path, "wb") as f:
            f.write(pack_fp32(c_values))

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
                "2",
                "--n",
                "3",
                "--k",
                "2",
                "--block-m",
                "2",
                "--block-n",
                "3",
            ] + list(extra_args or []),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_passes_when_c_is_tile_local_softmax_of_a_times_b(self):
        expected = softmax_rows([[1, 2, 3], [4, 5, 6]])
        flat = [item for row in expected for item in row]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_checker(tmpdir, flat)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[VERIFY-SOFTMAX] PASS", result.stdout)

    def test_fails_when_c_does_not_match_tile_local_softmax(self):
        wrong = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_checker(tmpdir, wrong)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[VERIFY-SOFTMAX] FAIL", result.stdout)

    def test_probability_reference_passes_for_valid_probability_rows(self):
        values = [0.2, 0.3, 0.5, 0.1, 0.1, 0.8]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_checker(tmpdir, values, ["--reference", "probability"])

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[VERIFY-SOFTMAX] PASS", result.stdout)
        self.assertIn("reference=probability", result.stdout)

    def test_probability_reference_fails_when_row_sum_is_not_one(self):
        values = [0.2, 0.3, 0.4, 0.1, 0.1, 0.8]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_checker(tmpdir, values, ["--reference", "probability"])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[VERIFY-SOFTMAX] FAIL", result.stdout)


if __name__ == "__main__":
    unittest.main()
