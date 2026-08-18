#!/usr/bin/env python3

import importlib.util
import math
import pathlib
import struct
import subprocess
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
RUNNER = HERE / "run_fused_attention_online.sh"
ATTENTION_CASE = HERE / "attention_case.py"


def load_attention_case():
    spec = importlib.util.spec_from_file_location("attention_case_d4", ATTENTION_CASE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_f32(path):
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data) // 4}f", data))


class FusedAttentionD4ContractTests(unittest.TestCase):
    def test_extreme_case_forces_second_tile_running_max_jump(self):
        case = load_attention_case()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            q_path, k_path, v_path = root / "q.bin", root / "k.bin", root / "v.bin"
            case.generate_case(
                64, 64, 64, q_path, k_path, v_path=v_path, extreme_logits=True
            )
            q, k, v = read_f32(q_path), read_f32(k_path), read_f32(v_path)
        scaled = [value / 8.0 for value in case.compute_qk(q, k, 64, 64, 64)]
        self.assertTrue(all(math.isfinite(value) for value in q + k + v + scaled))
        self.assertEqual(max(scaled[:32]), -100.0)
        self.assertEqual(max(scaled[32:64]), 100.0)
        self.assertTrue(all(math.isfinite(value) for value in
                            case.compute_attention(q, k, v, 64, 64, 64, False)))

    def test_extreme_runner_selects_d4_artifacts_and_generator_mode(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "--extreme-logits", "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fused_attention_d4_q64_k64_d64", result.stdout)
        self.assertIn("attention_case.py generate", result.stdout)
        self.assertIn("--extreme-logits", result.stdout)
        self.assertIn("fused_attention_d1", result.stdout)


if __name__ == "__main__":
    unittest.main()
