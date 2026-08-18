#!/usr/bin/env python3

import importlib.util
import pathlib
import struct
import subprocess
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
TESTS_DIR = HERE.parents[1]
BASE_DIR = HERE.parent / "mvm_noc_int_array"
RUNTIME = BASE_DIR / "golem_matmul_runtime.cpp"
GEMM = BASE_DIR / "gemm_matmul_op.h"
GENERATOR = TESTS_DIR / "tools" / "gen_hbm_init.py"
RUNNER = TESTS_DIR / "run_noc_dma_pipeline.sh"
MAKEFILE = HERE / "Makefile"


def load_attention_case():
    path = HERE / "attention_case.py"
    spec = importlib.util.spec_from_file_location("attention_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransposeContractTests(unittest.TestCase):
    def test_runtime_accepts_only_logical_b_transpose(self):
        text = RUNTIME.read_text()
        self.assertIn("op->transpose_a != 0", text)
        self.assertIn("op->transpose_b != 0 && op->transpose_b != 1", text)
        self.assertIn("op.transpose_b ? op.n : op.k", text)
        self.assertIn("op.transpose_b ? op.k : op.n", text)

    def test_tensor_loader_maps_logical_b_column_to_native_k_row(self):
        text = GEMM.read_text()
        self.assertIn("bool transpose_b", text)
        self.assertIn("tensors.transpose_b", text)
        self.assertIn("n_base) * tensors.b_stride0", text)
        self.assertIn("k_base + i) * tensors.b_stride1", text)

    def test_hbm_packer_loads_native_n_by_k_tensor(self):
        text = GENERATOR.read_text()
        self.assertIn('MATMUL_OP_DESC["transpose_b"] not in (0, 1)', text)
        self.assertIn("b_rows = GEMM_N if transpose_b else GEMM_K", text)
        self.assertIn("b_cols = GEMM_K if transpose_b else GEMM_N", text)
        self.assertIn("transpose_b=transpose_b", text)

    def test_base_runner_forwards_transpose_b(self):
        text = RUNNER.read_text()
        self.assertIn("--transpose-b", text)
        self.assertIn('export GOLEM_MATMUL_TRANSPOSE_B="$GOLEM_MATMUL_TRANSPOSE_B"', text)

    def test_attention_makefile_selects_riscv_output_directory(self):
        text = MAKEFILE.read_text()
        self.assertIn("ARCH ?= riscv64", text)
        self.assertIn("ARCH=$(ARCH)", text)


class AttentionCaseTests(unittest.TestCase):
    def test_native_k_index_and_full_qk_verification(self):
        case = load_attention_case()
        self.assertEqual(case.native_k_index(2, 3, 5), 13)
        q = [1.0, 2.0, 3.0, 4.0]
        k = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        expected = case.compute_qk(q, k, queries=2, keys=3, head_dim=2)
        self.assertEqual(expected, [1.0, 2.0, 3.0, 3.0, 4.0, 7.0])

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            q_path = root / "q.bin"
            k_path = root / "k.bin"
            out_path = root / "qk.bin"
            q_path.write_bytes(struct.pack("<4f", *q))
            k_path.write_bytes(struct.pack("<6f", *k))
            out_path.write_bytes(struct.pack("<6f", *expected))
            result = case.verify_qk(q_path, k_path, out_path, 2, 3, 2)
            self.assertEqual(result["mismatches"], 0)
            self.assertEqual(result["checked"], 6)

    def test_verifier_crops_padded_output_row_stride(self):
        case = load_attention_case()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            q_path = root / "q.bin"
            k_path = root / "k.bin"
            out_path = root / "qk.bin"
            q_path.write_bytes(struct.pack("<4f", 1.0, 0.0, 0.0, 1.0))
            k_path.write_bytes(
                struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
            )
            out_path.write_bytes(
                struct.pack("<8f", 1.0, 3.0, 99.0, 99.0, 2.0, 4.0, 99.0, 99.0)
            )
            result = case.verify_qk(
                q_path, k_path, out_path, 2, 2, 2, storage_keys=4
            )
            self.assertEqual(result["mismatches"], 0)

    def test_runner_dry_run_uses_native_k_and_transpose_b(self):
        runner = HERE / "run_muticore_attention.sh"
        result = subprocess.run(
            [
                "bash",
                str(runner),
                "--queries",
                "64",
                "--keys",
                "64",
                "--head-dim",
                "64",
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--transpose-b 1", result.stdout)
        self.assertIn("native_k_64x64_padded64.bin", result.stdout)
        self.assertIn("--tensor-a", result.stdout)
        self.assertIn("--gemm-cores 16", result.stdout)
        self.assertIn("GOLEM_GROUP_MANAGER_ENABLE=1", result.stdout)
        self.assertIn("GOLEM_CTRL_LINK_ENABLE=1", result.stdout)
        self.assertNotIn("--tensor-a-file", result.stdout)
        self.assertNotIn("--gemm-num-cores", result.stdout)

    def test_runner_pads_non_divisible_key_tail(self):
        runner = HERE / "run_muticore_attention.sh"
        result = subprocess.run(
            ["bash", str(runner), "--keys", "70", "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--storage-keys 80", result.stdout)
        self.assertIn("--gemm-n 80", result.stdout)
        self.assertIn("--orig-n 70", result.stdout)
        self.assertIn("--gemm-block-n 16", result.stdout)


if __name__ == "__main__":
    unittest.main()
