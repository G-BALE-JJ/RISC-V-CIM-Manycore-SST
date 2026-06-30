#!/usr/bin/env python3

import os
import subprocess
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SfuSoftmaxPipelineWrapperTest(unittest.TestCase):
    def read_wrapper(self):
        with open(os.path.join(SCRIPT_DIR, "run_noc_dma_softmax_sfu_pipeline.sh"), "r", encoding="utf-8") as source_file:
            return source_file.read()

    def run_wrapper(self, *args):
        env = os.environ.copy()
        env.pop("GOLEM_ARCH_SCRIPT", None)
        cmd = [
            os.path.join(SCRIPT_DIR, "run_noc_dma_softmax_sfu_pipeline.sh"),
            "--groups",
            "4",
            "--num-cores",
            "16",
            "--gemm-cores",
            "16",
            "--num-mem-nodes",
            "9",
            "--mesh-dim-x",
            "8",
            "--gemm-m",
            "128",
            "--gemm-n",
            "128",
            "--gemm-k",
            "128",
            "--gemm-block-m",
            "64",
            "--gemm-block-n",
            "64",
            "--gemm-block-k",
            "64",
            "--dry-run",
            *args,
        ]
        return subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_wrapper_launches_sfu_binary_through_base_pipeline(self):
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("VANADIS_EXE=", result.stdout)
        self.assertIn("test_noc_dma_softmax_sfu", result.stdout)
        self.assertIn("GOLEM_SFU_ENABLE=1", result.stdout)
        self.assertIn("GOLEM_MATMUL_DTYPE=fp32", result.stdout)

    def test_wrapper_uses_full_row_sfu_checker_not_tile_local_checker(self):
        source = self.read_wrapper()

        self.assertIn("verify_softmax_sfu_against_golden.py", source)
        self.assertNotIn("verify_softmax_tile_against_golden.py", source)
        self.assertIn("--reference", source)

    def test_wrapper_unpacks_sfu_output_with_gemm_tile_layout(self):
        source = self.read_wrapper()

        self.assertIn('GOLEM_GEMM_OUT_LAYOUT="${GOLEM_GEMM_OUT_LAYOUT:-colmajor_tile}"', source)
        self.assertIn('export GOLEM_GEMM_OUT_LAYOUT', source)
        self.assertIn('export GOLEM_MATMUL_BLOCK_M="$GOLEM_GEMM_BLOCK_M"', source)
        self.assertIn('export GOLEM_MATMUL_BLOCK_N="$GOLEM_GEMM_BLOCK_N"', source)
        self.assertLess(
            source.index('export GOLEM_MATMUL_M="$GOLEM_GEMM_M"'),
            source.index('"$TESTS_DIR/run_noc_dma_pipeline.sh"'),
        )

    def test_wrapper_build_metadata_tracks_sfu_runtime_sources(self):
        source = self.read_wrapper()

        self.assertIn("golem_softmax_sfu_runtime.cpp", source)
        self.assertIn("golem_softmax_sfu_runtime.h", source)
        self.assertIn("test_noc_dma_softmax_sfu.cpp", source)
        self.assertIn("gemm_matmul_op.h", source)
        self.assertIn("test_noc_dma_softmax_sfu.build.env", source)

    def test_sst_shim_sets_ld_library_path_for_local_sst(self):
        with open(os.path.join(SCRIPT_DIR, "bin", "sst"), "r", encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertIn("SST_SOFTMAX_LD_LIBRARY_PATH", source)
        self.assertIn("/data4/jjgong/miniconda3/lib", source)
        self.assertIn("/data4/jjgong/local/sstcore/lib", source)
        self.assertIn('exec "$REAL_SST_BIN" "$@"', source)

    def test_private_wrapper_options_are_not_forwarded_to_base_pipeline(self):
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("Unknown option: --group-manager-enable", result.stdout)
        self.assertNotIn("Unknown option: --ctrl-link-enable", result.stdout)


if __name__ == "__main__":
    unittest.main()
