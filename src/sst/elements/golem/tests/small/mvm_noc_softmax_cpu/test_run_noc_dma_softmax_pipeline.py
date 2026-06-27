#!/usr/bin/env python3
import os
import subprocess
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCH_SHIM = os.path.join(SCRIPT_DIR, "ncores_selfcom_dma_softmax_archive.py")
SOFTMAX_ENTRY = os.path.join(SCRIPT_DIR, "test_noc_dma_softmax.cpp")


class SoftmaxPipelineWrapperTest(unittest.TestCase):
    def run_wrapper(self, *args):
        env = os.environ.copy()
        env.pop("GOLEM_ARCH_SCRIPT", None)
        cmd = [
            os.path.join(SCRIPT_DIR, "run_noc_dma_softmax_pipeline.sh"),
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

    def test_16core_ctrl_link_disabled_uses_non_ctrl_architecture(self):
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_CTRL_LINK_ENABLE=0", result.stdout)
        self.assertIn("GOLEM_GROUP_MANAGER_ENABLE=0", result.stdout)
        self.assertIn("GOLEM_REQUEST_SCHEDULER_ENABLE=0", result.stdout)
        self.assertIn("GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0", result.stdout)
        self.assertIn("small/mvm_noc_softmax_cpu/ncores_selfcom_dma_softmax_archive.py", result.stdout)
        self.assertNotIn("architecture/ncores_selfcom_dma_ctrl.py", result.stdout)

    def test_16core_archive_architecture_exports_memory_routers_before_sst(self):
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_MEMORY_ROUTERS=24,0,1,2,3,4,5,6,7", result.stdout)

    def test_16core_archive_architecture_uses_realistic_dma_retry_window(self):
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_DMA_READ_RETRY_TICKS=256", result.stdout)

    def test_probability_reference_enables_fast_single_core_probability_mode(self):
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_SOFTMAX_VERIFY_REFERENCE=probability", result.stdout)
        self.assertIn("GOLEM_SOFTMAX_FAST_PROBABILITY=1", result.stdout)

    def test_archive_shim_expands_directory_memnic_buffers_for_dma_responses(self):
        with open(ARCH_SHIM, "r", encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertIn('"network_input_buffer_size": os.getenv("GOLEM_NOC_INPUT_BUF_SIZE", "512KB")', source)
        self.assertIn('"network_output_buffer_size": os.getenv("GOLEM_NOC_OUTPUT_BUF_SIZE", "512KB")', source)
        self.assertIn('"golem_dma_response_drain_limit": os.getenv("GOLEM_DMA_RESPONSE_DRAIN_LIMIT", "0")', source)

    def test_private_wrapper_options_are_not_forwarded_to_base_pipeline(self):
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("Unknown option: --group-manager-enable", result.stdout)
        self.assertNotIn("Unknown option: --ctrl-link-enable", result.stdout)

    def test_riscv_entry_uses_requested_core_id_as_logical_core(self):
        with open(SOFTMAX_ENTRY, "r", encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertIn(
            "const int softmax_core_id = read_requested_core_from_argv(argc, argv);",
            source,
        )
        self.assertIn("const int executor_core_id = bind_and_resolve_core_from_argv_or_exit(argc, argv, TOTAL_CORES);", source)
        self.assertIn("return run_tile_local_softmax_for_core(executor_core_id, softmax_core_id, op_desc);", source)
        self.assertNotIn("run_tile_local_softmax_for_core(core_id, op_desc);", source)

    def test_single_core_softmax_uses_executor_core_for_local_dma_and_task_desc_for_tile_addresses(self):
        with open(SOFTMAX_ENTRY, "r", encoding="utf-8") as source_file:
            entry_source = source_file.read()
        with open(os.path.join(SCRIPT_DIR, "golem_softmax_single_core.cpp"), "r", encoding="utf-8") as source_file:
            single_core_source = source_file.read()

        self.assertIn("int run_tile_local_softmax_for_core(int executor_core_id, int softmax_core_id", entry_source)
        self.assertIn("executor_core_id,", entry_source)
        self.assertIn("softmax_core_id,", entry_source)
        self.assertIn("const int64_t m_tile = row / block_m;", single_core_source)
        self.assertIn("const int tile_task_id = static_cast<int>(m_tile * n_tiles + n_tile);", single_core_source)
        self.assertIn("const GemmTaskDescriptor tile_desc = gemm_task_desc_for_task(executor_core_id, tile_task_id, cfg);", single_core_source)
        self.assertIn("dma_remote_load_to_gm(executor_core_id, row_hbm_addr, local_tmp_gm, row_bytes);", single_core_source)

    def test_single_core_softmax_has_fast_probability_path_for_smoke_tests(self):
        with open(os.path.join(SCRIPT_DIR, "golem_softmax_single_core.cpp"), "r", encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertIn('std::getenv("GOLEM_SOFTMAX_FAST_PROBABILITY")', source)
        self.assertIn("row_data[max_col] = 1.0f;", source)
        self.assertIn("if (fast_probability_mode)", source)

    def test_makefile_rebuilds_binary_when_single_core_softmax_changes(self):
        with open(os.path.join(SCRIPT_DIR, "Makefile"), "r", encoding="utf-8") as source_file:
            source = source_file.read()

        target_line = "$(OUT_DIR)/$(PROG):"
        dependency_lines = [line for line in source.splitlines() if line.startswith(target_line)]
        self.assertEqual(len(dependency_lines), 1, source)
        self.assertIn("golem_softmax_single_core.cpp", dependency_lines[0])
        self.assertIn("golem_softmax_single_core.h", dependency_lines[0])


if __name__ == "__main__":
    unittest.main()
