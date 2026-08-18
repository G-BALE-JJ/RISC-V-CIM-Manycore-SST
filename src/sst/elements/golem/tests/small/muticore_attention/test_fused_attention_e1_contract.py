#!/usr/bin/env python3

import pathlib
import subprocess
import unittest

from verify_fused_attention_scale_stats import ticks_to_cycles


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[6]
RUNNER = HERE / "run_fused_attention_scale.sh"


class FusedAttentionE1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rocc = (REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalog.h").read_text()
        cls.rocc_float = (
            REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalogFloat.h"
        ).read_text()
        cls.rocc_int = (
            REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalogInt.h"
        ).read_text()
        cls.runtime = (HERE / "golem_attention_runtime.cpp").read_text()
        cls.generator = (
            REPO_ROOT / "src/sst/elements/golem/tests/tools/gen_hbm_init.py"
        ).read_text()
        cls.globalmemory = (
            REPO_ROOT / "src/sst/elements/golem/globalmemory/globalmemory.h"
        ).read_text()
        cls.stats_verifier = (
            HERE / "verify_fused_attention_scale_stats.py"
        ).read_text()

    def test_manager_fans_out_to_explicit_workers_and_deduplicates_completion(self):
        self.assertIn("workerCoreIds", self.rocc)
        self.assertIn("workersDispatched", self.rocc)
        self.assertIn("workersCompleted", self.rocc)
        self.assertIn("message.workerSlot", self.rocc)
        self.assertIn("completionBitmap", self.rocc)

    def test_legacy_single_worker_keeps_the_complete_query_band(self):
        self.assertIn(
            "state.desc.worker_count == 1 ? state.desc.queries", self.rocc
        )

    def test_scale_worker_streams_striped_kv_tiles(self):
        self.assertIn("attentionStreamKv", self.rocc)
        self.assertIn("loadAttentionKeyTile", self.rocc)
        self.assertIn("kv_node_stride_bytes", self.rocc)
        self.assertIn("kv_rows_per_node", self.rocc)
        self.assertIn("ATTENTION_E1_WINDOW_BYTES", self.rocc)

    def test_guest_maps_four_managers_to_sixteen_noncontiguous_workers(self):
        self.assertIn("GOLEM_ATTENTION_SCALE", self.runtime)
        self.assertIn("4 + manager_id", self.runtime)
        self.assertIn("8 + manager_id", self.runtime)
        self.assertIn("12 + manager_id", self.runtime)
        self.assertIn("16 + manager_id", self.runtime)
        self.assertIn("manager_id * GOLEM_ATTENTION_GM_STRIDE", self.runtime)

    def test_hbm_generator_has_four_node_attention_striping(self):
        self.assertIn("ATTENTION_HBM_STRIPED", self.generator)
        self.assertIn("attention_q_band", self.generator)
        self.assertIn("attention_kv_band", self.generator)

    def test_scale_runner_selects_s256_four_manager_configuration(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "--scale-point", "e2", "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fused_attention_e2_s256_d64", result.stdout)
        self.assertIn("--groups 4", result.stdout)
        self.assertIn("--num-cores 20", result.stdout)
        self.assertIn("--num-mem-nodes 5", result.stdout)
        self.assertIn("GOLEM_DMA_READ_RETRY_TICKS=4096", result.stdout)
        self.assertIn("GOLEM_DMA_READ_MAX_RETRIES=32", result.stdout)
        self.assertIn("verify_fused_attention_scale_stats.py", result.stdout)

    def test_e3_runner_selects_s1024_d128(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "--scale-point", "e3", "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fused_attention_e3_s1024_d128", result.stdout)
        self.assertIn("--queries 1024 --keys 1024 --head-dim 128", result.stdout)
        self.assertIn("GOLEM_ATTENTION_HEAD_DIM=128", result.stdout)
        self.assertIn("--array-in 128", result.stdout)
        self.assertIn("scale-e3", result.stdout)

    def test_e3_worker_uses_dynamic_head_dimension_and_eight_pv_panels(self):
        self.assertIn("attentionDimensionPanels", self.rocc)
        self.assertIn("state.dispatch.headDim * sizeof(float)", self.rocc)
        self.assertIn("request.headDim = state.dispatch.headDim", self.rocc)
        self.assertIn("ATTENTION_E3_WINDOW_BYTES", self.rocc)

    def test_e3_manager_distributes_sixty_four_rows_per_worker(self):
        self.assertIn("rowsPerWorker", self.rocc)
        self.assertIn("desc.queries == 256 && desc.keys == 1024", self.rocc)
        self.assertIn("desc.head_dim == 128", self.rocc)

    def test_e3_generator_and_stats_are_shape_aware(self):
        self.assertIn("ATTENTION_HEAD_DIM", self.generator)
        self.assertIn('"e3"', self.stats_verifier)
        self.assertIn("65536", self.stats_verifier)

    def test_e4_runner_selects_s2048_d128(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fused_attention_e4_s2048_d128", result.stdout)
        self.assertIn("--queries 2048 --keys 2048 --head-dim 128", result.stdout)
        self.assertIn("--band-rows 512", result.stdout)
        self.assertIn("scale-e4", result.stdout)

    def test_e4_rocc_accepts_bounded_s2048_worker_and_manager_shapes(self):
        self.assertIn(
            "message.expectedRows == 128 && message.expectedCols == 2048", self.rocc
        )
        self.assertIn("desc.queries == 512 && desc.keys == 2048", self.rocc)
        self.assertIn("desc.kv_rows_per_node == 512", self.rocc)
        self.assertIn("ATTENTION_E3_WINDOW_BYTES", self.rocc)

    def test_e4_stats_profile_has_exact_per_worker_activity(self):
        self.assertIn('"e4"', self.stats_verifier)
        self.assertIn('"qk": 16384', self.stats_verifier)
        self.assertIn('"pv": 65536', self.stats_verifier)
        self.assertIn('"jobs": 512', self.stats_verifier)
        self.assertIn('"rows": 8192', self.stats_verifier)
        self.assertIn('"scaled": 262144', self.stats_verifier)

    def test_scale_descriptor_names_root_and_manager_slot(self):
        self.assertIn("tensor_root_core", self.runtime)
        self.assertIn("tensor_manager_slot", self.runtime)
        self.assertIn("tensor_manager_count", self.runtime)

    def test_manager_band_completion_uses_distinct_transport_message(self):
        self.assertIn("AttentionManagerComplete", self.globalmemory)
        self.assertIn("managerCompletionBitmap", self.rocc)
        self.assertIn("handleAttentionManagerBandCompletion", self.rocc)

    def test_scale_stats_require_one_root_tensor_completion(self):
        self.assertIn("attention_tensor_jobs_completed", self.stats_verifier)
        self.assertIn("attention_manager_bands_completed", self.stats_verifier)
        self.assertIn("core == 0", self.stats_verifier)

    def test_attention_lifecycle_stats_define_strict_accelerator_completion(self):
        for statistic in (
            "attention_manager_descriptor_accept_tick",
            "attention_manager_dispatch_tick",
            "attention_manager_local_complete_tick",
            "attention_manager_band_completion_received_tick",
            "attention_tensor_complete_tick",
            "attention_manager_wait_observed_tick",
        ):
            self.assertIn(statistic, self.rocc)
            self.assertIn(statistic, self.rocc_float)
            self.assertIn(statistic, self.rocc_int)
            self.assertIn(statistic, self.stats_verifier)
        self.assertIn("accelerator_completion_cycles", self.stats_verifier)
        self.assertIn("wait_return_cycles", self.stats_verifier)
        self.assertIn("attention_lifecycle.json", RUNNER.read_text())

    def test_lifecycle_ticks_are_converted_to_accelerator_cycles(self):
        self.assertEqual(ticks_to_cycles(97_176_395, 2_300_000_000, 10**12), 223_506)
        self.assertEqual(ticks_to_cycles(0, 2_300_000_000, 10**12), 0)
        dry_run = subprocess.run(
            ["bash", str(RUNNER), "--dry-run"], text=True,
            capture_output=True, check=False
        )
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertIn("--accelerator-clock 1.0GHz", dry_run.stdout)


if __name__ == "__main__":
    unittest.main()
