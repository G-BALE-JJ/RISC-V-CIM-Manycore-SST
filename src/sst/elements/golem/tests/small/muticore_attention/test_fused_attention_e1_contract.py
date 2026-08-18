#!/usr/bin/env python3

import pathlib
import subprocess
import unittest

import attention_case
import verify_fused_attention_scale_output
from verify_fused_attention_scale_stats import (
    PROFILES,
    summarize_worker_critical_path,
    ticks_to_cycles,
)


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
        cls.compute_array = (
            REPO_ROOT / "src/sst/elements/golem/array/computeArray.h"
        ).read_text()
        cls.mvm_array = (
            REPO_ROOT / "src/sst/elements/golem/array/mvmComputeArray.h"
        ).read_text()
        cls.crosssim_array = (
            REPO_ROOT / "src/sst/elements/golem/array/crossSimComputeArray.h"
        ).read_text()
        cls.cpu_builder = (
            REPO_ROOT / "src/sst/elements/golem/tests/architecture/cpu_builder.py"
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

    def test_e5_runner_selects_s4096_d128_with_expensive_run_gate(self):
        dry_run = subprocess.run(
            ["bash", str(RUNNER), "--scale-point", "e5", "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertIn("fused_attention_e5_s4096_d128", dry_run.stdout)
        self.assertIn("--queries 4096 --keys 4096 --head-dim 128", dry_run.stdout)
        self.assertIn("--band-rows 1024", dry_run.stdout)
        self.assertIn("scale-e5", dry_run.stdout)
        self.assertIn("timeout 28800", dry_run.stdout)

        blocked = subprocess.run(
            ["bash", str(RUNNER), "--scale-point", "e5"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(blocked.returncode, 2)
        self.assertIn("--allow-expensive", blocked.stderr)

    def test_e5_rocc_accepts_bounded_s4096_worker_and_manager_shapes(self):
        self.assertIn(
            "message.expectedRows == 256 && message.expectedCols == 4096", self.rocc
        )
        self.assertIn("desc.queries == 1024 && desc.keys == 4096", self.rocc)
        self.assertIn("desc.kv_rows_per_node == 1024", self.rocc)
        self.assertIn("ATTENTION_E3_WINDOW_BYTES", self.rocc)

    def test_e5_stats_profile_has_exact_per_worker_activity(self):
        self.assertEqual(PROFILES["e5"], {
            "qk": 65536,
            "pv": 262144,
            "jobs": 2048,
            "qblocks": 16,
            "rows": 32768,
            "scaled": 1048576,
        })

    def test_scale_output_blocked_reference_matches_scalar_attention(self):
        queries, keys, head_dim = 5, 7, 4
        q = [((index * 5) % 17 - 8) / 16.0
             for index in range(queries * head_dim)]
        k = [((index * 7) % 19 - 9) / 16.0
             for index in range(keys * head_dim)]
        v = [((index * 11) % 23 - 11) / 16.0
             for index in range(keys * head_dim)]

        scalar = attention_case.compute_attention(
            q, k, v, queries, keys, head_dim, False
        )
        blocked = verify_fused_attention_scale_output.compute_attention_blocked(
            q, k, v, queries, keys, head_dim, query_block_rows=2
        )
        self.assertEqual(len(blocked), len(scalar))
        for got, want in zip(blocked, scalar):
            self.assertAlmostEqual(got, want, places=12)

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
            "attention_worker_dispatch_accept_tick",
            "attention_worker_qk_tile_complete_tick",
            "attention_worker_softmax_tile_complete_tick",
            "attention_worker_pv_tile_complete_tick",
            "attention_worker_output_dma_ack_tick",
        ):
            self.assertIn(statistic, self.rocc)
            self.assertIn(statistic, self.rocc_float)
            self.assertIn(statistic, self.rocc_int)
            self.assertIn(statistic, self.stats_verifier)
        self.assertIn("accelerator_completion_cycles", self.stats_verifier)
        self.assertIn("wait_return_cycles", self.stats_verifier)
        self.assertIn("system_frontier", self.stats_verifier)
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

    def test_worker_critical_path_uses_one_slowest_worker(self):
        observed = {
            ("core4:rocc", "attention_worker_dispatch_accept_tick"): 105,
            ("core5:rocc", "attention_worker_dispatch_accept_tick"): 110,
            ("core5:rocc", "attention_worker_qk_tile_complete_tick"): 370,
            ("core5:rocc", "attention_worker_softmax_tile_complete_tick"): 400,
            ("core5:rocc", "attention_worker_pv_tile_complete_tick"): 470,
        }
        minima = {
            ("core5:rocc", "attention_worker_qk_tile_complete_tick"): 150,
        }
        maxima = {
            ("core4:rocc", "attention_worker_qk_tile_complete_tick"): 250,
            ("core4:rocc", "attention_worker_softmax_tile_complete_tick"): 270,
            ("core4:rocc", "attention_worker_pv_tile_complete_tick"): 280,
            ("core4:rocc", "attention_worker_output_dma_ack_tick"): 300,
            ("core5:rocc", "attention_worker_qk_tile_complete_tick"): 220,
            ("core5:rocc", "attention_worker_softmax_tile_complete_tick"): 240,
            ("core5:rocc", "attention_worker_pv_tile_complete_tick"): 290,
            ("core5:rocc", "attention_worker_output_dma_ack_tick"): 310,
        }

        critical_path = summarize_worker_critical_path(
            observed, minima, maxima, worker_cores=range(4, 6),
            accelerator_clock_hz=1_000, timebase_ticks_per_second=1_000,
        )

        self.assertEqual(critical_path["slowest_worker_core"], 5)
        self.assertEqual(
            critical_path["milestone_ticks"],
            {
                "dispatch_accept": 110,
                "final_qk_tile_complete": 220,
                "final_softmax_tile_complete": 240,
                "final_pv_tile_complete": 290,
                "final_output_dma_ack": 310,
            },
        )
        self.assertEqual(
            critical_path["stage_cycles"],
            {
                "dispatch_to_final_qk": 110,
                "final_qk_to_final_softmax": 20,
                "final_softmax_to_final_pv": 50,
                "final_pv_to_output_dma_ack": 20,
            },
        )
        self.assertEqual(
            critical_path["aggregate_online_pipeline_cycles"],
            {
                "dispatch_to_first_qk": 40,
                "all_qk_to_softmax": 30,
                "all_softmax_to_pv": 70,
                "inter_tile_pv_to_next_qk": 40,
                "final_pv_to_output_dma_ack": 20,
            },
        )

    def test_pv_matrix_broadcast_is_explicit_and_opt_in(self):
        for source in (self.compute_array, self.mvm_array, self.crosssim_array):
            self.assertIn("programMatrixGroupAsync", source)
        self.assertIn("attention_pv_matrix_broadcast", self.rocc)
        self.assertIn("GOLEM_ATTENTION_PV_MATRIX_BROADCAST", self.cpu_builder)
        self.assertIn("attention_pv_matrix_broadcasts", self.stats_verifier)

        default_run = subprocess.run(
            ["bash", str(RUNNER), "--scale-point", "e2", "--dry-run"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(default_run.returncode, 0, default_run.stderr)
        self.assertIn("GOLEM_ATTENTION_PV_MATRIX_BROADCAST=0", default_run.stdout)

        optimized_run = subprocess.run(
            [
                "bash", str(RUNNER), "--scale-point", "e2",
                "--pv-matrix-broadcast", "--dry-run",
            ],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(optimized_run.returncode, 0, optimized_run.stderr)
        self.assertIn("GOLEM_ATTENTION_PV_MATRIX_BROADCAST=1", optimized_run.stdout)
        self.assertIn("--pv-matrix-broadcast", optimized_run.stdout)


if __name__ == "__main__":
    unittest.main()
