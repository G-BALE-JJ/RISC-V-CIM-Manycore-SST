import dataclasses
import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("plot_sfu_phase4f_large_scale.py")
SPEC = importlib.util.spec_from_file_location("phase4f_large_scale", SCRIPT)
phase4f = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase4f)


class Phase4FLargeScaleContractTest(unittest.TestCase):
    def test_canonical_network_matches_gemm_preset(self):
        self.assertEqual(
            phase4f.CANONICAL_NETWORK,
            {
                "GOLEM_NOC_LINK_BW": "1200GB/s",
                "GOLEM_NOC_XBAR_BW": "1200GB/s",
                "GOLEM_DIRCTRL_HIGHLINK_BW": "1200GB/s",
                "GOLEM_NOC_INPUT_BUF_SIZE": "512KB",
                "GOLEM_NOC_OUTPUT_BUF_SIZE": "512KB",
                "GOLEM_NOC_FLIT_SIZE": "128B",
                "GOLEM_GM_BUFFER_LENGTH": "1024KB",
                "GOLEM_NOC_INTER_ROUTER_NO_CUT": "0",
                "GOLEM_NOC_LOCAL_NO_CUT": "0",
            },
        )

    def test_transport_and_virtual_networks_are_fixed(self):
        self.assertEqual(phase4f.TRANSPORT, "explicit_noc")
        self.assertEqual(phase4f.NUM_VNS, 3)
        self.assertEqual(phase4f.REDUCTION_VN, 0)
        self.assertEqual(phase4f.DMA_RESPONSE_VN, 0)

    def test_point_models_are_frozen_and_record_fields_are_planned(self):
        self.assertEqual(
            [field.name for field in dataclasses.fields(phase4f.PointSpec)],
            [
                "stage",
                "rows",
                "dim",
                "worker_cores",
                "band_cores",
                "mem_node_size",
                "timeout_sec",
            ],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(phase4f.PointRecord)],
            [
                "spec",
                "run_id",
                "chunk_elems",
                "cooperative_groups",
                "transport",
                "reduction_vn",
                "num_vns",
                "dma_response_vn",
                "noc_link_bw",
                "noc_xbar_bw",
                "dirctrl_highlink_bw",
                "noc_input_buffer",
                "noc_output_buffer",
                "gm_buffer",
                "flit_size",
                "retry_ticks",
                "max_retries",
                "status",
                "exit_code",
                "artifact_validation",
                "golden_checked",
                "golden_mismatches",
                "transport_events",
                "transport_immediate",
                "transport_queued",
                "transport_rejected",
                "transport_stale",
                "inbox_high_water",
                "latency_avg_cycles",
                "latency_max_cycles",
                "total_send_packets",
                "total_send_bits",
                "total_xbar_stalls",
                "simulated_time_us",
                "wall_time_sec",
                "dma_timeout_retry",
                "dma_timeout_exhausted",
                "dma_write_timeout_retry",
                "output_sha256",
                "child_root",
            ],
        )
        self.assertTrue(phase4f.PointSpec.__dataclass_params__.frozen)
        self.assertTrue(phase4f.PointRecord.__dataclass_params__.frozen)
        point = phase4f.DEFAULT_POINTS[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            point.rows = 32

    def test_default_points_have_canonical_order_and_derived_values(self):
        identities = [
            (point.rows, point.dim, point.worker_cores, point.band_cores)
            for point in phase4f.DEFAULT_POINTS
        ]
        self.assertEqual(
            identities,
            [
                (16, 512, 16, 16),
                (16, 1024, 16, 16),
                (16, 2048, 16, 16),
                (16, 4096, 16, 16),
                (16, 4096, 4, 4),
                (16, 4096, 8, 8),
                (64, 4096, 16, 16),
                (256, 4096, 16, 16),
            ],
        )
        self.assertEqual([point.stage for point in phase4f.DEFAULT_POINTS], list("AAAABBCC"))
        self.assertEqual(
            [point.mem_node_size for point in phase4f.DEFAULT_POINTS],
            [134217728, 134217728, 268435456, 268435456,
             268435456, 268435456, 268435456, 268435456],
        )
        self.assertEqual(
            [point.timeout_sec for point in phase4f.DEFAULT_POINTS],
            [900, 1800, 2400, 3600, 3600, 3600, 7200, 14400],
        )
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(identities.count((16, 4096, 16, 16)), 1)

    def test_resolve_point_returns_canonical_identity(self):
        self.assertIs(
            phase4f.resolve_point(16, 4096, 16, 16),
            phase4f.DEFAULT_POINTS[3],
        )
        self.assertEqual(phase4f.resolve_point(16, 4096, 4, 4).stage, "B")
        self.assertEqual(phase4f.resolve_point(256, 4096, 16, 16).stage, "C")

    def test_parse_point_list_uses_defaults_and_preserves_override_order(self):
        self.assertIs(phase4f.parse_point_list(None), phase4f.DEFAULT_POINTS)
        self.assertEqual(
            phase4f.parse_point_list("16:4096:8:8 16:512:16:16"),
            (phase4f.DEFAULT_POINTS[5], phase4f.DEFAULT_POINTS[0]),
        )

    def test_invalid_points_are_rejected(self):
        invalid_values = [
            "",
            "16:512:16",
            "16:512:16:16:1",
            "16:512:sixteen:16",
            "0:512:16:16",
            "16:-512:16:16",
            "16:4096:1:1",
            "16:4096:2:2",
            "16:4096:4:8",
            "32:4096:16:16",
            "16:8192:16:16",
            "16:512:16:16 16:512:16:16",
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                phase4f.parse_point_list(value)

        invalid_args = [
            (0, 512, 16, 16),
            (16, 4096, 1, 1),
            (16, 4096, 4, 8),
            (64, 4096, 8, 8),
        ]
        for args in invalid_args:
            with self.subTest(args=args), self.assertRaises(ValueError):
                phase4f.resolve_point(*args)


if __name__ == "__main__":
    unittest.main()
