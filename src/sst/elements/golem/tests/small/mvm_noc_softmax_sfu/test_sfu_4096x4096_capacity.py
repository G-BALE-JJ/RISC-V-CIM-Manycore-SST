#!/usr/bin/env python3

import pathlib
import sys
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sfu_4096x4096_capacity as capacity


class CapacityContractTest(unittest.TestCase):
    def test_default_points_are_the_fixed_ordered_ladder(self):
        self.assertEqual(
            tuple(
                (
                    point.rows,
                    point.dim,
                    point.worker_cores,
                    point.band_cores,
                    point.mem_node_size,
                    point.timeout_sec,
                    point.rowmajor_region_end,
                )
                for point in capacity.DEFAULT_POINTS
            ),
            (
                (512, 4096, 16, 16, 268435456, 3600, 37748736),
                (1024, 4096, 16, 16, 268435456, 7200, 58720256),
                (2048, 4096, 16, 16, 268435456, 10800, 100663296),
                (4096, 4096, 16, 16, 268435456, 14400, 184549376),
            ),
        )
        self.assertEqual(capacity.parse_point_list(None), capacity.DEFAULT_POINTS)

    def test_point_list_accepts_only_nonempty_ordered_prefixes(self):
        for length in range(1, len(capacity.DEFAULT_POINTS) + 1):
            value = " ".join(
                f"{point.rows}:{point.dim}:{point.worker_cores}:{point.band_cores}"
                for point in capacity.DEFAULT_POINTS[:length]
            )
            self.assertEqual(
                capacity.parse_point_list(value), capacity.DEFAULT_POINTS[:length]
            )

        invalid = (
            "",
            "1024:4096:16:16",
            "512:4096:16:16 2048:4096:16:16",
            "512:4096:16:16 512:4096:16:16",
            "512:8192:16:16",
            "512:4096:8:8",
            "512:4096:16:8",
            "4097:4096:16:16",
            "512:4096:16",
            "rows:4096:16:16",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                capacity.parse_point_list(value)

    def test_final_point_capacity_and_counter_formulas(self):
        evidence = capacity.derive_capacity(capacity.DEFAULT_POINTS[-1])
        self.assertEqual(evidence.elements, 16_777_216)
        self.assertEqual(evidence.tensor_bytes, 67_108_864)
        self.assertEqual(evidence.expected_reduction_each, 65_536)
        self.assertEqual(evidence.expected_transport_total, 262_144)
        self.assertEqual(evidence.expected_dma_ops, 65_536)
        self.assertEqual(evidence.expected_dma_bytes, 67_108_864)
        self.assertEqual(evidence.bias_base, 268_419_072)
        self.assertEqual(evidence.layout_margin_bytes, 83_869_696)

    def test_every_point_fits_the_fixed_memory_node(self):
        for point in capacity.DEFAULT_POINTS:
            with self.subTest(rows=point.rows):
                evidence = capacity.derive_capacity(point)
                self.assertGreater(evidence.layout_margin_bytes, 0)
                self.assertLess(point.rowmajor_region_end, evidence.bias_base)


if __name__ == "__main__":
    unittest.main()
