#!/usr/bin/env python3

import os
import pathlib
import sys
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TOOLS_DIR = os.path.join(TESTS_DIR, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from softmax_hbm_layout import softmax_row_location


class SoftmaxHbmLayoutTest(unittest.TestCase):
    def test_band_striping_keeps_each_tile_band_contiguous(self):
        nodes = [1, 2, 3, 4]
        self.assertEqual(softmax_row_location(0, 64, nodes, "band_striped"), (1, 0))
        self.assertEqual(softmax_row_location(63, 64, nodes, "band_striped"), (1, 63))
        self.assertEqual(softmax_row_location(64, 64, nodes, "band_striped"), (2, 0))
        self.assertEqual(softmax_row_location(255, 64, nodes, "band_striped"), (4, 63))
        self.assertEqual(softmax_row_location(256, 64, nodes, "band_striped"), (1, 64))

    def test_single_node_layout_remains_backward_compatible(self):
        self.assertEqual(softmax_row_location(255, 64, [1, 2, 3, 4], "single_node"), (1, 255))

    def test_striped_alias_is_shared_with_guest(self):
        guest = pathlib.Path(
            os.path.join(SCRIPT_DIR, "..", "mvm_noc_softmax_sfu", "test_noc_dma_softmax_sfu.cpp"),
        ).read_text()
        self.assertEqual(softmax_row_location(64, 64, [1, 2, 3, 4], "striped"), (2, 0))
        self.assertIn('std::strcmp(softmax_hbm_layout, "striped") == 0', guest)

    def test_generator_unpacker_and_guest_use_the_layout_contract(self):
        generator = pathlib.Path(TOOLS_DIR, "gen_hbm_init.py").read_text()
        unpacker = pathlib.Path(TOOLS_DIR, "unpack_c_from_hbm.py").read_text()
        guest = pathlib.Path(
            os.path.join(SCRIPT_DIR, "..", "mvm_noc_softmax_sfu", "test_noc_dma_softmax_sfu.cpp"),
        ).read_text()
        runner = pathlib.Path(SCRIPT_DIR, "run_muticore_softmax.sh").read_text()
        self.assertIn("from softmax_hbm_layout import softmax_row_location", generator)
        self.assertIn("from softmax_hbm_layout import softmax_row_location", unpacker)
        self.assertIn("GOLEM_SFU_SOFTMAX_HBM_LAYOUT", guest)
        self.assertIn("GOLEM_SFU_SOFTMAX_HBM_LAYOUT=band_striped", runner)


if __name__ == "__main__":
    unittest.main()
