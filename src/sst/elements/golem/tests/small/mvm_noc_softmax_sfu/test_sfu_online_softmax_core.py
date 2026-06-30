#!/usr/bin/env python3

import os
import re
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLEM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SFU_H = os.path.join(GOLEM_DIR, "sfu", "sfu.h")
SFU_CC = os.path.join(GOLEM_DIR, "sfu", "sfu.cc")


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class SfuOnlineSoftmaxCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = read(SFU_H)
        cls.source = read(SFU_CC)

    def test_operation_state_tracks_tile_stats_and_payload(self):
        self.assertIn("struct SFUTileRowStats", self.header)
        self.assertIn("uint32_t global_row", self.header)
        self.assertIn("double tile_m", self.header)
        self.assertIn("double tile_l", self.header)
        self.assertIn("std::vector<SFUTileRowStats> rowStats", self.header)
        self.assertIn("std::vector<float> inputTile", self.header)
        self.assertIn("bool normalizeReady", self.header)

    def test_sfu_reads_descriptor_and_input_tile_from_global_memory(self):
        self.assertIn("readSoftmaxDescriptor", self.header)
        self.assertIn("readInputTile", self.header)
        self.assertIn("globalMem_->rd_from_globalmem(descAddr", self.source)
        self.assertIn("globalMem_->rd_from_globalmem(desc.local_input_gm_addr", self.source)
        self.assertIn("std::memcpy(desc", self.source)
        self.assertIn("std::memcpy(&value", self.source)
        self.assertNotIn("desc.local_input_gm_addr = descAddr", self.source)

    def test_tile_stats_use_stable_exp_sum_per_valid_row(self):
        self.assertIn("computeTileStats", self.header)
        compute_body = re.search(
            r"void\s+SFU::computeTileStats\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(compute_body)
        body = compute_body.group("body")
        self.assertIn("std::max", body)
        self.assertIn("std::exp", body)
        self.assertIn("tile_l", body)
        self.assertIn("statSoftmaxRows_->addData", body)

    def test_sfu_softmax_uses_gemm_tile_packed_column_layout(self):
        self.assertIn("tilePackedIndex", self.source)
        self.assertIn("static_cast<size_t>(col) * desc.block_m + row", self.source)
        self.assertNotIn("static_cast<size_t>(row) * desc.block_n + col", self.source)

    def test_reducer_uses_online_softmax_merge_and_wait_can_stall(self):
        self.assertIn("struct SoftmaxReducerRowState", self.source)
        self.assertIn("softmaxReducerRows", self.source)
        self.assertIn("mergeTileStats", self.header)
        self.assertIn("normalizeTile", self.header)
        self.assertIn("l_acc * std::exp", self.source)
        self.assertIn("row.tile_l * std::exp", self.source)
        wait_body = re.search(
            r"bool\s+SFU::wait\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(wait_body)
        body = wait_body.group("body")
        self.assertIn("statCrossTileWaitCycles_->addData", body)
        self.assertIn("return false", body)

    def test_normalize_writes_full_row_softmax_tile_to_output_gm(self):
        self.assertIn("globalMem_->wr_to_globalmem", self.source)
        self.assertIn("desc.local_output_gm_addr", self.source)
        self.assertIn("std::exp(static_cast<double>(state->inputTile[idx])", self.source)
        self.assertIn("/ rowState.l_acc", self.source)
        self.assertIn("static_cast<float>(normalized)", self.source)


if __name__ == "__main__":
    unittest.main()
