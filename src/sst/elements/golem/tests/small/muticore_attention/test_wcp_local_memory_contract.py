#!/usr/bin/env python3

import pathlib
import re
import unittest


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[6]
WCP = REPO_ROOT / "src/sst/elements/golem/workercmdproc/workercmdproc.h"


class WcpLocalMemoryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WCP.read_text()

    def test_operand_load_is_callback_ordered_local_memory_access(self):
        load = re.search(
            r"bool beginTilePayloadLoad\(.*?bool beginArrayProgramming",
            self.source,
            re.S,
        )
        self.assertIsNotNone(load)
        self.assertIn("localReadAsync", load.group(0))
        self.assertIn("LocalMemoryClient::WCP", load.group(0))
        self.assertNotIn("rd_from_globalmem", load.group(0))
        self.assertIn("operandLoadPending_", self.source)

    def test_partial_c_spills_to_reserved_local_accumulator_window(self):
        self.assertNotIn("partialCTiles_", self.source)
        self.assertIn("partialCAddress", self.source)
        self.assertIn("header_.local_accum_gm_addr", self.source)
        self.assertIn("beginPartialCStore", self.source)
        self.assertIn("beginPartialCLoad", self.source)
        self.assertIn("localWriteAsync", self.source)
        self.assertIn("localReadAsync", self.source)


if __name__ == "__main__":
    unittest.main()
