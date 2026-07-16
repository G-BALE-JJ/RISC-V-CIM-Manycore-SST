#!/usr/bin/env python3

import os
import re
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SST_ELEMENTS_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "..")
)
MEMNIC_BASE = os.path.join(
    SST_ELEMENTS_DIR, "memHierarchy", "memNICBase.h"
)


def read_memnic_base():
    with open(MEMNIC_BASE, "r", encoding="utf-8") as source:
        return source.read()


class MemNicDmaResponseVnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = read_memnic_base()

    def test_declares_dma_response_vn_as_an_eli_parameter(self):
        self.assertRegex(
            self.text,
            r'\{\s*"golem_dma_response_vn"\s*,[^\n]+\}',
        )

    def test_distinguishes_explicit_value_from_derived_default(self):
        self.assertIn("bool golem_dma_response_vn_found", self.text)
        self.assertRegex(
            self.text,
            r'params\.find<uint32_t>\(\s*"golem_dma_response_vn"\s*,[\s\S]*?golem_dma_response_vn_found\s*\)',
        )

    def test_unset_value_derives_from_num_vns(self):
        self.assertRegex(
            self.text,
            r'golem_dma_response_vn_found[\s\S]*?num_vns\s*>=\s*2\s*\?\s*1\s*:\s*0',
        )

    def test_rejects_a_response_vn_outside_num_vns(self):
        self.assertRegex(
            self.text,
            r'golem_dma_response_vn\s*>=\s*num_vns[\s\S]*?fatal\(',
        )
        self.assertIn("golem_dma_response_vn", self.text)

    def test_trace_reports_resolved_response_vn_and_num_vns(self):
        self.assertIn("resolved golem_dma_response_vn=%", self.text)
        self.assertIn("num_vns=%", self.text)


if __name__ == "__main__":
    unittest.main()
