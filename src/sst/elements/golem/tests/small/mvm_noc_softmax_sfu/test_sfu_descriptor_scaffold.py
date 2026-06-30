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


class SfuDescriptorScaffoldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = read(SFU_H)
        cls.source = read(SFU_CC)

    def test_declares_softmax_tile_descriptor_abi(self):
        self.assertIn("struct SFUSoftmaxTileDesc", self.header)
        for field in [
            "uint64_t job_id",
            "uint64_t local_input_gm_addr",
            "uint64_t local_output_gm_addr",
            "uint32_t global_m",
            "uint32_t global_n",
            "uint32_t block_m",
            "uint32_t block_n",
            "uint32_t m_tile",
            "uint32_t n_tile",
            "uint32_t valid_m",
            "uint32_t valid_n",
            "uint32_t n_tiles_per_row",
            "uint32_t elem_bytes",
            "uint32_t flags",
        ]:
            self.assertIn(field, self.header)
        self.assertRegex(self.header, r"static_assert\s*\(\s*sizeof\s*\(\s*SFUSoftmaxTileDesc\s*\)\s*==\s*72")

    def test_declares_softmax_status_values_and_inflight_record(self):
        self.assertIn("enum class SFUStatus", self.header)
        self.assertIn("Success = 0", self.header)
        self.assertIn("InvalidDescriptor = 2", self.header)
        self.assertIn("UnsupportedElemBytes = 3", self.header)
        self.assertIn("struct SoftmaxOpState", self.header)
        self.assertIn("SFUSoftmaxTileDesc desc", self.header)
        self.assertIn("std::unordered_map<uint64_t, SoftmaxOpState>", self.header)

    def test_issue_records_descriptor_addr_and_tag_instead_of_ignoring_them(self):
        issue_body = re.search(
            r"bool\s+SFU::issueSoftmaxTile\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(issue_body)
        body = issue_body.group("body")
        self.assertNotIn("(void)descAddr", body)
        self.assertNotIn("(void)tag", body)
        self.assertIn("descAddr", body)
        self.assertIn("tag", body)
        self.assertIn("pendingSoftmaxOps_", body)

    def test_wait_uses_tagged_status_and_retires_operation(self):
        wait_body = re.search(
            r"bool\s+SFU::wait\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(wait_body)
        body = wait_body.group("body")
        self.assertNotIn("(void)tag", body)
        self.assertIn("pendingSoftmaxOps_.find(tag)", body)
        self.assertIn("pendingSoftmaxOps_.erase", body)
        self.assertIn("static_cast<uint64_t>", body)


if __name__ == "__main__":
    unittest.main()
