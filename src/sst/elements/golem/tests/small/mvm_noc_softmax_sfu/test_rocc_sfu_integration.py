#!/usr/bin/env python3

import os
import re
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLEM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
ROCC = os.path.join(GOLEM_DIR, "rocc", "roccAnalog.h")


def read_rocc():
    with open(ROCC, "r", encoding="utf-8") as f:
        return f.read()


class RoCCSfuIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = read_rocc()

    def test_existing_gemm_func7_values_are_unchanged(self):
        expected = {
            "GOLEM_ROCC_FUNC7_TILE_MVM_BATCH": "0x11",
            "GOLEM_ROCC_FUNC7_TILE_WAIT_BATCH": "0x12",
            "GOLEM_ROCC_FUNC7_TILE_GM2IMAT_BCAST": "0x13",
            "GOLEM_ROCC_FUNC7_TILE_GM2IVEC_BATCH": "0x14",
            "GOLEM_ROCC_FUNC7_WCP_START": "0x15",
            "GOLEM_ROCC_FUNC7_WCP_WAIT": "0x16",
        }
        for name, value in expected.items():
            self.assertRegex(
                self.text,
                rf"constexpr\s+uint8_t\s+{name}\s*=\s*{value}\s*;",
                f"{name} must remain {value}",
            )

    def test_declares_sfu_func7_values_after_existing_gemm_values(self):
        self.assertRegex(
            self.text,
            r"constexpr\s+uint8_t\s+GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE\s*=\s*0x17\s*;",
        )
        self.assertRegex(
            self.text,
            r"constexpr\s+uint8_t\s+GOLEM_ROCC_FUNC7_SFU_WAIT\s*=\s*0x18\s*;",
        )
        self.assertRegex(
            self.text,
            r"constexpr\s+uint8_t\s+GOLEM_ROCC_FUNC7_SFU_PRIMITIVE\s*=\s*0x19\s*;",
        )
        self.assertRegex(
            self.text,
            r"constexpr\s+uint8_t\s+GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_WAIT\s*=\s*0x1a\s*;",
        )

    def test_loads_sfu_only_when_enabled_and_binds_resources(self):
        self.assertIn("#include <sst/elements/golem/sfu/sfu.h>", self.text)
        self.assertIn('params.find<int>("sfuEnable", 0)', self.text)
        self.assertIn('loadUserSubComponent<SST::Golem::SFUAPI>', self.text)
        self.assertIn('"sfu"', self.text)
        self.assertIn("sfu->bindGlobalMemory(globalMem)", self.text)
        self.assertIn("sfu->setCoreInfo", self.text)
        self.assertIn("sfuEnable=1", self.text)

    def test_sfu_has_lifecycle_and_member_pointer(self):
        self.assertIn("SST::Golem::SFUAPI *sfu", self.text)
        self.assertIn("if (sfu) {\n        sfu->init(phase);", self.text)
        self.assertIn("if (sfu) {\n            sfu->setup();", self.text)
        self.assertIn("if (sfu) {\n            sfu->complete(phase);", self.text)
        self.assertIn("if (sfu) {\n            sfu->finish();", self.text)

    def test_tick_dispatches_sfu_commands_without_using_busy_path(self):
        self.assertIn("tryIssueSfuSoftmaxTileCommand", self.text)
        self.assertIn("tryWaitSfuCommand", self.text)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE", self.text)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_WAIT", self.text)
        self.assertIn("sfu->issueSoftmaxTile", self.text)
        self.assertIn("sfu->wait", self.text)

    def test_tick_dispatches_sfu_primitive_commands_without_using_busy_path(self):
        self.assertIn("tryIssueSfuPrimitiveCommand", self.text)
        self.assertIn("tryWaitSfuPrimitiveCommand", self.text)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_PRIMITIVE", self.text)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_WAIT", self.text)
        self.assertIn("sfu->issuePrimitive", self.text)
        self.assertIn("sfu->wait", self.text)

    def test_tick_dispatches_sfu_primitive_batch_commands_without_using_busy_path(self):
        self.assertIn("tryIssueSfuPrimitiveBatchCommand", self.text)
        self.assertIn("tryWaitSfuPrimitiveBatchCommand", self.text)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH", self.text)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH_WAIT", self.text)
        self.assertIn("sfu->issuePrimitiveBatch", self.text)
        self.assertIn("sfu->wait", self.text)


if __name__ == "__main__":
    unittest.main()
