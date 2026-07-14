#!/usr/bin/env python3

import os
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLEM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
CPU_BUILDER = os.path.join(GOLEM_DIR, "tests", "architecture", "cpu_builder.py")


def read_cpu_builder():
    with open(CPU_BUILDER, "r", encoding="utf-8") as f:
        return f.read()


class CpuBuilderSfuMountTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = read_cpu_builder()

    def test_declares_sfu_enable_env_with_default_disabled(self):
        self.assertIn('sfu_enable = int(os.getenv("GOLEM_SFU_ENABLE", "0")) != 0', self.text)

    def test_passes_sfu_enable_param_to_rocc(self):
        self.assertIn('cpu_rocc.addParam("sfuEnable", 1 if sfu_enable else 0)', self.text)

    def test_passes_active_worker_count_to_rocc_and_sfu(self):
        self.assertIn(
            'cpu_rocc.addParam("active_worker_cores", active_worker_cores)',
            self.text,
        )
        self.assertIn('"active_worker_cores": active_worker_cores', self.text)

    def test_mounts_sfu_only_when_enabled(self):
        self.assertIn("if sfu_enable:", self.text)
        self.assertIn('sfu = cpu_rocc.setSubComponent("sfu", "golem.SFU")', self.text)
        self.assertIn('"core_id": cpuId', self.text)
        self.assertIn('"active_worker_cores": active_worker_cores', self.text)
        self.assertIn('sfu_max_inflight = os.getenv("GOLEM_SFU_MAX_INFLIGHT", "8")', self.text)
        self.assertIn('"max_inflight": sfu_max_inflight', self.text)

    def test_passes_distributed_reduction_transport_to_sfu(self):
        self.assertIn(
            'sfu_distributed_reduction_transport = os.getenv("GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT", "shared")',
            self.text,
        )
        self.assertIn(
            '"distributed_reduction_transport": sfu_distributed_reduction_transport',
            self.text,
        )

    def test_passes_reduction_vn_only_to_global_memory(self):
        self.assertIn(
            'sfu_reduction_vn = os.getenv("GOLEM_SFU_REDUCTION_VN", "")',
            self.text,
        )
        self.assertIn('gm_params["reduction_vn"] = sfu_reduction_vn', self.text)
        self.assertNotIn('"reduction_vn": sfu_reduction_vn', self.text)


if __name__ == "__main__":
    unittest.main()
