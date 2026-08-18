#!/usr/bin/env python3

import importlib.util
import pathlib
import subprocess
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[6]
ROCC = REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalog.h"
SFU_H = REPO_ROOT / "src/sst/elements/golem/sfu/sfu.h"
SFU_CC = REPO_ROOT / "src/sst/elements/golem/sfu/sfu.cc"
GM_H = REPO_ROOT / "src/sst/elements/golem/globalmemory/globalmemory.h"
GUEST_H = HERE / "golem_attention_runtime.h"
GUEST_CC = HERE / "golem_attention_runtime.cpp"
RUNNER = HERE / "run_fused_attention.sh"
CASE_TOOL = HERE / "attention_case.py"
STATS_TOOL = HERE / "verify_fused_attention_stats.py"


def load_case_tool():
    spec = importlib.util.spec_from_file_location("attention_case", CASE_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_stats_tool():
    spec = importlib.util.spec_from_file_location("verify_fused_attention_stats", STATS_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FusedAttentionC1ContractTests(unittest.TestCase):
    def test_attention_v1_has_distinct_manager_issue_wait_and_topology(self):
        rocc = ROCC.read_text()
        guest = GUEST_H.read_text()
        for token in (
            "GolemAttentionDescV1",
            "GOLEM_ATTENTION_DESC_MAGIC",
            "GOLEM_ATTENTION_DESC_VERSION",
            "GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_JOB",
            "GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_WAIT",
            "ManagerAttentionJobState",
            "AttentionWorkerState",
        ):
            self.assertIn(token, rocc + guest)
        self.assertIn("SFUWorkerTopologyMapV1", rocc)

    def test_worker_uses_real_qk_and_pv_arrays_and_local_sfu_tile(self):
        rocc = ROCC.read_text()
        sfu_h = SFU_H.read_text()
        for token in (
            "programMatrixAsync",
            "programInputAsync",
            "beginComputation",
            "readOutputAsync",
            "issueAttentionTile",
            "attention_qk_array_ops",
            "attention_pv_array_ops",
        ):
            self.assertIn(token, rocc + sfu_h)
        self.assertNotIn("compute_attention", rocc)
        self.assertIn("state.panel = 0", rocc)

    def test_fused_tile_uses_bounded_sfu_lanes_and_local_gm_only(self):
        header = SFU_H.read_text()
        source = SFU_CC.read_text()
        self.assertIn("AttentionTileRequest", header)
        self.assertIn("localTileMode", header)
        self.assertIn("issueAttentionTile", source)
        self.assertIn("localReadAsync", source)
        self.assertIn("localWriteAsync", source)
        self.assertIn("rowEngineVectorLanes_", source)

    def test_recycled_physical_context_rebinds_online_row_state(self):
        source = SFU_CC.read_text()
        assign_row = source.split("void SFU::issueTensorInputDma", 1)[1].split(
            "void SFU::beginTensorRowStage", 1
        )[0]
        self.assertIn("worker.attentionKeyTile == 0", assign_row)
        self.assertIn("online.globalRow = context.row", assign_row)

    def test_attention_transport_and_window_are_explicit_and_bounded(self):
        gm = GM_H.read_text()
        rocc = ROCC.read_text()
        self.assertIn("AttentionDispatch", gm)
        self.assertIn("AttentionComplete", gm)
        self.assertIn("ATTENTION_C1_WINDOW_BYTES = 26752", rocc)
        self.assertIn("attentionWindowOffset_", rocc)
        self.assertIn("attentionWindowBytes_", rocc)

    def test_fused_verifier_reports_zero_score_probability_hbm_bytes(self):
        case = load_case_tool()
        traffic = case.attention_logical_hbm_traffic(32, 32, 64, fused=True)
        self.assertEqual(traffic["score_tensor_write_bytes"], 0)
        self.assertEqual(traffic["score_tensor_read_bytes"], 0)
        self.assertEqual(traffic["probability_tensor_write_bytes"], 0)
        self.assertEqual(traffic["probability_tensor_read_bytes"], 0)

    def test_c1_runner_is_fixed_to_single_tile_noncausal_contract(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--queries 32", result.stdout)
        self.assertIn("--keys 32", result.stdout)
        self.assertIn("--head-dim 64", result.stdout)
        self.assertIn("GOLEM_ATTENTION_FUSED=1", result.stdout)
        self.assertIn("GOLEM_GROUP_MANAGER_ENABLE=1", result.stdout)
        self.assertIn("GOLEM_CTRL_LINK_ENABLE=0", result.stdout)
        self.assertIn("verify-attention", result.stdout)
        self.assertIn("--fused", result.stdout)
        self.assertIn("verify_fused_attention_stats.py", result.stdout)

    def test_c1_stats_verifier_rejects_incomplete_second_query_block(self):
        stats = load_stats_tool()
        header = "ComponentName,StatisticName,Sum.u64\n"
        rows = [
            f"{component},{stat},{48 if stat == 'attention_qk_array_ops' else expected}\n"
            for (component, stat), expected in stats.EXPECTED.items()
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "stats.csv"
            path.write_text(header + "".join(rows), encoding="ascii")
            result = stats.verify_stats(path)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["mismatches"]["core1:rocc/attention_qk_array_ops"],
            {"expected": 64, "actual": 48},
        )


if __name__ == "__main__":
    unittest.main()
