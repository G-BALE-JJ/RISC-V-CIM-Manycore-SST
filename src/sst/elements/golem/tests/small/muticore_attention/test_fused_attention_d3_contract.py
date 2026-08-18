#!/usr/bin/env python3

import importlib.util
import pathlib
import subprocess
import unittest


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[6]
ROCC = REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalog.h"
RUNNER = HERE / "run_fused_attention_online.sh"
STATS_TOOL = HERE / "verify_fused_attention_online_stats.py"


def load_stats_tool():
    spec = importlib.util.spec_from_file_location("verify_d3_stats", STATS_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FusedAttentionD3ContractTests(unittest.TestCase):
    def test_partial_runner_selects_d3_shape_guest_and_verification(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "--partial", "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fused_attention_d3_partial", result.stdout)
        self.assertIn("--queries 20 --keys 70", result.stdout)
        self.assertIn("--partial", result.stdout)

    def test_partial_stats_count_only_valid_query_and_key_lanes(self):
        expected = load_stats_tool().expected_sums(partial=True)
        self.assertEqual(expected[("core1:rocc", "attention_qk_array_ops")], 140)
        self.assertEqual(expected[("core1:rocc", "attention_pv_array_ops")], 240)
        self.assertEqual(expected[("core1:rocc:sfu", "sfu_attention_jobs")], 6)
        self.assertEqual(expected[("core1:rocc:sfu", "sfu_softmax_rows")], 60)
        self.assertEqual(
            expected[("core1:rocc:sfu", "sfu_attention_scaled_elements")], 1400
        )

    def test_worker_uses_ceil_tiles_and_runtime_valid_bounds(self):
        source = ROCC.read_text()
        self.assertIn("attentionQueryRows(state)", source)
        self.assertIn("attentionKeyCols(state)", source)
        self.assertIn("attentionQueryBlocks(state)", source)
        self.assertIn("attentionKeyPanels(state)", source)
        self.assertNotIn(
            "state.dispatch.expectedRows / state.dispatch.queryBlockRows", source
        )


if __name__ == "__main__":
    unittest.main()
