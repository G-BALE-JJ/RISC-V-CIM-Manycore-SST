#!/usr/bin/env python3

import importlib.util
import pathlib
import subprocess
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[6]
ROCC = REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalog.h"
GUEST_H = HERE / "golem_attention_runtime.h"
GUEST_CC = HERE / "golem_attention_runtime.cpp"
RUNNER = HERE / "run_fused_attention_online.sh"
STATS_TOOL = HERE / "verify_fused_attention_online_stats.py"


def load_stats_tool():
    spec = importlib.util.spec_from_file_location("verify_d2_stats", STATS_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FusedAttentionD2ContractTests(unittest.TestCase):
    def test_causal_flag_reaches_sfu_and_allows_only_known_descriptor_bits(self):
        source = ROCC.read_text()
        guest = GUEST_H.read_text() + GUEST_CC.read_text()
        self.assertIn("GOLEM_ATTENTION_FLAG_CAUSAL", guest)
        self.assertIn("GOLEM_ATTENTION_CAUSAL", guest)
        self.assertIn("attentionKeyTilesForQueryBlock", source)
        self.assertIn("request.causal = attentionCausal(state)", source)
        self.assertIn("~GOLEM_ATTENTION_FLAG_CAUSAL", source)

    def test_causal_runner_selects_d2_guest_and_verification(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "--causal", "1", "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fused_attention_d2_causal", result.stdout)
        self.assertIn("--causal 1", result.stdout)
        self.assertIn("verify_fused_attention_online_stats.py", result.stdout)
        self.assertIn("--causal", result.stdout)

    def test_causal_stats_require_future_tile_skip_and_exact_diagonal_masks(self):
        stats = load_stats_tool()
        expected = stats.expected_sums(causal=True)
        self.assertEqual(expected[("core1:rocc", "attention_qk_array_ops")], 192)
        self.assertEqual(expected[("core1:rocc", "attention_pv_array_ops")], 384)
        self.assertEqual(expected[("core1:rocc:sfu", "sfu_attention_jobs")], 6)
        self.assertEqual(expected[("core1:rocc:sfu", "sfu_softmax_rows")], 96)
        self.assertEqual(
            expected[("core1:rocc:sfu", "sfu_attention_masked_elements")], 992
        )
        header = "ComponentName,StatisticName,Sum.u64,Count.u64\n"
        rows = [
            f"{component},{name},{value},{1 if 'rsqrt' in name else value}\n"
            for (component, name), value in expected.items()
        ]
        rows.append("core1:rocc:sfu,sfu_attention_rsqrt_ready_tick,1,1\n")
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "stats.csv"
            path.write_text(header + "".join(rows), encoding="ascii")
            result = stats.verify_stats(path, causal=True)
        self.assertEqual(result["status"], "PASS", result)


if __name__ == "__main__":
    unittest.main()
