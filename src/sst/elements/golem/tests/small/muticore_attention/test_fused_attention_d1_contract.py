#!/usr/bin/env python3

import importlib.util
import math
import pathlib
import subprocess
import unittest


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[6]
ROCC = REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalog.h"
SFU_H = REPO_ROOT / "src/sst/elements/golem/sfu/sfu.h"
SFU_CC = REPO_ROOT / "src/sst/elements/golem/sfu/sfu.cc"
CASE_TOOL = HERE / "attention_case.py"
RUNNER = HERE / "run_fused_attention_online.sh"


def load_case_tool():
    spec = importlib.util.spec_from_file_location("attention_case", CASE_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FusedAttentionD1ContractTests(unittest.TestCase):
    def test_online_merge_rescales_old_output_when_second_tile_raises_max(self):
        case = load_case_tool()
        merged = case.merge_online_tile(0.0, 2.0, 10.0, 3.0, final_tile=True)
        expected_l = 2.0 * math.exp(-10.0) + 3.0
        self.assertAlmostEqual(merged["m"], 10.0)
        self.assertAlmostEqual(merged["l"], expected_l)
        self.assertAlmostEqual(merged["old_output_scale"], math.exp(-10.0) / expected_l)
        self.assertAlmostEqual(merged["tile_weight_scale"], 1.0 / expected_l)

    def test_sfu_uses_bounded_online_context_registers_and_one_job_rsqrt(self):
        header = SFU_H.read_text()
        source = SFU_CC.read_text()
        for token in (
            "AttentionOnlineRowContext",
            "attentionOnlineContexts_",
            "request.firstTileForJob",
            "request.keyTile",
            "request.keyTiles",
            "oldOutputScale",
            "tileWeightScale",
        ):
            self.assertIn(token, header + source)
        self.assertIn("std::exp(online.m - mNew)", source)
        self.assertIn("std::exp(context.rowMax - mNew)", source)

    def test_rocc_runs_four_query_blocks_two_key_tiles_and_restores_oacc(self):
        rocc = ROCC.read_text()
        for token in (
            "ATTENTION_D1_WINDOW_BYTES",
            "state.keyTile",
            "attentionQueryBlocks(state)",
            "attentionKeyTilesForQueryBlock(state)",
            "writeOutputAsync",
            "configureOutputMode(arrayId, 1)",
        ):
            self.assertIn(token, rocc)

    def test_qk_resets_array_output_mode_after_pv_accumulation(self):
        rocc = ROCC.read_text()
        qk_compute = rocc.split("void programAttentionQkInput()", 1)[1].split(
            "void readAttentionQkOutput()", 1
        )[0]
        self.assertIn("configureOutputMode(arrayId, 0)", qk_compute)

    def test_d1_runner_is_fixed_to_s64_two_key_tiles(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--queries 64", result.stdout)
        self.assertIn("--keys 64", result.stdout)
        self.assertIn("GOLEM_SFU_ROW_CONTEXTS=16", result.stdout)
        self.assertIn("verify_fused_attention_online_stats.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
