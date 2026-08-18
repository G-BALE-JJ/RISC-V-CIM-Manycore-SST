#!/usr/bin/env python3

import importlib.util
import math
import pathlib
import unittest


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[6]
CASE_TOOL = HERE / "attention_case.py"


def load_case_tool():
    spec = importlib.util.spec_from_file_location("attention_case", CASE_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AttentionPhaseBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_case_tool()

    def test_noncausal_attention_applies_inverse_sqrt_scale(self):
        q = [1.0, 0.0, 0.0, 1.0]
        k = [1.0, 0.0, 0.0, 1.0]
        v = [10.0, 0.0, 0.0, 20.0]
        actual = self.case.compute_attention(q, k, v, 2, 2, 2, False)
        weight = math.exp(1.0 / math.sqrt(2.0))
        high = weight / (weight + 1.0)
        low = 1.0 / (weight + 1.0)
        expected = [10.0 * high, 20.0 * low, 10.0 * low, 20.0 * high]
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, places=6)

    def test_causal_attention_masks_future_keys(self):
        q = [1.0, 0.0, 0.0, 1.0]
        k = [1.0, 0.0, 0.0, 1.0]
        v = [10.0, 0.0, 0.0, 20.0]
        actual = self.case.compute_attention(q, k, v, 2, 2, 2, True)
        self.assertEqual(actual[:2], [10.0, 0.0])
        self.assertGreater(actual[2], 0.0)
        self.assertGreater(actual[3], 0.0)

    def test_sfu_exposes_versioned_attention_mode(self):
        header = (REPO_ROOT / "src/sst/elements/golem/sfu/sfu.h").read_text()
        source = (REPO_ROOT / "src/sst/elements/golem/sfu/sfu.cc").read_text()
        self.assertIn("SFU_SOFTMAX_JOB_PARAMS_VERSION_ATTENTION", header)
        self.assertIn("SFU_SOFTMAX_PARAMS_FLAG_ATTENTION", header)
        self.assertIn("attentionRsqrtReadyTick", header)
        self.assertIn("std::sqrt", source)
        self.assertIn("attentionCausal", source)

    def test_materialized_runner_contains_three_real_sst_stages(self):
        runner = (HERE / "run_materialized_attention.sh").read_text()
        self.assertIn("run_muticore_attention.sh", runner)
        self.assertIn("run_muticore_softmax.sh", runner)
        self.assertIn("run_noc_dma_pipeline.sh", runner)
        self.assertIn("--attention-head-dim", runner)
        self.assertIn("verify-attention", runner)

    def test_softmax_runner_accepts_raw_scores_and_attention_mode(self):
        runner = (
            REPO_ROOT
            / "src/sst/elements/golem/tests/small/muticore_softmax/run_muticore_softmax.sh"
        ).read_text()
        self.assertIn("--logits-file", runner)
        self.assertIn("GOLEM_SFU_ATTENTION_HEAD_DIM", runner)
        self.assertIn("GOLEM_SFU_ATTENTION_CAUSAL", runner)
        architecture = (
            REPO_ROOT
            / "src/sst/elements/golem/tests/small/mvm_noc_softmax_cpu/ncores_selfcom_dma_softmax_archive.py"
        ).read_text()
        self.assertIn('"GOLEM_SFU_ATTENTION_HEAD_DIM"', architecture)
        self.assertIn('"GOLEM_SFU_ATTENTION_CAUSAL"', architecture)


if __name__ == "__main__":
    unittest.main()
