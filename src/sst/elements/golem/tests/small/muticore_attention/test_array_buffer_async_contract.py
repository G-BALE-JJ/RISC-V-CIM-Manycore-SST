#!/usr/bin/env python3

import pathlib
import unittest


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[6]
ARRAY_API = REPO_ROOT / "src/sst/elements/golem/array/computeArray.h"
WCP = REPO_ROOT / "src/sst/elements/golem/workercmdproc/workercmdproc.h"
PIPELINE_CONFIG = REPO_ROOT / (
    "src/sst/elements/golem/tests/small/mvm_noc_int_array/pipeline_config.h"
)


class ArrayBufferAsyncContractTest(unittest.TestCase):
    def test_compute_array_exposes_bounded_byte_scaled_buffer_operations(self):
        source = ARRAY_API.read_text(encoding="utf-8")

        for parameter in (
            "arrayBufferBaseLatencyCycles",
            "arrayBufferBytesPerCycle",
            "arrayBufferPorts",
            "arrayBufferQueueDepth",
        ):
            self.assertIn(parameter, source)
        self.assertIn("programOperandsAsync", source)
        self.assertIn("readOutputAsync", source)
        self.assertIn("writeOutputAsync", source)
        self.assertIn("enqueueBufferTransfer", source)

    def test_wcp_uses_async_array_ports_and_address_based_final_dma(self):
        source = WCP.read_text(encoding="utf-8")

        self.assertNotIn("array_->setMatrixItem", source)
        self.assertNotIn("array_->setVectorItem", source)
        self.assertNotIn("array_->getOutputVector", source)
        self.assertIn("array_->programOperandsAsync", source)
        self.assertIn("array_->readOutputAsync", source)
        self.assertIn("array_->writeOutputAsync", source)
        self.assertIn("current_.local_out_gm_addr", source)
        self.assertIn("dma_write_from_globalmem_to_host", source)

    def test_local_output_window_holds_a_complete_c_tile(self):
        source = PIPELINE_CONFIG.read_text(encoding="utf-8")

        self.assertIn(
            "std::max(OUT_VEC_BYTES, GEMM_BLOCK_OUT_TILE_BYTES)", source
        )


if __name__ == "__main__":
    unittest.main()
