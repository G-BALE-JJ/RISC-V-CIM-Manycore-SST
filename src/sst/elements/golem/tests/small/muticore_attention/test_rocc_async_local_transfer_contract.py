#!/usr/bin/env python3

import pathlib
import re
import unittest


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[6]
ROCC = REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalog.h"
ARRAY_API = REPO_ROOT / "src/sst/elements/golem/array/computeArray.h"


class RoccAsyncLocalTransferContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rocc = ROCC.read_text(encoding="utf-8")
        cls.array_api = ARRAY_API.read_text(encoding="utf-8")

    def _function(self, name, next_name):
        match = re.search(
            rf"void {name}\(.*?void {next_name}\(", self.rocc, re.S
        )
        self.assertIsNotNone(match)
        return match.group(0)

    def test_array_supports_independent_async_operand_programming(self):
        self.assertIn("programMatrixAsync", self.array_api)
        self.assertIn("programInputAsync", self.array_api)
        self.assertIn("readOutputBytesAsync", self.array_api)

    def test_blocking_legacy_commands_wait_for_real_local_and_array_transfers(self):
        output_store = self._function("OutputvectorStore", "IntputvectorLoad")
        vector_load = self._function("IntputvectorLoad", "InputMatrixLoad")
        matrix_load = self._function("InputMatrixLoad", "SetRemoteLength")
        engine = re.search(
            r"struct AsyncArrayLoadState.*?void enqueueResponse", self.rocc, re.S
        )
        self.assertIsNotNone(engine)
        engine = engine.group(0)

        self.assertIn("progressLegacyOutputStore", output_store)
        self.assertIn("readOutputBytesAsync", engine)
        self.assertIn("localWriteAsync", engine)
        self.assertNotIn("getOutputVector", engine)
        self.assertNotIn("wr_to_globalmem", engine)
        self.assertNotIn("latency_mvm_ovec2gm", engine)
        output_engine = re.search(
            r"void progressLegacyOutputStore\(\).*?void enqueueResponse",
            self.rocc,
            re.S,
        )
        self.assertIsNotNone(output_engine)
        self.assertNotIn("static_cast<T>(values", output_engine.group(0))

        for source, array_method, latency in (
            (vector_load, "programInputAsync", "latency_mvm_gm2ivec"),
            (matrix_load, "programMatrixAsync", "latency_mvm_gm2imat"),
        ):
            self.assertIn("beginBlockingArrayLoad", source)
            self.assertIn("localReadAsync", engine)
            self.assertIn("LocalMemoryClient::RoCC", engine)
            self.assertIn(array_method, engine)
            self.assertNotIn("rd_from_globalmem", engine)
            self.assertNotIn("setVectorItem", engine)
            self.assertNotIn("setMatrixItem", engine)
            self.assertNotIn(latency, engine)

    def test_nonblocking_array_loads_are_callback_driven(self):
        state_and_progress = re.search(
            r"struct AsyncArrayLoadState.*?void enqueueResponse", self.rocc, re.S
        )
        self.assertIsNotNone(state_and_progress)
        source = state_and_progress.group(0)

        self.assertNotIn("ready_cycle", source)
        self.assertNotIn("rd_from_globalmem", source)
        self.assertNotIn("setMatrixItem", source)
        self.assertNotIn("setVectorItem", source)
        self.assertIn("localReadAsync", source)
        self.assertIn("LocalMemoryClient::RoCC", source)
        self.assertIn("programMatrixAsync", source)
        self.assertIn("programInputAsync", source)


if __name__ == "__main__":
    unittest.main()
