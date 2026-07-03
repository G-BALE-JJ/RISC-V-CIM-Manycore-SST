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


class SfuPrimitiveCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = read(SFU_H)
        cls.source = read(SFU_CC)

    def test_sfu_api_exposes_standalone_primitive_issue(self):
        self.assertIn("virtual bool issuePrimitive(uint64_t descAddr, uint64_t tag) = 0", self.header)
        self.assertIn("bool issuePrimitive(uint64_t descAddr, uint64_t tag) override", self.header)
        self.assertIn("std::unordered_map<uint64_t, PrimitiveOpState> pendingPrimitiveOps_", self.header)

    def test_primitive_state_and_helpers_are_declared(self):
        self.assertIn("struct PrimitiveOpState", self.header)
        self.assertIn("SFUPrimitiveDesc desc", self.header)
        self.assertIn("std::vector<float> input0", self.header)
        self.assertIn("std::vector<float> output", self.header)
        for helper in [
            "readPrimitiveDescriptor",
            "validatePrimitiveDescriptor",
            "readPrimitiveInput",
            "executePrimitive",
            "writePrimitiveOutput",
            "primitiveProcessedElems",
        ]:
            self.assertIn(helper, self.header)

    def test_issue_primitive_reads_executes_and_records_tagged_state(self):
        issue_body = re.search(
            r"bool\s+SFU::issuePrimitive\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(issue_body)
        body = issue_body.group("body")
        self.assertIn("readPrimitiveDescriptor", body)
        self.assertIn("validatePrimitiveDescriptor", body)
        self.assertIn("readPrimitiveInput", body)
        self.assertIn("executePrimitive", body)
        self.assertIn("writePrimitiveOutput", body)
        self.assertIn("pendingPrimitiveOps_[tag]", body)
        self.assertIn("statOpsIssued_->addData", body)
        self.assertIn("statPrimitiveElems_->addData(primitiveProcessedElems", body)

    def test_repeat_chunk_primitive_records_logical_processed_elements(self):
        self.assertIn("sfu_primitive_elems", self.header)
        self.assertIn("statPrimitiveElems_", self.header)
        self.assertIn("GOLEM_SFU_PRIMITIVE_FLAG_REPEAT_CHUNK", self.source)
        self.assertIn("primitiveProcessedElems", self.source)
        self.assertIn("desc.input1_gm_addr", self.source)

    def test_wait_retires_primitive_operations(self):
        wait_body = re.search(
            r"bool\s+SFU::wait\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(wait_body)
        body = wait_body.group("body")
        self.assertIn("pendingPrimitiveOps_.find(tag)", body)
        self.assertIn("pendingPrimitiveOps_.erase", body)

    def test_unary_primitive_executes_exp_log_and_reciprocal_fp32(self):
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::EXP)", self.source)
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::LOG)", self.source)
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::RECIPROCAL)", self.source)
        self.assertIn("std::exp", self.source)
        self.assertIn("std::log", self.source)
        self.assertIn("1.0f / value", self.source)

    def test_phase9c_unary_primitive_executes_rsqrt_tanh_and_sigmoid_fp32(self):
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::RSQRT)", self.source)
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::TANH)", self.source)
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::SIGMOID)", self.source)
        self.assertIn("1.0f / static_cast<float>(std::sqrt", self.source)
        self.assertIn("std::tanh", self.source)
        self.assertIn("1.0f / (1.0f + static_cast<float>(std::exp", self.source)

    def test_contiguous_fp32_primitive_uses_bulk_local_gm_transfers(self):
        read_body = re.search(
            r"bool\s+SFU::readPrimitiveInput\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        write_body = re.search(
            r"bool\s+SFU::writePrimitiveOutput\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(read_body)
        self.assertIsNotNone(write_body)
        self.assertIn("stride == sizeof(float)", read_body.group("body"))
        self.assertIn("static_cast<size_t>(desc.elem_count) * sizeof(float)", read_body.group("body"))
        self.assertIn("std::memcpy(values->data()", read_body.group("body"))
        self.assertIn("stride == sizeof(float)", write_body.group("body"))
        self.assertIn("values.size() * sizeof(float)", write_body.group("body"))
        self.assertIn("std::memcpy(raw.data()", write_body.group("body"))


if __name__ == "__main__":
    unittest.main()
