#!/usr/bin/env python3

import os
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLEM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))


def read_rel(path):
    with open(os.path.join(GOLEM_DIR, path), "r", encoding="utf-8") as f:
        return f.read()


class SfuComponentScaffoldTest(unittest.TestCase):
    def test_sfu_header_defines_api_and_component(self):
        text = read_rel("sfu/sfu.h")
        self.assertIn("class SFUAPI", text)
        self.assertIn("SST_ELI_REGISTER_SUBCOMPONENT_API(SST::Golem::SFUAPI)", text)
        self.assertIn("class SFU", text)
        self.assertIn('"golem",', text)
        self.assertIn('"SFU"', text)
        self.assertIn("issueSoftmaxTile", text)
        self.assertIn("wait", text)
        self.assertIn("bindGlobalMemory", text)
        self.assertIn("setCoreInfo", text)

    def test_sfu_source_implements_minimal_methods(self):
        text = read_rel("sfu/sfu.cc")
        self.assertIn("SFU::SFU", text)
        self.assertIn("bool SFU::issueSoftmaxTile", text)
        self.assertIn("bool SFU::wait", text)
        self.assertIn("void SFU::bindGlobalMemory", text)
        self.assertIn("void SFU::setCoreInfo", text)

    def test_golem_registration_includes_sfu_without_touching_old_paths(self):
        golem_cc = read_rel("golem.cc")
        makefile = read_rel("Makefile.am")
        self.assertIn("#include <sst/elements/golem/sfu/sfu.h>", golem_cc)
        self.assertIn("sfu/sfu.h", makefile)
        self.assertIn("sfu/sfu.cc", makefile)

    def test_golem_eli_aggregation_includes_global_memory_registration(self):
        golem_cc = read_rel("golem.cc")
        self.assertIn("#include <sst/elements/golem/globalmemory/globalmemory.h>", golem_cc)


if __name__ == "__main__":
    unittest.main()
