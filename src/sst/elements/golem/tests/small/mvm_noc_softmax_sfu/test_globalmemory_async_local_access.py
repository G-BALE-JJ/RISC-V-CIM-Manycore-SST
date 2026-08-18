#!/usr/bin/env python3

import os
import re
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLEM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
GLOBAL_MEMORY_H = os.path.join(GOLEM_DIR, "globalmemory", "globalmemory.h")
GLOBAL_MEMORY_CC = os.path.join(GOLEM_DIR, "globalmemory", "globalmemory.cc")
CPU_BUILDER = os.path.join(GOLEM_DIR, "tests", "architecture", "cpu_builder.py")


def read(path):
    with open(path, "r", encoding="utf-8") as source_file:
        return source_file.read()


class GlobalMemoryAsyncLocalAccessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = read(GLOBAL_MEMORY_H)
        cls.source = read(GLOBAL_MEMORY_CC)
        cls.builder = read(CPU_BUILDER)

    def test_api_exposes_tagged_async_local_read_and_write(self):
        self.assertIn("enum class LocalMemoryClient", self.header)
        self.assertIn("using LocalReadCallback", self.header)
        self.assertIn("using LocalWriteCallback", self.header)
        self.assertIn("virtual bool localReadAsync", self.header)
        self.assertIn("virtual bool localWriteAsync", self.header)

    def test_component_declares_bounded_local_scheduler_parameters(self):
        for name in [
            "local_access_clock",
            "local_access_base_latency_cycles",
            "local_access_bytes_per_cycle",
            "local_access_read_ports",
            "local_access_write_ports",
            "local_access_queue_depth",
            "local_access_max_request_bytes",
        ]:
            self.assertIn(name, self.header)
            self.assertIn(name, self.source)

    def test_scheduler_has_finite_queues_ports_and_completion_events(self):
        for token in [
            "struct PendingLocalAccess",
            "localReadQueue_",
            "localWriteQueue_",
            "localAccessPending_",
            "tryIssueLocalAccesses",
            "handleLocalAccessEvent",
            '"LocalAccessSelf"',
        ]:
            self.assertIn(token, self.header + self.source)
        self.assertRegex(
            self.source,
            r"localAccessPending_\.size\(\)\s*>=\s*localAccessQueueDepth_",
        )
        self.assertIn("ceilDivLocalAccess", self.source)

    def test_async_completion_reports_identity_and_accounts_bytes(self):
        for token in [
            "request.tag",
            "localReadBytes_",
            "localWriteBytes_",
            "localQueueRejected_",
            "localAccessQueueHighWater_",
            "localReadQueueCycles_",
            "localWriteQueueCycles_",
            "GOLEM_LOCAL_GM_STATS",
        ]:
            self.assertIn(token, self.header + self.source)

    def test_local_only_fallback_implements_the_same_api(self):
        local_class = re.search(
            r"class GlobalMemoryLocal.*?\n\};",
            self.header,
            re.S,
        )
        self.assertIsNotNone(local_class)
        self.assertIn("localReadAsync", local_class.group(0))
        self.assertIn("localWriteAsync", local_class.group(0))
        self.assertIn("GlobalMemoryLocal::localReadAsync", self.source)
        self.assertIn("GlobalMemoryLocal::localWriteAsync", self.source)

    def test_cpu_builder_wires_local_scheduler_knobs(self):
        for env_name in [
            "GOLEM_LOCAL_GM_BASE_LATENCY_CYCLES",
            "GOLEM_LOCAL_GM_BYTES_PER_CYCLE",
            "GOLEM_LOCAL_GM_READ_PORTS",
            "GOLEM_LOCAL_GM_WRITE_PORTS",
            "GOLEM_LOCAL_GM_QUEUE_DEPTH",
            "GOLEM_LOCAL_GM_MAX_REQUEST_BYTES",
        ]:
            self.assertIn(env_name, self.builder)
        for param_name in [
            '"local_access_clock": cpu_clock',
            '"local_access_base_latency_cycles"',
            '"local_access_bytes_per_cycle"',
            '"local_access_read_ports"',
            '"local_access_write_ports"',
            '"local_access_queue_depth"',
            '"local_access_max_request_bytes"',
        ]:
            self.assertIn(param_name, self.builder)

    def test_dma_read_completion_lands_through_async_local_write(self):
        self.assertIn("completeDmaReadToLocalMemory", self.header)
        self.assertIn("completeDmaReadToLocalMemory", self.source)
        self.assertIn("LocalMemoryClient::Dma", self.source)

        receive = re.search(
            r"bool GlobalMemoryImplement::handle_receives.*?\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(receive)
        dma_complete = receive.group(0).split(
            "NetworkDataEvent::DMA_READ_COMPLETE", 1
        )[1]
        self.assertIn("completeDmaReadToLocalMemory", dma_complete)


if __name__ == "__main__":
    unittest.main()
