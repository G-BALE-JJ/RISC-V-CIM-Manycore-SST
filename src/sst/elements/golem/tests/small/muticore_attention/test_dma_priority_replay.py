import unittest

from dma_priority_replay import replay_trace_events


class DmaPriorityReplayTest(unittest.TestCase):
    def test_compressed_pressure_trace_drains_all_classes(self):
        events = [
            (100, "attention_kv_prefetch"),
            (101, "attention_output"),
            (102, "attention_query"),
            (103, "attention_kv"),
        ]
        summary = replay_trace_events(events, arrival_quantum=10)
        self.assertEqual(summary["completion_order"], ["consumer", "query", "output", "prefetch"])
        self.assertEqual(summary["completed"], 4)
        self.assertTrue(summary["exactly_once"])
        self.assertTrue(summary["drained"])


if __name__ == "__main__":
    unittest.main()
