import tempfile
import unittest
from pathlib import Path

from dma_kind_trace import parse_events, parse_trace


class DmaKindTraceTest(unittest.TestCase):
    def test_counts_read_and_output_kinds(self):
        text = "\n".join(
            [
                "send READ_RESP cycle=1 req=1 kind=1",
                "send READ_RESP cycle=2 req=2 kind=2",
                "send READ_RESP cycle=3 req=3 kind=3",
                "send WRITE_COMPLETE cycle=4 req=0 kind=4",
                "send WRITE_COMPLETE cycle=5 req=0 kind=0",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.log"
            path.write_text(text, encoding="utf-8")
            result = parse_trace(path)

        self.assertEqual(result["counts"]["attention_query"], 1)
        self.assertEqual(result["counts"]["attention_kv"], 1)
        self.assertEqual(result["counts"]["attention_kv_prefetch"], 1)
        self.assertEqual(result["counts"]["attention_output"], 1)
        self.assertEqual(result["unknown_events"], 1)

    def test_parse_events_preserves_cycle_and_semantic_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.log"
            path.write_text(
                "send WRITE_COMPLETE cycle=17 req=0 kind=4\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_events(path), [(17, "attention_output")])


if __name__ == "__main__":
    unittest.main()
