import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[7]
MEMNIC = REPO_ROOT / "src/sst/elements/memHierarchy/memNICBase.h"


class MemnicDmaPriorityContractTest(unittest.TestCase):
    def test_priority_is_opt_in_and_uses_semantic_kind(self):
        source = MEMNIC.read_text(encoding="utf-8")
        self.assertIn("GOLEM_DMA_RESPONSE_PRIORITY_ENABLE", source)
        self.assertIn("golem_dma_response_priority_enable", source)
        self.assertIn("DmaRequestKind::AttentionKv", source)
        self.assertIn("DmaRequestKind::AttentionQuery", source)
        self.assertIn("DmaRequestKind::AttentionOutput", source)
        self.assertIn("DmaRequestKind::AttentionKvPrefetch", source)
        self.assertIn("priority_reorders", source)


if __name__ == "__main__":
    unittest.main()
