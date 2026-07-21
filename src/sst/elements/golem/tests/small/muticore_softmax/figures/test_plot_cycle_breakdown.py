import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from plot_cycle_breakdown_1024x4096 import critical_path_from_ticks


class CriticalPathTicksTest(unittest.TestCase):
    def test_causal_endpoints_sum_to_actual_accelerator_completion(self):
        ticks = {
            "descriptor_accept": 232392660,
            "first_worker_dispatch": 232397226,
            "first_input_dma_ready": 232508506,
            "last_compute_done": 261442892,
            "final_output_dma_ack": 261481110,
            "accelerator_complete": 261504650,
        }

        segments = critical_path_from_ticks(ticks)

        self.assertEqual(
            segments,
            [
                ("Band to first worker", 11),
                ("First input DMA", 256),
                ("DMA-fed row pipeline", 66549),
                ("Final output DMA", 88),
                ("Completion delivery", 54),
            ],
        )
        self.assertEqual(sum(cycles for _, cycles in segments), 66958)

    def test_plot_sources_do_not_restore_the_decoupled_completion_gap(self):
        figures_dir = Path(__file__).parent
        source = "\n".join(
            (figures_dir / name).read_text(encoding="utf-8")
            for name in (
                "plot_cycle_breakdown_1024x4096.py",
                "plot_cycle_breakdown_1024x4096_split.py",
            )
        )

        self.assertNotIn("modeled_compute_ready", source)
        self.assertNotIn("Model completion gap", source)
        self.assertNotIn("model completion gap", source)


if __name__ == "__main__":
    unittest.main()
