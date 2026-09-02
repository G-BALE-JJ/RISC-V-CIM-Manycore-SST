import pathlib
import os
import subprocess
import unittest


HERE = pathlib.Path(__file__).resolve().parent
WRAPPER = HERE / "run_flash_attention.sh"
SCALE_RUNNER = HERE / "run_fused_attention_scale.sh"
UNIFIED_RUNNER = HERE.parents[6] / "scripts" / "test_flash_attention.sh"


class FlashAttentionBaselineContractTest(unittest.TestCase):
    def test_wrapper_selects_verified_e3_by_default(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("fused_attention_e3_s1024_d128", result.stdout)
        self.assertIn("PROFILE=e3", WRAPPER.read_text())

    def test_wrapper_targets_local_scale_runner(self):
        text = WRAPPER.read_text()
        self.assertIn('SCALE_RUNNER="$SCRIPT_DIR/run_fused_attention_scale.sh"', text)
        self.assertNotIn("RISC-V-CIM-Manycore-SST", text)

    def test_scale_runner_rejects_multirank_archive_execution(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            env={**os.environ, "GOLEM_MPI_RANKS": "2"},
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scale attention MPI is not enabled", result.stderr)

    def test_unified_runner_uses_worktree_install_and_e3(self):
        text = UNIFIED_RUNNER.read_text()
        self.assertIn('source "$SCRIPT_DIR/env_local_install.sh"', text)
        self.assertIn('run_flash_attention.sh" --timeout', text)
        self.assertIn('export SST_LIB_PATH="$WORKTREE_ROOT/install/lib/sst-elements-library"', text)
        self.assertIn("Build it first with scripts/build_and_install_local.sh", text)
        self.assertNotIn("BUILD_ARGS", text)
        self.assertNotIn("SKIP_BUILD", text)


if __name__ == "__main__":
    unittest.main()
