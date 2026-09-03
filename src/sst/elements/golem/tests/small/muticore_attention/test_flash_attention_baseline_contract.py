import json
import pathlib
import os
import subprocess
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verify_attention_mpi_partition import expected_component_ranks


WRAPPER = HERE / "run_flash_attention.sh"
SCALE_RUNNER = HERE / "run_fused_attention_scale.sh"
UNIFIED_RUNNER = HERE.parents[6] / "scripts" / "test_flash_attention.sh"
BUILD_SCRIPT = HERE.parents[6] / "scripts" / "build_and_install_local.sh"
ARCHIVE_ARCH = HERE.parents[1] / "architecture" / "archive" / "ncores_selfcom_dma.py"


class FlashAttentionBaselineContractTest(unittest.TestCase):
    def write_placement_manifest(self, directory, mpi_ranks, overrides=None):
        component_ranks = expected_component_ranks(mpi_ranks)
        component_ranks.update(overrides or {})
        placement_file = pathlib.Path(directory) / "attention_mpi_placement.json"
        placement_file.write_text(
            json.dumps(
                {"mpi_ranks": mpi_ranks, "component_ranks": component_ranks}
            ),
            encoding="ascii",
        )
        return placement_file

    def write_ranked_stats(self, directory, mpi_ranks):
        stats_file = pathlib.Path(directory) / "stats_selfcom.txt"
        for rank in range(mpi_ranks):
            rows = ["ComponentName,Rank"]
            rows.extend(
                f"core{core_id}:rocc,{rank}"
                for core_id in range(20)
                if core_id % mpi_ranks == rank
            )
            rows.extend(
                f"rtr_{router_id},{rank}"
                for router_id in range(28)
                if (router_id % mpi_ranks if router_id < 24 else 0) == rank
            )
            stats_file.with_name(f"stats_selfcom_{rank}.txt").write_text(
                "\n".join(rows) + "\n", encoding="ascii"
            )
        return stats_file

    def run_partition_verifier(self, stats_file, placement_file, mpi_ranks):
        result = subprocess.run(
            [
                "python3",
                str(HERE / "verify_attention_mpi_partition.py"),
                "--stats-file",
                str(stats_file),
                "--mpi-ranks",
                str(mpi_ranks),
                "--placement-file",
                str(placement_file),
            ],
            capture_output=True,
            text=True,
        )
        return result, json.loads(result.stdout)

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

    def test_scale_runner_forwards_multirank_execution(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            env={**os.environ, "GOLEM_MPI_RANKS": "2"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mpi-ranks 2", result.stdout)
        self.assertIn("--mpi-partitioner sst.self", result.stdout)
        self.assertIn("--lib-path=", result.stdout)
        self.assertIn("/install/lib/sst-elements-library", result.stdout)
        self.assertIn("architecture/archive/ncores_selfcom_dma.py", result.stdout)
        self.assertIn("GOLEM_ATTENTION_QUERY_BLOCK_MPI=1", result.stdout)
        self.assertIn("verify_attention_mpi_partition.py", result.stdout)
        self.assertNotIn("/data/shun/", result.stdout)
        self.assertNotIn("make -C", result.stdout)

    def test_scale_runner_ignores_polluting_generic_sst_args(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            env={
                **os.environ,
                "GOLEM_MPI_RANKS": "2",
                "GOLEM_SST_ARGS": "--partitioner=sst.simple --lib-path=/tmp/foreign",
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mpi-partitioner sst.self", result.stdout)
        self.assertNotIn("sst.simple", result.stdout)
        self.assertNotIn("/tmp/foreign", result.stdout)

    def test_archive_architecture_supports_sst_mpi_partitioning(self):
        text = ARCHIVE_ARCH.read_text()
        self.assertIn('MPI_PARTITIONING = _env_flag("GOLEM_MPI_PARTITIONING", False)', text)
        self.assertIn('DRAMSIM3_OUT_DIR = os.getenv(', text)
        self.assertIn('node_output_dir = os.path.join(DRAMSIM3_OUT_DIR, f"node{idx}")', text)
        self.assertGreaterEqual(text.count("if not MPI_PARTITIONING:"), 3)
        self.assertIn("def attention_rank_for_core(core_id: int) -> int:", text)
        self.assertIn("def set_attention_cpu_rank(prefix: str, rank: int) -> None:", text)
        self.assertIn("component = sst.findComponentByName(component_name)", text)
        self.assertIn("for router_id, router in enumerate(noc.routers):", text)
        self.assertIn("set_attention_cpu_rank(", text)
        self.assertIn("def set_attention_component_rank(", text)
        self.assertIn("GOLEM_ATTENTION_PLACEMENT_FILE", text)

    def test_unified_runner_uses_worktree_install_and_e3(self):
        text = UNIFIED_RUNNER.read_text()
        self.assertIn('source "$SCRIPT_DIR/env_local_install.sh"', text)
        self.assertIn('"$ATTENTION_DIR/run_flash_attention.sh"', text)
        self.assertIn('--timeout "$TIMEOUT" --artifact-root "$ARTIFACT_ROOT"', text)
        self.assertIn('export SST_LIB_PATH="$WORKTREE_ROOT/install/lib/sst-elements-library"', text)
        self.assertIn('--mpi-ranks) MPI_RANKS="$2"', text)
        self.assertIn('baseline/e3/mpi2/result.json', text)
        self.assertIn('baseline/e3/mpi4/result.json', text)
        self.assertIn('"verification.score_probability_hbm_bytes"', text)
        self.assertIn("Build it first with scripts/build_and_install_local.sh", text)
        self.assertNotIn("export SST_SOFTMAX_LD_LIBRARY_PATH=", text)
        self.assertNotIn("BUILD_ARGS", text)
        self.assertNotIn("SKIP_BUILD", text)

    def test_build_script_builds_attention_guests(self):
        text = BUILD_SCRIPT.read_text()
        self.assertIn('make -C "$ATTENTION_DIR"', text)
        self.assertIn("scale-e2 scale-e3 scale-e4 scale-e5", text)
        self.assertNotIn('make -C "$SCRIPT_DIR"', SCALE_RUNNER.read_text())

    def test_partition_verifier_accepts_query_block_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            stats_file = pathlib.Path(directory) / "stats_selfcom.txt"
            placement_file = self.write_placement_manifest(directory, 2)
            for rank in range(2):
                rows = ["ComponentName,Rank"]
                rows.extend(
                    f"core{core_id}:rocc,{rank}"
                    for core_id in range(20)
                    if core_id % 2 == rank
                )
                rows.extend(
                    f"rtr_{router_id},{rank}"
                    for router_id in range(28)
                    if (router_id % 2 if router_id < 24 else 0) == rank
                )
                stats_file.with_name(f"stats_selfcom_{rank}.txt").write_text(
                    "\n".join(rows) + "\n", encoding="ascii"
                )
            result = subprocess.run(
                [
                    "python3",
                    str(HERE / "verify_attention_mpi_partition.py"),
                    "--stats-file",
                    str(stats_file),
                    "--mpi-ranks",
                    "2",
                    "--placement-file",
                    str(placement_file),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"status": "PASS"', result.stdout)

    def test_partition_verifier_rejects_misplaced_core(self):
        with tempfile.TemporaryDirectory() as directory:
            stats_file = pathlib.Path(directory) / "stats_selfcom.txt"
            placement_file = self.write_placement_manifest(directory, 2)
            for rank in range(2):
                rows = ["ComponentName,Rank"]
                rows.extend(
                    f"core{core_id}:rocc,{0 if core_id == 1 else rank}"
                    for core_id in range(20)
                    if core_id % 2 == rank
                )
                rows.extend(
                    f"rtr_{router_id},{rank}"
                    for router_id in range(28)
                    if (router_id % 2 if router_id < 24 else 0) == rank
                )
                stats_file.with_name(f"stats_selfcom_{rank}.txt").write_text(
                    "\n".join(rows) + "\n", encoding="ascii"
                )
            result = subprocess.run(
                [
                    "python3",
                    str(HERE / "verify_attention_mpi_partition.py"),
                    "--stats-file",
                    str(stats_file),
                    "--mpi-ranks",
                    "2",
                    "--placement-file",
                    str(placement_file),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('"status": "FAIL"', result.stdout)

    def test_partition_verifier_accepts_four_rank_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            stats_file = pathlib.Path(directory) / "stats_selfcom.txt"
            placement_file = self.write_placement_manifest(directory, 4)
            for rank in range(4):
                rows = ["ComponentName,Rank"]
                rows.extend(
                    f"core{core_id}:rocc,{rank}"
                    for core_id in range(20)
                    if core_id % 4 == rank
                )
                rows.extend(
                    f"rtr_{router_id},{rank}"
                    for router_id in range(28)
                    if (router_id % 4 if router_id < 24 else 0) == rank
                )
                stats_file.with_name(f"stats_selfcom_{rank}.txt").write_text(
                    "\n".join(rows) + "\n", encoding="ascii"
                )
            result = subprocess.run(
                [
                    "python3",
                    str(HERE / "verify_attention_mpi_partition.py"),
                    "--stats-file",
                    str(stats_file),
                    "--mpi-ranks",
                    "4",
                    "--placement-file",
                    str(placement_file),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"status": "PASS"', result.stdout)

    def test_partition_verifier_rejects_misplaced_memory_component(self):
        with tempfile.TemporaryDirectory() as directory:
            stats_file = pathlib.Path(directory) / "stats_selfcom.txt"
            placement_file = self.write_placement_manifest(
                directory, 2, {"memory_2": 0}
            )
            for rank in range(2):
                rows = ["ComponentName,Rank"]
                rows.extend(
                    f"core{core_id}:rocc,{rank}"
                    for core_id in range(20)
                    if core_id % 2 == rank
                )
                rows.extend(
                    f"rtr_{router_id},{rank}"
                    for router_id in range(28)
                    if (router_id % 2 if router_id < 24 else 0) == rank
                )
                stats_file.with_name(f"stats_selfcom_{rank}.txt").write_text(
                    "\n".join(rows) + "\n", encoding="ascii"
                )
            result = subprocess.run(
                [
                    "python3",
                    str(HERE / "verify_attention_mpi_partition.py"),
                    "--stats-file",
                    str(stats_file),
                    "--mpi-ranks",
                    "2",
                    "--placement-file",
                    str(placement_file),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('"memory_2"', result.stdout)
            self.assertIn('"status": "FAIL"', result.stdout)

    def test_partition_verifier_rejects_manifest_integrity_errors(self):
        for case in ("missing", "unexpected", "rank_count"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                stats_file = self.write_ranked_stats(directory, 2)
                placement_file = self.write_placement_manifest(directory, 2)
                placement = json.loads(placement_file.read_text(encoding="ascii"))
                if case == "missing":
                    del placement["component_ranks"]["os"]
                elif case == "unexpected":
                    placement["component_ranks"]["foreign_component"] = 0
                else:
                    placement["mpi_ranks"] = 4
                placement_file.write_text(json.dumps(placement), encoding="ascii")
                result, report = self.run_partition_verifier(
                    stats_file, placement_file, 2
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(report["status"], "FAIL")

    def test_partition_verifier_rejects_rank_file_integrity_errors(self):
        for case in ("missing", "unexpected", "reported_rank"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                stats_file = self.write_ranked_stats(directory, 2)
                placement_file = self.write_placement_manifest(directory, 2)
                if case == "missing":
                    stats_file.with_name("stats_selfcom_1.txt").unlink()
                elif case == "unexpected":
                    stats_file.with_name("stats_selfcom_2.txt").write_text(
                        "ComponentName,Rank\n", encoding="ascii"
                    )
                else:
                    rank_one = stats_file.with_name("stats_selfcom_1.txt")
                    rank_one.write_text(
                        rank_one.read_text(encoding="ascii").replace(",1\n", ",0\n"),
                        encoding="ascii",
                    )
                result, report = self.run_partition_verifier(
                    stats_file, placement_file, 2
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
