#!/usr/bin/env python3

import csv
import dataclasses
import fcntl
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sfu_4096x4096_capacity as capacity
import test_sfu_phase4f_large_scale as phase4f_test_support


RUNNER = SCRIPT_DIR / "run_sfu_4096x4096_capacity.sh"


class CapacityContractTest(unittest.TestCase):
    def test_default_points_are_the_fixed_ordered_ladder(self):
        self.assertEqual(
            tuple(
                (
                    point.rows,
                    point.dim,
                    point.worker_cores,
                    point.band_cores,
                    point.mem_node_size,
                    point.timeout_sec,
                    point.rowmajor_region_end,
                )
                for point in capacity.DEFAULT_POINTS
            ),
            (
                (512, 4096, 16, 16, 268435456, 3600, 37748736),
                (1024, 4096, 16, 16, 268435456, 7200, 58720256),
                (2048, 4096, 16, 16, 268435456, 10800, 100663296),
                (4096, 4096, 16, 16, 268435456, 14400, 184549376),
            ),
        )
        self.assertEqual(capacity.parse_point_list(None), capacity.DEFAULT_POINTS)

    def test_point_list_accepts_only_nonempty_ordered_prefixes(self):
        for length in range(1, len(capacity.DEFAULT_POINTS) + 1):
            value = " ".join(
                f"{point.rows}:{point.dim}:{point.worker_cores}:{point.band_cores}"
                for point in capacity.DEFAULT_POINTS[:length]
            )
            self.assertEqual(
                capacity.parse_point_list(value), capacity.DEFAULT_POINTS[:length]
            )

        invalid = (
            "",
            "1024:4096:16:16",
            "512:4096:16:16 2048:4096:16:16",
            "512:4096:16:16 512:4096:16:16",
            "512:8192:16:16",
            "512:4096:8:8",
            "512:4096:16:8",
            "4097:4096:16:16",
            "512:4096:16",
            "rows:4096:16:16",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                capacity.parse_point_list(value)

    def test_final_point_capacity_and_counter_formulas(self):
        evidence = capacity.derive_capacity(capacity.DEFAULT_POINTS[-1])
        self.assertEqual(evidence.elements, 16_777_216)
        self.assertEqual(evidence.tensor_bytes, 67_108_864)
        self.assertEqual(evidence.expected_reduction_each, 65_536)
        self.assertEqual(evidence.expected_transport_total, 262_144)
        self.assertEqual(evidence.expected_dma_ops, 65_536)
        self.assertEqual(evidence.expected_dma_bytes, 67_108_864)
        self.assertEqual(evidence.bias_base, 268_419_072)
        self.assertEqual(evidence.layout_margin_bytes, 83_869_696)

    def test_every_point_fits_the_fixed_memory_node(self):
        for point in capacity.DEFAULT_POINTS:
            with self.subTest(rows=point.rows):
                evidence = capacity.derive_capacity(point)
                self.assertGreater(evidence.layout_margin_bytes, 0)
                self.assertLess(point.rowmajor_region_end, evidence.bias_base)


class CapacityResourceAndEvidenceTest(unittest.TestCase):
    def test_resource_snapshot_enforces_disk_memory_and_tmpdir(self):
        valid = capacity.ResourceSnapshot(
            artifact_free_bytes=16 * 1024**3,
            available_memory_bytes=8 * 1024**3,
            tmpdir=pathlib.Path("/data4/jjgong/tmp"),
            tmpdir_writable=True,
        )
        capacity.validate_resource_snapshot(valid)

        invalid = (
            dataclasses.replace(valid, artifact_free_bytes=16 * 1024**3 - 1),
            dataclasses.replace(valid, available_memory_bytes=8 * 1024**3 - 1),
            dataclasses.replace(valid, tmpdir=pathlib.Path("/tmp")),
            dataclasses.replace(valid, tmpdir_writable=False),
        )
        for snapshot in invalid:
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                capacity.validate_resource_snapshot(snapshot)

    def test_preflight_csv_is_complete_and_deterministic(self):
        snapshot = capacity.ResourceSnapshot(
            artifact_free_bytes=32 * 1024**3,
            available_memory_bytes=64 * 1024**3,
            tmpdir=pathlib.Path("/data4/jjgong/tmp"),
            tmpdir_writable=True,
        )
        with tempfile.TemporaryDirectory(dir="/data4/jjgong/tmp") as temp:
            first = pathlib.Path(temp) / "first.csv"
            second = pathlib.Path(temp) / "second.csv"
            capacity.write_preflight_csv(capacity.DEFAULT_POINTS, snapshot, first)
            capacity.write_preflight_csv(capacity.DEFAULT_POINTS, snapshot, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotIn(b"\r\n", first.read_bytes())
            with first.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            list(rows[0]),
            [
                "rows", "dim", "worker_cores", "band_cores",
                "mem_node_size", "timeout_sec", "elements", "tensor_bytes",
                "rowmajor_region_end", "bias_base", "layout_margin_bytes",
                "expected_reduction_each", "expected_transport_total",
                "expected_dma_ops", "expected_dma_bytes",
                "artifact_free_bytes", "available_memory_bytes", "tmpdir",
                "status",
            ],
        )
        self.assertEqual([int(row["rows"]) for row in rows], [512, 1024, 2048, 4096])
        self.assertTrue(all(row["status"] == "PASS" for row in rows))

    def test_collect_child_reuses_generic_phase4f_parser(self):
        point = capacity.DEFAULT_POINTS[0]
        spec = capacity.to_phase4f_spec(point)
        with tempfile.TemporaryDirectory(dir="/data4/jjgong/tmp") as temp:
            root = pathlib.Path(temp)
            child = phase4f_test_support.SyntheticChild(root / "child", spec)
            manifest = root / "capacity_manifest.csv"
            record = capacity.collect_child(
                child.root, point, child.verifier, manifest
            )
            loaded = capacity.phase4f.load_parent_manifest(manifest)

        self.assertEqual(record.spec.stage, "CAP")
        self.assertEqual(record.golden_checked, point.rows * point.dim)
        self.assertEqual(loaded, [record])

    @staticmethod
    def record_for(point, child_root):
        evidence = capacity.derive_capacity(point)
        return capacity.phase4f.PointRecord(
            spec=capacity.to_phase4f_spec(point),
            run_id=(
                f"sfu_job_dist_r{point.rows}_d{point.dim}_w16_bc16_g1_vn0"
            ),
            chunk_elems=256,
            cooperative_groups=1,
            transport="explicit_noc",
            reduction_vn=0,
            num_vns=3,
            dma_response_vn=0,
            noc_link_bw="1200GB/s",
            noc_xbar_bw="1200GB/s",
            dirctrl_highlink_bw="1200GB/s",
            noc_input_buffer="512KB",
            noc_output_buffer="512KB",
            gm_buffer="1024KB",
            flit_size="128B",
            inter_router_no_cut=0,
            local_no_cut=0,
            retry_ticks=1024,
            max_retries=8,
            status="PASS",
            exit_code=0,
            artifact_validation="PASS",
            golden_checked=evidence.elements,
            golden_mismatches=0,
            transport_events=evidence.expected_transport_total,
            transport_immediate=evidence.expected_transport_total,
            transport_queued=0,
            transport_rejected=0,
            transport_stale=0,
            inbox_high_water=4,
            latency_avg_cycles=10.0,
            latency_max_cycles=20,
            total_send_packets=100,
            total_send_bits=1000,
            total_xbar_stalls=0,
            simulated_time_us=float(point.rows),
            wall_time_sec=float(point.rows * 2),
            dma_timeout_retry=0,
            dma_timeout_exhausted=0,
            dma_write_timeout_retry=0,
            output_sha256=hashlib.sha256(str(point.rows).encode()).hexdigest(),
            child_root=str(child_root),
        )

    def test_report_reparses_all_four_children_and_is_deterministic(self):
        with tempfile.TemporaryDirectory(dir="/data4/jjgong/tmp") as temp:
            root = pathlib.Path(temp)
            records = []
            for point in capacity.DEFAULT_POINTS:
                child_root = root / "children" / f"r{point.rows}" / "attempt-0001"
                child_root.mkdir(parents=True)
                record = self.record_for(point, child_root)
                capacity.phase4f.upsert_parent_manifest(
                    root / "capacity_manifest.csv", record
                )
                records.append(record)

            first = root / "report-a"
            second = root / "report-b"
            with mock.patch.object(
                capacity.phase4f, "parse_child_point", side_effect=records + records
            ) as parser:
                capacity.write_capacity_report(root, first, pathlib.Path("verifier.py"))
                capacity.write_capacity_report(root, second, pathlib.Path("verifier.py"))
            self.assertEqual(parser.call_count, 8)
            for name in (
                "sfu_4096x4096_capacity_source_data.csv",
                "sfu_4096x4096_capacity_summary.md",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            summary = (first / "sfu_4096x4096_capacity_summary.md").read_text(
                encoding="utf-8"
            )

        self.assertIn("16,777,216", summary)
        self.assertIn("65,536", summary)
        self.assertIn("262,144", summary)
        self.assertIn("67,108,864", summary)

    def test_report_rejects_incomplete_matrix(self):
        with tempfile.TemporaryDirectory(dir="/data4/jjgong/tmp") as temp:
            root = pathlib.Path(temp)
            point = capacity.DEFAULT_POINTS[0]
            record = self.record_for(point, root / "child")
            capacity.phase4f.upsert_parent_manifest(
                root / "capacity_manifest.csv", record
            )
            with self.assertRaises(ValueError):
                capacity.write_capacity_report(
                    root, root / "report", pathlib.Path("verifier.py")
                )


class CapacityParentRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/data4/jjgong/tmp")
        self.base = pathlib.Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def env(self, root, **updates):
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith("GOLEM_") and key != "TMPDIR"
        }
        env.update(
            {
                "TMPDIR": "/data4/jjgong/tmp",
                "GOLEM_SFU_CAPACITY_ROOT": str(root),
                "GOLEM_SFU_CAPACITY_DRY_RUN": "1",
            }
        )
        env.update({key: str(value) for key, value in updates.items()})
        return env

    def run_parent(self, root, env=None):
        return subprocess.run(
            ["/bin/bash", str(RUNNER)],
            cwd=RUNNER.parent,
            env=env or self.env(root),
            check=False,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def attempt(root, point, number=1):
        return (
            pathlib.Path(root)
            / "children"
            / f"r{point.rows}_d4096_w16_b16"
            / f"attempt-{number:04d}"
        )

    @staticmethod
    def marker(root, point):
        return pathlib.Path(root) / "completed" / f"r{point.rows}_d4096_w16_b16.marker"

    def test_default_dry_run_expands_exact_ladder_and_watchdogs(self):
        root = self.base / "dry-run"
        result = self.run_parent(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [
            line for line in result.stdout.splitlines()
            if line.startswith("[SFU-CAPACITY][DRY-RUN]")
        ]
        self.assertEqual(len(lines), 4, result.stdout)
        for point, line in zip(capacity.DEFAULT_POINTS, lines):
            self.assertIn(f"point={point.rows}:4096:16:16", line)
            self.assertIn(f"timeout_sec={point.timeout_sec}", line)
            self.assertIn("mem_node_size=268435456", line)
            child_root = self.attempt(root, point)
            self.assertIn(f"child_root={child_root}", line)
            with (child_root / "sweep_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "DRYRUN")
            self.assertEqual(rows[0]["timeout_sec"], str(point.timeout_sec))

    def test_conflicts_invalid_prefix_schema_and_lock_fail_closed(self):
        conflicts = {
            "GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT": "modeled_noc",
            "GOLEM_SFU_VN_SWEEP": "0",
            "GOLEM_SFU_REDUCTION_VN": "1",
            "GOLEM_DMA_RESPONSE_VN": "1",
            "GOLEM_NOC_LINK_BW": "25GB/s",
            "GOLEM_NOC_XBAR_BW": "25GB/s",
            "GOLEM_DIRCTRL_HIGHLINK_BW": "25GB/s",
            "GOLEM_NOC_INPUT_BUF_SIZE": "8KB",
            "GOLEM_NOC_OUTPUT_BUF_SIZE": "8KB",
            "GOLEM_NOC_FLIT_SIZE": "64B",
            "GOLEM_GM_BUFFER_LENGTH": "512KB",
            "GOLEM_SFU_DISTRIBUTED_CHUNK_ELEMS": "128",
            "GOLEM_SFU_DISTRIBUTED_STAGING_ROWS": "8",
            "GOLEM_SFU_DISTRIBUTED_JOB_ROWS": "8",
            "GOLEM_SFU_DISTRIBUTED_RETRY_TICKS": "2048",
            "GOLEM_SFU_DISTRIBUTED_MAX_RETRIES": "9",
            "GOLEM_SFU_CAPACITY_STOP_ON_FAIL": "0",
            "TMPDIR": "/tmp",
        }
        for index, (name, value) in enumerate(conflicts.items()):
            with self.subTest(name=name):
                root = self.base / f"conflict-{index}"
                result = self.run_parent(root, self.env(root, **{name: value}))
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(name, result.stderr)
                self.assertFalse((root / "children").exists())

        for index, points in enumerate(
            (
                "",
                "1024:4096:16:16",
                "512:4096:16:16 2048:4096:16:16",
                "512:8192:16:16",
            )
        ):
            root = self.base / f"invalid-points-{index}"
            result = self.run_parent(
                root, self.env(root, GOLEM_SFU_CAPACITY_POINT_LIST=points)
            )
            self.assertEqual(result.returncode, 2, result.stderr)

        old = self.base / "old-schema"
        old.mkdir()
        (old / "parent_schema").write_text("old\n", encoding="utf-8")
        result = self.run_parent(old)
        self.assertEqual(result.returncode, 2)
        self.assertIn("schema", result.stderr)

        locked = self.base / "locked"
        locked.mkdir()
        with (locked / ".capacity.lock").open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_parent(locked)
        self.assertEqual(result.returncode, 2)
        self.assertIn("locked", result.stderr)

    def fake_child_bin(self, outcome):
        helper = self.base / f"fake_child_{outcome.lower()}.py"
        helper.write_text(
            "import importlib.util, os, pathlib, struct\n"
            f"test_file = pathlib.Path({str(pathlib.Path(__file__).resolve())!r})\n"
            "spec = importlib.util.spec_from_file_location('capacity_test_fixture', test_file)\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "root = pathlib.Path(os.environ['GOLEM_SWEEP_ROOT'])\n"
            "rows, dim, workers, bands = map(int, os.environ['GOLEM_SFU_DISTRIBUTED_POINT_LIST'].split(':'))\n"
            "point = module.capacity.resolve_point(rows, dim, workers, bands)\n"
            "phase_spec = module.capacity.to_phase4f_spec(point)\n"
            f"outcome = {outcome!r}\n"
            "if outcome == 'PASS':\n"
            "    child = module.phase4f_test_support.SyntheticChild(root, phase_spec)\n"
            "    zero = struct.pack('<f', 0.0)\n"
            "    uniform = struct.pack('<f', 1.0 / point.dim)\n"
            "    (root / 'inputs' / f'softmax_logits_{point.rows}x{point.dim}.bin').write_bytes(zero * (point.rows * point.dim))\n"
            "    (root / 'outputs' / f'{child.run_id}.bin').write_bytes(uniform * (point.rows * point.dim))\n"
            "else:\n"
            "    (root / 'logs').mkdir(parents=True, exist_ok=True)\n"
            "    message = 'No network interfaces were found for out-of-band communications\\nMPI_Init failed\\n' if outcome == 'ENVIRONMENT_FAIL' else 'watchdog timeout\\n'\n"
            "    (root / 'logs' / 'timeout.log').write_text(message)\n",
            encoding="utf-8",
        )
        fake_bin = self.base / f"fake-bin-{outcome.lower()}"
        fake_bin.mkdir()
        fake_bash = fake_bin / "bash"
        exit_code = {"PASS": 0, "TIMEOUT": 124, "ENVIRONMENT_FAIL": 1}[outcome]
        fake_bash.write_text(
            "#!/bin/sh\n"
            "printf '%s|%s\\n' \"${GOLEM_SFU_DISTRIBUTED_POINT_LIST}\" \"${GOLEM_SWEEP_ROOT}\" >> \"${FAKE_CHILD_LOG}\"\n"
            f"{sys.executable} {helper}\n"
            "helper_rc=$?\n"
            "[ \"${helper_rc}\" -eq 0 ] || exit \"${helper_rc}\"\n"
            f"exit {exit_code}\n",
            encoding="utf-8",
        )
        fake_bash.chmod(0o755)
        return fake_bin

    def test_watchdog_timeout_is_recorded_and_never_auto_retried(self):
        root = self.base / "timeout"
        child_log = self.base / "timeout-child.log"
        env = self.env(
            root,
            GOLEM_SFU_CAPACITY_DRY_RUN="0",
            GOLEM_SFU_CAPACITY_POINT_LIST="512:4096:16:16 1024:4096:16:16",
            FAKE_CHILD_LOG=child_log,
        )
        env["PATH"] = f"{self.fake_child_bin('TIMEOUT')}:{env['PATH']}"
        first = self.run_parent(root, env)
        self.assertEqual(first.returncode, 124, first.stderr)
        self.assertEqual(len(child_log.read_text().splitlines()), 1)
        status = (root / "capacity_status.csv").read_text(encoding="utf-8")
        self.assertIn("TIMEOUT,124", status)
        self.assertIn("timeout.log", status)
        marker = self.marker(root, capacity.DEFAULT_POINTS[0]).read_text(encoding="utf-8")
        self.assertIn("state=TIMEOUT", marker)
        self.assertIn("exit_code=124", marker)
        self.assertIn("wall_time_sec=", marker)
        self.assertIn("log_path=", marker)

        second = self.run_parent(root, env)
        self.assertEqual(second.returncode, 124, second.stderr)
        self.assertEqual(len(child_log.read_text().splitlines()), 1)
        self.assertIn("recorded TIMEOUT", second.stderr)

    def test_dryrun_then_pass_uses_new_attempt_and_cached_pass_revalidates(self):
        point = capacity.DEFAULT_POINTS[0]
        root = self.base / "resume"
        dry_env = self.env(
            root, GOLEM_SFU_CAPACITY_POINT_LIST="512:4096:16:16"
        )
        dry = self.run_parent(root, dry_env)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertTrue(self.attempt(root, point, 1).is_dir())

        child_log = self.base / "pass-child.log"
        real_env = self.env(
            root,
            GOLEM_SFU_CAPACITY_DRY_RUN="0",
            GOLEM_SFU_CAPACITY_POINT_LIST="512:4096:16:16",
            FAKE_CHILD_LOG=child_log,
        )
        real_env["PATH"] = f"{self.fake_child_bin('PASS')}:{real_env['PATH']}"
        passed = self.run_parent(root, real_env)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self.assertTrue(self.attempt(root, point, 2).is_dir())
        self.assertEqual(len(child_log.read_text().splitlines()), 1)

        cached = self.run_parent(root, real_env)
        self.assertEqual(cached.returncode, 0, cached.stderr)
        self.assertIn("validated cached PASS", cached.stdout)
        self.assertEqual(len(child_log.read_text().splitlines()), 1)

    def test_mpi_environment_failure_is_archived_before_explicit_rerun(self):
        point = capacity.DEFAULT_POINTS[0]
        root = self.base / "environment-fail"
        child_log = self.base / "environment-child.log"
        env = self.env(
            root,
            GOLEM_SFU_CAPACITY_DRY_RUN="0",
            GOLEM_SFU_CAPACITY_POINT_LIST="512:4096:16:16",
            FAKE_CHILD_LOG=child_log,
        )
        env["PATH"] = f"{self.fake_child_bin('ENVIRONMENT_FAIL')}:{env['PATH']}"
        failed = self.run_parent(root, env)
        self.assertEqual(failed.returncode, 1, failed.stderr)
        marker = self.marker(root, point)
        self.assertIn("state=ENVIRONMENT_FAIL", marker.read_text(encoding="utf-8"))

        retry_env = dict(env)
        retry_env["PATH"] = f"{self.fake_child_bin('PASS')}:{os.environ['PATH']}"
        passed = self.run_parent(root, retry_env)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self.assertTrue(self.attempt(root, point, 2).is_dir())
        archives = list((root / "completed").glob("*.environment-fail.marker"))
        self.assertEqual(len(archives), 1)
        self.assertIn("state=ENVIRONMENT_FAIL", archives[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
