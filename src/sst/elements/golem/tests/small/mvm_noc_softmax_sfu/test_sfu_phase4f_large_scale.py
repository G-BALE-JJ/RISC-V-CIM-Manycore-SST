import dataclasses
import csv
import fcntl
import hashlib
import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).with_name("plot_sfu_phase4f_large_scale.py")
RUNNER = pathlib.Path(__file__).with_name(
    "run_sfu_phase4f_large_scale_explicit_noc.sh"
)
SPEC = importlib.util.spec_from_file_location("phase4f_large_scale", SCRIPT)
phase4f = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase4f)


class Phase4FLargeScaleContractTest(unittest.TestCase):
    def test_canonical_network_matches_gemm_preset(self):
        self.assertEqual(
            phase4f.CANONICAL_NETWORK,
            {
                "GOLEM_NOC_LINK_BW": "1200GB/s",
                "GOLEM_NOC_XBAR_BW": "1200GB/s",
                "GOLEM_DIRCTRL_HIGHLINK_BW": "1200GB/s",
                "GOLEM_NOC_INPUT_BUF_SIZE": "512KB",
                "GOLEM_NOC_OUTPUT_BUF_SIZE": "512KB",
                "GOLEM_NOC_FLIT_SIZE": "128B",
                "GOLEM_GM_BUFFER_LENGTH": "1024KB",
                "GOLEM_NOC_INTER_ROUTER_NO_CUT": "0",
                "GOLEM_NOC_LOCAL_NO_CUT": "0",
            },
        )

    def test_transport_and_virtual_networks_are_fixed(self):
        self.assertEqual(phase4f.TRANSPORT, "explicit_noc")
        self.assertEqual(phase4f.NUM_VNS, 3)
        self.assertEqual(phase4f.REDUCTION_VN, 0)
        self.assertEqual(phase4f.DMA_RESPONSE_VN, 0)

    def test_point_models_are_frozen_and_record_fields_are_planned(self):
        self.assertEqual(
            [field.name for field in dataclasses.fields(phase4f.PointSpec)],
            [
                "stage",
                "rows",
                "dim",
                "worker_cores",
                "band_cores",
                "mem_node_size",
                "timeout_sec",
            ],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(phase4f.PointRecord)],
            [
                "spec",
                "run_id",
                "chunk_elems",
                "cooperative_groups",
                "transport",
                "reduction_vn",
                "num_vns",
                "dma_response_vn",
                "noc_link_bw",
                "noc_xbar_bw",
                "dirctrl_highlink_bw",
                "noc_input_buffer",
                "noc_output_buffer",
                "gm_buffer",
                "flit_size",
                "retry_ticks",
                "max_retries",
                "status",
                "exit_code",
                "artifact_validation",
                "golden_checked",
                "golden_mismatches",
                "transport_events",
                "transport_immediate",
                "transport_queued",
                "transport_rejected",
                "transport_stale",
                "inbox_high_water",
                "latency_avg_cycles",
                "latency_max_cycles",
                "total_send_packets",
                "total_send_bits",
                "total_xbar_stalls",
                "simulated_time_us",
                "wall_time_sec",
                "dma_timeout_retry",
                "dma_timeout_exhausted",
                "dma_write_timeout_retry",
                "output_sha256",
                "child_root",
            ],
        )
        self.assertTrue(phase4f.PointSpec.__dataclass_params__.frozen)
        self.assertTrue(phase4f.PointRecord.__dataclass_params__.frozen)
        point = phase4f.DEFAULT_POINTS[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            point.rows = 32

    def test_default_points_have_canonical_order_and_derived_values(self):
        identities = [
            (point.rows, point.dim, point.worker_cores, point.band_cores)
            for point in phase4f.DEFAULT_POINTS
        ]
        self.assertEqual(
            identities,
            [
                (16, 512, 16, 16),
                (16, 1024, 16, 16),
                (16, 2048, 16, 16),
                (16, 4096, 16, 16),
                (16, 4096, 4, 4),
                (16, 4096, 8, 8),
                (64, 4096, 16, 16),
                (256, 4096, 16, 16),
            ],
        )
        self.assertEqual([point.stage for point in phase4f.DEFAULT_POINTS], list("AAAABBCC"))
        self.assertEqual(
            [point.mem_node_size for point in phase4f.DEFAULT_POINTS],
            [134217728, 134217728, 268435456, 268435456,
             268435456, 268435456, 268435456, 268435456],
        )
        self.assertEqual(
            [point.timeout_sec for point in phase4f.DEFAULT_POINTS],
            [900, 1800, 2400, 3600, 3600, 3600, 7200, 14400],
        )
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(identities.count((16, 4096, 16, 16)), 1)

    def test_resolve_point_returns_canonical_identity(self):
        self.assertIs(
            phase4f.resolve_point(16, 4096, 16, 16),
            phase4f.DEFAULT_POINTS[3],
        )
        self.assertEqual(phase4f.resolve_point(16, 4096, 4, 4).stage, "B")
        self.assertEqual(phase4f.resolve_point(256, 4096, 16, 16).stage, "C")

    def test_parse_point_list_uses_defaults_and_preserves_override_order(self):
        self.assertIs(phase4f.parse_point_list(None), phase4f.DEFAULT_POINTS)
        self.assertEqual(
            phase4f.parse_point_list("16:4096:8:8 16:512:16:16"),
            (phase4f.DEFAULT_POINTS[5], phase4f.DEFAULT_POINTS[0]),
        )

    def test_invalid_points_are_rejected(self):
        invalid_values = [
            "",
            "16:512:16",
            "16:512:16:16:1",
            "16:512:sixteen:16",
            "0:512:16:16",
            "16:-512:16:16",
            "16:4096:1:1",
            "16:4096:2:2",
            "16:4096:4:8",
            "32:4096:16:16",
            "16:8192:16:16",
            "16:512:16:16 16:512:16:16",
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                phase4f.parse_point_list(value)

        invalid_args = [
            (0, 512, 16, 16),
            (16, 4096, 1, 1),
            (16, 4096, 4, 8),
            (64, 4096, 8, 8),
        ]
        for args in invalid_args:
            with self.subTest(args=args), self.assertRaises(ValueError):
                phase4f.resolve_point(*args)


class SyntheticChild:
    MANIFEST_FIELDS = [
        "run_id", "rows", "dim", "chunk_elems", "worker_cores",
        "band_cores", "cooperative_groups", "reduction_vn", "num_vns",
        "dma_response_vn", "staging_rows", "job_rows", "retry_ticks",
        "max_retries", "status", "exit_code", "timeout_sec",
        "artifact_validation",
    ]

    def __init__(self, root: pathlib.Path, spec: phase4f.PointSpec):
        self.root = root
        self.spec = spec
        self.run_id = (
            f"sfu_job_dist_r{spec.rows}_d{spec.dim}_w{spec.worker_cores}"
            f"_bc{spec.band_cores}_g1_vn0"
        )
        self.stats_dir = root / "stats" / "overlap0" / self.run_id
        self.stdout_dir = root / "stdout" / "overlap0" / self.run_id
        self.log = root / "logs" / f"test_default_{self.run_id}.log"
        self.verifier = root / "verifier.py"
        self.manifest_row = {
            "run_id": self.run_id,
            "rows": str(spec.rows),
            "dim": str(spec.dim),
            "chunk_elems": "256",
            "worker_cores": str(spec.worker_cores),
            "band_cores": str(spec.band_cores),
            "cooperative_groups": "1",
            "reduction_vn": "0",
            "num_vns": "3",
            "dma_response_vn": "0",
            "staging_rows": "4",
            "job_rows": "4",
            "retry_ticks": "1024",
            "max_retries": "8",
            "status": "PASS",
            "exit_code": "0",
            "timeout_sec": str(spec.timeout_sec),
            "artifact_validation": "PASS",
        }
        self._build()

    @staticmethod
    def write_csv(path: pathlib.Path, fields: list[str], rows: list[dict[str, object]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_manifest(self, rows=None, fields=None):
        selected_fields = fields or self.MANIFEST_FIELDS
        self.write_csv(
            self.root / "sweep_manifest.csv",
            selected_fields,
            [
                {field: row[field] for field in selected_fields}
                for row in (rows or [self.manifest_row])
            ],
        )

    def write_verifier(self, checked=None, mismatches=0, exit_code=0):
        checked = checked if checked is not None else self.spec.rows * self.spec.dim
        self.verifier.write_text(
            "import sys\n"
            f"print('[VERIFY-SFU-SOFTMAX] PASS checked={checked} mismatches={mismatches}')\n"
            f"raise SystemExit({exit_code})\n",
            encoding="utf-8",
        )

    def _build(self):
        for path in (
            self.stats_dir,
            self.stdout_dir,
            self.root / "logs",
            self.root / "inputs",
            self.root / "outputs",
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.write_manifest()
        self.write_verifier()
        for name in ("a.bin", "b.bin", f"softmax_logits_{self.spec.rows}x{self.spec.dim}.bin"):
            (self.root / "inputs" / name).write_bytes(b"fixture")
        (self.root / "outputs" / f"{self.run_id}.bin").write_bytes(
            b"\0" * (self.spec.rows * self.spec.dim * 4)
        )
        topology = (
            "[NoC] input_buf_size=512KB, output_buf_size=512KB, "
            "link_bw=1200GB/s, xbar_bw=1200GB/s, flit_size=128B\n"
            "[NoC] inter_router_no_cut=0, local_no_cut=0\n"
            "[GOLEM] GlobalMemory link buffer_length=1024KB\n"
            "GlobalMemory VN mapping: request_vn=0 response_vn=1 reduction_vn=0 (num_vns=3)\n"
            "resolved golem_dma_response_vn=0 num_vns=3 explicit=1\n"
        )
        self.log.write_text(
            "".join(f"[component0] {line}\n" for line in topology.splitlines())
            + "".join(f"[component1] {line}\n" for line in topology.splitlines())
            + "Simulation is complete, simulated time: 2.5 ms\n",
            encoding="utf-8",
        )
        pass_line = (
            f"mode=sfu-standalone-job-softmax rows={self.spec.rows} dim={self.spec.dim} "
            f"worker_cores={self.spec.worker_cores} staging_rows=4 job_rows=4 "
            f"band_cores={self.spec.band_cores} distributed_columns=1 PASS\n"
        )
        for core in range(self.spec.band_cores):
            (self.stdout_dir / f"stdout-{core}").write_text(pass_line, encoding="utf-8")

        expected_worker_rows = self.spec.rows * self.spec.worker_cores
        stats = []
        for statistic in (
            "sfu_ops_issued", "sfu_job_softmax_max_chunks",
            "sfu_job_softmax_sum_chunks", "sfu_job_softmax_norm_chunks",
        ):
            for core in range(self.spec.band_cores):
                stats.append(self.stat(statistic, 1, component=f"core{core}"))
        for statistic in (
            "sfu_reduction_max_requests", "sfu_reduction_max_responses",
            "sfu_reduction_sum_requests", "sfu_reduction_sum_responses",
        ):
            stats.append(self.stat(statistic, expected_worker_rows))
        transport_total = 4 * expected_worker_rows
        stats.extend([
            self.stat("sfu_reduction_transport_received", transport_total),
            self.stat("gmem_reduction_send_immediate", transport_total - 7),
            self.stat("gmem_reduction_send_queued", 7),
            self.stat("gmem_reduction_send_rejected", 0),
            self.stat("gmem_reduction_received", transport_total),
            self.stat("sfu_reduction_transport_stale_dropped", 0),
            self.stat("sfu_reduction_transport_latency_cycles", 90, count=3, maximum=40),
            self.stat("sfu_reduction_transport_latency_cycles", 110, count=2, maximum=70),
            self.stat("sfu_reduction_transport_inbox_high_water", 3),
            self.stat("sfu_reduction_transport_inbox_high_water", 5),
        ])
        self.write_stats(stats)
        dma_ops = expected_worker_rows
        dma_bytes = self.spec.rows * self.spec.dim * 4
        self.write_metrics("dma_summary.csv", "sum", {
            "read_issue_count": dma_ops,
            "write_issue_count": dma_ops,
            "completion": dma_ops,
            "write_completion": dma_ops,
            "read_bytes_total": dma_bytes,
            "write_bytes_total": dma_bytes,
            "timeout_retry": 0,
            "timeout_exhausted": 0,
            "write_timeout_retry": 0,
        })
        self.write_metrics("noc_summary.csv", "value", {
            "total_send_packets": 1234,
            "total_send_bits": 98765,
            "total_xbar_stalls": 42,
        })
        self.write_csv(self.root / "stats" / "run_summary.csv", [
            "run_id", "noc_link_bw", "noc_xbar_bw", "noc_flit_size",
            "dirctrl_highlink_bw", "mem_node_size_bytes", "wall_time_sec",
        ], [{
            "run_id": self.run_id,
            "noc_link_bw": "1200GB/s",
            "noc_xbar_bw": "1200GB/s",
            "noc_flit_size": "128B",
            "dirctrl_highlink_bw": "1200GB/s",
            "mem_node_size_bytes": self.spec.mem_node_size,
            "wall_time_sec": "12.75",
        }])

    @staticmethod
    def stat(name, total, count=1, maximum=None, component="component"):
        return {
            "ComponentName": component,
            "StatisticName": name,
            "Sum.u64": total,
            "Count.u64": count,
            "Max.u64": total if maximum is None else maximum,
        }

    def write_stats(self, rows):
        self.write_csv(
            self.stats_dir / "stats_selfcom.txt",
            ["ComponentName", "StatisticName", "Sum.u64", "Count.u64", "Max.u64"],
            rows,
        )

    def read_stats(self):
        with (self.stats_dir / "stats_selfcom.txt").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def write_metrics(self, name, value_field, values):
        self.write_csv(
            self.stats_dir / name,
            ["metric", value_field],
            [{"metric": metric, value_field: value} for metric, value in values.items()],
        )


class Phase4FParentRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def env(self, root, **updates):
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith("GOLEM_")
        }
        env.update({
            "GOLEM_PHASE4F_LARGE_SCALE_ROOT": str(root),
            "GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN": "1",
        })
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
    def child_base(root, spec):
        identity = (
            f"stage_{spec.stage}_r{spec.rows}_d{spec.dim}_"
            f"w{spec.worker_cores}_b{spec.band_cores}"
        )
        return pathlib.Path(root) / "children" / identity

    @classmethod
    def child_root(cls, root, spec, attempt=1):
        return cls.child_base(root, spec) / f"attempt-{attempt:04d}"

    @staticmethod
    def marker(root, spec):
        identity = (
            f"stage_{spec.stage}_r{spec.rows}_d{spec.dim}_"
            f"w{spec.worker_cores}_b{spec.band_cores}"
        )
        return pathlib.Path(root) / "completed" / f"{identity}.marker"

    def test_default_dry_run_orchestrates_exact_canonical_matrix(self):
        root = self.base / "default"
        result = self.run_parent(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [
            line for line in result.stdout.splitlines()
            if line.startswith("[PHASE4F][DRY-RUN] ")
        ]
        self.assertEqual(len(lines), 8, result.stdout)
        for spec, line in zip(phase4f.DEFAULT_POINTS, lines):
            identity = f"{spec.rows}:{spec.dim}:{spec.worker_cores}:{spec.band_cores}"
            child_root = self.child_root(root, spec, 1)
            self.assertIn(f"point={identity}", line)
            self.assertIn(f"stage={spec.stage}", line)
            self.assertIn(f"mem_node_size={spec.mem_node_size}", line)
            self.assertIn(f"timeout_sec={spec.timeout_sec}", line)
            self.assertIn(f"child_root={child_root}", line)
            manifest = child_root / "sweep_manifest.csv"
            self.assertTrue(manifest.is_file(), line)
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "DRYRUN")
            self.assertEqual(rows[0]["timeout_sec"], str(spec.timeout_sec))
        self.assertEqual(len({line.split(" point=", 1)[1] for line in lines}), 8)

    def test_conflicting_fixed_environment_is_rejected_before_child_artifact(self):
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
            "GOLEM_NOC_INTER_ROUTER_NO_CUT": "1",
            "GOLEM_NOC_LOCAL_NO_CUT": "1",
            "GOLEM_SFU_DISTRIBUTED_CHUNK_ELEMS": "128",
            "GOLEM_SFU_DISTRIBUTED_STAGING_ROWS": "8",
            "GOLEM_SFU_DISTRIBUTED_JOB_ROWS": "8",
            "GOLEM_SFU_DISTRIBUTED_RETRY_TICKS": "2048",
            "GOLEM_SFU_DISTRIBUTED_MAX_RETRIES": "9",
            "GOLEM_SFU_DISTRIBUTED_PIPELINE_ARGS": "--dry-run",
        }
        for index, (name, value) in enumerate(conflicts.items()):
            with self.subTest(name=name):
                root = self.base / f"conflict-{index}"
                result = self.run_parent(root, self.env(root, **{name: value}))
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(name, result.stderr)
                self.assertFalse((root / "children").exists())

    def test_root_point_schema_and_lock_guards(self):
        relative = self.run_parent(
            self.base / "unused",
            self.env("relative-root"),
        )
        self.assertEqual(relative.returncode, 2)
        self.assertIn("absolute", relative.stderr)

        for index, points in enumerate((
            "",
            "16:512:16:16 16:512:16:16",
            "16:512:16",
            "16:4096:1:1",
        )):
            root = self.base / f"points-{index}"
            result = self.run_parent(root, self.env(
                root, GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST=points,
            ))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse((root / "children").exists())

        old = self.base / "old-schema"
        old.mkdir()
        (old / "parent_schema").write_text("phase4f-parent-v0\n", encoding="utf-8")
        result = self.run_parent(old)
        self.assertEqual(result.returncode, 2)
        self.assertIn("schema", result.stderr)

        old_manifest = self.base / "old-manifest"
        old_manifest.mkdir()
        (old_manifest / "parent_schema").write_text(
            "phase4f-parent-v1\n", encoding="utf-8"
        )
        (old_manifest / "large_scale_manifest.csv").write_text(
            "old,header\n1,2\n", encoding="utf-8"
        )
        result = self.run_parent(old_manifest)
        self.assertEqual(result.returncode, 2)
        self.assertIn("manifest schema", result.stderr)

        locked = self.base / "locked"
        locked.mkdir()
        lock_path = locked / ".phase4f.lock"
        with lock_path.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_parent(locked)
        self.assertEqual(result.returncode, 2)
        self.assertIn("locked", result.stderr)

    def _fake_bash(self, exit_code):
        fake_bin = self.base / f"fake-bin-{exit_code}"
        fake_bin.mkdir(exist_ok=True)
        fake = fake_bin / "bash"
        fake.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"${GOLEM_SFU_DISTRIBUTED_POINT_LIST}\" >> \"${FAKE_CHILD_LOG}\"\n"
            f"exit {exit_code}\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        return fake_bin

    def _fake_child_bash(self, outcome):
        helper = self.base / "fake_child.py"
        helper.write_text(
            "import csv, importlib.util, os, pathlib, struct, sys\n"
            f"test_file = pathlib.Path({str(pathlib.Path(__file__).resolve())!r})\n"
            "module_spec = importlib.util.spec_from_file_location('phase4f_test_fixture', test_file)\n"
            "module = importlib.util.module_from_spec(module_spec)\n"
            "module_spec.loader.exec_module(module)\n"
            "root = pathlib.Path(os.environ['GOLEM_SWEEP_ROOT'])\n"
            "rows, dim, workers, bands = map(int, os.environ['GOLEM_SFU_DISTRIBUTED_POINT_LIST'].split(':'))\n"
            "point = module.phase4f.resolve_point(rows, dim, workers, bands)\n"
            "outcome = os.environ['FAKE_CHILD_OUTCOME']\n"
            "if outcome == 'PASS':\n"
            "    child = module.SyntheticChild(root, point)\n"
            "    zero = struct.pack('<f', 0.0)\n"
            "    uniform = struct.pack('<f', 1.0 / point.dim)\n"
            "    (root / 'inputs' / 'a.bin').write_bytes(zero * (point.rows * point.dim))\n"
            "    (root / 'inputs' / 'b.bin').write_bytes(zero * (point.dim * point.dim))\n"
            "    (root / 'inputs' / f'softmax_logits_{point.rows}x{point.dim}.bin').write_bytes(zero * (point.rows * point.dim))\n"
            "    (root / 'outputs' / f'{child.run_id}.bin').write_bytes(uniform * (point.rows * point.dim))\n"
            "else:\n"
            "    run_id = f'sfu_job_dist_r{rows}_d{dim}_w{workers}_bc{bands}_g1_vn0'\n"
            "    row = {\n"
            "        'run_id': run_id, 'rows': rows, 'dim': dim, 'chunk_elems': 256,\n"
            "        'worker_cores': workers, 'band_cores': bands, 'cooperative_groups': 1,\n"
            "        'reduction_vn': 0, 'num_vns': 3, 'dma_response_vn': 0,\n"
            "        'staging_rows': 4, 'job_rows': 4, 'retry_ticks': 1024,\n"
            "        'max_retries': 8, 'status': outcome,\n"
            "        'exit_code': 124 if outcome == 'TIMEOUT' else 9,\n"
            "        'timeout_sec': point.timeout_sec, 'artifact_validation': 'NOT_RUN',\n"
            "    }\n"
            "    module.SyntheticChild.write_csv(root / 'sweep_manifest.csv', module.SyntheticChild.MANIFEST_FIELDS, [row])\n",
            encoding="utf-8",
        )
        fake_bin = self.base / f"fake-child-{outcome.lower()}"
        fake_bin.mkdir(exist_ok=True)
        fake = fake_bin / "bash"
        exit_code = {"PASS": 0, "TIMEOUT": 124, "FAIL": 9}[outcome]
        fake.write_text(
            "#!/bin/sh\n"
            "printf '%s|%s\\n' \"${GOLEM_SFU_DISTRIBUTED_POINT_LIST}\" \"${GOLEM_SWEEP_ROOT}\" >> \"${FAKE_CHILD_LOG}\"\n"
            f"FAKE_CHILD_OUTCOME={outcome} {sys.executable} {helper}\n"
            "helper_rc=$?\n"
            "[ \"${helper_rc}\" -eq 0 ] || exit \"${helper_rc}\"\n"
            f"exit {exit_code}\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        return fake_bin

    def test_stop_on_fail_and_stage_c_always_stop_without_downgrade(self):
        cases = (
            ("A", "16:512:16:16 16:1024:16:16", "1"),
            ("C", "64:4096:16:16 256:4096:16:16", "0"),
        )
        for index, (stage, points, stop_on_fail) in enumerate(cases):
            with self.subTest(stage=stage):
                root = self.base / f"stop-{index}"
                child_log = self.base / f"child-{index}.log"
                env = self.env(
                    root,
                    GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN="0",
                    GOLEM_PHASE4F_LARGE_SCALE_STOP_ON_FAIL=stop_on_fail,
                    GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST=points,
                    FAKE_CHILD_LOG=child_log,
                )
                env["PATH"] = f"{self._fake_bash(124)}:{env['PATH']}"
                result = self.run_parent(root, env)
                self.assertEqual(result.returncode, 124, result.stderr)
                self.assertEqual(child_log.read_text(encoding="utf-8").splitlines(), [points.split()[0]])
                status = (root / "point_status.csv").read_text(encoding="utf-8")
                self.assertIn("TIMEOUT,124", status)
                self.assertIn("1200GB/s", status)
                self.assertIn("1024,8", status)

    def test_dryrun_to_real_pass_uses_new_attempt_and_cached_pass_revalidates_it(self):
        spec = phase4f.DEFAULT_POINTS[0]
        root = self.base / "resume"
        env = self.env(
            root,
            GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="16:512:16:16",
        )
        dry = self.run_parent(root, env)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        marker = self.marker(root, spec)
        marker_text = marker.read_text(encoding="utf-8")
        self.assertIn("state=DRYRUN", marker_text)
        self.assertIn("schema=phase4f-parent-v1", marker_text)
        self.assertIn("child_runner_sha256=", marker_text)
        self.assertIn("pipeline_args_sha256=", marker_text)
        attempt1 = self.child_root(root, spec, 1)
        self.assertIn(f"child_root={attempt1}", marker_text)
        with (attempt1 / "sweep_manifest.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual([row["status"] for row in csv.DictReader(handle)], ["DRYRUN"])

        child_log = self.base / "resume-child.log"
        real_env = dict(
            env,
            GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN="0",
            FAKE_CHILD_LOG=str(child_log),
        )
        real_env["PATH"] = f"{self._fake_child_bash('PASS')}:{real_env['PATH']}"
        passed = self.run_parent(root, real_env)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        attempt2 = self.child_root(root, spec, 2)
        self.assertTrue(attempt1.is_dir())
        self.assertTrue(attempt2.is_dir())
        self.assertIn(f"child_root={attempt2}", marker.read_text(encoding="utf-8"))
        self.assertEqual(child_log.read_text(encoding="utf-8").splitlines(), [f"16:512:16:16|{attempt2}"])
        records = phase4f.load_parent_manifest(root / "large_scale_manifest.csv")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].child_root, str(attempt2))

        cached_env = dict(env, GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN="0")
        cached = self.run_parent(root, cached_env)
        self.assertEqual(cached.returncode, 0, cached.stderr)
        self.assertIn("validated cached PASS", cached.stdout)
        self.assertEqual(child_log.read_text(encoding="utf-8").splitlines(), [f"16:512:16:16|{attempt2}"])

        valid_marker = marker.read_text(encoding="utf-8")
        manifest = root / "large_scale_manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["wall_time_sec"] = "999.0"
        SyntheticChild.write_csv(manifest, phase4f.PARENT_MANIFEST_FIELDS, rows)
        manifest_before = manifest.read_bytes()
        marker.write_text(
            re.sub(r"output_sha256=[0-9a-f]{64}", f"output_sha256={'0' * 64}", valid_marker),
            encoding="utf-8",
        )
        hash_drift = self.run_parent(root, cached_env)
        self.assertEqual(hash_drift.returncode, 3, hash_drift.stderr)
        self.assertIn("hash drift", hash_drift.stderr)
        self.assertEqual(manifest.read_bytes(), manifest_before)
        marker.write_text(valid_marker, encoding="utf-8")

        child = SyntheticChild(attempt2, spec)
        output = attempt2 / "outputs" / f"{child.run_id}.bin"
        output.write_bytes(b"bad")
        rejected = self.run_parent(root, cached_env)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn(str(attempt2), rejected.stderr)
        self.assertNotIn("validated cached PASS", rejected.stdout)

    def test_timeout_and_fail_recovery_allocate_fresh_attempts(self):
        spec = phase4f.DEFAULT_POINTS[0]
        for outcome in ("TIMEOUT", "FAIL"):
            with self.subTest(outcome=outcome):
                root = self.base / f"recover-{outcome.lower()}"
                child_log = self.base / f"recover-{outcome.lower()}.log"
                env = self.env(
                    root,
                    GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN="0",
                    GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="16:512:16:16",
                    FAKE_CHILD_LOG=child_log,
                )
                env["PATH"] = f"{self._fake_child_bash(outcome)}:{env['PATH']}"
                failed = self.run_parent(root, env)
                self.assertEqual(failed.returncode, 124 if outcome == "TIMEOUT" else 9)
                attempt1 = self.child_root(root, spec, 1)
                with (attempt1 / "sweep_manifest.csv").open(newline="", encoding="utf-8") as handle:
                    self.assertEqual([row["status"] for row in csv.DictReader(handle)], [outcome])

                pass_env = dict(env)
                pass_env["PATH"] = f"{self._fake_child_bash('PASS')}:{os.environ['PATH']}"
                passed = self.run_parent(root, pass_env)
                self.assertEqual(passed.returncode, 0, passed.stderr)
                attempt2 = self.child_root(root, spec, 2)
                self.assertTrue(attempt1.is_dir())
                self.assertTrue(attempt2.is_dir())
                self.assertIn(f"child_root={attempt2}", self.marker(root, spec).read_text(encoding="utf-8"))

    @staticmethod
    def _tree_snapshot(root):
        return [
            (
                str(path.relative_to(root)),
                "symlink" if path.is_symlink() else "dir" if path.is_dir() else "file",
                path.read_bytes() if path.is_file() and not path.is_symlink() else b"",
            )
            for path in sorted(root.rglob("*"))
        ]

    def test_symlinked_attempt_ancestors_are_rejected_before_external_access(self):
        target = phase4f.DEFAULT_POINTS[0]

        root = self.base / "symlink-new"
        initialized = self.run_parent(
            root,
            self.env(
                root,
                GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="16:1024:16:16",
            ),
        )
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        external = self.base / "external-new"
        external.mkdir()
        self.child_base(root, target).symlink_to(external, target_is_directory=True)
        external_before = self._tree_snapshot(external)
        rejected = self.run_parent(
            root,
            self.env(
                root,
                GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="16:512:16:16",
            ),
        )
        self.assertIn(rejected.returncode, (2, 3), rejected.stderr)
        self.assertIn("symlink", rejected.stderr)
        self.assertEqual(self._tree_snapshot(external), external_before)

        resume_root = self.base / "symlink-resume"
        child_log = self.base / "symlink-resume.log"
        pass_env = self.env(
            resume_root,
            GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN="0",
            GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="16:512:16:16",
            FAKE_CHILD_LOG=child_log,
        )
        pass_env["PATH"] = f"{self._fake_child_bash('PASS')}:{pass_env['PATH']}"
        passed = self.run_parent(resume_root, pass_env)
        self.assertEqual(passed.returncode, 0, passed.stderr)

        attempt_base = self.child_base(resume_root, target)
        external_resume = self.base / "external-resume"
        external_resume.mkdir()
        linked_base = external_resume / "linked-base"
        attempt_base.rename(linked_base)
        attempt_base.symlink_to(linked_base, target_is_directory=True)
        external_resume_before = self._tree_snapshot(external_resume)

        manifest = resume_root / "large_scale_manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        manifest_rows[0]["wall_time_sec"] = "999.0"
        SyntheticChild.write_csv(manifest, phase4f.PARENT_MANIFEST_FIELDS, manifest_rows)
        manifest_before = manifest.read_bytes()

        cached_env = self.env(
            resume_root,
            GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN="0",
            GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="16:512:16:16",
        )
        rejected = self.run_parent(resume_root, cached_env)
        self.assertIn(rejected.returncode, (2, 3), rejected.stderr)
        self.assertIn("symlink", rejected.stderr)
        self.assertEqual(self._tree_snapshot(external_resume), external_resume_before)
        self.assertEqual(manifest.read_bytes(), manifest_before)

    def test_damaged_marker_and_signature_hash_drift_fail_closed(self):
        spec = phase4f.DEFAULT_POINTS[0]
        for index, replacement in enumerate(("garbage\n", None)):
            root = self.base / f"marker-{index}"
            env = self.env(
                root,
                GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="16:512:16:16",
            )
            first = self.run_parent(root, env)
            self.assertEqual(first.returncode, 0, first.stderr)
            marker = self.marker(root, spec)
            if replacement is None:
                replacement = marker.read_text(encoding="utf-8").replace(
                    "noc_link_bw=1200GB/s", "noc_link_bw=25GB/s"
                )
            marker.write_text(replacement, encoding="utf-8")
            result = self.run_parent(root, env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("marker", result.stderr)

        root = self.base / "marker-child-root"
        env = self.env(
            root,
            GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="16:512:16:16",
        )
        first = self.run_parent(root, env)
        self.assertEqual(first.returncode, 0, first.stderr)
        marker = self.marker(root, spec)
        marker.write_text(
            re.sub(
                r"^child_root=.*$",
                f"child_root={root / 'outside' / 'attempt-0001'}",
                marker.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            ),
            encoding="utf-8",
        )
        rejected = self.run_parent(root, env)
        self.assertEqual(rejected.returncode, 2, rejected.stderr)
        self.assertIn("marker", rejected.stderr)


class Phase4FArtifactParserTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name) / "child"
        self.spec = phase4f.DEFAULT_POINTS[0]
        self.child = SyntheticChild(self.root, self.spec)

    def tearDown(self):
        self.temp.cleanup()

    def assert_context_error(self, field, action):
        with self.assertRaises(ValueError) as caught:
            action()
        message = str(caught.exception)
        self.assertIn(str(self.root), message)
        self.assertIn(self.child.run_id, message)
        self.assertIn(field, message)

    def test_manifest_selects_unique_canonical_and_ignores_cached_row(self):
        cached = dict(self.child.manifest_row, artifact_validation="CACHED")
        self.child.write_manifest([self.child.manifest_row, cached])
        selected = phase4f.select_child_manifest_row(self.root, self.spec)
        self.assertEqual(selected, self.child.manifest_row)

    def test_manifest_rejects_duplicate_old_schema_and_identity_mismatch(self):
        cases = [
            ("artifact_validation", [self.child.manifest_row, self.child.manifest_row], None),
            ("sweep_manifest.csv", [self.child.manifest_row], self.child.MANIFEST_FIELDS[:-1]),
            ("dim", [dict(self.child.manifest_row, dim="1024")], None),
            ("reduction_vn", [dict(self.child.manifest_row, reduction_vn="1")], None),
        ]
        for field, rows, fields in cases:
            with self.subTest(field=field):
                self.child.write_manifest(rows, fields)
                self.assert_context_error(
                    field,
                    lambda: phase4f.select_child_manifest_row(self.root, self.spec),
                )

    def test_manifest_rejects_noncanonical_run_id_even_with_vn_suffix(self):
        self.child.write_manifest([dict(self.child.manifest_row, run_id="foo_vn0")])
        with self.assertRaisesRegex(
            ValueError,
            rf"child_root={re.escape(str(self.root))} run_id=foo_vn0 field=run_id",
        ):
            phase4f.select_child_manifest_row(self.root, self.spec)

    def test_manifest_rejects_previous_incorrect_run_id_template(self):
        old_run_ids = (
            (
                f"sfu_unified_job_r{self.spec.rows}_d{self.spec.dim}_c256_"
                f"w{self.spec.worker_cores}_b{self.spec.band_cores}_g1_vn0"
            ),
            (
                f"sfu_job_dist_r{self.spec.rows}_d{self.spec.dim}_c256_"
                f"w{self.spec.worker_cores}_bc{self.spec.band_cores}_g1_vn0"
            ),
        )
        for old_run_id in old_run_ids:
            with self.subTest(old_run_id=old_run_id):
                self.child.write_manifest([
                    dict(self.child.manifest_row, run_id=old_run_id)
                ])
                with self.assertRaisesRegex(ValueError, r"field=run_id"):
                    phase4f.select_child_manifest_row(self.root, self.spec)

    def test_parse_canonical_child_point_aggregates_all_evidence(self):
        record = phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
        self.assertEqual(record.run_id, self.child.run_id)
        self.assertEqual(record.cooperative_groups, 1)
        self.assertEqual(record.golden_checked, self.spec.rows * self.spec.dim)
        self.assertEqual(record.transport_events, 4 * self.spec.rows * self.spec.worker_cores)
        self.assertEqual(record.transport_immediate, record.transport_events - 7)
        self.assertEqual(record.transport_queued, 7)
        self.assertEqual(record.inbox_high_water, 5)
        self.assertEqual(record.latency_avg_cycles, 40.0)
        self.assertEqual(record.latency_max_cycles, 70)
        self.assertEqual(record.total_send_packets, 1234)
        self.assertEqual(record.simulated_time_us, 2500.0)
        self.assertEqual(record.wall_time_sec, 12.75)
        self.assertEqual(len(record.output_sha256), 64)

    def test_parse_rejects_network_and_golden_failures(self):
        self.child.log.write_text(
            self.child.log.read_text(encoding="utf-8").replace("link_bw=1200GB/s", "link_bw=999GB/s"),
            encoding="utf-8",
        )
        self.assert_context_error(
            "noc_profile", lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
        )
        self.child._build()
        self.child.write_verifier(checked=self.spec.rows * self.spec.dim - 1)
        self.assert_context_error(
            "golden_checked", lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
        )
        self.child.write_verifier(mismatches=1)
        self.assert_context_error(
            "golden_mismatches", lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
        )

    def test_log_evidence_requires_legal_value_boundaries(self):
        replacements = (
            ("flit_size=128B", "flit_size=128BAD"),
            ("local_no_cut=0", "local_no_cut=01"),
            ("buffer_length=1024KB", "buffer_length=1024KB_bad"),
            ("explicit=1", "explicit=10"),
        )
        for valid, invalid in replacements:
            with self.subTest(invalid=invalid):
                self.child._build()
                self.child.log.write_text(
                    self.child.log.read_text(encoding="utf-8").replace(valid, invalid),
                    encoding="utf-8",
                )
                self.assert_context_error(
                    "noc_profile",
                    lambda: phase4f.parse_child_point(
                        self.root, self.spec, self.child.verifier
                    ),
                )

    def test_parse_rejects_reduction_dma_and_physical_core_failures(self):
        stats = self.child.read_stats()
        next(row for row in stats if row["StatisticName"] == "gmem_reduction_received")["Sum.u64"] = "1"
        self.child.write_stats(stats)
        self.assert_context_error(
            "gmem_reduction_received",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )
        self.child._build()
        self.child.write_metrics("dma_summary.csv", "sum", {
            "read_issue_count": self.spec.rows * self.spec.worker_cores,
            "write_issue_count": self.spec.rows * self.spec.worker_cores,
            "completion": self.spec.rows * self.spec.worker_cores,
            "write_completion": self.spec.rows * self.spec.worker_cores,
            "read_bytes_total": 1,
            "write_bytes_total": self.spec.rows * self.spec.dim * 4,
            "timeout_retry": 1, "timeout_exhausted": 0, "write_timeout_retry": 0,
        })
        self.assert_context_error(
            "read_bytes_total",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )
        self.child._build()
        (self.child.stdout_dir / "stdout-0").unlink()
        self.assert_context_error(
            "physical_pass_cores",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )

    def test_parse_rejects_duplicate_log_completion_and_run_summary(self):
        duplicate = self.root / "logs" / f"duplicate_{self.child.run_id}.log"
        duplicate.write_text(self.child.log.read_text(encoding="utf-8"), encoding="utf-8")
        self.assert_context_error(
            "sst_log", lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
        )
        duplicate.unlink()
        with self.child.log.open("a", encoding="utf-8") as handle:
            handle.write("Simulation is complete, simulated time: 3 s\n")
        self.assert_context_error(
            "simulated_time_us",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )
        self.child._build()
        summary = self.root / "stats" / "run_summary.csv"
        with summary.open("a", encoding="utf-8") as handle:
            handle.write(summary.read_text(encoding="utf-8").splitlines()[1] + "\n")
        self.assert_context_error(
            "run_summary.csv",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )

    def test_output_size_is_gated(self):
        (self.root / "outputs" / f"{self.child.run_id}.bin").write_bytes(b"bad")
        self.assert_context_error(
            "output_size", lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
        )

    def test_run_summary_network_and_dma_retry_are_gated(self):
        summary = self.root / "stats" / "run_summary.csv"
        text = summary.read_text(encoding="utf-8")
        summary.write_text(text.replace("1200GB/s", "999GB/s", 1), encoding="utf-8")
        self.assert_context_error(
            "noc_link_bw",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )
        self.child._build()
        dma_ops = self.spec.rows * self.spec.worker_cores
        dma_bytes = self.spec.rows * self.spec.dim * 4
        self.child.write_metrics("dma_summary.csv", "sum", {
            "read_issue_count": dma_ops, "write_issue_count": dma_ops,
            "completion": dma_ops, "write_completion": dma_ops,
            "read_bytes_total": dma_bytes, "write_bytes_total": dma_bytes,
            "timeout_retry": 1, "timeout_exhausted": 0, "write_timeout_retry": 0,
        })
        self.assert_context_error(
            "timeout_retry",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )

    def test_negative_u64_noc_and_time_values_are_rejected(self):
        stats = self.child.read_stats()
        target = next(
            row for row in stats
            if row["StatisticName"] == "sfu_reduction_max_requests"
        )
        target["Sum.u64"] = str(self.spec.rows * self.spec.worker_cores + 1)
        stats.append(dict(target, ComponentName="negative", **{"Sum.u64": "-1"}))
        self.child.write_stats(stats)
        self.assert_context_error(
            "sfu_reduction_max_requests.Sum.u64",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )
        self.child._build()
        self.child.write_metrics("noc_summary.csv", "value", {
            "total_send_packets": 1234,
            "total_send_bits": 98765,
            "total_xbar_stalls": -1,
        })
        self.assert_context_error(
            "total_xbar_stalls",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )
        self.child._build()
        summary = self.root / "stats" / "run_summary.csv"
        summary.write_text(
            summary.read_text(encoding="utf-8").replace("12.75", "-12.75"),
            encoding="utf-8",
        )
        self.assert_context_error(
            "wall_time_sec",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )
        self.child._build()
        self.child.log.write_text(
            self.child.log.read_text(encoding="utf-8").replace("2.5 ms", "-2.5 ms"),
            encoding="utf-8",
        )
        self.assert_context_error(
            "simulated_time_us",
            lambda: phase4f.parse_child_point(self.root, self.spec, self.child.verifier),
        )

    def test_completion_time_supports_us_ms_and_s(self):
        for unit, expected in (("us", 2.5), ("ms", 2500.0), ("s", 2500000.0)):
            with self.subTest(unit=unit):
                text = self.child.log.read_text(encoding="utf-8")
                text = re.sub(
                    r"Simulation is complete, simulated time: .*",
                    f"Simulation is complete, simulated time: 2.5 {unit}",
                    text,
                )
                self.child.log.write_text(text, encoding="utf-8")
                record = phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
                self.assertEqual(record.simulated_time_us, expected)

    def test_parent_manifest_atomic_upsert_and_round_trip(self):
        manifest = pathlib.Path(self.temp.name) / "large_scale_manifest.csv"
        record = phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
        phase4f.upsert_parent_manifest(manifest, record)
        self.assertEqual(phase4f.load_parent_manifest(manifest), [record])
        replacement = dataclasses.replace(record, wall_time_sec=99.5)
        phase4f.upsert_parent_manifest(manifest, replacement)
        self.assertEqual(phase4f.load_parent_manifest(manifest), [replacement])
        self.assertEqual(len(manifest.read_text(encoding="utf-8").splitlines()), 2)
        self.assertFalse(list(manifest.parent.glob(f".{manifest.name}.*.tmp")))

    def test_parent_manifest_rejects_old_schema_duplicate_and_malformed_without_damage(self):
        manifest = pathlib.Path(self.temp.name) / "large_scale_manifest.csv"
        record = phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
        phase4f.upsert_parent_manifest(manifest, record)
        good = manifest.read_bytes()
        manifest.write_text("old,header\n1,2\n", encoding="utf-8")
        old = manifest.read_bytes()
        with self.assertRaises(ValueError):
            phase4f.upsert_parent_manifest(manifest, record)
        self.assertEqual(manifest.read_bytes(), old)
        manifest.write_bytes(good + good.splitlines(keepends=True)[1])
        with self.assertRaises(ValueError):
            phase4f.load_parent_manifest(manifest)
        manifest.write_bytes(b"\xff\xfe")
        with self.assertRaises(ValueError):
            phase4f.load_parent_manifest(manifest)

    def test_parent_manifest_rejects_negative_counter_and_time_values(self):
        manifest = pathlib.Path(self.temp.name) / "large_scale_manifest.csv"
        record = phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
        phase4f.upsert_parent_manifest(manifest, record)
        with manifest.open(newline="", encoding="utf-8") as handle:
            canonical = list(csv.DictReader(handle))[0]
        for field in ("total_send_packets", "simulated_time_us", "wall_time_sec"):
            with self.subTest(field=field):
                bad = dict(canonical, **{field: "-1"})
                SyntheticChild.write_csv(
                    manifest, phase4f.PARENT_MANIFEST_FIELDS, [bad]
                )
                with self.assertRaisesRegex(ValueError, rf"field={field}"):
                    phase4f.load_parent_manifest(manifest)

    def test_collect_cli_uses_parser_and_upserts_manifest(self):
        manifest = pathlib.Path(self.temp.name) / "parent.csv"
        result = subprocess.run([
            sys.executable, str(SCRIPT), "collect",
            "--child-root", str(self.root), "--stage", self.spec.stage,
            "--rows", str(self.spec.rows), "--dim", str(self.spec.dim),
            "--workers", str(self.spec.worker_cores), "--bands", str(self.spec.band_cores),
            "--parent-manifest", str(manifest), "--verifier", str(self.child.verifier),
        ], check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = phase4f.load_parent_manifest(manifest)[0]
        self.assertEqual(
            result.stdout,
            f"run_id={record.run_id} output_sha256={record.output_sha256}\n",
        )
        self.assertEqual(result.stderr, "")


class Phase4FReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.records = []
        for index, spec in enumerate(phase4f.DEFAULT_POINTS):
            child = SyntheticChild(self.root / f"child-{index}", spec)
            record = phase4f.parse_child_point(child.root, spec, child.verifier)
            simulated_time = {
                (16, 512, 16): 800.0,
                (16, 1024, 16): 1400.0,
                (16, 2048, 16): 2500.0,
                (16, 4096, 16): 4000.0,
                (16, 4096, 4): 10000.0,
                (16, 4096, 8): 6000.0,
                (64, 4096, 16): 15000.0,
                (256, 4096, 16): 56000.0,
            }[(spec.rows, spec.dim, spec.worker_cores)]
            self.records.append(dataclasses.replace(
                record,
                simulated_time_us=simulated_time,
                latency_avg_cycles=float(20 + index),
                latency_max_cycles=40 + index,
                total_send_packets=1000 + index,
                total_send_bits=100000 + index,
                total_xbar_stalls=10 + index,
            ))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def failure(record, status="TIMEOUT", exit_code=124):
        return dataclasses.replace(
            record,
            status=status,
            exit_code=exit_code,
            artifact_validation=status,
            golden_checked=None,
            golden_mismatches=None,
            transport_events=None,
            transport_immediate=None,
            transport_queued=None,
            transport_rejected=None,
            transport_stale=None,
            inbox_high_water=None,
            latency_avg_cycles=None,
            latency_max_cycles=None,
            total_send_packets=None,
            total_send_bits=None,
            total_xbar_stalls=None,
            simulated_time_us=None,
            wall_time_sec=None,
            dma_timeout_retry=None,
            dma_timeout_exhausted=None,
            dma_write_timeout_retry=None,
            output_sha256=None,
        )

    @staticmethod
    def marker_text(record):
        spec = record.spec
        child_hash = "a" * 64
        pipeline_hash = "b" * 64
        signature = ";".join((
            "schema=phase4f-parent-v1",
            f"stage={spec.stage}",
            f"rows={spec.rows}",
            f"dim={spec.dim}",
            f"workers={spec.worker_cores}",
            f"bands={spec.band_cores}",
            "cooperative_groups=1",
            "transport=explicit_noc",
            "request_vn=0",
            "ordinary_response_vn=1",
            "reduction_vn=0",
            "num_vns=3",
            "dma_response_vn=0",
            "noc_link_bw=1200GB/s",
            "noc_xbar_bw=1200GB/s",
            "dirctrl_highlink_bw=1200GB/s",
            "noc_input_buffer=512KB",
            "noc_output_buffer=512KB",
            "gm_buffer=1024KB",
            "flit_size=128B",
            "inter_router_no_cut=0",
            "local_no_cut=0",
            f"mem_node_size={spec.mem_node_size}",
            f"timeout_sec={spec.timeout_sec}",
            "chunk=256",
            "staging_rows=4",
            "job_rows=4",
            "retry_ticks=1024",
            "max_retries=8",
            f"child_runner_sha256={child_hash}",
            f"pipeline_args_sha256={pipeline_hash}",
        ))
        signature_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        return (
            "schema=phase4f-parent-v1\nstate=PASS\n"
            f"signature_sha256={signature_hash}\nsignature={signature}\n"
            f"child_runner_sha256={child_hash}\n"
            f"pipeline_args_sha256={pipeline_hash}\n"
            f"child_root={record.child_root}\n"
            f"output_sha256={record.output_sha256}\n"
        )

    def test_complete_matrix_and_lifecycle_gates(self):
        phase4f.validate_complete_matrix(self.records)
        invalid = (
            self.records[:-1],
            self.records + [self.records[0]],
            [dataclasses.replace(self.records[0], noc_link_bw="25GB/s")] + self.records[1:],
            [dataclasses.replace(self.records[0], transport_stale=1)] + self.records[1:],
        )
        for records in invalid:
            with self.subTest(length=len(records)), self.assertRaises(ValueError):
                phase4f.validate_complete_matrix(records)

    def test_metrics_use_four_worker_baseline_without_single_worker(self):
        metrics = phase4f.derive_metrics(self.records)
        self.assertEqual(metrics["r16_d4096_w4_time_per_row_us"], 625.0)
        self.assertAlmostEqual(metrics["r16_d4096_w4_time_per_element_us"], 10000 / 65536)
        self.assertEqual(metrics["r16_d4096_w4_speedup"], 1.0)
        self.assertAlmostEqual(metrics["r16_d4096_w8_speedup"], 10 / 6)
        self.assertAlmostEqual(metrics["r16_d4096_w8_efficiency"], (10 / 6) / 2)
        self.assertEqual(metrics["r16_d4096_w16_speedup"], 2.5)
        self.assertEqual(metrics["r16_d4096_w16_efficiency"], 2.5 / 4)
        self.assertAlmostEqual(metrics["r16_d4096_w8_marginal_gain"], 0.4)
        self.assertAlmostEqual(metrics["r16_d4096_w16_marginal_gain"], 1 / 3)
        self.assertFalse(any(
            re.search(r"(?:^|_)w1(?:_|$)", key) or "single" in key
            for key in metrics
        ))

    def test_failed_outcome_round_trips_without_performance(self):
        outcomes = self.records[:-1] + [self.failure(self.records[-1])]
        phase4f.validate_complete_matrix(outcomes)
        path = self.root / "source.csv"
        phase4f.write_source_csv(outcomes, path)
        loaded = phase4f.load_source_csv(path)
        self.assertEqual(loaded, outcomes)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        row = rows[0]
        failed = next(item for item in rows if item["status"] == "TIMEOUT")
        self.assertEqual(failed["simulated_time_us"], "")
        self.assertEqual(failed["time_per_row_us"], "")
        self.assertEqual(failed["time_per_element_us"], "")
        self.assertEqual(row["status"], "PASS")

    def test_source_csv_is_sorted_and_byte_deterministic(self):
        first = self.root / "first.csv"
        second = self.root / "second.csv"
        phase4f.write_source_csv(list(reversed(self.records)), first)
        phase4f.write_source_csv(self.records, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(phase4f.load_source_csv(first), self.records)

    def test_exports_are_english_editable_and_deterministic(self):
        prefix_a = self.root / "a" / "sfu_phase4f_large_scale"
        prefix_b = self.root / "b" / "sfu_phase4f_large_scale"
        phase4f.render_figure(self.records, prefix_a)
        phase4f.render_figure(self.records, prefix_b)
        for suffix in (".svg", ".pdf", ".png"):
            self.assertTrue(prefix_a.with_suffix(suffix).is_file())
        svg = prefix_a.with_suffix(".svg").read_text(encoding="utf-8")
        self.assertIn("<text", svg)
        self.assertIn("Dimension scaling", svg)
        self.assertIn("Worker scaling", svg)
        self.assertIn("Row scaling", svg)
        self.assertIn("NoC and correctness", svg)
        self.assertIn("1200 GB/s", svg)
        self.assertIn("us/element", svg)
        self.assertNotIn("single-worker", svg.lower())
        self.assertEqual(
            prefix_a.with_suffix(".svg").read_bytes(),
            prefix_b.with_suffix(".svg").read_bytes(),
        )
        pdf = prefix_a.with_suffix(".pdf").read_bytes()
        self.assertIn(b"/FontFile2", pdf)
        from PIL import Image
        with Image.open(prefix_a.with_suffix(".png")) as image:
            self.assertAlmostEqual(image.width / image.height, 16 / 9, places=2)
            self.assertAlmostEqual(image.info["dpi"][0], 300, delta=1)

    def test_qa_distinguishes_measurement_derived_and_unavailable(self):
        outcomes = self.records[:-1] + [self.failure(self.records[-1])]
        output = self.root / "qa.md"
        phase4f.write_qa(outcomes, output)
        text = output.read_text(encoding="utf-8")
        self.assertIn("Measurements", text)
        self.assertIn("Derived metrics", text)
        self.assertIn("Unavailable outcomes", text)
        self.assertIn("TIMEOUT", text)
        self.assertIn("256:4096:16:16", text)
        self.assertIn("1200GB/s", text)
        self.assertIn("golden", text.lower())
        self.assertIn("transport", text.lower())
        self.assertIn("DMA", text)

    def test_report_cli_reparses_pass_and_publishes_atomically(self):
        experiment = self.root / "experiment"
        report = self.root / "report"
        experiment.mkdir()
        manifest = experiment / "large_scale_manifest.csv"
        for record in self.records:
            phase4f.upsert_parent_manifest(manifest, record)
        SyntheticChild.write_csv(
            experiment / "point_status.csv",
            phase4f.POINT_STATUS_FIELDS,
            [phase4f._status_row(record) for record in self.records],
        )
        (experiment / "completed").mkdir()
        for record in self.records:
            marker = experiment / "completed" / phase4f._marker_name(record.spec)
            marker.write_text(
                self.marker_text(record),
                encoding="utf-8",
            )
        with mock.patch.object(phase4f, "parse_child_point", side_effect=self.records) as parse:
            self.assertEqual(phase4f.main([
                "--root", str(experiment), "--output-dir", str(report),
                "--verifier", str(self.root / "verifier.py"),
            ]), 0)
        self.assertEqual(parse.call_count, 8)
        self.assertEqual(len(list(report.iterdir())), 5)

        failed_report = self.root / "failed-report"
        bad_status = list(self.records)
        bad_status[0] = dataclasses.replace(bad_status[0], noc_link_bw="25GB/s")
        SyntheticChild.write_csv(
            experiment / "point_status.csv",
            phase4f.POINT_STATUS_FIELDS,
            [phase4f._status_row(record) for record in bad_status],
        )
        with self.assertRaises(ValueError):
            phase4f.main([
                "--root", str(experiment), "--output-dir", str(failed_report),
                "--verifier", str(self.root / "verifier.py"),
            ])
        self.assertFalse(failed_report.exists())


if __name__ == "__main__":
    unittest.main()
