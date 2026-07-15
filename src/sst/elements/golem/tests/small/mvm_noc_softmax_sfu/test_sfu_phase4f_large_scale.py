import dataclasses
import csv
import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("plot_sfu_phase4f_large_scale.py")
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
            f"_bc{spec.band_cores}_g4_vn0"
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
            "cooperative_groups": "4",
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
        self.log.write_text(
            "[NoC] input_buf_size=512KB, output_buf_size=512KB, "
            "link_bw=1200GB/s, xbar_bw=1200GB/s, flit_size=128B\n"
            "[NoC] inter_router_no_cut=0, local_no_cut=0\n"
            "[GOLEM] GlobalMemory link buffer_length=1024KB\n"
            "GlobalMemory VN mapping: request_vn=0 response_vn=1 reduction_vn=0 (num_vns=3)\n"
            "resolved golem_dma_response_vn=0 num_vns=3 explicit=1\n"
            "Simulation is complete, simulated time: 2.5 ms\n",
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

    def test_parse_canonical_child_point_aggregates_all_evidence(self):
        record = phase4f.parse_child_point(self.root, self.spec, self.child.verifier)
        self.assertEqual(record.run_id, self.child.run_id)
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


if __name__ == "__main__":
    unittest.main()
