import csv
import dataclasses
import importlib.util
import pathlib
import stat
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[7]
SCRIPT = (
    REPO_ROOT
    / "src/sst/elements/golem/tests/artifacts/sweeps"
    / "sfu_phase4e_group_report_20260715/figures"
    / "plot_sfu_phase4e_group_figure.py"
)
SPEC = importlib.util.spec_from_file_location("phase4e_figure", SCRIPT)
phase4e_figure = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase4e_figure)


MANIFEST_FIELDS = [
    "run_id",
    "rows",
    "dim",
    "chunk_elems",
    "worker_cores",
    "band_cores",
    "cooperative_groups",
    "reduction_vn",
    "num_vns",
    "dma_response_vn",
    "status",
    "exit_code",
    "artifact_validation",
]


class ArtifactFixture:
    def __init__(self, base: pathlib.Path):
        self.sweeps_root = base / "sweeps"
        self.sweeps_root.mkdir()
        self.verifier = base / "verify_softmax_sfu_against_golden.py"
        self.verifier.write_text(
            """#!/usr/bin/env python3
import argparse
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--c-file', required=True)
args, _ = parser.parse_known_args()
mode = pathlib.Path(args.c_file).read_text().strip()
if mode == 'failure':
    print('[VERIFY-SFU-SOFTMAX] FAIL checked=8192 mismatches=1')
    raise SystemExit(2)
if mode == 'missing':
    raise SystemExit(0)
checked = 8000 if mode == 'wrong_checked' else 8192
mismatches = 1 if mode == 'mismatch' else 0
status = 'PASS' if mismatches == 0 else 'FAIL'
print(f'[VERIFY-SFU-SOFTMAX] {status} checked={checked} mismatches={mismatches}')
""",
            encoding="utf-8",
        )
        self.verifier.chmod(self.verifier.stat().st_mode | stat.S_IXUSR)

    @staticmethod
    def root_name(transport: str, vn: int | None) -> str:
        if transport == "modeled-NoC":
            return "sfu_phase4e_modeled_control_matrix_20260715"
        return f"sfu_phase4e_explicit_vn{vn}_matrix_20260715"

    @staticmethod
    def run_id(transport: str, workers: int, vn: int | None) -> str:
        suffix = "" if transport == "modeled-NoC" else f"_vn{vn}"
        return f"sfu_job_dist_r16_d512_w{workers}_bc{workers}_g1{suffix}"

    def create_root(
        self,
        transport: str = "explicit-NoC",
        vn: int | None = 0,
        workers: tuple[int, ...] = (4, 8, 16),
        cached_rows: bool = True,
        verifier_mode: str = "pass",
    ) -> pathlib.Path:
        root = self.sweeps_root / self.root_name(transport, vn)
        root.mkdir(parents=True)
        (root / "inputs").mkdir()
        (root / "logs").mkdir()
        (root / "outputs").mkdir()
        for name in ("a.bin", "b.bin", "softmax_logits_16x512.bin"):
            (root / "inputs" / name).write_bytes(b"fixture")

        rows = []
        for worker_count in workers:
            run_id = self.run_id(transport, worker_count, vn)
            row = {
                "run_id": run_id,
                "rows": "16",
                "dim": "512",
                "chunk_elems": "256",
                "worker_cores": str(worker_count),
                "band_cores": str(worker_count),
                "cooperative_groups": "1",
                "reduction_vn": str(0 if vn is None else vn),
                "num_vns": "3",
                "dma_response_vn": "0",
                "status": "PASS",
                "exit_code": "0",
                "artifact_validation": "PASS",
            }
            rows.append(row)
            if cached_rows:
                rows.append({**row, "artifact_validation": "CACHED"})
            self._create_point(root, row, transport, verifier_mode)
        self.write_manifest(root, rows)
        return root

    @staticmethod
    def write_manifest(root: pathlib.Path, rows: list[dict[str, str]]) -> None:
        with (root / "sweep_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def manifest_rows(root: pathlib.Path) -> list[dict[str, str]]:
        with (root / "sweep_manifest.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _create_point(
        self,
        root: pathlib.Path,
        row: dict[str, str],
        transport: str,
        verifier_mode: str,
    ) -> None:
        run_id = row["run_id"]
        workers = int(row["worker_cores"])
        (root / "logs" / f"test_default_{run_id}.log").write_text(
            "Simulation is complete, simulated time: 407.75 us\n", encoding="utf-8"
        )
        (root / "outputs" / f"{run_id}.bin").write_text(verifier_mode, encoding="utf-8")
        stats_dir = root / "stats" / "overlap0" / run_id
        stats_dir.mkdir(parents=True)
        events = 4 * 16 * workers
        stats_rows = [
            self._stat("sfu_reduction_transport_received", events // 4, events // 4, 1),
            self._stat("sfu_reduction_transport_received", events - events // 4, events - events // 4, 1),
            self._stat("sfu_reduction_transport_latency_cycles", 600, 2, 400),
            self._stat("sfu_reduction_transport_latency_cycles", 900, 3, 700),
            self._stat("sfu_reduction_transport_inbox_high_water", 2, 2, 1),
            self._stat("sfu_reduction_transport_inbox_high_water", 4, 4, 1),
            self._stat("sfu_reduction_transport_stale_dropped", 0, 0, 0),
            self._stat("gmem_reduction_send_queued", 0, 0, 0),
            self._stat("gmem_reduction_send_rejected", 0, 0, 0),
        ]
        with (stats_dir / "stats_selfcom.txt").open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "ComponentName",
                "StatisticName",
                "StatisticSubId",
                "StatisticType",
                "SimTime",
                "Rank",
                "Sum.u64",
                "SumSQ.u64",
                "Count.u64",
                "Min.u64",
                "Max.u64",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(stats_rows)
        with (stats_dir / "noc_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerows(
                [
                    ["total_send_packets", 1234 + workers],
                    ["total_send_bits", 5678 + workers],
                    ["total_xbar_stalls", 90 + workers],
                ]
            )
        with (stats_dir / "dma_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["metric", "mean", "median", "p95", "min", "max", "sum"]
            )
            writer.writeheader()
            for metric in ("timeout_retry", "timeout_exhausted", "write_timeout_retry"):
                writer.writerow(
                    {"metric": metric, "mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0, "sum": 0}
                )

    @staticmethod
    def _stat(name: str, total: int, count: int, maximum: int) -> dict[str, str | int]:
        return {
            "ComponentName": "component",
            "StatisticName": name,
            "StatisticSubId": "",
            "StatisticType": "Accumulator",
            "SimTime": "407750000",
            "Rank": "0",
            "Sum.u64": total,
            "SumSQ.u64": "0",
            "Count.u64": count,
            "Min.u64": "0",
            "Max.u64": maximum,
        }


class Phase4EParserTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ArtifactFixture(pathlib.Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_selects_one_canonical_row_and_ignores_cached_row(self):
        root = self.fixture.create_root()
        selected = phase4e_figure.select_manifest_rows(root, "explicit-NoC", 0)
        self.assertEqual([4, 8, 16], [int(row["worker_cores"]) for row in selected])
        self.assertTrue(all(row["artifact_validation"] == "PASS" for row in selected))

    def test_selection_rejects_missing_worker_point(self):
        root = self.fixture.create_root(workers=(4, 8))
        with self.assertRaisesRegex(ValueError, "worker"):
            phase4e_figure.select_manifest_rows(root, "explicit-NoC", 0)

    def test_selection_rejects_duplicate_pass_pass_rows(self):
        root = self.fixture.create_root(cached_rows=False)
        rows = self.fixture.manifest_rows(root)
        rows.append(dict(rows[0]))
        self.fixture.write_manifest(root, rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            phase4e_figure.select_manifest_rows(root, "explicit-NoC", 0)

    def test_selection_rejects_non_pass_status(self):
        root = self.fixture.create_root(cached_rows=False)
        rows = self.fixture.manifest_rows(root)
        rows[0]["status"] = "FAIL"
        self.fixture.write_manifest(root, rows)
        with self.assertRaisesRegex(ValueError, "status"):
            phase4e_figure.select_manifest_rows(root, "explicit-NoC", 0)

    def test_selection_rejects_non_pass_artifact_validation(self):
        root = self.fixture.create_root(cached_rows=False)
        rows = self.fixture.manifest_rows(root)
        rows[0]["artifact_validation"] = "FAIL"
        self.fixture.write_manifest(root, rows)
        with self.assertRaisesRegex(ValueError, "artifact_validation"):
            phase4e_figure.select_manifest_rows(root, "explicit-NoC", 0)

    def test_selection_rejects_unexpected_worker_count(self):
        root = self.fixture.create_root(workers=(4, 8, 32))
        with self.assertRaisesRegex(ValueError, "worker"):
            phase4e_figure.select_manifest_rows(root, "explicit-NoC", 0)

    def test_selection_rejects_vn_root_mismatch(self):
        root = self.fixture.create_root(vn=0)
        with self.assertRaisesRegex(ValueError, "VN|root"):
            phase4e_figure.select_manifest_rows(root, "explicit-NoC", 1)

    def test_selection_wraps_manifest_decode_failure_with_context(self):
        root = self.fixture.create_root(cached_rows=False)
        (root / "sweep_manifest.csv").write_bytes(b"\xff")
        with self.assertRaisesRegex(
            ValueError, r"root=.*run_id=<manifest> field=sweep_manifest\.csv"
        ):
            phase4e_figure.select_manifest_rows(root, "explicit-NoC", 0)

    def test_selection_wraps_malformed_csv_failure_with_context(self):
        root = self.fixture.create_root(cached_rows=False)
        (root / "sweep_manifest.csv").write_text(
            'run_id,rows\n"unterminated', encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ValueError, r"root=.*run_id=<manifest> field=sweep_manifest\.csv"
        ):
            phase4e_figure.select_manifest_rows(root, "explicit-NoC", 0)

    def test_parse_point_converts_all_supported_time_units(self):
        root = self.fixture.create_root(cached_rows=False)
        row = self.fixture.manifest_rows(root)[0]
        log = next((root / "logs").glob(f"*{row['run_id']}*.log"))
        for value, unit, expected in ((407.75, "us", 407.75), (1.25, "ms", 1250.0), (2.0, "s", 2_000_000.0)):
            with self.subTest(unit=unit):
                log.write_text(
                    f"Simulation is complete, simulated time: {value} {unit}\n", encoding="utf-8"
                )
                record = phase4e_figure.parse_point(root, row, "explicit-NoC", 0, self.fixture.verifier)
                self.assertEqual(expected, record.simulated_time_us)

    def test_parse_point_wraps_sst_log_decode_failure_with_context(self):
        root = self.fixture.create_root(cached_rows=False)
        row = self.fixture.manifest_rows(root)[0]
        log = next((root / "logs").glob(f"*{row['run_id']}*.log"))
        log.write_bytes(
            b"\xffSimulation is complete, simulated time: 407.75 us\n"
        )
        with self.assertRaisesRegex(
            ValueError, r"root=.*run_id=.* field=simulated_time"
        ):
            phase4e_figure.parse_point(root, row, "explicit-NoC", 0, self.fixture.verifier)

    def test_parse_point_aggregates_transport_statistics(self):
        root = self.fixture.create_root(cached_rows=False)
        row = self.fixture.manifest_rows(root)[0]
        record = phase4e_figure.parse_point(root, row, "explicit-NoC", 0, self.fixture.verifier)
        self.assertEqual(256, record.transport_events)
        self.assertEqual(300.0, record.latency_average_cycles)
        self.assertEqual(700, record.latency_max_cycles)
        self.assertEqual(4, record.inbox_high_water)
        self.assertEqual(0, record.stale)
        self.assertEqual(0, record.queued)
        self.assertEqual(0, record.rejected)

    def test_parse_point_reads_noc_and_dma_summaries(self):
        root = self.fixture.create_root(cached_rows=False)
        row = self.fixture.manifest_rows(root)[0]
        record = phase4e_figure.parse_point(root, row, "explicit-NoC", 0, self.fixture.verifier)
        self.assertEqual((1238, 5682, 94), (record.total_send_packets, record.total_send_bits, record.total_xbar_stalls))
        self.assertEqual((0, 0, 0), (record.dma_timeout_retry, record.dma_timeout_exhausted, record.dma_write_timeout_retry))

    def test_modeled_point_keeps_transport_fields_zero(self):
        root = self.fixture.create_root(transport="modeled-NoC", vn=None, cached_rows=False)
        row = self.fixture.manifest_rows(root)[0]
        record = phase4e_figure.parse_point(root, row, "modeled-NoC", None, self.fixture.verifier)
        self.assertEqual((0, 0.0, 0, 0, 0, 0, 0), (record.transport_events, record.latency_average_cycles, record.latency_max_cycles, record.inbox_high_water, record.queued, record.rejected, record.stale))
        self.assertEqual(1238, record.total_send_packets)

    def test_modeled_point_rejects_empty_stats_evidence(self):
        root = self.fixture.create_root(transport="modeled-NoC", vn=None, cached_rows=False)
        row = self.fixture.manifest_rows(root)[0]
        stats_path = root / "stats" / "overlap0" / row["run_id"] / "stats_selfcom.txt"
        stats_path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(
            ValueError, r"root=.*run_id=.* field=stats_selfcom\.txt"
        ):
            phase4e_figure.parse_point(root, row, "modeled-NoC", None, self.fixture.verifier)

    def test_modeled_point_rejects_malformed_stats_value(self):
        root = self.fixture.create_root(transport="modeled-NoC", vn=None, cached_rows=False)
        row = self.fixture.manifest_rows(root)[0]
        stats_path = root / "stats" / "overlap0" / row["run_id"] / "stats_selfcom.txt"
        with stats_path.open(newline="", encoding="utf-8") as handle:
            stats_rows = list(csv.DictReader(handle))
        stats_rows[0]["Sum.u64"] = "not-an-integer"
        with stats_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=stats_rows[0].keys())
            writer.writeheader()
            writer.writerows(stats_rows)
        with self.assertRaisesRegex(
            ValueError, r"root=.*run_id=.* field=stats_selfcom\.txt\.Sum\.u64"
        ):
            phase4e_figure.parse_point(root, row, "modeled-NoC", None, self.fixture.verifier)

    def test_parse_point_wraps_stats_decode_failure_with_context(self):
        root = self.fixture.create_root(cached_rows=False)
        row = self.fixture.manifest_rows(root)[0]
        stats_path = root / "stats" / "overlap0" / row["run_id"] / "stats_selfcom.txt"
        stats_path.write_bytes(b"\xff")
        with self.assertRaisesRegex(
            ValueError, r"root=.*run_id=.* field=stats_selfcom\.txt"
        ):
            phase4e_figure.parse_point(root, row, "explicit-NoC", 0, self.fixture.verifier)

    def test_verifier_requires_successful_pass_evidence(self):
        for mode, message in (
            ("failure", "verifier"),
            ("missing", "output"),
            ("wrong_checked", "checked"),
            ("mismatch", "mismatches"),
        ):
            with self.subTest(mode=mode):
                case_dir = pathlib.Path(self.temporary.name) / mode
                case_dir.mkdir()
                case_fixture = ArtifactFixture(case_dir)
                root = case_fixture.create_root(cached_rows=False, verifier_mode=mode)
                row = case_fixture.manifest_rows(root)[0]
                with self.assertRaisesRegex(ValueError, message):
                    phase4e_figure.parse_point(root, row, "explicit-NoC", 0, case_fixture.verifier)

    def test_loads_and_validates_exact_twelve_point_matrix(self):
        for vn in (0, 1, 2):
            self.fixture.create_root(vn=vn)
        self.fixture.create_root(transport="modeled-NoC", vn=None)
        records = phase4e_figure.load_and_validate_matrix(self.fixture.sweeps_root, self.fixture.verifier)
        self.assertEqual(12, len(records))
        explicit = [record for record in records if record.transport == "explicit-NoC"]
        self.assertEqual({0, 1, 2}, {record.reduction_vn for record in explicit})
        self.assertEqual({4, 8, 16}, {record.workers for record in records})
        for record in explicit:
            self.assertEqual(4 * 16 * record.workers, record.transport_events)
        for record in records:
            self.assertEqual((0, 0, 0, 0, 0, 0), (record.dma_timeout_retry, record.dma_timeout_exhausted, record.dma_write_timeout_retry, record.queued, record.rejected, record.stale))

    def test_matrix_rejects_bad_event_total(self):
        roots = [self.fixture.create_root(vn=vn) for vn in (0, 1, 2)]
        self.fixture.create_root(transport="modeled-NoC", vn=None)
        row = self.fixture.manifest_rows(roots[0])[0]
        stats_dir = roots[0] / "stats" / "overlap0" / row["run_id"]
        stats_path = stats_dir / "stats_selfcom.txt"
        stats_path.write_text(
            stats_path.read_text(encoding="utf-8").replace(
                "sfu_reduction_transport_received,,Accumulator,407750000,0,64,",
                "sfu_reduction_transport_received,,Accumulator,407750000,0,63,",
                1,
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "transport_events"):
            phase4e_figure.load_and_validate_matrix(self.fixture.sweeps_root, self.fixture.verifier)

    def test_matrix_rejects_nonzero_failure_total(self):
        roots = [self.fixture.create_root(vn=vn) for vn in (0, 1, 2)]
        self.fixture.create_root(transport="modeled-NoC", vn=None)
        row = self.fixture.manifest_rows(roots[0])[0]
        dma_path = roots[0] / "stats" / "overlap0" / row["run_id"] / "dma_summary.csv"
        with dma_path.open(newline="", encoding="utf-8") as handle:
            dma_rows = list(csv.DictReader(handle))
        dma_rows[0]["sum"] = "1"
        with dma_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=dma_rows[0].keys())
            writer.writeheader()
            writer.writerows(dma_rows)
        with self.assertRaisesRegex(ValueError, "dma_timeout_retry"):
            phase4e_figure.load_and_validate_matrix(self.fixture.sweeps_root, self.fixture.verifier)


class Phase4ESourceDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.temporary.name)
        self.fixture = ArtifactFixture(self.base)
        for vn in (2, 0, 1):
            self.fixture.create_root(vn=vn)
        self.fixture.create_root(transport="modeled-NoC", vn=None)
        self._set_contract_latency_values()

    def tearDown(self):
        self.temporary.cleanup()

    def load_records(self):
        return phase4e_figure.load_and_validate_matrix(
            self.fixture.sweeps_root, self.fixture.verifier
        )

    def _set_contract_latency_values(self):
        averages = {4: 9440, 8: 12100, 16: 15583}
        for vn in (0, 1, 2):
            root = self.fixture.sweeps_root / self.fixture.root_name("explicit-NoC", vn)
            for row in self.fixture.manifest_rows(root)[::2]:
                workers = int(row["worker_cores"])
                path = root / "stats" / "overlap0" / row["run_id"] / "stats_selfcom.txt"
                with path.open(newline="", encoding="utf-8") as handle:
                    stats_rows = list(csv.DictReader(handle))
                for stats_row in stats_rows:
                    if (
                        stats_row["StatisticName"] == "sfu_reduction_transport_latency_cycles"
                        and stats_row["Count.u64"] == "3"
                    ):
                        stats_row["Sum.u64"] = str(averages[workers] * 5 - 600)
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=stats_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(stats_rows)

    def test_parse_only_cli_writes_complete_deterministically_ordered_csv(self):
        output_dir = self.base / "output"
        result = phase4e_figure.main(
            [
                "--sweeps-root",
                str(self.fixture.sweeps_root),
                "--output-dir",
                str(output_dir),
                "--parse-only",
            ],
            verifier=self.fixture.verifier,
        )
        self.assertEqual(0, result)

        csv_path = output_dir / "sfu_phase4e_group_figure_source_data.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        expected_fields = [field.name for field in dataclasses.fields(phase4e_figure.PointRecord)]
        self.assertEqual(expected_fields + ["single_simulation_outcome"], reader.fieldnames)
        self.assertTrue(all(row["single_simulation_outcome"] == "1" for row in rows))
        self.assertEqual(
            [
                ("modeled-NoC", "", "4"),
                ("modeled-NoC", "", "8"),
                ("modeled-NoC", "", "16"),
                ("explicit-NoC", "0", "4"),
                ("explicit-NoC", "0", "8"),
                ("explicit-NoC", "0", "16"),
                ("explicit-NoC", "1", "4"),
                ("explicit-NoC", "1", "8"),
                ("explicit-NoC", "1", "16"),
                ("explicit-NoC", "2", "4"),
                ("explicit-NoC", "2", "8"),
                ("explicit-NoC", "2", "16"),
            ],
            [(row["transport"], row["reduction_vn"], row["workers"]) for row in rows],
        )
        self.assertEqual([], list(output_dir.glob("*.png")))
        self.assertEqual([], list(output_dir.glob("*.pdf")))
        self.assertEqual([], list(output_dir.glob("*.svg")))

    def test_source_csv_round_trips_every_point_record_field(self):
        records = self.load_records()
        csv_path = self.base / "source.csv"
        phase4e_figure.write_source_csv(records, csv_path)
        self.assertNotIn(b"\r\n", csv_path.read_bytes())
        restored = phase4e_figure.load_source_csv(csv_path)
        self.assertEqual(len(records), len(restored))
        for original, actual in zip(records, restored):
            for field in dataclasses.fields(phase4e_figure.PointRecord):
                expected_value = getattr(original, field.name)
                actual_value = getattr(actual, field.name)
                if isinstance(expected_value, float):
                    self.assertAlmostEqual(expected_value, actual_value, delta=1e-9)
                else:
                    self.assertEqual(expected_value, actual_value)

    def test_derived_contracts_match_approved_runtime_and_latency_claims(self):
        modeled_runtime = {4: 400.0, 8: 410.0, 16: 420.0}
        explicit_runtime = {4: 400.24, 8: 410.20, 16: 420.10}
        latency = {4: 9440.0, 8: 12100.0, 16: 15583.0}
        records = [
            dataclasses.replace(
                record,
                simulated_time_us=(
                    modeled_runtime[record.workers]
                    if record.transport == "modeled-NoC"
                    else explicit_runtime[record.workers]
                ),
                latency_average_cycles=(
                    0.0
                    if record.transport == "modeled-NoC"
                    else latency[record.workers]
                ),
            )
            for record in self.load_records()
        ]
        derived = phase4e_figure.validate_derived_contracts(records)
        self.assertEqual(0.0, derived.explicit_runtime_spread)
        self.assertAlmostEqual(0.06, derived.max_explicit_modeled_pct, places=12)
        self.assertLess(derived.max_explicit_modeled_pct, 0.061)
        self.assertAlmostEqual(
            (15583.0 / 9440.0 - 1.0) * 100.0,
            derived.latency_growth_pct,
            places=12,
        )
        self.assertEqual(65, round(derived.latency_growth_pct))

    def test_derived_contracts_reject_nonidentical_vn_runtime(self):
        records = self.load_records()
        records[3] = dataclasses.replace(
            records[3], simulated_time_us=records[3].simulated_time_us + 0.001
        )
        with self.assertRaisesRegex(ValueError, "VN runtime"):
            phase4e_figure.validate_derived_contracts(records)

    def test_derived_contracts_reject_each_numerical_gate_violation(self):
        records = self.load_records()
        modeled_four = next(
            record.simulated_time_us
            for record in records
            if record.transport == "modeled-NoC" and record.workers == 4
        )
        excessive_delta = [
            dataclasses.replace(record, simulated_time_us=modeled_four * 1.000611)
            if record.transport == "explicit-NoC" and record.workers == 4
            else record
            for record in records
        ]
        with self.assertRaisesRegex(ValueError, "0.061"):
            phase4e_figure.validate_derived_contracts(excessive_delta)

        wrong_growth = [
            dataclasses.replace(record, latency_average_cycles=9440.0)
            if record.transport == "explicit-NoC" and record.workers == 16
            else record
            for record in records
        ]
        with self.assertRaisesRegex(ValueError, "latency growth"):
            phase4e_figure.validate_derived_contracts(wrong_growth)

        unequal_latency = list(records)
        unequal_latency[3] = dataclasses.replace(
            unequal_latency[3],
            latency_average_cycles=unequal_latency[3].latency_average_cycles + 1.0,
        )
        with self.assertRaisesRegex(ValueError, "VN latency"):
            phase4e_figure.validate_derived_contracts(unequal_latency)

    def test_source_csv_loader_rejects_trailing_columns(self):
        csv_path = self.base / "source.csv"
        phase4e_figure.write_source_csv(self.load_records(), csv_path)
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        lines[1] += ",extra"
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "malformed"):
            phase4e_figure.load_source_csv(csv_path)

    def test_source_csv_loader_rejects_nonfinite_floats(self):
        csv_path = self.base / "source.csv"
        phase4e_figure.write_source_csv(self.load_records(), csv_path)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        rows[0]["simulated_time_us"] = "nan"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            phase4e_figure.load_source_csv(csv_path)


if __name__ == "__main__":
    unittest.main()
