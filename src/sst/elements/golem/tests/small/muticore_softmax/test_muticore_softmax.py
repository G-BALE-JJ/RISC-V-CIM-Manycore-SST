#!/usr/bin/env python3

import json
import os
import subprocess
import tempfile
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(SCRIPT_DIR, "run_muticore_softmax.sh")
PARSER = os.path.join(SCRIPT_DIR, "parse_muticore_softmax.py")


class MuticoreSoftmaxTest(unittest.TestCase):
    def test_dry_run_selects_one_row_engine_job_per_tile_for_1024x4096(self):
        result = subprocess.run(
            [RUNNER, "--rows", "1024", "--cols", "4096", "--dry-run"],
            cwd=SCRIPT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        for contract in [
            "GOLEM_SFU_JOB_SOFTMAX_ROW_ENGINE=1",
            "GOLEM_SFU_JOB_SOFTMAX_TENSOR_CONTROLLER=1",
            "GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS=0",
            "GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES=1",
            "GOLEM_SFU_JOB_SOFTMAX_BAND_CORES=16",
            "GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS=64",
            "GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS=64",
            "--global-stride-kb 2304",
            "VANADIS_CPU_CLOCK=2.3GHz",
            "GOLEM_DMA_BURST_BYTES=262144",
            "/guest/muticore_softmax_r1024_d4096_row_engine_",
            "/test_noc_dma_softmax_sfu",
        ]:
            self.assertIn(contract, result.stdout)

    def test_dry_run_reserves_tensor_controller_scratch_for_64x4096(self):
        result = subprocess.run(
            [RUNNER, "--rows", "64", "--cols", "4096", "--dry-run"],
            cwd=SCRIPT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("global_stride_kb=1280", result.stdout)
        self.assertIn("--global-stride-kb 1280", result.stdout)

    def test_each_attempt_isolates_guest_stdout_and_stats(self):
        with open(RUNNER, encoding="utf-8") as source_file:
            source = source_file.read()
        self.assertIn('attempt_id="${run_id}_$(date +%Y%m%d_%H%M%S)_$$"', source)
        self.assertIn('attempt_stats="$artifact_root/stats/$attempt_id"', source)
        self.assertIn('attempt_stdout="$artifact_root/stdout/$attempt_id"', source)
        self.assertIn('GOLEM_SFU_GUEST_SNAPSHOT="$artifact_root/guest/$attempt_id/', source)
        self.assertIn("--require-contract", source)

    def test_runner_exposes_noc_bandwidth_sensitivity_knob(self):
        with open(RUNNER, encoding="utf-8") as source_file:
            source = source_file.read()
        self.assertIn('GOLEM_SOFTMAX_NOC_LINK_BW:-1200GB/s', source)
        self.assertIn('--noc-link-bw "$noc_link_bw"', source)
        self.assertIn('--noc-xbar-bw "$noc_xbar_bw"', source)

    def test_tensor_parser_requires_the_expected_number_of_control_completions(self):
        with open(PARSER, encoding="utf-8") as source_file:
            source = source_file.read()
        self.assertIn("expected_control_events = min(args.rows, len(components))", source)
        self.assertIn('timeline_event_counts["completion_received"] == expected_control_events', source)

    def test_parser_takes_critical_max_across_sfus_not_global_sum(self):
        rows = []
        for core in range(16):
            modeled = 66061
            issue = 1000 + core * 100
            ready = 43751652 + core * 100
            component = f"core{core}:rocc:sfu"
            rows.extend(
                [
                    f"core{core},cycles,1,Accumulator,538140890,0,1237105,1237105,1237105,1,1",
                    f"{component},sfu_row_engine_jobs,,Accumulator,0,0,1,1,1,1,1",
                    f"{component},sfu_row_engine_rows,,Accumulator,0,0,64,4096,1,64,64",
                    f"{component},sfu_row_engine_completed_jobs,,Accumulator,0,0,1,1,1,1,1",
                    f"{component},sfu_row_engine_max_cycles,,Accumulator,0,0,16384,0,1,16384,16384",
                    f"{component},sfu_row_engine_exp_sum_cycles,,Accumulator,0,0,65536,0,1,65536,65536",
                    f"{component},sfu_row_engine_normalize_cycles,,Accumulator,0,0,16384,0,1,16384,16384",
                    f"{component},sfu_row_engine_max_start_cycles,,Accumulator,0,0,0,0,1,0,0",
                    f"{component},sfu_row_engine_max_end_cycles,,Accumulator,0,0,64768,0,1,64768,64768",
                    f"{component},sfu_row_engine_exp_sum_start_cycles,,Accumulator,0,0,256,0,1,256,256",
                    f"{component},sfu_row_engine_exp_sum_end_cycles,,Accumulator,0,0,65536,0,1,65536,65536",
                    f"{component},sfu_row_engine_normalize_start_cycles,,Accumulator,0,0,1280,0,1,1280,1280",
                    f"{component},sfu_row_engine_normalize_end_cycles,,Accumulator,0,0,66048,0,1,66048,66048",
                    f"{component},sfu_row_engine_modeled_cycles,,Accumulator,0,0,{modeled},0,1,{modeled},{modeled}",
                    f"{component},sfu_row_engine_issue_tick,,Accumulator,0,0,{issue},0,1,{issue},{issue}",
                    f"{component},sfu_row_engine_ready_tick,,Accumulator,0,0,{ready},0,1,{ready},{ready}",
                    f"{component},sfu_reduction_max_requests,,Accumulator,0,0,0,0,0,0,0",
                    f"{component},sfu_reduction_sum_requests,,Accumulator,0,0,0,0,0,0,0",
                ]
            )
        with tempfile.TemporaryDirectory() as tmp:
            stats = os.path.join(tmp, "stats.txt")
            output = os.path.join(tmp, "result.json")
            with open(stats, "w", encoding="utf-8") as stats_file:
                stats_file.write("\n".join(rows))
            result = subprocess.run(
                [
                    "python3",
                    PARSER,
                    "--stats",
                    stats,
                    "--rows",
                    "1024",
                    "--cols",
                    "4096",
                    "--output",
                    output,
                    "--require-contract",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            with open(output, "r", encoding="utf-8") as result_file:
                parsed = json.load(result_file)
        self.assertEqual(parsed["row_engine_jobs"], 16)
        self.assertEqual(parsed["rows_completed"], 1024)
        self.assertEqual(parsed["modeled_critical_cycles"], 66061)
        self.assertEqual(parsed["vector_max_active_critical_cycles"], 16384)
        self.assertEqual(parsed["exp_sum_active_critical_cycles"], 65536)
        self.assertEqual(parsed["normalize_active_critical_cycles"], 16384)
        self.assertEqual(parsed["stage_cycles"]["max_start"], 0)
        self.assertEqual(parsed["stage_cycles"]["max_end"], 64768)
        self.assertEqual(parsed["stage_cycles"]["exp_sum_start"], 256)
        self.assertEqual(parsed["stage_cycles"]["exp_sum_end"], 65536)
        self.assertEqual(parsed["stage_cycles"]["normalize_start"], 1280)
        self.assertEqual(parsed["stage_cycles"]["normalize_end"], 66048)
        self.assertEqual(parsed["stage_cycles"]["sequential_active"], 98304)
        self.assertEqual(parsed["stage_cycles"]["temporal_span"], 66048)
        self.assertEqual(parsed["stage_cycles"]["overlap_cycles"], 32256)
        self.assertEqual(parsed["reduction_request_messages"], 0)
        self.assertEqual(parsed["vanadis_critical_cycles"], 1237105)
        self.assertEqual(parsed["sst_simulated_time_ps"], 538140890)
        self.assertEqual(parsed["whole_architecture_cycles_at_clock"], 1237725)
        self.assertEqual(parsed["noc_simulated_window_cycles_at_clock"], 1237725)
        self.assertNotEqual(parsed["modeled_critical_cycles"], 1056976)

    def test_parser_does_not_claim_rows_from_an_incomplete_tile(self):
        rows = []
        for core, completed in [(0, 1), (1, 0)]:
            component = f"core{core}:rocc:sfu"
            rows.extend(
                [
                    f"{component},sfu_row_engine_jobs,,Accumulator,0,0,1,1,1,1,1",
                    f"{component},sfu_row_engine_rows,,Accumulator,0,0,64,4096,1,64,64",
                    f"{component},sfu_row_engine_completed_jobs,,Accumulator,0,0,{completed},0,1,{completed},{completed}",
                ]
            )
        with tempfile.TemporaryDirectory() as tmp:
            stats = os.path.join(tmp, "stats.txt")
            output = os.path.join(tmp, "result.json")
            with open(stats, "w", encoding="utf-8") as stats_file:
                stats_file.write("\n".join(rows))
            result = subprocess.run(
                [
                    "python3",
                    PARSER,
                    "--stats",
                    stats,
                    "--rows",
                    "128",
                    "--cols",
                    "4096",
                    "--output",
                    output,
                    "--require-contract",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            with open(output, "r", encoding="utf-8") as result_file:
                parsed = json.load(result_file)
        self.assertEqual(parsed["completed_jobs"], 1)
        self.assertEqual(parsed["rows_completed"], 64)
        self.assertFalse(parsed["contract_pass"])

    def test_tensor_parser_ignores_idle_sfu_zero_issue_ticks(self):
        rows = []
        for core in range(16):
            component = f"core{core}:rocc:sfu"
            active = core == 3
            rows.extend([
                f"{component},sfu_row_engine_jobs,,Accumulator,0,0,{1 if active else 0},0,{1 if active else 0},0,{1 if active else 0}",
                f"{component},sfu_row_engine_rows,,Accumulator,0,0,{1024 if active else 0},0,{1 if active else 0},0,{1024 if active else 0}",
                f"{component},sfu_row_engine_completed_jobs,,Accumulator,0,0,{1 if active else 0},0,{1 if active else 0},0,{1 if active else 0}",
                f"{component},sfu_row_engine_modeled_cycles,,Accumulator,0,0,{66061 if active else 0},0,{1 if active else 0},0,{66061 if active else 0}",
                f"{component},sfu_row_engine_issue_tick,,Accumulator,0,0,{100000 if active else 0},0,{1 if active else 0},{100000 if active else 0},{100000 if active else 0}",
                f"{component},sfu_row_engine_ready_tick,,Accumulator,0,0,{130150 if active else 0},0,{1 if active else 0},0,{130150 if active else 0}",
                f"{component},sfu_row_engine_completion_observed_tick,,Accumulator,0,0,{130150 if active else 0},0,{1 if active else 0},0,{130150 if active else 0}",
                f"{component},sfu_tensor_band_dispatch_tick,,Accumulator,0,0,{100100 if active else 0},0,{16 if active else 0},{100040 if active else 0},{100160 if active else 0}",
                f"{component},sfu_tensor_worker_dispatch_tick,,Accumulator,0,0,{100240 if active else 0},0,{16 if active else 0},{100200 if active else 0},{100280 if active else 0}",
                f"{component},sfu_tensor_input_dma_ready_tick,,Accumulator,0,0,{101300 if active else 0},0,{1024 if active else 0},{101000 if active else 0},{101600 if active else 0}",
                f"{component},sfu_tensor_max_start_tick,,Accumulator,0,0,{101310 if active else 0},0,{1024 if active else 0},{101010 if active else 0},{101610 if active else 0}",
                f"{component},sfu_tensor_max_done_tick,,Accumulator,0,0,{101420 if active else 0},0,{1024 if active else 0},{101120 if active else 0},{101720 if active else 0}",
                f"{component},sfu_tensor_exp_sum_start_tick,,Accumulator,0,0,{101420 if active else 0},0,{1024 if active else 0},{101120 if active else 0},{101720 if active else 0}",
                f"{component},sfu_tensor_exp_sum_done_tick,,Accumulator,0,0,{101700 if active else 0},0,{1024 if active else 0},{101400 if active else 0},{102000 if active else 0}",
                f"{component},sfu_tensor_normalize_start_tick,,Accumulator,0,0,{101700 if active else 0},0,{1024 if active else 0},{101400 if active else 0},{102000 if active else 0}",
                f"{component},sfu_tensor_normalize_done_tick,,Accumulator,0,0,{102000 if active else 0},0,{1024 if active else 0},{101500 if active else 0},{102500 if active else 0}",
                f"{component},sfu_tensor_compute_done_tick,,Accumulator,0,0,{102000 if active else 0},0,{1024 if active else 0},{101500 if active else 0},{102500 if active else 0}",
                f"{component},sfu_tensor_output_dma_ack_tick,,Accumulator,0,0,{120000 if active else 0},0,{1024 if active else 0},{110000 if active else 0},{130000 if active else 0}",
                f"{component},sfu_tensor_completion_received_tick,,Accumulator,0,0,{130100 if active else 0},0,{16 if active else 0},{130050 if active else 0},{130150 if active else 0}",
                f"{component},sfu_tensor_guest_wait_observed_tick,,Accumulator,0,0,{150000 if active else 0},0,{1 if active else 0},{150000 if active else 0},{150000 if active else 0}",
            ])
        with tempfile.TemporaryDirectory() as tmp:
            stats = os.path.join(tmp, "stats.txt")
            output = os.path.join(tmp, "result.json")
            stdout_dir = os.path.join(tmp, "stdout")
            os.mkdir(stdout_dir)
            with open(stats, "w", encoding="utf-8") as stats_file:
                stats_file.write("\n".join(rows))
            with open(os.path.join(stdout_dir, "stdout-100"), "w", encoding="utf-8") as stdout_file:
                stdout_file.write(
                    "[SOFTMAX-ROW-ENGINE] core=3 rows=1024 start_cycle=200 "
                    "end_cycle=400 cycles=200 output_dma_completion=1 tensor_controller=1 "
                    "launch_start_cycle=210 descriptors_ready_cycle=220 params_write_done_cycle=230 "
                    "desc_write_done_cycle=240 issue_return_cycle=250 "
                    "wait_start_cycle=260 wait_return_cycle=400\n"
                )
            result = subprocess.run([
                "python3", PARSER, "--stats", stats, "--rows", "1024",
                "--cols", "4096", "--output", output, "--stdout-dir", stdout_dir,
                "--tensor-controller",
                "--require-contract",
            ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, result.stdout)
            with open(output, encoding="utf-8") as result_file:
                parsed = json.load(result_file)
        self.assertEqual(parsed["issue_to_ready_ticks"], 30150)
        self.assertEqual(parsed["accelerator_latency_cycles"], 70)
        self.assertEqual(parsed["issue_to_completion_observed_cycles"], 70)
        self.assertEqual(parsed["timeline_ticks"]["descriptor_accept"], 100000)
        self.assertEqual(parsed["timeline_ticks"]["accelerator_complete"], 130150)
        self.assertEqual(parsed["timeline_ticks"]["first_band_dispatch"], 100040)
        self.assertEqual(parsed["timeline_ticks"]["last_band_dispatch"], 100160)
        self.assertEqual(parsed["timeline_ticks"]["first_worker_dispatch"], 100200)
        self.assertEqual(parsed["timeline_ticks"]["first_input_dma_ready"], 101000)
        self.assertEqual(parsed["timeline_ticks"]["last_input_dma_ready"], 101600)
        self.assertEqual(parsed["timeline_ticks"]["first_max_start"], 101010)
        self.assertEqual(parsed["timeline_ticks"]["last_max_done"], 101720)
        self.assertEqual(parsed["timeline_ticks"]["last_exp_sum_done"], 102000)
        self.assertEqual(parsed["timeline_ticks"]["last_normalize_done"], 102500)
        self.assertEqual(parsed["timeline_ticks"]["last_compute_done"], 102500)
        self.assertEqual(parsed["timeline_ticks"]["final_output_dma_ack"], 130000)
        self.assertEqual(parsed["timeline_ticks"]["last_completion_received"], 130150)
        self.assertEqual(parsed["timeline_ticks"]["guest_wait_observed"], 150000)
        self.assertEqual(parsed["timeline_cycles"]["descriptor_to_final_output_ack"], 69)
        self.assertEqual(parsed["timeline_cycles"]["final_output_ack_to_completion_received"], 1)
        self.assertEqual(parsed["timeline_cycles"]["completion_received_to_guest_wait_observed"], 46)
        self.assertEqual(parsed["actual_stage_windows_cycles"]["max"], 2)
        self.assertEqual(parsed["actual_stage_windows_cycles"]["exp_sum"], 3)
        self.assertEqual(parsed["actual_stage_windows_cycles"]["normalize"], 3)
        self.assertEqual(parsed["timeline_cycles"]["guest_start_to_descriptor_accept"], 30)
        self.assertEqual(parsed["timeline_cycles"]["guest_wait_observed_to_guest_end"], 55)
        self.assertEqual(parsed["timeline_event_counts"]["band_dispatch"], 16)
        self.assertEqual(parsed["timeline_event_counts"]["worker_dispatch"], 16)
        self.assertEqual(parsed["timeline_event_counts"]["input_dma_ready"], 1024)
        self.assertEqual(parsed["timeline_event_counts"]["max_done"], 1024)
        self.assertEqual(parsed["timeline_event_counts"]["exp_sum_done"], 1024)
        self.assertEqual(parsed["timeline_event_counts"]["normalize_done"], 1024)
        self.assertEqual(parsed["timeline_event_counts"]["compute_done"], 1024)
        self.assertEqual(parsed["timeline_event_counts"]["output_dma_ack"], 1024)
        self.assertEqual(parsed["timeline_event_counts"]["completion_received"], 16)
        self.assertEqual(parsed["timeline_event_counts"]["guest_wait_observed"], 1)
        self.assertEqual(parsed["guest_launch_cycles"]["descriptor_construction"], 10)
        self.assertEqual(parsed["guest_launch_cycles"]["params_write"], 10)
        self.assertEqual(parsed["guest_launch_cycles"]["descriptor_write"], 10)
        self.assertEqual(parsed["guest_launch_cycles"]["issue_return"], 10)
        self.assertEqual(parsed["guest_launch_cycles"]["wait_entry_to_return"], 140)


if __name__ == "__main__":
    unittest.main()
