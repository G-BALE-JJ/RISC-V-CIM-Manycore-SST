#!/usr/bin/env python3

import ast
import os
import subprocess
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCH_SHIM = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "mvm_noc_softmax_cpu",
        "ncores_selfcom_dma_softmax_archive.py",
    )
)
BASE_ARCHIVE = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "architecture", "archive", "ncores_selfcom_dma.py")
)


class SfuSoftmaxPipelineWrapperTest(unittest.TestCase):
    def read_wrapper(self):
        with open(os.path.join(SCRIPT_DIR, "run_noc_dma_softmax_sfu_pipeline.sh"), "r", encoding="utf-8") as source_file:
            return source_file.read()

    def read_archive_shim(self):
        with open(ARCH_SHIM, "r", encoding="utf-8") as source_file:
            return source_file.read()

    def test_archive_shim_keeps_three_vns_and_pins_dma_responses_to_vn0(self):
        source = self.read_archive_shim()

        self.assertIn("'\"num_vns\": 3,\\n'", source)
        self.assertIn(
            "os.getenv(\"GOLEM_DMA_RESPONSE_VN\", \"1\")",
            source,
        )
        self.assertIn("'            \"golem_dma_response_vn\": \"0\",'", source)
        self.assertNotIn('"num_vns": 1,\\n', source)

    def test_archive_shim_requires_exactly_one_directory_memnic_fragment(self):
        source = self.read_archive_shim()

        self.assertIn("directory_memnic_fragment", source)
        self.assertIn(
            "directory_memnic_fragment_count = source.count(directory_memnic_fragment)",
            source,
        )
        self.assertIn("if directory_memnic_fragment_count != 1", source)
        self.assertIn("raise RuntimeError", source)

        tree = ast.parse(source)
        fragment = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name)
                and target.id == "directory_memnic_fragment"
                for target in node.targets
            ):
                fragment = ast.literal_eval(node.value)
                break
        self.assertIsNotNone(fragment)
        with open(BASE_ARCHIVE, "r", encoding="utf-8") as source_file:
            archive_source = source_file.read()
        self.assertEqual(archive_source.count(fragment), 1)

    def test_archive_shim_reports_effective_global_memory_buffer(self):
        source = self.read_archive_shim()

        self.assertIn(
            'gm_buffer_length = os.getenv("GOLEM_GM_BUFFER_LENGTH", "64KB")',
            source,
        )
        self.assertIn(
            'print(f"[GOLEM] GlobalMemory link buffer_length={gm_buffer_length}")',
            source,
        )

    def run_wrapper(self, *args):
        env = os.environ.copy()
        env.pop("GOLEM_ARCH_SCRIPT", None)
        cmd = [
            os.path.join(SCRIPT_DIR, "run_noc_dma_softmax_sfu_pipeline.sh"),
            "--groups",
            "4",
            "--num-cores",
            "16",
            "--gemm-cores",
            "16",
            "--num-mem-nodes",
            "9",
            "--mesh-dim-x",
            "8",
            "--gemm-m",
            "128",
            "--gemm-n",
            "128",
            "--gemm-k",
            "128",
            "--gemm-block-m",
            "64",
            "--gemm-block-n",
            "64",
            "--gemm-block-k",
            "64",
            "--dry-run",
            *args,
        ]
        return subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def run_wrapper_with_env(self, extra_env, *args):
        env = os.environ.copy()
        env.pop("GOLEM_ARCH_SCRIPT", None)
        env.update(extra_env)
        cmd = [
            os.path.join(SCRIPT_DIR, "run_noc_dma_softmax_sfu_pipeline.sh"),
            "--groups",
            "4",
            "--num-cores",
            "16",
            "--gemm-cores",
            "16",
            "--num-mem-nodes",
            "9",
            "--mesh-dim-x",
            "8",
            "--gemm-m",
            "128",
            "--gemm-n",
            "128",
            "--gemm-k",
            "128",
            "--gemm-block-m",
            "64",
            "--gemm-block-n",
            "64",
            "--gemm-block-k",
            "64",
            "--dry-run",
            *args,
        ]
        return subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_wrapper_launches_sfu_binary_through_base_pipeline(self):
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("VANADIS_EXE=", result.stdout)
        self.assertIn("test_noc_dma_softmax_sfu", result.stdout)
        self.assertIn("GOLEM_SFU_ENABLE=1", result.stdout)
        self.assertIn("GOLEM_MATMUL_DTYPE=fp32", result.stdout)

    def test_wrapper_preserves_skip_build_for_base_pipeline(self):
        result = self.run_wrapper_with_env(
            {"GOLEM_SKIP_BUILD": "1"},
            "--group-manager-enable",
            "0",
            "--ctrl-link-enable",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_SKIP_BUILD=1", result.stdout)

    def test_wrapper_perf_profile_keeps_real_sst_but_disables_expensive_checks(self):
        result = self.run_wrapper_with_env(
            {
                "GOLEM_SFU_PERF_PROFILE": "1",
                "GOLEM_SFU_PRIMITIVE_SOFTMAX": "1",
            },
            "--group-manager-enable",
            "0",
            "--ctrl-link-enable",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_SFU_PERF_PROFILE=1", result.stdout)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_VERIFY=0", result.stdout)
        self.assertIn("GOLEM_SST_ENABLE_ALL_STATS=0", result.stdout)
        self.assertIn("GOLEM_BENCH_DISABLE_SST_STATS=1", result.stdout)
        self.assertIn("sst --num-threads=1", result.stdout)

    def test_wrapper_perf_profile_preserves_explicit_build_request(self):
        result = self.run_wrapper_with_env(
            {
                "GOLEM_SFU_PERF_PROFILE": "1",
                "GOLEM_SFU_PRIMITIVE_SOFTMAX": "1",
                "GOLEM_SKIP_BUILD": "0",
            },
            "--group-manager-enable",
            "0",
            "--ctrl-link-enable",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_SFU_PERF_PROFILE=1", result.stdout)
        self.assertIn("GOLEM_SKIP_BUILD=0", result.stdout)

    def test_wrapper_exports_softmax_row_block_env(self):
        result = self.run_wrapper_with_env(
            {
                "GOLEM_SFU_PRIMITIVE_SOFTMAX": "1",
                "GOLEM_SFU_PRIMITIVE_SOFTMAX_ROW_BLOCK": "4",
            },
            "--group-manager-enable",
            "0",
            "--ctrl-link-enable",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_ROW_BLOCK=4", result.stdout)

    def test_wrapper_exports_softmax_pipeline_depth_env(self):
        result = self.run_wrapper_with_env(
            {
                "GOLEM_SFU_PRIMITIVE_SOFTMAX": "1",
                "GOLEM_SFU_PRIMITIVE_SOFTMAX_PIPELINE_DEPTH": "2",
            },
            "--group-manager-enable",
            "0",
            "--ctrl-link-enable",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_PIPELINE_DEPTH=2", result.stdout)

    def test_runner_exposes_unified_sfu_job_softmax_switch(self):
        script = self.read_wrapper()
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX", script)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_CHUNK_ELEMS", script)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES", script)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS", script)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS", script)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_BAND_CORES", script)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS", script)

    def test_wrapper_exports_distributed_unified_job_softmax_env(self):
        result = self.run_wrapper_with_env(
            {
                "GOLEM_SFU_JOB_SOFTMAX": "1",
                "GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM": "1",
                "GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS": "1",
            },
            "--group-manager-enable",
            "0",
            "--ctrl-link-enable",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS=1", result.stdout)

    def test_workload_calls_unified_sfu_job_helper_when_enabled(self):
        with open(os.path.join(SCRIPT_DIR, "test_noc_dma_softmax_sfu.cpp"), "r", encoding="utf-8") as source_file:
            source = source_file.read()
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX", source)
        self.assertIn("golemRunStandaloneSoftmaxSfuJobForCore", source)
        self.assertIn("dispatch=sfu-unified-job-softmax", source)

    def test_wrapper_uses_full_row_sfu_checker_not_tile_local_checker(self):
        source = self.read_wrapper()

        self.assertIn("verify_softmax_sfu_against_golden.py", source)
        self.assertNotIn("verify_softmax_tile_against_golden.py", source)
        self.assertIn("--reference", source)

    def test_wrapper_can_recover_completed_sst_run_with_offline_verifier(self):
        source = self.read_wrapper()
        result = self.run_wrapper(
            "--recover-completed-run",
            "--verify-softmax",
            "--softmax-reference",
            "probability",
            "--softmax-c-file",
            os.path.join(SCRIPT_DIR, "data", "recover_c.bin"),
            "--group-manager-enable",
            "0",
            "--ctrl-link-enable",
            "0",
        )

        self.assertIn('GOLEM_SFU_RECOVER_COMPLETED_RUN="${GOLEM_SFU_RECOVER_COMPLETED_RUN:-0}"', source)
        self.assertIn("--recover-completed-run", source)
        self.assertIn("run_sfu_softmax_offline_verify()", source)
        self.assertLess(
            source.index('if [[ "$GOLEM_SFU_RECOVER_COMPLETED_RUN" != "0" ]]'),
            source.index("pushd \"$SCRIPT_DIR\" >/dev/null"),
        )
        self.assertLess(
            source.index("run_sfu_softmax_offline_verify()"),
            source.index('"$TESTS_DIR/run_noc_dma_pipeline.sh"'),
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("GOLEM_SFU_RECOVER_COMPLETED_RUN=1", result.stdout)
        self.assertIn("[SFU][DRY-RUN] python3", result.stdout)
        self.assertNotIn("VANADIS_EXE=", result.stdout)

    def test_wrapper_skips_offline_verify_when_sfu_guest_reports_failure(self):
        source = self.read_wrapper()

        self.assertIn("detect_sfu_guest_failure()", source)
        self.assertIn("DMA_LOAD_FAILED", source)
        self.assertIn("standalone unified job failed", source)
        self.assertIn("[SFU][ERROR] guest reported failure; skip softmax verifier", source)
        self.assertIn('run_summary_file="${GOLEM_RUN_SUMMARY_CSV:-$GOLEM_ARTIFACT_ROOT/stats/run_summary.csv}"', source)
        self.assertIn('resolved_run_id="${GOLEM_RUN_ID:-}"', source)
        self.assertIn("tail -n 1", source)
        self.assertIn("awk -F,", source)
        self.assertIn('derived_stdout_dir="$GOLEM_ARTIFACT_ROOT/stdout/overlap${GOLEM_DMA_OVERLAP:-0}/$resolved_run_id"', source)
        self.assertIn('log_file="${LOG_FILE:-${GOLEM_PRESET_LOG:-test.log}}"', source)
        self.assertIn('derived_log_path="$GOLEM_ARTIFACT_ROOT/logs/${log_stem}_${resolved_run_id}${log_ext}"', source)
        failure_gate = 'if [[ "$HAS_DRY_RUN" -eq 0 ]] && detect_sfu_guest_failure; then'
        verifier_gate = 'if [[ "$GOLEM_VERIFY_SOFTMAX" -eq 1 && "$HAS_DRY_RUN" -eq 0 ]]; then'
        self.assertLess(
            source.index('"$TESTS_DIR/run_noc_dma_pipeline.sh"'),
            source.index(failure_gate),
        )
        self.assertLess(
            source.index(failure_gate),
            source.rindex(verifier_gate),
        )

    def test_wrapper_unpacks_sfu_output_with_gemm_tile_layout(self):
        source = self.read_wrapper()

        self.assertIn('GOLEM_GEMM_OUT_LAYOUT="${GOLEM_GEMM_OUT_LAYOUT:-colmajor_tile}"', source)
        self.assertIn('export GOLEM_GEMM_OUT_LAYOUT', source)
        self.assertIn('export GOLEM_MATMUL_BLOCK_M="$GOLEM_GEMM_BLOCK_M"', source)
        self.assertIn('export GOLEM_MATMUL_BLOCK_N="$GOLEM_GEMM_BLOCK_N"', source)
        self.assertLess(
            source.index('export GOLEM_MATMUL_M="$GOLEM_GEMM_M"'),
            source.index('"$TESTS_DIR/run_noc_dma_pipeline.sh"'),
        )

    def test_wrapper_build_metadata_tracks_sfu_runtime_sources(self):
        source = self.read_wrapper()

        self.assertIn("golem_softmax_sfu_runtime.cpp", source)
        self.assertIn("golem_softmax_sfu_runtime.h", source)
        self.assertIn("test_noc_dma_softmax_sfu.cpp", source)
        self.assertIn("gemm_matmul_op.h", source)
        self.assertIn("test_noc_dma_softmax_sfu.build.env", source)

    def test_wrapper_rebuilds_if_binary_is_newer_than_build_metadata(self):
        source = self.read_wrapper()

        self.assertIn('[[ "$SFU_BIN" -nt "$SFU_BUILD_ENV" ]]', source)
        self.assertIn("return 1", source)

    def test_sst_shim_sets_ld_library_path_for_local_sst(self):
        with open(os.path.join(SCRIPT_DIR, "bin", "sst"), "r", encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertIn("SST_SOFTMAX_LD_LIBRARY_PATH", source)
        self.assertIn("/data4/jjgong/miniconda3/lib", source)
        self.assertIn("/data4/jjgong/local/sstcore/lib", source)
        self.assertIn('exec "$REAL_SST_BIN" "$@"', source)

    def test_private_wrapper_options_are_not_forwarded_to_base_pipeline(self):
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("Unknown option: --group-manager-enable", result.stdout)
        self.assertNotIn("Unknown option: --ctrl-link-enable", result.stdout)

    def test_wrapper_exports_interleaved_experiment_knob(self):
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")
        source = self.read_wrapper()
        with open(
            os.path.join(SCRIPT_DIR, "..", "mvm_noc_softmax_cpu", "ncores_selfcom_dma_softmax_archive.py"),
            "r",
            encoding="utf-8",
        ) as archive_file:
            archive_source = archive_file.read()

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('GOLEM_SFU_INTERLEAVE_GEMM="${GOLEM_SFU_INTERLEAVE_GEMM:-0}"', source)
        self.assertIn("GOLEM_SFU_INTERLEAVE_GEMM", result.stdout)
        self.assertIn("GOLEM_SFU_INTERLEAVE_GEMM", source)
        self.assertIn('"GOLEM_SFU_INTERLEAVE_GEMM"', archive_source)

    def test_wrapper_exports_standalone_softmax_knob_and_logits_file(self):
        result = self.run_wrapper(
            "--group-manager-enable",
            "0",
            "--ctrl-link-enable",
            "0",
            "--softmax-logits-file",
            os.path.join(SCRIPT_DIR, "data", "standalone_logits.bin"),
        )
        source = self.read_wrapper()
        with open(
            os.path.join(SCRIPT_DIR, "..", "mvm_noc_softmax_cpu", "ncores_selfcom_dma_softmax_archive.py"),
            "r",
            encoding="utf-8",
        ) as archive_file:
            archive_source = archive_file.read()

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('GOLEM_SFU_STANDALONE_SOFTMAX="${GOLEM_SFU_STANDALONE_SOFTMAX:-0}"', source)
        self.assertIn('GOLEM_SOFTMAX_LOGITS_FILE="${GOLEM_SOFTMAX_LOGITS_FILE:-}"', source)
        self.assertIn("--softmax-logits-file", source)
        self.assertIn("GOLEM_SFU_STANDALONE_SOFTMAX", result.stdout)
        self.assertIn("GOLEM_SOFTMAX_LOGITS_FILE", result.stdout)
        self.assertIn("GOLEM_SFU_STANDALONE_SOFTMAX", source)
        self.assertIn('"GOLEM_SFU_STANDALONE_SOFTMAX"', archive_source)

    def test_wrapper_uses_logits_golden_for_standalone_softmax(self):
        source = self.read_wrapper()

        self.assertIn("GOLEM_SFU_STANDALONE_SOFTMAX", source)
        self.assertIn('GOLEM_SOFTMAX_VERIFY_REFERENCE="logits"', source)
        self.assertIn("--logits-file", source)
        self.assertIn("$GOLEM_SOFTMAX_LOGITS_FILE", source)

    def test_wrapper_normalizes_relative_softmax_logits_file_to_script_dir(self):
        source = self.read_wrapper()

        self.assertIn("normalize_path_under_script_dir", source)
        self.assertIn('GOLEM_SOFTMAX_LOGITS_FILE="$(normalize_path_under_script_dir "$GOLEM_SOFTMAX_LOGITS_FILE")"', source)

    def test_wrapper_uses_shape_specific_default_softmax_logits_file(self):
        source = self.read_wrapper()

        self.assertIn('softmax_logits_${GOLEM_GEMM_M}x${GOLEM_GEMM_N}.bin', source)
        self.assertNotIn('GOLEM_SOFTMAX_LOGITS_FILE="$GOLEM_TENSOR_DIR/softmax_logits.bin"', source)

    def test_ctrl_architecture_forwards_sfu_mode_knobs_to_guest(self):
        with open(
            os.path.join(SCRIPT_DIR, "..", "..", "architecture", "ncores_selfcom_dma_ctrl.py"),
            "r",
            encoding="utf-8",
        ) as arch_file:
            arch_source = arch_file.read()

        self.assertIn('"GOLEM_SFU_INTERLEAVE_GEMM"', arch_source)
        self.assertIn('"GOLEM_SFU_STANDALONE_SOFTMAX"', arch_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX"', arch_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_CHUNK_ELEMS"', arch_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES"', arch_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS"', arch_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS"', arch_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_BAND_CORES"', arch_source)
        self.assertIn('"GOLEM_SFU_PRIMITIVE_SMOKE"', arch_source)

    def test_archive_shim_forwards_sfu_primitive_smoke_to_guest(self):
        with open(
            os.path.join(SCRIPT_DIR, "..", "mvm_noc_softmax_cpu", "ncores_selfcom_dma_softmax_archive.py"),
            "r",
            encoding="utf-8",
        ) as archive_file:
            archive_source = archive_file.read()

        self.assertIn('"GOLEM_SFU_PRIMITIVE_SMOKE"', archive_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX"', archive_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_CHUNK_ELEMS"', archive_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES"', archive_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS"', archive_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS"', archive_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_BAND_CORES"', archive_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS"', archive_source)

    def test_wrapper_and_architectures_forward_scaled_primitive_smoke_knobs(self):
        source = self.read_wrapper()
        with open(
            os.path.join(SCRIPT_DIR, "..", "..", "architecture", "ncores_selfcom_dma_ctrl.py"),
            "r",
            encoding="utf-8",
        ) as arch_file:
            arch_source = arch_file.read()
        with open(
            os.path.join(SCRIPT_DIR, "..", "mvm_noc_softmax_cpu", "ncores_selfcom_dma_softmax_archive.py"),
            "r",
            encoding="utf-8",
        ) as archive_file:
            archive_source = archive_file.read()

        for knob in (
            "GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS",
            "GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS",
        ):
            self.assertIn(f'{knob}="${{{knob}:-', source)
            self.assertIn(f"export {knob}", source)
            self.assertIn(knob, self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0").stdout)
            self.assertIn(f'"{knob}"', arch_source)
            self.assertIn(f'"{knob}"', archive_source)

    def test_wrapper_and_architectures_forward_hbm_streaming_primitive_knobs(self):
        source = self.read_wrapper()
        result = self.run_wrapper("--group-manager-enable", "0", "--ctrl-link-enable", "0")
        with open(
            os.path.join(SCRIPT_DIR, "..", "..", "architecture", "ncores_selfcom_dma_ctrl.py"),
            "r",
            encoding="utf-8",
        ) as arch_file:
            arch_source = arch_file.read()
        with open(
            os.path.join(SCRIPT_DIR, "..", "mvm_noc_softmax_cpu", "ncores_selfcom_dma_softmax_archive.py"),
            "r",
            encoding="utf-8",
        ) as archive_file:
            archive_source = archive_file.read()

        self.assertEqual(result.returncode, 0, result.stdout)
        for knob in (
            "GOLEM_SFU_PRIMITIVE_HBM_STREAM",
            "GOLEM_SFU_PRIMITIVE_HBM_ELEMS",
            "GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS",
            "GOLEM_SFU_PRIMITIVE_HBM_OPS",
        ):
            self.assertIn(f'{knob}="${{{knob}:-', source)
            self.assertIn(f"export {knob}", source)
            self.assertIn(knob, result.stdout)
            self.assertIn(f'"{knob}"', arch_source)
            self.assertIn(f'"{knob}"', archive_source)


if __name__ == "__main__":
    unittest.main()
