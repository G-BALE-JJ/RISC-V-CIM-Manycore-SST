#!/usr/bin/env python3

import csv
import os
import re
import subprocess
import tempfile
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_local(path):
    with open(os.path.join(SCRIPT_DIR, path), "r", encoding="utf-8") as f:
        return f.read()


def read_repo_relative(path):
    with open(os.path.join(SCRIPT_DIR, path), "r", encoding="utf-8") as f:
        return f.read()


class SfuWorkloadScaffoldTest(unittest.TestCase):
    def test_ex_instr_declares_sfu_rocc_wrappers(self):
        text = read_local("ex_instr.h")
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE = 0x17", text)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_WAIT = 0x18", text)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_PRIMITIVE = 0x19", text)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_WAIT = 0x1a", text)
        self.assertIn("sfu_softmax_tile", text)
        self.assertIn("sfu_wait", text)
        self.assertIn("sfu_primitive", text)
        self.assertIn("sfu_primitive_wait", text)
        self.assertIn(".insn r 0x0b, 7", text)

    def test_runtime_header_declares_descriptor_and_entry_point(self):
        text = read_local("golem_softmax_sfu_runtime.h")
        self.assertIn("struct SFUSoftmaxTileDesc", text)
        self.assertIn("golemRunSoftmaxSfuForCore", text)
        self.assertIn("job_id", text)
        self.assertIn("n_tiles_per_row", text)

    def test_runtime_header_declares_primitive_descriptor_for_guest(self):
        text = read_local("golem_softmax_sfu_runtime.h")
        self.assertIn("enum class SFUPrimitiveOp", text)
        self.assertIn("EXP = 0x01", text)
        self.assertIn("LOG = 0x02", text)
        self.assertIn("RECIPROCAL = 0x03", text)
        self.assertIn("struct SFUPrimitiveDesc", text)
        self.assertIn("input0_gm_addr", text)
        self.assertIn("output_gm_addr", text)
        self.assertRegex(text, r"static_assert\s*\(\s*sizeof\s*\(\s*SFUPrimitiveDesc\s*\)\s*==\s*64")

    def test_runtime_uses_existing_dma_and_two_phase_issue_wait(self):
        text = read_local("golem_softmax_sfu_runtime.cpp")
        self.assertIn("dma_remote_load_to_gm", text)
        self.assertIn("write_sfu_desc_to_gm", text)
        self.assertIn("sfu_softmax_tile", text)
        self.assertIn("sfu_wait", text)
        self.assertIn("remote_store", text)
        self.assertLess(text.index("sfu_softmax_tile"), text.index("sfu_wait"))

    def test_workload_enables_sfu_mode_and_keeps_fp32_boundary(self):
        text = read_local("test_noc_dma_softmax_sfu.cpp")
        self.assertIn("golemRunSoftmaxSfuForCore", text)
        self.assertIn("GOLEM_DTYPE_FP32", text)
        self.assertIn("mode=sfu", text)

    def test_workload_can_skip_sfu_for_gemm_output_diagnostics(self):
        text = read_local("test_noc_dma_softmax_sfu.cpp")
        self.assertIn('read_i64_env_or_default("GOLEM_SFU_SKIP_SOFTMAX", 0)', text)
        self.assertIn("skip_softmax != 0", text)
        self.assertLess(text.index("run_gemm_for_core"), text.index("GOLEM_SFU_SKIP_SOFTMAX"))
        self.assertLess(text.index("GOLEM_SFU_SKIP_SOFTMAX"), text.index("golemRunSoftmaxSfuForCore"))

    def test_sfu_runtime_separates_executor_local_gm_from_worker_assignment(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        runtime_header = read_local("golem_softmax_sfu_runtime.h")
        runtime_source = read_local("golem_softmax_sfu_runtime.cpp")

        self.assertIn("resolve_executor_core_from_argv_or_exit(argc, argv, requested_core_id)", entry_source)
        self.assertIn("run_gemm_for_core(executor_core_id, requested_core_id, op_desc)", entry_source)
        self.assertIn("make_gemm_runtime_context(executor_core_id)", entry_source)
        self.assertIn("gemm_worker_slot_for_core(worker_core_id)", entry_source)
        self.assertIn("gemm_descriptor_for_task(executor_core_id, task_id, cfg)", entry_source)
        self.assertIn("gemm_tiled<float>(executor_core_id, desc, rt, &stats)", entry_source)
        self.assertNotIn("golemRunMatmul(kernel, &a_desc, &b_desc, &c_desc)", entry_source)
        self.assertIn(
            "&softmax_desc, executor_core_id, requested_core_id, &cfg, job_id",
            entry_source,
        )
        self.assertIn("int executor_core_id,", runtime_header)
        self.assertIn("int worker_core_id,", runtime_header)
        self.assertIn("gemm_worker_slot_for_core(worker_core_id)", runtime_source)
        self.assertIn("gemm_task_desc_for_task(executor_core_id, task_id, *cfg)", runtime_source)
        self.assertIn("gm_addr(executor_core_id, SFU_DESC_GM_OFFSET)", runtime_source)
        self.assertIn("gm_addr(executor_core_id, SFU_INPUT_GM_OFFSET)", runtime_source)
        self.assertIn("gm_addr(executor_core_id, SFU_OUTPUT_GM_OFFSET)", runtime_source)
        self.assertIn("dma_remote_load_to_gm(executor_core_id,", runtime_source)
        self.assertIn("write_sfu_desc_to_gm(executor_core_id,", runtime_source)

    def test_runtime_has_interleaved_local_accum_experiment_mode(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        runtime_header = read_local("golem_softmax_sfu_runtime.h")
        runtime_source = read_local("golem_softmax_sfu_runtime.cpp")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_INTERLEAVE_GEMM", 0)', entry_source)
        self.assertIn("golemRunSoftmaxSfuTileFromLocalAccum", runtime_header)
        self.assertIn("local_input_gm_addr = local_accum_gm", runtime_source)
        self.assertIn("skip_hbm_reload", runtime_source)
        self.assertIn("mode=sfu-interleaved-local-accum", entry_source)
        self.assertLess(
            entry_source.index("gemm_tiled<float>(executor_core_id, desc, rt, &stats)"),
            entry_source.index("golemRunSoftmaxSfuTileFromLocalAccum"),
        )

    def test_interleaved_mode_keeps_default_bulk_softmax_path_available(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        entry_body = re.search(
            r"int\s+run_riscv_gemm_softmax_sfu\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_host_smoke",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(entry_body)
        body = entry_body.group("body")

        self.assertIn("run_gemm_for_core(executor_core_id, requested_core_id, op_desc)", body)
        self.assertIn("golemRunSoftmaxSfuForCore(", body)
        self.assertLess(
            body.index('read_i64_env_or_default("GOLEM_SFU_INTERLEAVE_GEMM", 0)'),
            body.index("run_gemm_for_core(executor_core_id, requested_core_id, op_desc)"),
        )

    def test_workload_has_standalone_softmax_only_mode_before_gemm(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        runtime_header = read_local("golem_softmax_sfu_runtime.h")
        runtime_source = read_local("golem_softmax_sfu_runtime.cpp")
        entry_body = re.search(
            r"int\s+run_riscv_gemm_softmax_sfu\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_host_smoke",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(entry_body)
        body = entry_body.group("body")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_STANDALONE_SOFTMAX", 0)', body)
        self.assertIn("mode=sfu-standalone-softmax", body)
        self.assertIn("golemRunStandaloneSoftmaxSfuForCore", runtime_header)
        self.assertIn("golemRunStandaloneSoftmaxSfuForCore", runtime_source)
        self.assertIn("task.c_base_mm", runtime_source)
        self.assertLess(
            body.index('read_i64_env_or_default("GOLEM_SFU_STANDALONE_SOFTMAX", 0)'),
            body.index("run_gemm_for_core(executor_core_id, requested_core_id, op_desc)"),
        )
        self.assertLess(
            body.index("golemRunStandaloneSoftmaxSfuForCore"),
            body.index("run_gemm_for_core(executor_core_id, requested_core_id, op_desc)"),
        )

    def test_runtime_exposes_unified_softmax_job_helper(self):
        runtime_h = read_local("golem_softmax_sfu_runtime.h")
        runtime_cc = read_local("golem_softmax_sfu_runtime.cpp")
        self.assertIn("golemRunStandaloneSoftmaxSfuJobForCore", runtime_h)
        self.assertIn("SFUJobDesc", runtime_cc)
        self.assertIn("static_cast<uint32_t>(SFUJobOp::SOFTMAX_ROW)", runtime_cc)
        self.assertIn("sfu_job(desc_gm", runtime_cc)
        self.assertIn("sfu_job_wait(tag)", runtime_cc)

    def test_runtime_unified_job_helper_writes_distributed_worker_fields(self):
        runtime_h = read_local("golem_softmax_sfu_runtime.h")
        runtime_cc = read_local("golem_softmax_sfu_runtime.cpp")
        for parameter in (
            "uint32_t worker_slot",
            "uint32_t owner_core",
            "uint32_t flags",
        ):
            self.assertIn(parameter, runtime_h)
            self.assertIn(parameter, runtime_cc)
        self.assertIn("desc.reserved0 = worker_slot", runtime_cc)
        self.assertIn("desc.owner_core = owner_core", runtime_cc)
        self.assertIn("desc.flags = flags", runtime_cc)

    def test_standalone_logits_can_use_unified_sfu_job_without_gemm(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        entry_body = re.search(
            r"int\s+run_riscv_gemm_softmax_sfu\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_host_smoke",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(entry_body)
        body = entry_body.group("body")

        self.assertIn("mode=sfu-standalone-job-softmax", entry_source)
        self.assertIn("run_standalone_unified_job_softmax_for_core", entry_source)
        self.assertIn("golemRunStandaloneSoftmaxSfuJobForCore", entry_source)
        self.assertLess(
            body.index('read_i64_env_or_default("GOLEM_SFU_STANDALONE_SOFTMAX", 0)'),
            body.index('read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX", 0)'),
        )
        standalone_job_index = body.index("run_standalone_unified_job_softmax_for_core")
        gemm_job_index = body.index("run_gemm_unified_job_softmax_for_core")
        self.assertLess(standalone_job_index, gemm_job_index)

    def test_standalone_unified_job_streams_hbm_logits_in_row_bands(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS", entry_source)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS", entry_source)
        self.assertIn("run_standalone_unified_job_softmax_band_for_core", entry_source)
        self.assertIn("sub_desc.outer", entry_source)
        self.assertIn("MatmulRuntimeConfig sub_cfg = cfg", entry_source)
        self.assertIn("sub_cfg.m = sub_job_rows", entry_source)
        self.assertIn("row_band_begin", entry_source)
        self.assertIn("row_band_rows", entry_source)
        self.assertIn("trace_bands", entry_source)
        self.assertIn("band_stage=load", entry_source)
        self.assertIn("band_stage=job", entry_source)
        self.assertIn("band_stage=store", entry_source)
        self.assertIn("band_stage=done", entry_source)
        self.assertIn("const bool full_tile_band", entry_source)
        self.assertIn("if (!full_tile_band)", entry_source)
        self.assertIn("band_matrix_bytes", entry_source)
        self.assertIn("band_matrix_bytes_aligned", entry_source)
        self.assertIn("std::vector<float> row_band", entry_source)
        self.assertIn("const int global_row = row_band_begin + r", entry_source)

        standalone_body = re.search(
            r"int\s+run_standalone_unified_job_softmax_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+read_requested_core_from_argv",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(standalone_body)
        body = standalone_body.group("body")
        self.assertNotIn("matrix_bytes_aligned + matrix_bytes_aligned", body)
        self.assertIn("band_matrix_bytes_aligned + band_matrix_bytes_aligned + tile_bytes_aligned", body)
        self.assertIn("for (int row_band_begin = 0; row_band_begin < cfg.m;", body)
        self.assertLess(
            body.index("run_standalone_unified_job_softmax_band_for_core"),
            body.rindex("mode=sfu-standalone-job-softmax"),
        )

    def test_standalone_unified_job_distributes_row_bands_across_cores(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")
        arch_source = read_repo_relative("../../architecture/ncores_selfcom_dma_ctrl.py")

        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_BAND_CORES", entry_source)
        self.assertIn("band_core_count", entry_source)
        self.assertIn("requested_core_id >= band_core_count", entry_source)
        self.assertIn("band_index % band_core_count", entry_source)
        self.assertIn("band_slot != requested_core_id", entry_source)
        self.assertIn("band_cores=%d", entry_source)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_BAND_CORES", wrapper_source)
        self.assertIn("export GOLEM_SFU_JOB_SOFTMAX_BAND_CORES", wrapper_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_BAND_CORES"', arch_source)

    def test_standalone_unified_job_splits_staging_band_into_smaller_sfu_jobs(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")
        arch_source = read_repo_relative("../../architecture/ncores_selfcom_dma_ctrl.py")

        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS", entry_source)
        self.assertIn("job_rows_per_issue", entry_source)
        self.assertIn("for (int job_row_begin = 0; job_row_begin < row_band_rows;", entry_source)
        self.assertIn("sub_job_offset_bytes", entry_source)
        self.assertIn("input_gm + sub_job_offset_bytes", entry_source)
        self.assertIn("output_gm + sub_job_offset_bytes", entry_source)
        self.assertIn("sub_desc.outer = static_cast<uint64_t>(sub_job_rows)", entry_source)
        self.assertIn("sub_cfg.block_m = sub_job_rows", entry_source)
        self.assertIn("band_stage=subjob", entry_source)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS", wrapper_source)
        self.assertIn("export GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS", wrapper_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS"', arch_source)

    def test_standalone_unified_job_can_stream_direct_rowmajor_hbm(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")
        arch_source = read_repo_relative("../../architecture/ncores_selfcom_dma_ctrl.py")

        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM", entry_source)
        self.assertIn("direct_rowmajor_hbm", entry_source)
        self.assertIn("OFF_SFU_SOFTMAX_ROWMAJOR_BASE", entry_source)
        self.assertIn("rowmajor_input_hbm", entry_source)
        self.assertIn("rowmajor_output_hbm", entry_source)
        self.assertIn("run_standalone_unified_job_softmax_direct_band_for_core", entry_source)
        self.assertIn("band_stage=direct-load", entry_source)
        self.assertIn("band_stage=direct-store", entry_source)
        self.assertIn("mode=sfu-standalone-job-softmax", entry_source)
        self.assertIn("direct_rowmajor_hbm=%d", entry_source)

        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM", wrapper_source)
        self.assertIn("export GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM", wrapper_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM"', arch_source)

    def test_direct_rowmajor_hbm_can_distribute_columns_across_physical_sfus(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")
        arch_source = read_repo_relative("../../architecture/ncores_selfcom_dma_ctrl.py")

        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS", entry_source)
        self.assertIn("run_standalone_unified_job_softmax_distributed_direct_for_core", entry_source)
        self.assertIn("cooperative_group_count", entry_source)
        self.assertIn("cooperative_group_id", entry_source)
        self.assertIn("worker_slot", entry_source)
        self.assertIn("slice_begin", entry_source)
        self.assertIn("slice_end", entry_source)
        self.assertIn("SFU_JOB_FLAG_DISTRIBUTED_COLUMNS", entry_source)
        self.assertIn("distributed_columns=%d", entry_source)

        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS", wrapper_source)
        self.assertIn("export GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS", wrapper_source)
        self.assertIn('"GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS"', arch_source)

    def test_distributed_direct_path_signals_abort_after_dma_guard_failure(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        direct_fn = re.search(
            r"int\s+run_standalone_unified_job_softmax_distributed_direct_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_standalone_unified_job_softmax_direct_band_for_core",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(direct_fn)
        body = direct_fn.group("body")
        guard_index = body.index("if (!direct_dma_load_guard_passed(input_gm, compact_bytes))")
        self.assertIn("SFU_JOB_FLAG_DISTRIBUTED_ABORT", body[guard_index:])
        abort_index = body.index("SFU_JOB_FLAG_DISTRIBUTED_ABORT", guard_index)
        return_index = body.index("return 1", guard_index)
        self.assertLess(abort_index, return_index)
        self.assertLess(body.index("const uint64_t sub_job_id"), guard_index)
        self.assertLess(body.index("const uint64_t sub_job_tag"), guard_index)
        self.assertIn("sub_job_id", body[guard_index:return_index])
        self.assertIn("sub_job_tag", body[guard_index:return_index])

    def test_distributed_local_gm_capacity_uses_compact_worker_slice(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        standalone_fn = re.search(
            r"int\s+run_standalone_unified_job_softmax_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+read_requested_core_from_argv",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(standalone_fn)
        body = standalone_fn.group("body")
        self.assertIn("local_buffer_cols", body)
        self.assertIn("distributed_worker_slot", body)
        self.assertIn("softmax_primitive_slice_for_worker", body)
        self.assertIn("static_cast<uint64_t>(local_buffer_rows) * local_buffer_cols", body)
        self.assertLess(body.index("uint64_t worker_cores"), body.index("local_buffer_elems"))

    def test_direct_rowmajor_hbm_streams_each_subjob_instead_of_full_band_dma(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        direct_fn = re.search(
            r"int\s+run_standalone_unified_job_softmax_direct_band_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_standalone_unified_job_softmax_for_core",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(direct_fn)
        body = direct_fn.group("body")

        self.assertIn("sub_job_bytes", body)
        self.assertIn("sub_job_input_gm", body)
        self.assertIn("sub_job_output_gm", body)
        self.assertIn("sub_job_input_hbm", body)
        self.assertIn("sub_job_output_hbm", body)
        self.assertLess(body.index("sub_job_bytes"), body.index("dma_remote_load_to_gm"))
        self.assertLess(body.index("dma_remote_load_to_gm"), body.index("golemRunStandaloneSoftmaxSfuJobForCore"))
        self.assertLess(body.index("golemRunStandaloneSoftmaxSfuJobForCore"), body.index("remote_store"))
        self.assertIn("set_len(sub_job_bytes)", body)
        self.assertNotIn("input_gm + sub_job_offset_bytes", body)
        self.assertNotIn("output_gm + sub_job_offset_bytes", body)

    def test_direct_rowmajor_hbm_aborts_when_dma_load_guard_detects_stale_local_gm(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        direct_fn = re.search(
            r"int\s+run_standalone_unified_job_softmax_direct_band_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_standalone_unified_job_softmax_for_core",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(direct_fn)
        body = direct_fn.group("body")

        self.assertIn("kDirectDmaLoadSentinel", entry_source)
        self.assertIn("prepare_direct_dma_load_guard", entry_source)
        self.assertIn("direct_dma_load_guard_passed", entry_source)
        self.assertIn("prepare_direct_dma_load_guard(sub_job_input_gm, sub_job_bytes)", body)
        self.assertIn("if (!direct_dma_load_guard_passed(sub_job_input_gm, sub_job_bytes))", body)
        self.assertIn("DMA_LOAD_FAILED", body)
        self.assertIn("return 1", body)
        self.assertLess(
            body.index("prepare_direct_dma_load_guard(sub_job_input_gm, sub_job_bytes)"),
            body.index("dma_remote_load_to_gm"),
        )
        self.assertLess(
            body.index("if (!direct_dma_load_guard_passed(sub_job_input_gm, sub_job_bytes))"),
            body.index("golemRunStandaloneSoftmaxSfuJobForCore"),
        )
        self.assertLess(
            body.index("DMA_LOAD_FAILED"),
            body.index("golemRunStandaloneSoftmaxSfuJobForCore"),
        )

    def test_unified_job_direct_sweep_uses_direct_rowmajor_standalone_job_path(self):
        sweep_source = read_local("run_sfu_unified_job_direct_sweep.sh")

        self.assertIn("GOLEM_SFU_STANDALONE_SOFTMAX=1", sweep_source)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX=1", sweep_source)
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM=1", sweep_source)
        self.assertIn("GOLEM_SOFTMAX_VERIFY_REFERENCE=logits", sweep_source)
        self.assertIn("GOLEM_A_REUSE_N_TILES=1", sweep_source)
        self.assertIn("GOLEM_B_REUSE_M_TILES=1", sweep_source)
        self.assertIn("--verify-softmax", sweep_source)
        self.assertIn("--group-manager-enable 0", sweep_source)
        self.assertIn("--ctrl-link-enable 0", sweep_source)
        self.assertIn("run_noc_dma_softmax_sfu_pipeline.sh", sweep_source)
        self.assertNotIn("GOLEM_SFU_PRIMITIVE_SOFTMAX=1", sweep_source)

    def test_unified_job_direct_sweep_has_stable_and_pressure_profiles(self):
        sweep_source = read_local("run_sfu_unified_job_direct_sweep.sh")

        self.assertIn("GOLEM_SFU_JOB_DIRECT_PROFILE", sweep_source)
        self.assertIn("stable512", sweep_source)
        self.assertIn("stable1024", sweep_source)
        self.assertIn("stable2048", sweep_source)
        self.assertIn("stable4096", sweep_source)
        self.assertIn("pressure1024", sweep_source)
        self.assertIn("pressure4096_jr4_rt384", sweep_source)
        self.assertIn("pressure4096_jr4_rt512", sweep_source)
        self.assertIn("pressure4096_jr4_rt704", sweep_source)
        self.assertIn("pressure4096_jr4_rt768", sweep_source)
        self.assertIn("pressure4096_jr4_rt1024", sweep_source)
        self.assertIn('run_point 1024 8 8 320 8 pass "$(timeout_for_dim 1024)" "stable1024"', sweep_source)
        self.assertIn("stable1024 band_cores=8 retry_ticks=320 max_retries=8 expect=pass", sweep_source)
        self.assertIn('run_point 2048 8 4 384 8 pass "$(timeout_for_dim 2048)" "stable2048"', sweep_source)
        self.assertIn("stable2048 band_cores=8 job_rows=4 retry_ticks=384 max_retries=8 expect=pass", sweep_source)
        self.assertIn('run_point 4096 8 2 384 8 pass "$(timeout_for_dim 4096)" "stable4096" 268435456', sweep_source)
        self.assertIn("stable4096 band_cores=8 job_rows=2 retry_ticks=384 max_retries=8 mem_node_size=268435456 expect=pass", sweep_source)
        self.assertIn('run_point 4096 8 4 384 8 fail "$(timeout_for_dim 4096)" "pressure4096_jr4_rt384" 268435456', sweep_source)
        self.assertIn("pressure4096_jr4_rt384 band_cores=8 job_rows=4 retry_ticks=384 max_retries=8 mem_node_size=268435456 expect=fail", sweep_source)
        self.assertIn('run_point 4096 8 4 512 8 fail "$(timeout_for_dim 4096)" "pressure4096_jr4_rt512" 268435456', sweep_source)
        self.assertIn("pressure4096_jr4_rt512 band_cores=8 job_rows=4 retry_ticks=512 max_retries=8 mem_node_size=268435456 expect=fail", sweep_source)
        self.assertIn('run_point 4096 8 4 704 8 pass "$(timeout_for_dim 4096)" "pressure4096_jr4_rt704" 268435456', sweep_source)
        self.assertIn("pressure4096_jr4_rt704 band_cores=8 job_rows=4 retry_ticks=704 max_retries=8 mem_node_size=268435456 expect=pass", sweep_source)
        self.assertIn('run_point 4096 8 4 768 8 pass "$(timeout_for_dim 4096)" "pressure4096_jr4_rt768" 268435456', sweep_source)
        self.assertIn("pressure4096_jr4_rt768 band_cores=8 job_rows=4 retry_ticks=768 max_retries=8 mem_node_size=268435456 expect=pass", sweep_source)
        self.assertIn('run_point 4096 8 4 1024 8 pass "$(timeout_for_dim 4096)" "pressure4096_jr4_rt1024" 268435456', sweep_source)
        self.assertIn("pressure4096_jr4_rt1024 band_cores=8 job_rows=4 retry_ticks=1024 max_retries=8 mem_node_size=268435456 expect=pass", sweep_source)
        self.assertIn('local mem_node_size="${9:-134217728}"', sweep_source)
        self.assertIn('--mem-node-size "$mem_node_size"', sweep_source)
        self.assertIn('echo "${GOLEM_TIMEOUT_4096:-2400}"', sweep_source)
        self.assertIn("mem_node_size=$mem_node_size", sweep_source)
        self.assertIn('run_point 1024 8 8 256 8 fail "$(timeout_for_dim 1024)" "pressure1024"', sweep_source)
        self.assertIn("pressure1024 band_cores=8 retry_ticks=256 max_retries=8 expect=fail", sweep_source)
        self.assertIn("expect=fail", sweep_source)

    def test_unified_job_direct_sweep_supports_point_list_manifest_and_dry_run(self):
        sweep_source = read_local("run_sfu_unified_job_direct_sweep.sh")

        self.assertIn("GOLEM_SFU_JOB_DIRECT_POINT_LIST", sweep_source)
        self.assertIn("dim:band_cores:job_rows:retry_ticks:max_retries:expect", sweep_source)
        self.assertIn("sweep_manifest.csv", sweep_source)
        self.assertIn(
            "run_id,dim,band_cores,job_rows,retry_ticks,max_retries,expect,status,exit_code",
            sweep_source,
        )
        self.assertIn("GOLEM_DRY_RUN_SWEEP", sweep_source)
        self.assertIn("EXPECTED_FAIL", sweep_source)
        self.assertIn("UNEXPECTED_PASS", sweep_source)
        self.assertIn("TIMEOUT", sweep_source)

    def test_unified_job_direct_sweep_normalizes_artifact_root_to_absolute_path(self):
        sweep_source = read_local("run_sfu_unified_job_direct_sweep.sh")

        self.assertIn('mkdir -p "$SWEEP_ROOT"', sweep_source)
        self.assertIn('SWEEP_ROOT="$(cd "$SWEEP_ROOT" && pwd)"', sweep_source)
        self.assertLess(
            sweep_source.index('SWEEP_ROOT="$(cd "$SWEEP_ROOT" && pwd)"'),
            sweep_source.index('MANIFEST="$SWEEP_ROOT/sweep_manifest.csv"'),
        )

    def test_distributed_scaling_sweep_uses_fixed_unified_job_architecture_knobs(self):
        sweep_source = read_local("run_sfu_unified_job_distributed_scaling.sh")

        for setting in (
            "GOLEM_SFU_STANDALONE_SOFTMAX=1",
            "GOLEM_SFU_JOB_SOFTMAX=1",
            "GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM=1",
            "GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS=1",
            "GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT:-modeled_noc",
            'GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT="$REDUCTION_TRANSPORT"',
            "GOLEM_SFU_PRIMITIVE_SOFTMAX=0",
        ):
            self.assertIn(setting, sweep_source)
        self.assertIn('ROWS="${GOLEM_SFU_DISTRIBUTED_ROWS:-16}"', sweep_source)
        self.assertIn('STAGING_ROWS="${GOLEM_SFU_DISTRIBUTED_STAGING_ROWS:-4}"', sweep_source)
        self.assertIn('JOB_ROWS="${GOLEM_SFU_DISTRIBUTED_JOB_ROWS:-4}"', sweep_source)
        self.assertIn('CHUNK_ELEMS="${GOLEM_SFU_DISTRIBUTED_CHUNK_ELEMS:-256}"', sweep_source)
        self.assertIn('RETRY_TICKS="${GOLEM_SFU_DISTRIBUTED_RETRY_TICKS:-1024}"', sweep_source)
        self.assertIn('MAX_RETRIES="${GOLEM_SFU_DISTRIBUTED_MAX_RETRIES:-8}"', sweep_source)
        self.assertNotIn("GOLEM_SFU_PRIMITIVE_SOFTMAX=1", sweep_source)

    def test_distributed_scaling_sweep_requires_observable_reduction_transport(self):
        script = os.path.join(SCRIPT_DIR, "run_sfu_unified_job_distributed_scaling.sh")
        sweep_source = read_local("run_sfu_unified_job_distributed_scaling.sh")

        self.assertIn(
            "distributed scaling requires modeled_noc or explicit_noc reduction transport",
            sweep_source,
        )
        self.assertIn('case "$REDUCTION_TRANSPORT" in', sweep_source)

        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env.update(
                {
                    "GOLEM_SWEEP_ROOT": temp_dir,
                    "GOLEM_DRY_RUN_SWEEP": "1",
                    "GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT": "shared",
                }
            )
            result = subprocess.run(
                ["bash", script],
                cwd=SCRIPT_DIR,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires modeled_noc or explicit_noc reduction transport", result.stderr)

    def test_distributed_scaling_sweep_accepts_explicit_noc_and_signs_points(self):
        script = os.path.join(SCRIPT_DIR, "run_sfu_unified_job_distributed_scaling.sh")

        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env.update(
                {
                    "GOLEM_SWEEP_ROOT": temp_dir,
                    "GOLEM_DRY_RUN_SWEEP": "1",
                    "GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT": "explicit_noc",
                    "GOLEM_SFU_DISTRIBUTED_POINT_LIST": "16:512:4:4",
                }
            )
            result = subprocess.run(
                ["bash", "-c", f'source "{script}"; point_signature 16 512 4 4'],
                cwd=SCRIPT_DIR,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("reduction_transport=explicit_noc", result.stdout.splitlines()[-1])

    def test_distributed_scaling_sweep_validates_sfu_and_dma_lifecycle(self):
        sweep_source = read_local("run_sfu_unified_job_distributed_scaling.sh")

        self.assertIn("validate_point_artifacts", sweep_source)
        self.assertIn("sfu_ops_issued", sweep_source)
        self.assertIn("sfu_job_softmax_max_chunks", sweep_source)
        self.assertIn("sfu_job_softmax_sum_chunks", sweep_source)
        self.assertIn("sfu_job_softmax_norm_chunks", sweep_source)
        self.assertIn("sfu_reduction_max_requests", sweep_source)
        self.assertIn("sfu_reduction_max_responses", sweep_source)
        self.assertIn("sfu_reduction_sum_requests", sweep_source)
        self.assertIn("sfu_reduction_sum_responses", sweep_source)
        self.assertIn("timeout_retry", sweep_source)
        self.assertIn("timeout_exhausted", sweep_source)
        self.assertIn("read_issue_count", sweep_source)
        self.assertIn("write_issue_count", sweep_source)
        self.assertIn("completion", sweep_source)
        self.assertIn("write_completion", sweep_source)
        self.assertIn("rows,dim,chunk_elems,worker_cores,band_cores,cooperative_groups", sweep_source)
        self.assertIn("GOLEM_SFU_DISTRIBUTED_POINT_LIST", sweep_source)
        self.assertIn("rows:dim:worker_cores:band_cores", sweep_source)
        self.assertIn(".pass", sweep_source)

    def test_distributed_scaling_explicit_noc_validates_transport_event_totals(self):
        sweep_source = read_local("run_sfu_unified_job_distributed_scaling.sh")

        self.assertIn('if [[ "$REDUCTION_TRANSPORT" == "explicit_noc" ]]', sweep_source)
        self.assertIn("local expected_transport_events=$(( expected_worker_rows * 4 ))", sweep_source)
        self.assertIn("sfu_reduction_transport_received", sweep_source)
        self.assertIn('require_equal "SFU reduction transport receives"', sweep_source)
        self.assertNotIn(
            'transport_received="$(sfu_stat_sum "$stats_file" gmem_reduction_received)"',
            sweep_source,
        )
        self.assertIn("gmem_reduction_send_immediate", sweep_source)
        self.assertIn("gmem_reduction_send_queued", sweep_source)
        self.assertIn("gmem_reduction_send_rejected", sweep_source)
        self.assertIn("gmem_reduction_received", sweep_source)
        self.assertIn('if rg -q ",gmem_reduction_send_immediate," "$stats_file"', sweep_source)

    def test_gemm_architecture_keeps_sfu_disabled_except_explicit_softmax_runs(self):
        architecture_source = read_repo_relative("../../architecture/cpu_builder.py")
        sweep_source = read_local("run_sfu_unified_job_distributed_scaling.sh")

        self.assertIn(
            'sfu_enable = int(os.getenv("GOLEM_SFU_ENABLE", "0")) != 0',
            architecture_source,
        )
        self.assertIn("if sfu_enable:", architecture_source)
        self.assertIn("GOLEM_SFU_ENABLE=1", sweep_source)

    def test_distributed_scaling_sweep_scopes_generated_artifacts_to_its_root(self):
        sweep_source = read_local("run_sfu_unified_job_distributed_scaling.sh")

        self.assertIn('GOLEM_HBM_DIR="$SWEEP_ROOT/hbm"', sweep_source)
        self.assertIn('GOLEM_RUN_SUMMARY_CSV="$SWEEP_ROOT/stats/run_summary.csv"', sweep_source)
        self.assertIn('GOLEM_SOFTMAX_C_FILE="$SWEEP_ROOT/outputs/${run_id}.bin"', sweep_source)
        self.assertIn('GOLEM_TENSOR_DIR="$SWEEP_ROOT/inputs"', sweep_source)
        self.assertIn("GOLEM_SKIP_BUILD=0", sweep_source)
        self.assertIn("GOLEM_SFU_PERF_PROFILE=0", sweep_source)
        self.assertIn("GOLEM_HBM_DUMP_OUTPUT=1", sweep_source)

    def test_distributed_scaling_sweep_dry_run_expands_representative_matrix(self):
        script = os.path.join(SCRIPT_DIR, "run_sfu_unified_job_distributed_scaling.sh")
        expected = {
            ("16", "512", "4", "4", "1"),
            ("16", "512", "4", "16", "4"),
            ("16", "512", "8", "16", "2"),
            ("16", "512", "16", "16", "1"),
            ("16", "1024", "4", "4", "1"),
            ("16", "1024", "4", "16", "4"),
            ("16", "1024", "8", "16", "2"),
            ("16", "1024", "16", "16", "1"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env.update(
                {
                    "GOLEM_SWEEP_ROOT": temp_dir,
                    "GOLEM_DRY_RUN_SWEEP": "1",
                }
            )
            result = subprocess.run(
                ["bash", script],
                cwd=SCRIPT_DIR,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with open(os.path.join(temp_dir, "sweep_manifest.csv"), newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        actual = {
            (
                row["rows"],
                row["dim"],
                row["worker_cores"],
                row["band_cores"],
                row["cooperative_groups"],
            )
            for row in rows
        }
        self.assertEqual(actual, expected)
        self.assertTrue(all(row["status"] == "DRYRUN" for row in rows))

    def test_distributed_scaling_validator_rejects_an_early_counter_mismatch(self):
        script = os.path.join(SCRIPT_DIR, "run_sfu_unified_job_distributed_scaling.sh")
        run_id = "sfu_job_dist_r16_d512_w4_bc4_g1"
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout_dir = os.path.join(temp_dir, "stdout", "overlap0", run_id)
            stats_dir = os.path.join(temp_dir, "stats", "overlap0", run_id)
            output_dir = os.path.join(temp_dir, "outputs")
            os.makedirs(stdout_dir)
            os.makedirs(stats_dir)
            os.makedirs(output_dir)
            with open(os.path.join(output_dir, f"{run_id}.bin"), "wb") as f:
                f.truncate(16 * 512 * 4)

            pass_line = (
                "[SOFTMAX] mode=sfu-standalone-job-softmax rows=16 dim=512 "
                "worker_cores=4 staging_rows=4 job_rows=4 band_cores=4 "
                "distributed_columns=1 PASS\n"
            )
            for core in range(3):
                with open(os.path.join(stdout_dir, f"stdout-{core}"), "w", encoding="utf-8") as f:
                    f.write(pass_line)

            stat_values = {
                "sfu_ops_issued": 4,
                "sfu_job_softmax_max_chunks": 16,
                "sfu_job_softmax_sum_chunks": 16,
                "sfu_job_softmax_norm_chunks": 16,
                "sfu_partial_submits": 32,
                "sfu_partial_done": 16,
                "sfu_reduction_max_requests": 16,
                "sfu_reduction_max_responses": 16,
                "sfu_reduction_sum_requests": 16,
                "sfu_reduction_sum_responses": 16,
                "sfu_retry_events": 0,
            }
            with open(os.path.join(stats_dir, "stats_selfcom.txt"), "w", encoding="utf-8") as f:
                for core in range(16):
                    for stat, value in stat_values.items():
                        actual = value if core < 4 else 0
                        f.write(f"core{core}:rocc:sfu,{stat},,Accumulator,0,0,{actual},0,0,0,0\n")

            dma_values = {
                "read_issue_count": 64,
                "write_issue_count": 64,
                "completion": 64,
                "write_completion": 64,
                "read_bytes_total": 32768,
                "write_bytes_total": 32768,
                "timeout_retry": 0,
                "timeout_exhausted": 0,
                "write_timeout_retry": 0,
            }
            with open(os.path.join(stats_dir, "dma_summary.csv"), "w", encoding="utf-8") as f:
                f.write("metric,mean,median,p95,min,max,sum\n")
                for metric, value in dma_values.items():
                    f.write(f"{metric},0,0,0,0,0,{value}\n")

            env = os.environ.copy()
            env.update({"GOLEM_SWEEP_ROOT": temp_dir, "GOLEM_DRY_RUN_SWEEP": "1"})
            command = (
                f'source "{script}"; '
                f'if validate_point_artifacts "{run_id}" 16 512 4 4; then exit 0; else exit 7; fi'
            )
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=SCRIPT_DIR,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("physical PASS cores expected=4 actual=3", result.stderr)

    def test_distributed_scaling_sweep_does_not_trust_a_stale_marker(self):
        script = os.path.join(SCRIPT_DIR, "run_sfu_unified_job_distributed_scaling.sh")
        run_id = "sfu_job_dist_r16_d512_w4_bc4_g1"
        with tempfile.TemporaryDirectory() as temp_dir:
            completed_dir = os.path.join(temp_dir, "completed")
            os.makedirs(completed_dir)
            with open(os.path.join(completed_dir, f"{run_id}.pass"), "w", encoding="utf-8") as f:
                f.write("signature=stale\n")
            env = os.environ.copy()
            env.update(
                {
                    "GOLEM_SWEEP_ROOT": temp_dir,
                    "GOLEM_DRY_RUN_SWEEP": "1",
                    "GOLEM_SFU_DISTRIBUTED_POINT_LIST": "16:512:4:4",
                }
            )
            result = subprocess.run(
                ["bash", script],
                cwd=SCRIPT_DIR,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            with open(os.path.join(temp_dir, "sweep_manifest.csv"), newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows[-1]["status"], "DRYRUN")
        self.assertNotEqual(rows[-1]["artifact_validation"], "CACHED")

    def test_distributed_scaling_signature_includes_pipeline_arguments(self):
        script = os.path.join(SCRIPT_DIR, "run_sfu_unified_job_distributed_scaling.sh")
        signatures = []
        for pipeline_args in ("--noc-link-bw 100GB/s", "--noc-link-bw 200GB/s"):
            with tempfile.TemporaryDirectory() as temp_dir:
                env = os.environ.copy()
                env.update(
                    {
                        "GOLEM_SWEEP_ROOT": temp_dir,
                        "GOLEM_DRY_RUN_SWEEP": "1",
                        "GOLEM_SFU_DISTRIBUTED_POINT_LIST": "16:512:4:4",
                        "GOLEM_SFU_DISTRIBUTED_PIPELINE_ARGS": pipeline_args,
                    }
                )
                command = f'source "{script}"; point_signature 16 512 4 4'
                result = subprocess.run(
                    ["bash", "-c", command],
                    cwd=SCRIPT_DIR,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                signatures.append(result.stdout.splitlines()[-1])

        self.assertNotEqual(signatures[0], signatures[1])

    def test_distributed_scaling_runner_pins_its_verified_output_after_overrides(self):
        sweep_source = read_local("run_sfu_unified_job_distributed_scaling.sh")

        pipeline_args_index = sweep_source.index('"${pipeline_args[@]}"')
        output_arg = '--softmax-c-file "$SWEEP_ROOT/outputs/${run_id}.bin"'
        self.assertIn(output_arg, sweep_source)
        self.assertLess(pipeline_args_index, sweep_source.index(output_arg))

    def test_distributed_scaling_cache_rejects_a_corrupted_output_tensor(self):
        script = os.path.join(SCRIPT_DIR, "run_sfu_unified_job_distributed_scaling.sh")
        run_id = "sfu_job_dist_r16_d512_w4_bc4_g1"
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout_dir = os.path.join(temp_dir, "stdout", "overlap0", run_id)
            stats_dir = os.path.join(temp_dir, "stats", "overlap0", run_id)
            completed_dir = os.path.join(temp_dir, "completed")
            output_dir = os.path.join(temp_dir, "outputs")
            os.makedirs(stdout_dir)
            os.makedirs(stats_dir)
            os.makedirs(completed_dir)
            os.makedirs(output_dir)
            with open(os.path.join(output_dir, f"{run_id}.bin"), "wb") as f:
                f.truncate(16 * 512 * 4)

            pass_line = (
                "[SOFTMAX] mode=sfu-standalone-job-softmax rows=16 dim=512 "
                "worker_cores=4 staging_rows=4 job_rows=4 band_cores=4 "
                "distributed_columns=1 PASS\n"
            )
            for core in range(4):
                with open(os.path.join(stdout_dir, f"stdout-{core}"), "w", encoding="utf-8") as f:
                    f.write(pass_line)

            per_core_stats = {
                "sfu_ops_issued": 4,
                "sfu_job_softmax_max_chunks": 16,
                "sfu_job_softmax_sum_chunks": 16,
                "sfu_job_softmax_norm_chunks": 16,
                "sfu_partial_submits": 32,
                "sfu_partial_done": 16,
                "sfu_reduction_max_requests": 16,
                "sfu_reduction_max_responses": 16,
                "sfu_reduction_sum_requests": 16,
                "sfu_reduction_sum_responses": 16,
                "sfu_retry_events": 0,
            }
            with open(os.path.join(stats_dir, "stats_selfcom.txt"), "w", encoding="utf-8") as f:
                for core in range(16):
                    for stat, value in per_core_stats.items():
                        actual = value if core < 4 else 0
                        f.write(f"core{core}:rocc:sfu,{stat},,Accumulator,0,0,{actual},0,0,0,0\n")

            dma_values = {
                "read_issue_count": 64,
                "write_issue_count": 64,
                "completion": 64,
                "write_completion": 64,
                "read_bytes_total": 32768,
                "write_bytes_total": 32768,
                "timeout_retry": 0,
                "timeout_exhausted": 0,
                "write_timeout_retry": 0,
            }
            with open(os.path.join(stats_dir, "dma_summary.csv"), "w", encoding="utf-8") as f:
                f.write("metric,mean,median,p95,min,max,sum\n")
                for metric, value in dma_values.items():
                    f.write(f"{metric},0,0,0,0,0,{value}\n")

            env = os.environ.copy()
            env.update(
                {
                    "GOLEM_SWEEP_ROOT": temp_dir,
                    "GOLEM_DRY_RUN_SWEEP": "1",
                    "GOLEM_SFU_DISTRIBUTED_POINT_LIST": "16:512:4:4",
                }
            )
            signature_result = subprocess.run(
                ["bash", "-c", f'source "{script}"; point_signature 16 512 4 4'],
                cwd=SCRIPT_DIR,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                signature_result.returncode,
                0,
                signature_result.stdout + signature_result.stderr,
            )
            signature = signature_result.stdout.splitlines()[-1]
            with open(os.path.join(completed_dir, f"{run_id}.pass"), "w", encoding="utf-8") as f:
                f.write(f"signature={signature}\noutput_sha256=corrupted\n")

            result = subprocess.run(
                ["bash", script],
                cwd=SCRIPT_DIR,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            with open(os.path.join(temp_dir, "sweep_manifest.csv"), newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows[-1]["status"], "DRYRUN")
        self.assertNotEqual(rows[-1]["artifact_validation"], "CACHED")

    def test_distributed_scaling_default_root_is_collision_resistant(self):
        sweep_source = read_local("run_sfu_unified_job_distributed_scaling.sh")

        self.assertIn("%N", sweep_source)
        self.assertIn("$$", sweep_source)

    def test_workload_has_local_gm_primitive_smoke_before_softmax_modes(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SMOKE", 0)', entry_source)
        self.assertIn("mode=sfu-primitive-smoke", entry_source)
        self.assertIn("run_sfu_primitive_smoke_for_core", entry_source)
        self.assertIn("sfu_primitive(", entry_source)
        self.assertIn("sfu_primitive_wait", entry_source)
        self.assertIn("SFUPrimitiveOp::EXP", entry_source)
        self.assertIn("SFUPrimitiveOp::LOG", entry_source)
        self.assertIn("SFUPrimitiveOp::RECIPROCAL", entry_source)
        self.assertIn("SFUPrimitiveOp::RSQRT", entry_source)
        self.assertIn("SFUPrimitiveOp::TANH", entry_source)
        self.assertIn("SFUPrimitiveOp::SIGMOID", entry_source)
        self.assertIn("EXP,LOG,RECIPROCAL,RSQRT,TANH,SIGMOID", entry_source)
        self.assertLess(
            entry_source.index('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SMOKE", 0)'),
            entry_source.index('read_i64_env_or_default("GOLEM_SFU_STANDALONE_SOFTMAX", 0)'),
        )
        self.assertIn("GOLEM_SFU_PRIMITIVE_SMOKE", wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_SMOKE", wrapper_source)

    def test_primitive_smoke_can_scale_total_elements_with_chunking(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS", 4)', entry_source)
        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS", 0)', entry_source)
        self.assertIn("primitive_smoke_chunk_elems", entry_source)
        self.assertIn("kPrimitiveDefaultChunkElems", entry_source)
        self.assertIn("set_len(chunk_bytes)", entry_source)
        self.assertIn("mm2gm", entry_source)
        self.assertIn("gm2mm", entry_source)
        self.assertIn("total_elems", entry_source)
        self.assertIn("chunk_elems", entry_source)
        self.assertIn("chunks", entry_source)
        self.assertIn("processed_elems", entry_source)
        self.assertIn("kPrimitiveFlagRepeatChunk", entry_source)
        self.assertIn(".input1_gm_addr = processed_elem_count", entry_source)
        self.assertIn(".flags = primitive_flags_for_processed_elems", entry_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS", wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS", wrapper_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS", wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS", wrapper_source)

    def test_workload_has_hbm_streaming_primitive_benchmark_before_local_smoke(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_STREAM", 0)', entry_source)
        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_ELEMS", 64)', entry_source)
        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS", 0)', entry_source)
        self.assertIn("run_sfu_primitive_hbm_stream_for_core", entry_source)
        self.assertIn("mode=sfu-primitive-hbm-stream", entry_source)
        self.assertIn("dma_remote_load_to_gm(executor_core_id,", entry_source)
        self.assertIn("remote_store(", entry_source)
        self.assertIn("hbm_read_bytes", entry_source)
        self.assertIn("hbm_write_bytes", entry_source)
        self.assertIn("SFUPrimitiveOp::EXP", entry_source)
        self.assertLess(
            entry_source.index('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_STREAM", 0)'),
            entry_source.index('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SMOKE", 0)'),
        )
        for knob in (
            "GOLEM_SFU_PRIMITIVE_HBM_STREAM",
            "GOLEM_SFU_PRIMITIVE_HBM_ELEMS",
            "GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS",
        ):
            self.assertIn(knob, wrapper_source)
            self.assertIn(f"export {knob}", wrapper_source)

    def test_hbm_streaming_primitive_supports_configurable_multi_op_list(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")

        self.assertIn('read_string_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_OPS", "EXP")', entry_source)
        self.assertIn("parse_sfu_primitive_hbm_ops", entry_source)
        self.assertIn("primitive_op_name", entry_source)
        self.assertIn("input_hbm_base", entry_source)
        self.assertIn("hbm_init_write_bytes", entry_source)
        self.assertIn("ops=", entry_source)
        for op_name in ("EXP", "LOG", "RECIPROCAL", "RSQRT", "TANH", "SIGMOID"):
            self.assertIn(f'"{op_name}"', entry_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_HBM_OPS", wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_HBM_OPS", wrapper_source)

    def test_hbm_streaming_primitive_supports_batch_mode(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")
        ex_instr_source = read_local("ex_instr.h")
        sweep_source = read_local("run_sfu_hbm_chunk_batch_sweep.sh")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_BATCH", 1)', entry_source)
        self.assertIn("run_hbm_stream_sfu_primitive_batch_case", entry_source)
        self.assertIn("sfu_primitive_batch(", entry_source)
        self.assertIn("sfu_primitive_batch_wait", entry_source)
        self.assertIn("SFUPrimitiveBatchDesc", entry_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_HBM_BATCH", wrapper_source)
        self.assertIn('GOLEM_SFU_PRIMITIVE_HBM_BATCH="${GOLEM_SFU_PRIMITIVE_HBM_BATCH:-1}"', wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_HBM_BATCH", wrapper_source)
        self.assertIn("default architecture path", wrapper_source)
        self.assertIn("legacy/debug fallback", wrapper_source)
        self.assertIn("GOLEM_SFU_RUN_LEGACY_NONBATCH", sweep_source)
        self.assertIn('GOLEM_SFU_PRIMITIVE_HBM_BATCH="1"', sweep_source)
        self.assertNotIn("for batch in 0 1", sweep_source)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH", ex_instr_source)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH_WAIT", ex_instr_source)

    def test_workload_has_softmax_primitive_pipeline_mode(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")
        arch_source = read_repo_relative("../../architecture/ncores_selfcom_dma_ctrl.py")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX", 0)', entry_source)
        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_ROWS", 1)', entry_source)
        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_DIM", 256)', entry_source)
        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_CHUNK_ELEMS", 0)', entry_source)
        self.assertIn("run_sfu_primitive_softmax_for_core", entry_source)
        self.assertIn("mode=sfu-primitive-softmax", entry_source)
        self.assertIn("SFUPrimitiveOp::REDUCE_MAX", entry_source)
        self.assertIn("SFUPrimitiveOp::REDUCE_SUM", entry_source)
        self.assertIn("primitive_stages=4", entry_source)
        self.assertIn("local_steps=3", entry_source)
        self.assertIn("max_abs_diff", entry_source)
        self.assertLess(
            entry_source.index('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX", 0)'),
            entry_source.index('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_STREAM", 0)'),
        )
        for knob in (
            "GOLEM_SFU_PRIMITIVE_SOFTMAX",
            "GOLEM_SFU_PRIMITIVE_SOFTMAX_ROWS",
            "GOLEM_SFU_PRIMITIVE_SOFTMAX_DIM",
            "GOLEM_SFU_PRIMITIVE_SOFTMAX_CHUNK_ELEMS",
            "GOLEM_SFU_PRIMITIVE_SOFTMAX_VERIFY",
        ):
            self.assertIn(knob, wrapper_source)
            self.assertIn(f"export {knob}", wrapper_source)
            self.assertIn(f'"{knob}"', arch_source)

    def test_softmax_primitive_uses_multicore_worker_slice_policy_for_large_dims(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")
        arch_source = read_repo_relative("../../architecture/ncores_selfcom_dma_ctrl.py")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_WORKER_CORES", 0)', entry_source)
        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_MULTICORE_MIN_DIM", 512)', entry_source)
        self.assertIn("resolve_softmax_primitive_worker_count", entry_source)
        self.assertIn("softmax_primitive_slice_for_worker", entry_source)
        self.assertIn("softmax_primitive_coord_worker_addr", entry_source)
        self.assertIn("softmax_primitive_publish_to_addr", entry_source)
        self.assertIn("gemm_worker_slot_for_core(executor_core_id)", entry_source)
        self.assertIn("const int actual_core_id = sched_getcpu()", entry_source)
        self.assertIn("worker_cores=%llu", entry_source)
        self.assertIn("dim_per_core=%llu", entry_source)
        self.assertIn("cross_core_reduce_stages=2", entry_source)
        self.assertIn(
            "softmax_primitive_wait_local_u64(\n            executor_core_id, kSoftmaxPrimitiveMboxGlobalReady",
            entry_source,
        )
        primitive_body = re.search(
            r"int\s+run_sfu_primitive_softmax_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nbool\s+run_hbm_stream_sfu_primitive_case",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(primitive_body)
        self.assertNotIn("if (requested_core_id != 0) {\n        return 0;\n    }", primitive_body.group("body"))
        self.assertNotIn("softmax_primitive_read_worker_u64", entry_source)

        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_WORKER_CORES", wrapper_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_MULTICORE_MIN_DIM", wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_SOFTMAX_WORKER_CORES", wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_SOFTMAX_MULTICORE_MIN_DIM", wrapper_source)
        self.assertIn('"GOLEM_SFU_PRIMITIVE_SOFTMAX_WORKER_CORES"', arch_source)
        self.assertIn('"GOLEM_SFU_PRIMITIVE_SOFTMAX_MULTICORE_MIN_DIM"', arch_source)

    def test_softmax_primitive_scalar_helpers_use_register_gm_path(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        read_body = re.search(
            r"float\s+read_fp32_from_gm\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            entry_source,
            re.S,
        )
        write_body = re.search(
            r"void\s+write_fp32_to_gm\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            entry_source,
            re.S,
        )

        self.assertIsNotNone(read_body)
        self.assertIsNotNone(write_body)
        self.assertIn("gm2reg", read_body.group("body"))
        self.assertIn("reg2gm", write_body.group("body"))
        self.assertNotIn("gm2mm", read_body.group("body"))
        self.assertNotIn("mm2gm", write_body.group("body"))

    def test_softmax_primitive_uses_coordinator_only_reciprocal_broadcast(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        primitive_body = re.search(
            r"int\s+run_sfu_primitive_softmax_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nbool\s+run_hbm_stream_sfu_primitive_case",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(primitive_body)
        body = primitive_body.group("body")

        self.assertIn("planned_groups_per_row * 3 + 1", body)
        self.assertNotIn("planned_groups_per_row * 3 + static_cast<uint64_t>(worker_cores)", body)
        self.assertIn("const float inv_sum = coordinator_reciprocal_and_broadcast", body)
        helper_body = re.search(
            r"float\s+coordinator_reciprocal_and_broadcast\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+resolve_softmax_primitive_worker_count",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(helper_body)
        self.assertIn("SFUPrimitiveOp::RECIPROCAL", helper_body.group("body"))
        self.assertIn("kSoftmaxPrimitiveMboxGlobalValue", helper_body.group("body"))

        reciprocal_index = body.index("coordinator_reciprocal_and_broadcast")
        coord_sum_index = body.index("double global_row_sum = 0.0;")
        wait_inv_sum_index = body.index(
            "softmax_primitive_wait_local_u64(\n"
            "            executor_core_id, kSoftmaxPrimitiveMboxGlobalReady, global_sum_seq)"
        )
        self.assertGreater(reciprocal_index, coord_sum_index)
        self.assertLess(reciprocal_index, wait_inv_sum_index)

    def test_softmax_primitive_supports_row_block_reduction(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")
        arch_source = read_repo_relative("../../architecture/ncores_selfcom_dma_ctrl.py")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_ROW_BLOCK"', entry_source)
        self.assertIn("resolve_softmax_primitive_row_block", entry_source)
        self.assertIn("softmax_primitive_block_worker_addr", entry_source)
        self.assertIn("softmax_primitive_block_global_addr", entry_source)
        self.assertIn("run_sfu_primitive_softmax_row_block_for_core", entry_source)
        self.assertIn("row_blocks", entry_source)
        self.assertIn("row_block=%llu", entry_source)
        self.assertIn("row_blocks=%llu", entry_source)
        self.assertIn("block_syncs=%llu", entry_source)
        self.assertIn("const uint64_t block_index = row_base / row_block", entry_source)
        self.assertIn("const uint64_t local_max_seq = block_index * 4 + 1", entry_source)
        self.assertIn("if (row_block <= 1)", entry_source)

        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_ROW_BLOCK", wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_SOFTMAX_ROW_BLOCK", wrapper_source)
        self.assertIn('"GOLEM_SFU_PRIMITIVE_SOFTMAX_ROW_BLOCK"', arch_source)

    def test_softmax_primitive_row_block_path_is_implemented(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        row_block_body = re.search(
            r"int\s+run_sfu_primitive_softmax_row_block_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_sfu_primitive_softmax_for_core",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(row_block_body)
        body = row_block_body.group("body")

        self.assertNotIn("row_block path is not implemented", body)
        self.assertIn("for (uint64_t row_base = 0; row_base < rows; row_base += row_block)", body)
        self.assertIn("const uint64_t block_rows = std::min(row_block, rows - row_base)", body)
        self.assertIn("kSoftmaxPrimitiveBlockLocalMaxBase", body)
        self.assertIn("kSoftmaxPrimitiveBlockLocalSumBase", body)
        self.assertIn("kSoftmaxPrimitiveBlockGlobalMaxReady", body)
        self.assertIn("kSoftmaxPrimitiveBlockGlobalSumReady", body)
        self.assertIn("kSoftmaxPrimitiveBlockInvSumBase", body)
        self.assertIn("SFUPrimitiveOp::REDUCE_MAX", body)
        self.assertIn("SFUPrimitiveOp::EXP", body)
        self.assertIn("SFUPrimitiveOp::REDUCE_SUM", body)
        self.assertIn("SFUPrimitiveOp::RECIPROCAL", body)
        self.assertIn("block_row_exp", body)
        self.assertIn("row_block=%llu", body)
        self.assertIn("block_syncs=%llu", body)

    def test_softmax_primitive_supports_row_pipeline_depth(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")
        arch_source = read_repo_relative("../../architecture/ncores_selfcom_dma_ctrl.py")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_PIPELINE_DEPTH"', entry_source)
        self.assertIn("resolve_softmax_primitive_pipeline_depth", entry_source)
        self.assertIn("SoftmaxRowPipelineState", entry_source)
        self.assertIn("SoftmaxRowPipelineStage", entry_source)
        self.assertIn("run_sfu_primitive_softmax_row_pipeline_for_core", entry_source)
        self.assertIn("pipeline_depth=%llu", entry_source)
        self.assertIn("pipeline_mode=row", entry_source)
        self.assertIn("if (row_block > 1)", entry_source)
        self.assertIn("if (pipeline_depth > 1)", entry_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_PIPELINE_DEPTH", wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_SOFTMAX_PIPELINE_DEPTH", wrapper_source)
        self.assertIn('"GOLEM_SFU_PRIMITIVE_SOFTMAX_PIPELINE_DEPTH"', arch_source)

    def test_softmax_primitive_pipeline_depth_uses_windowed_row_dispatch(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        pipeline_body = re.search(
            r"int\s+run_sfu_primitive_softmax_row_pipeline_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_sfu_primitive_softmax_for_core",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(pipeline_body)
        body = pipeline_body.group("body")

        self.assertIn("pipeline_window_rows", body)
        self.assertIn("dispatch=stage-row-state-machine", body)
        self.assertIn("stage_pipeline_states", body)
        self.assertIn("row_blocks = (rows + pipeline_window_rows - 1) / pipeline_window_rows", body)
        self.assertNotIn("dispatch=conservative-row", body)

    def test_softmax_primitive_windowed_row_uses_packed_two_row_sync(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        row_block_body = re.search(
            r"int\s+run_sfu_primitive_softmax_row_block_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_sfu_primitive_softmax_row_pipeline_for_core",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(row_block_body)
        body = row_block_body.group("body")

        self.assertIn("pack_two_fp32_to_reg", entry_source)
        self.assertIn("low_fp32_from_packed_reg", entry_source)
        self.assertIn("high_fp32_from_packed_reg", entry_source)
        self.assertIn("packed_two_row_sync", body)
        self.assertIn("if (packed_two_row_sync)", body)
        self.assertIn("!packed_two_row_sync", body)

    def test_softmax_primitive_windowed_row_batches_exp_sum_across_rows(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        row_block_body = re.search(
            r"int\s+run_sfu_primitive_softmax_row_block_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_sfu_primitive_softmax_row_pipeline_for_core",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(row_block_body)
        body = row_block_body.group("body")

        self.assertIn("cross_row_batch_rows", body)
        self.assertIn("cross_row_batch_items <= max_batch_items", body)
        self.assertIn("combined_slot", body)
        self.assertIn("cross_row_exp_descs", body)
        self.assertIn("cross_row_sum_descs", body)
        self.assertIn("issue_sfu_primitive_batch_descs(executor_core_id, cross_row_exp_descs", body)
        self.assertIn("issue_sfu_primitive_batch_descs(executor_core_id, cross_row_sum_descs", body)

    def test_softmax_primitive_coordinator_nbpoll_is_optional_negative_experiment_path(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        wrapper_source = read_local("run_noc_dma_softmax_sfu_pipeline.sh")
        arch_source = read_repo_relative("../../architecture/ncores_selfcom_dma_ctrl.py")
        row_block_body = re.search(
            r"int\s+run_sfu_primitive_softmax_row_block_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_sfu_primitive_softmax_row_pipeline_for_core",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(row_block_body)
        body = row_block_body.group("body")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_NBPOLL", 0)', entry_source)
        self.assertIn("coordinator_nbpoll", body)
        self.assertIn("softmax_primitive_poll_ready", entry_source)
        self.assertIn("observed_max_workers", body)
        self.assertIn("remaining_max_workers", body)
        self.assertIn("softmax_primitive_poll_ready(max_ready_addr, local_max_seq)", body)
        self.assertIn("observed_sum_workers", body)
        self.assertIn("remaining_sum_workers", body)
        self.assertIn("softmax_primitive_poll_ready(sum_ready_addr, local_sum_seq)", body)
        self.assertIn("dispatch=stage-row-state-machine", entry_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_NBPOLL", wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_SOFTMAX_NBPOLL", wrapper_source)
        self.assertIn('"GOLEM_SFU_PRIMITIVE_SOFTMAX_NBPOLL"', arch_source)

    def test_softmax_primitive_pipeline_uses_stage_level_state_machine(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        pipeline_body = re.search(
            r"int\s+run_sfu_primitive_softmax_row_pipeline_for_core\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nint\s+run_sfu_primitive_softmax_for_core",
            entry_source,
            re.S,
        )
        self.assertIsNotNone(pipeline_body)
        body = pipeline_body.group("body")

        self.assertIn("kSoftmaxPrimitiveStageWorkerStride", entry_source)
        self.assertIn("softmax_primitive_stage_worker_addr", entry_source)
        self.assertIn("softmax_primitive_stage_global_addr", entry_source)
        self.assertIn("advance_softmax_stage_local_max", entry_source)
        self.assertIn("advance_softmax_stage_exp_sum", entry_source)
        self.assertIn("advance_softmax_stage_normalize", entry_source)
        self.assertIn("SoftmaxRowPipelineStage::LOCAL_MAX_PUBLISHED", entry_source)
        self.assertIn("SoftmaxRowPipelineStage::GLOBAL_MAX_READY", entry_source)
        self.assertIn("SoftmaxRowPipelineStage::LOCAL_SUM_PUBLISHED", entry_source)
        self.assertIn("dispatch=stage-row-state-machine", body)
        self.assertIn("stage_pipeline_states", body)
        self.assertIn("pipeline_stage_cycles", body)
        self.assertIn("for (uint64_t row_base = 0; row_base < rows; row_base += pipeline_window_rows)", body)
        self.assertNotIn("row_block=pipeline_window_rows", body)
        self.assertNotIn("return run_sfu_primitive_softmax_row_block_for_core", body)

    def test_softmax_primitive_sweep_script_uses_batch_default_path(self):
        sweep_source = read_local("run_sfu_softmax_primitive_sweep.sh")

        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX=1", sweep_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_DIM", sweep_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_CHUNK_ELEMS", sweep_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_ROWS", sweep_source)
        self.assertIn("mode=sfu-primitive-softmax", sweep_source)
        self.assertIn("sizes=(128 256 512 1024 2048 4096)", sweep_source)
        self.assertNotIn("GOLEM_SFU_PRIMITIVE_HBM_BATCH=0", sweep_source)
        self.assertNotIn("nonbatch", sweep_source.lower())

    def test_softmax_primitive_sweep_reuses_hbm_after_first_point(self):
        sweep_source = read_local("run_sfu_softmax_primitive_sweep.sh")

        self.assertIn("GOLEM_SFU_SWEEP_REUSE_HBM", sweep_source)
        self.assertIn("hbm_config.env", sweep_source)
        self.assertIn("GOLEM_SKIP_TENSOR_GEN", sweep_source)
        self.assertIn("GOLEM_SKIP_HBM_GEN", sweep_source)
        self.assertIn('local skip_build="${GOLEM_SKIP_BUILD:-$reuse_hbm}"', sweep_source)

    def test_softmax_primitive_sweep_has_real_sst_perf_profile(self):
        sweep_source = read_local("run_sfu_softmax_primitive_sweep.sh")

        self.assertIn("GOLEM_SFU_PERF_PROFILE", sweep_source)
        self.assertIn("perf_verify", sweep_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_VERIFY", sweep_source)
        self.assertIn("GOLEM_BENCH_DISABLE_SST_STATS", sweep_source)
        self.assertIn("run_noc_dma_softmax_sfu_pipeline.sh", sweep_source)
        self.assertNotIn("timing-only", sweep_source.lower())

    def test_softmax_primitive_sweep_supports_focused_real_sst_point_lists(self):
        sweep_source = read_local("run_sfu_softmax_primitive_sweep.sh")

        self.assertIn("GOLEM_SFU_SOFTMAX_POINT_LIST", sweep_source)
        self.assertIn("run_explicit_point_list", sweep_source)
        self.assertIn("IFS=:", sweep_source)
        self.assertIn("rows:dim:chunk_elems:worker_cores", sweep_source)
        self.assertIn("GOLEM_SFU_SOFTMAX_SWEEP_DIMS", sweep_source)
        self.assertIn("GOLEM_SFU_SOFTMAX_CHUNK_SWEEP_DIM", sweep_source)
        self.assertIn("GOLEM_SFU_SOFTMAX_CHUNKS", sweep_source)
        self.assertIn("GOLEM_SFU_SOFTMAX_WORKERS", sweep_source)
        self.assertIn("GOLEM_SFU_SOFTMAX_MULTICORE_MATRIX", sweep_source)
        self.assertIn("worker_cores", sweep_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_WORKER_CORES", sweep_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_SOFTMAX_MULTICORE_MIN_DIM", sweep_source)
        self.assertIn("GOLEM_SFU_SOFTMAX_PIPELINE_ARGS", sweep_source)
        self.assertIn("pipeline_args", sweep_source)
        self.assertIn("read -r -a", sweep_source)

    def test_softmax_primitive_sweep_normalizes_artifact_root_to_absolute_path(self):
        sweep_source = read_local("run_sfu_softmax_primitive_sweep.sh")

        self.assertIn('mkdir -p "$SWEEP_ROOT"', sweep_source)
        self.assertIn('SWEEP_ROOT="$(cd "$SWEEP_ROOT" && pwd)"', sweep_source)
        self.assertLess(
            sweep_source.index('SWEEP_ROOT="$(cd "$SWEEP_ROOT" && pwd)"'),
            sweep_source.index('MANIFEST="$SWEEP_ROOT/sweep_manifest.csv"'),
        )

    def test_softmax_primitive_plotter_parses_pass_lines_and_writes_chinese_notes(self):
        plot_source = read_local("plot_sfu_softmax_primitive_sweep.py")

        self.assertIn("mode=sfu-primitive-softmax", plot_source)
        self.assertIn("max_abs_diff", plot_source)
        self.assertIn("max_row_sum_error", plot_source)
        self.assertIn("worker_cores", plot_source)
        self.assertIn("dim_per_core", plot_source)
        self.assertIn("simulated_time_us", plot_source)
        self.assertIn("wall_time_sec", plot_source)
        self.assertIn("softmax_primitive_summary.csv", plot_source)
        self.assertIn("softmax_primitive_notes.md", plot_source)
        self.assertIn("softmax_primitive_dse.svg", plot_source)
        self.assertIn("softmax_primitive_multirow.svg", plot_source)
        self.assertIn("softmax_primitive_timeout_diagnosis.md", plot_source)
        self.assertIn("write_focused_dse_svg", plot_source)
        self.assertIn("write_multirow_svg", plot_source)
        self.assertIn("write_timeout_diagnosis", plot_source)
        self.assertIn("estimated_rows_by_reads", plot_source)
        self.assertIn("实验结论", plot_source)
        self.assertIn("PASS 点摘要", plot_source)
        self.assertIn("chunk 覆盖说明", plot_source)
        self.assertIn("路线纠偏", plot_source)
        self.assertIn("multi-core cooperative", plot_source)
        self.assertNotIn("所有 PASS 点使用单核 batch-default primitive softmax 原型路径", plot_source)
        self.assertNotIn("2048 维度在 `chunk=512` 和 `chunk=2048` 下均 timeout", plot_source)

    def test_standalone_softmax_uses_row_band_issue_wait_window(self):
        runtime_source = read_local("golem_softmax_sfu_runtime.cpp")
        body_match = re.search(
            r"extern \"C\" golem_status_t golemRunSoftmaxSfuForCore\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\nextern \"C\" golem_status_t golemRunStandaloneSoftmaxSfuForCore",
            runtime_source,
            re.S,
        )
        self.assertIsNotNone(body_match)
        body = body_match.group("body")

        self.assertIn("SFU_SOFTMAX_ISSUE_WINDOW_TILES", runtime_source)
        self.assertIn("row_band_m_tiles", runtime_source)
        self.assertIn("issue_sfu_softmax_tile", runtime_source)
        self.assertIn("wait_and_store_pending_tiles", runtime_source)
        self.assertIn("std::vector<PendingTile> pending", body)
        self.assertLess(
            body.index("for (int m_tile_begin = 0;"),
            body.index("wait_and_store_pending_tiles"),
        )

    def test_hbm_generator_preloads_standalone_logits_into_c_tile_layout(self):
        source = read_repo_relative("../../tools/gen_hbm_init.py")

        self.assertIn("GOLEM_SFU_STANDALONE_SOFTMAX", source)
        self.assertIn("GOLEM_SOFTMAX_LOGITS_FILE", source)
        self.assertIn("--softmax-logits-file", source)
        self.assertIn("_build_softmax_logits_matrix", source)
        self.assertIn("_write_standalone_softmax_logits", source)
        self.assertIn("OFF_GEMM_OUT_BASE", source)
        self.assertIn("GEMM_OUT_STRIDE_MM", source)
        self.assertIn("macro_task_id = _macro_task_for_group(m_group, n_group)", source)
        self.assertIn("cc * BLOCK_M + r", source)

    def test_hbm_generator_can_preload_standalone_logits_as_direct_rowmajor(self):
        source = read_repo_relative("../../tools/gen_hbm_init.py")

        self.assertIn("SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM", source)
        self.assertIn("OFF_SFU_SOFTMAX_ROWMAJOR_BASE", source)
        self.assertIn("OFF_SFU_SOFTMAX_ROWMAJOR_OUT_BASE", source)
        self.assertIn("_write_standalone_softmax_rowmajor_logits", source)
        self.assertIn("softmax_rowmajor_logits_row", source)
        self.assertIn("Preloaded standalone softmax logits into direct row-major HBM region", source)
        self.assertIn("SFU softmax row-major HBM region exceeds bias area", source)

    def test_hbm_generator_preloads_safe_primitive_stream_inputs(self):
        source = read_repo_relative("../../tools/gen_hbm_init.py")

        self.assertIn("GOLEM_SFU_PRIMITIVE_HBM_STREAM", source)
        self.assertIn("_write_sfu_primitive_hbm_input", source)
        self.assertIn("_sfu_primitive_hbm_input_value", source)
        self.assertIn("Preloaded SFU primitive HBM stream input", source)
        self.assertIn("OFF_GEMM_OUT_BASE", source)
        self.assertIn("GEMM_OUT_STRIDE_MM", source)

    def test_explicit_gemm_baseline_keeps_columns_independent_on_single_array_path(self):
        text = read_repo_relative("../mvm_noc_int_array/gemm_matmul_op.h")
        body_match = re.search(
            r"static inline void gemm_tiled_baseline\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}\n\ntemplate <typename T>\nstatic inline void gemm_tiled_overlap",
            text,
            re.S,
        )
        self.assertIsNotNone(body_match)
        body = body_match.group("body")
        compute_pos = body.index("run_mvm_compute_only<T>")
        store_pos = body.index("outputvectorstore(accum_col_addr(desc, rt, n_col), 0)")
        self.assertLess(compute_pos, store_pos)
        outer_loop_pos = body.index("for (int n_col = 0; n_col < desc.block_n; ++n_col) {")
        k_loop_pos = body.index("for (int k = 0; k < desc.k_tiles; ++k) {")
        self.assertLess(outer_loop_pos, k_loop_pos)
        self.assertNotIn("MAT_BYTES + vec_block_bytes", body)
        self.assertIn("run_mvm_compute_only<T>(rt.local_mat, rt.local_vec_in, 0)", body)
        self.assertNotIn("run_mvm_compute_only<T>(rt.local_mat, vec_col_addr(rt, n_col), 0)", body)

    def test_makefile_builds_independent_sfu_workload(self):
        text = read_local("Makefile")
        self.assertIn("test_noc_dma_softmax_sfu", text)
        self.assertIn("golem_softmax_sfu_runtime.cpp", text)
        self.assertIn("../mvm_noc_int_array", text)


if __name__ == "__main__":
    unittest.main()
