#!/usr/bin/env python3

import os
import re
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

        self.assertIn("run_gemm_for_core(executor_core_id, requested_core_id, op_desc)", entry_source)
        self.assertIn("golemRunSoftmaxSfuForCore(", entry_source)
        self.assertLess(
            entry_source.index('read_i64_env_or_default("GOLEM_SFU_INTERLEAVE_GEMM", 0)'),
            entry_source.index("run_gemm_for_core(executor_core_id, requested_core_id, op_desc)"),
        )

    def test_workload_has_standalone_softmax_only_mode_before_gemm(self):
        entry_source = read_local("test_noc_dma_softmax_sfu.cpp")
        runtime_header = read_local("golem_softmax_sfu_runtime.h")
        runtime_source = read_local("golem_softmax_sfu_runtime.cpp")

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_STANDALONE_SOFTMAX", 0)', entry_source)
        self.assertIn("mode=sfu-standalone-softmax", entry_source)
        self.assertIn("golemRunStandaloneSoftmaxSfuForCore", runtime_header)
        self.assertIn("golemRunStandaloneSoftmaxSfuForCore", runtime_source)
        self.assertIn("task.c_base_mm", runtime_source)
        self.assertLess(
            entry_source.index('read_i64_env_or_default("GOLEM_SFU_STANDALONE_SOFTMAX", 0)'),
            entry_source.index("run_gemm_for_core(executor_core_id, requested_core_id, op_desc)"),
        )
        self.assertLess(
            entry_source.index("golemRunStandaloneSoftmaxSfuForCore"),
            entry_source.index("run_gemm_for_core(executor_core_id, requested_core_id, op_desc)"),
        )

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

        self.assertIn('read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_BATCH", 0)', entry_source)
        self.assertIn("run_hbm_stream_sfu_primitive_batch_case", entry_source)
        self.assertIn("sfu_primitive_batch(", entry_source)
        self.assertIn("sfu_primitive_batch_wait", entry_source)
        self.assertIn("SFUPrimitiveBatchDesc", entry_source)
        self.assertIn("GOLEM_SFU_PRIMITIVE_HBM_BATCH", wrapper_source)
        self.assertIn("export GOLEM_SFU_PRIMITIVE_HBM_BATCH", wrapper_source)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH", ex_instr_source)
        self.assertIn("GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH_WAIT", ex_instr_source)

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
