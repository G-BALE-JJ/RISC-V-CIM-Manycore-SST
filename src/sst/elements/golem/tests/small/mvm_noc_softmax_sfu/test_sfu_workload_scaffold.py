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
        self.assertIn("sfu_softmax_tile", text)
        self.assertIn("sfu_wait", text)
        self.assertIn(".insn r 0x0b, 7", text)

    def test_runtime_header_declares_descriptor_and_entry_point(self):
        text = read_local("golem_softmax_sfu_runtime.h")
        self.assertIn("struct SFUSoftmaxTileDesc", text)
        self.assertIn("golemRunSoftmaxSfuForCore", text)
        self.assertIn("job_id", text)
        self.assertIn("n_tiles_per_row", text)

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
