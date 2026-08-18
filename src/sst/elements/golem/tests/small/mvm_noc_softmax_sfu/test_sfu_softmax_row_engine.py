#!/usr/bin/env python3

import os
import re
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLEM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SFU_H = os.path.join(GOLEM_DIR, "sfu", "sfu.h")
SFU_CC = os.path.join(GOLEM_DIR, "sfu", "sfu.cc")
WORKLOAD = os.path.join(SCRIPT_DIR, "test_noc_dma_softmax_sfu.cpp")
RUNTIME = os.path.join(SCRIPT_DIR, "golem_softmax_sfu_runtime.cpp")
ROCC_H = os.path.join(GOLEM_DIR, "rocc", "roccAnalog.h")
GLOBAL_MEMORY_H = os.path.join(GOLEM_DIR, "globalmemory", "globalmemory.h")
ARCH_SHIM = os.path.join(
    GOLEM_DIR, "tests", "small", "mvm_noc_softmax_cpu",
    "ncores_selfcom_dma_softmax_archive.py",
)


def read(path):
    with open(path, "r", encoding="utf-8") as source_file:
        return source_file.read()


class SfuSoftmaxRowEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = read(SFU_H)
        cls.source = read(SFU_CC)
        cls.workload = read(WORKLOAD)
        cls.runtime = read(RUNTIME)
        cls.rocc = read(ROCC_H)
        cls.global_memory = read(GLOBAL_MEMORY_H)
        cls.arch_shim = read(ARCH_SHIM)

    def test_row_engine_state_uses_absolute_ready_tick(self):
        job_state = re.search(
            r"struct JobOpState \{(?P<body>.*?)\n    \};",
            self.header,
            re.S,
        )
        self.assertIsNotNone(job_state)
        for field in [
            "uint64_t rowEngineIssueTick",
            "uint64_t rowEngineStartTick",
            "uint64_t rowEngineReadyTick",
            "uint64_t rowEngineModeledCycles",
        ]:
            self.assertIn(field, job_state.group("body"))
        self.assertIn("getCurrentSimCycle() >= state.rowEngineReadyTick", self.source)

    def test_row_engine_cycle_model_has_separate_vector_and_exp_throughput(self):
        self.assertIn("rowEngineModeledCycles", self.source)
        self.assertIn("ceilDiv(desc.cols, rowEngineVectorLanes_)", self.source)
        self.assertIn("ceilDiv(desc.cols, rowEngineExpLanes_)", self.source)
        self.assertIn("rowEngineReductionTreeLatency_", self.source)
        self.assertIn("rowEngineExpLatency_", self.source)
        self.assertIn("rowEngineReciprocalLatency_", self.source)

    def test_row_engine_contexts_pipeline_independent_rows(self):
        self.assertIn("rowEngineContexts_ > 1", self.source)
        self.assertIn("std::max(maxPerRow, std::max(expSumPerRow, normalizePerRow))", self.source)
        self.assertIn("(rows - 1) * rowInterval", self.source)

    def test_row_engine_statistics_have_explicit_cycle_or_tick_units(self):
        for statistic in [
            "sfu_row_engine_jobs",
            "sfu_row_engine_rows",
            "sfu_row_engine_max_cycles",
            "sfu_row_engine_exp_sum_cycles",
            "sfu_row_engine_normalize_cycles",
            "sfu_row_engine_modeled_cycles",
            "sfu_row_engine_queue_wait_cycles",
            "sfu_row_engine_wait_polls",
            "sfu_row_engine_completed_jobs",
        ]:
            self.assertIn(statistic, self.header)
            self.assertIn(statistic, self.source)

    def test_direct_rowmajor_workload_can_request_row_engine_model(self):
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_ROW_ENGINE", self.workload)
        self.assertIn("SFU_JOB_FLAG_ROW_ENGINE_MODEL", self.workload)

    def test_row_engine_job_is_not_limited_by_legacy_gemm_tile_height(self):
        self.assertIn("MatmulRuntimeConfig validation_cfg", self.runtime)
        self.assertIn("flags & SFU_JOB_FLAG_ROW_ENGINE_MODEL", self.runtime)
        self.assertIn("validation_cfg.block_m = GEMM_BLOCK_M", self.runtime)

    def test_row_engine_defers_functional_write_until_ready_tick(self):
        self.assertIn("const bool rowEngineJob =", self.source)
        self.assertIn("!rowEngineJob && !executeJob(&state)", self.source)
        ready_branch = re.search(
            r"getCurrentSimCycle\(\) >= state\.rowEngineReadyTick\) \{(?P<body>.*?)\n\s*\} else",
            self.source,
            re.S,
        )
        self.assertIsNotNone(ready_branch)
        self.assertIn("executeJob(&state)", ready_branch.group("body"))

    def test_row_engine_uses_waitable_hbm_store_completion(self):
        self.assertIn("GOLEM_ROCC_FUNC7_REMOTE_STORE_WAIT", self.rocc)
        self.assertIn("dma_write_to_host_async", self.global_memory)
        self.assertIn("dma_completion_done", self.rocc)
        self.assertIn("remote_store_wait", self.workload)
        self.assertNotIn("SOFTMAX_STORE_DRAIN_CYCLES", self.workload)

    def test_rocc_defers_row_engine_wait_until_absolute_completion_tick(self):
        self.assertIn(
            "virtual bool completionTick(uint64_t tag, uint64_t* tick) const = 0",
            self.header,
        )
        self.assertIn("bool completionTick(uint64_t tag, uint64_t* tick) const override", self.header)
        self.assertIn("bool SFU::completionTick(uint64_t tag, uint64_t* tick) const", self.source)
        self.assertIn("sfu->completionTick(cmd->rs1, &completionTick)", self.rocc)
        self.assertIn("getCurrentSimCycle() < completionTick", self.rocc)
        self.assertIn("sfuWaitBlocked_", self.rocc)
        self.assertIn("sfuWaitBlockedUntilTick_", self.rocc)
        self.assertIn("next_cmd->cmd_id == sfuWaitBlockedCmdId_", self.rocc)

    def test_tensor_controller_has_versioned_abi_and_hardware_scheduler(self):
        self.assertIn("SFU_JOB_FLAG_TENSOR_ROW_ENGINE", self.header)
        self.assertIn("struct SFUSoftmaxJobParamsV1", self.header)
        self.assertIn("static_assert(sizeof(SFUSoftmaxJobParamsV1) == 64", self.header)
        self.assertIn("startTensorRowEngineJob", self.header)
        self.assertIn("dma_read_from_host_to_globalmem", self.source)
        self.assertIn("dma_write_from_globalmem_to_host", self.source)
        self.assertIn(".reserved0 = rows_per_band", self.runtime)
        self.assertIn("worker.nextRow++", self.source)
        self.assertIn("TensorRowDispatch", self.global_memory)
        self.assertIn("TensorRowComplete", self.global_memory)
        self.assertIn("handleTensorRowDispatch", self.source)
        self.assertIn("handleTensorRowComplete", self.source)
        self.assertIn("sendReductionMessage(dispatch.workerSlot, dispatch)", self.source)
        self.assertIn("issueTensorInputDma", self.source)
        self.assertIn("worker.rowsCompleted += 1", self.source)

    def test_tensor_row_engine_compute_is_dma_driven_and_causal(self):
        finish = re.search(
            r"bool SFU::finishTensorJobIfReady\(JobOpState\* state\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(finish)
        self.assertNotIn("getCurrentSimCycle() < state->rowEngineReadyTick", finish.group("body"))
        self.assertIn("state->rowEngineReadyTick = getCurrentSimCycle()", finish.group("body"))

        self.assertIn("configureSelfLink", self.source)
        self.assertIn('"RowEngineSelf"', self.source)
        self.assertIn("handleTensorRowEngineEvent", self.source)
        self.assertIn("TensorRowEngineStage::Max", self.source)
        self.assertIn("TensorRowEngineStage::ExpSum", self.source)
        self.assertIn("TensorRowEngineStage::Normalize", self.source)

        dispatch = re.search(
            r"void SFU::handleTensorRowDispatch.*?\n\}(?=\n\nvoid SFU::)",
            self.source,
            re.S,
        )
        self.assertIsNotNone(dispatch)
        self.assertNotIn("std::exp", dispatch.group(0))
        self.assertNotIn("dma_write_to_host", dispatch.group(0))

        stage_advance = re.search(
            r"void SFU::advanceTensorRowChunk.*?\n\}(?=\n\nvoid SFU::)",
            self.source,
            re.S,
        )
        self.assertIsNotNone(stage_advance)
        normalize = stage_advance.group(0).find("TensorRowEngineStage::Normalize")
        output_dma = stage_advance.group(0).find("completeTensorRow")
        self.assertGreaterEqual(normalize, 0)
        self.assertGreater(output_dma, normalize)

    def test_tensor_context_holds_exactly_one_row(self):
        self.assertNotIn("constexpr uint32_t rowsPerContext = 16", self.source)
        self.assertIn("context.scratchAddr = worker.scratchAddr + contextIndex * rowBytes", self.source)
        self.assertIn("message.expectedRows, rowEngineContexts_", self.source)

    def test_tensor_context_uses_bounded_lane_state_not_a_full_row(self):
        context = re.search(
            r"struct TensorWorkerState \{.*?struct Context \{(?P<body>.*?)\n        \};",
            self.header,
            re.S,
        )
        self.assertIsNotNone(context)
        body = context.group("body")
        self.assertNotIn("std::vector<float> values", body)
        self.assertIn("std::vector<float> laneValues", body)
        self.assertIn("TensorRowEngineStage stage", body)
        self.assertIn("uint32_t chunkBegin", body)
        self.assertIn("uint64_t pendingLocalTag", body)
        self.assertIn("laneValues.reserve(rowEngineVectorLanes_)", self.source)

    def test_tensor_stages_use_async_local_memory_and_address_output_dma(self):
        self.assertIn("localReadAsync", self.source)
        self.assertIn("localWriteAsync", self.source)
        self.assertIn("LocalMemoryClient::SFU", self.source)
        self.assertIn("dma_write_from_globalmem_to_host", self.global_memory)
        self.assertIn("dma_write_from_globalmem_to_host", self.source)

        input_dma = re.search(
            r"void SFU::issueTensorInputDma.*?\n\}(?=\n\nvoid SFU::)",
            self.source,
            re.S,
        )
        self.assertIsNotNone(input_dma)
        self.assertNotIn("rd_from_globalmem", input_dma.group(0))
        self.assertNotIn("values.resize", input_dma.group(0))

        event_handler = re.search(
            r"void SFU::handleTensorRowEngineEvent.*?\n\}(?=\n\nvoid SFU::)",
            self.source,
            re.S,
        )
        self.assertIsNotNone(event_handler)
        self.assertNotIn("context.values", event_handler.group(0))
        self.assertNotIn("std::vector<uint8_t> output", event_handler.group(0))

    def test_tensor_local_memory_statistics_cover_each_in_place_pass(self):
        self.assertIn("GOLEM_TENSOR_LOCAL_STATS", self.source)
        for statistic in [
            "sfu_tensor_max_local_read_bytes",
            "sfu_tensor_max_local_write_bytes",
            "sfu_tensor_exp_sum_local_read_bytes",
            "sfu_tensor_exp_sum_local_write_bytes",
            "sfu_tensor_normalize_local_read_bytes",
            "sfu_tensor_normalize_local_write_bytes",
            "sfu_tensor_lane_buffer_high_water",
            "sfu_tensor_local_retry_events",
        ]:
            self.assertIn(statistic, self.source)

    def test_tensor_completion_is_unique_and_matches_the_dispatched_band(self):
        self.assertIn("std::vector<uint8_t> tensorCompletionSeen", self.header)
        completion = re.search(
            r"void SFU::handleTensorRowComplete.*?\n\}(?=\n\nbool SFU::)",
            self.source,
            re.S,
        )
        self.assertIsNotNone(completion)
        body = completion.group(0)
        self.assertIn("message.jobId != state.desc.job_id", body)
        self.assertIn("message.workerSlot != band % state.desc.worker_cores", body)
        self.assertIn("state.tensorCompletionSeen[band] != 0", body)
        self.assertIn("state.tensorCompletionSeen[band] = 1", body)

    def test_tensor_controller_rejects_unsafe_transport_or_band_oversubscription(self):
        start = re.search(
            r"bool SFU::startTensorRowEngineJob.*?\n\}(?=\n\nbool SFU::)",
            self.source,
            re.S,
        )
        self.assertIsNotNone(start)
        body = start.group(0)
        self.assertIn("!explicitDistributedReductionEnabled()", body)
        self.assertIn("bands > state.desc.worker_cores", body)
        self.assertIn("rowBytes * contextCount > rowEngineScratchpadBytes_", body)

    def test_tensor_worker_rejection_returns_a_failed_completion(self):
        dispatch = re.search(
            r"void SFU::handleTensorRowDispatch.*?\n\}(?=\n\nuint64_t SFU::)",
            self.source,
            re.S,
        )
        self.assertIsNotNone(dispatch)
        self.assertIn("rejectTensorRowDispatch(message)", dispatch.group(0))
        self.assertIn("void SFU::rejectTensorRowDispatch", self.source)

    def test_tensor_jobs_cannot_alias_physical_contexts_across_tags(self):
        self.assertIn("tensorRowEngineBusy", self.source)
        self.assertIn("if (!tensorWorkerOps_.empty())", self.source)

    def test_tensor_controller_is_coordinator_only_in_guest(self):
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_TENSOR_CONTROLLER", self.workload)
        self.assertIn("SFU_JOB_FLAG_TENSOR_ROW_ENGINE", self.runtime)
        self.assertIn("golemRunTensorSoftmaxSfuJob", self.runtime)

    def test_single_chunk_dma_read_preserves_callback(self):
        global_memory_cc = read(os.path.join(GOLEM_DIR, "globalmemory", "globalmemory.cc"))
        self.assertIn("op.cb = ctx ? DmaCallback() : cb", global_memory_cc)

    def test_softmax_directory_memnic_uses_requested_highlink_bandwidth(self):
        self.assertIn("GOLEM_DIRCTRL_HIGHLINK_BW", self.arch_shim)
        self.assertIn("directory_memnic_bandwidth_fragment", self.arch_shim)


if __name__ == "__main__":
    unittest.main()
