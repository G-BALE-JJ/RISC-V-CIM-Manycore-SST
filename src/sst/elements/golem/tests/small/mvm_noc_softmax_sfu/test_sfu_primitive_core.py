#!/usr/bin/env python3

import os
import re
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLEM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SFU_H = os.path.join(GOLEM_DIR, "sfu", "sfu.h")
SFU_CC = os.path.join(GOLEM_DIR, "sfu", "sfu.cc")
GLOBAL_MEMORY_H = os.path.join(GOLEM_DIR, "globalmemory", "globalmemory.h")
GLOBAL_MEMORY_CC = os.path.join(GOLEM_DIR, "globalmemory", "globalmemory.cc")
ROCC_H = os.path.join(GOLEM_DIR, "rocc", "roccAnalog.h")


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class SfuPrimitiveCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = read(SFU_H)
        cls.source = read(SFU_CC)
        cls.global_memory_header = read(GLOBAL_MEMORY_H)
        cls.global_memory_source = read(GLOBAL_MEMORY_CC)
        cls.rocc_header = read(ROCC_H)

    def test_global_memory_exposes_serializable_reduction_transport_bridge(self):
        for token in (
            "enum class ReductionTransportMessageKind",
            "struct ReductionTransportMessage",
            "class ReductionTransportEvent",
            "sendReductionMessage",
            "setReductionMessageHandler",
            "reductionNetworkAvailable",
            "ImplementSerializable(SST::Golem::ReductionTransportEvent)",
        ):
            self.assertIn(token, self.global_memory_source + self.global_memory_header)

    def test_reduction_transport_event_uses_pool_compatible_allocation(self):
        self.assertIn(
            "new ReductionTransportEvent(stampedMessage)",
            self.global_memory_source,
        )
        self.assertNotRegex(
            self.global_memory_source,
            r"new\s*\(\s*std::nothrow\s*\)\s*ReductionTransportEvent",
        )

    def test_sfu_api_exposes_standalone_primitive_issue(self):
        self.assertIn("virtual bool issuePrimitive(uint64_t descAddr, uint64_t tag) = 0", self.header)
        self.assertIn("bool issuePrimitive(uint64_t descAddr, uint64_t tag) override", self.header)
        self.assertIn("std::unordered_map<uint64_t, PrimitiveOpState> pendingPrimitiveOps_", self.header)

    def test_primitive_state_and_helpers_are_declared(self):
        self.assertIn("struct PrimitiveOpState", self.header)
        self.assertIn("SFUPrimitiveDesc desc", self.header)
        self.assertIn("std::vector<float> input0", self.header)
        self.assertIn("std::vector<float> output", self.header)
        for helper in [
            "readPrimitiveDescriptor",
            "validatePrimitiveDescriptor",
            "readPrimitiveInput",
            "executePrimitive",
            "writePrimitiveOutput",
            "primitiveProcessedElems",
        ]:
            self.assertIn(helper, self.header)

    def test_issue_primitive_reads_executes_and_records_tagged_state(self):
        issue_body = re.search(
            r"bool\s+SFU::issuePrimitive\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(issue_body)
        body = issue_body.group("body")
        self.assertIn("readPrimitiveDescriptor", body)
        self.assertIn("validatePrimitiveDescriptor", body)
        self.assertIn("readPrimitiveInput", body)
        self.assertIn("executePrimitive", body)
        self.assertIn("writePrimitiveOutput", body)
        self.assertIn("pendingPrimitiveOps_[tag]", body)
        self.assertIn("statOpsIssued_->addData", body)
        self.assertIn("statPrimitiveElems_->addData(primitiveProcessedElems", body)

    def test_repeat_chunk_primitive_records_logical_processed_elements(self):
        self.assertIn("sfu_primitive_elems", self.header)
        self.assertIn("statPrimitiveElems_", self.header)
        self.assertIn("GOLEM_SFU_PRIMITIVE_FLAG_REPEAT_CHUNK", self.source)
        self.assertIn("primitiveProcessedElems", self.source)
        self.assertIn("desc.input1_gm_addr", self.source)

    def test_wait_retires_primitive_operations(self):
        wait_body = re.search(
            r"bool\s+SFU::wait\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(wait_body)
        body = wait_body.group("body")
        self.assertIn("pendingPrimitiveOps_.find(tag)", body)
        self.assertIn("pendingPrimitiveOps_.erase", body)

    def test_unary_primitive_executes_exp_log_and_reciprocal_fp32(self):
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::EXP)", self.source)
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::LOG)", self.source)
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::RECIPROCAL)", self.source)
        self.assertIn("std::exp", self.source)
        self.assertIn("std::log", self.source)
        self.assertIn("1.0f / value", self.source)

    def test_phase9c_unary_primitive_executes_rsqrt_tanh_and_sigmoid_fp32(self):
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::RSQRT)", self.source)
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::TANH)", self.source)
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::SIGMOID)", self.source)
        self.assertIn("1.0f / static_cast<float>(std::sqrt", self.source)
        self.assertIn("std::tanh", self.source)
        self.assertIn("1.0f / (1.0f + static_cast<float>(std::exp", self.source)

    def test_reduction_primitives_execute_reduce_max_and_sum_to_scalar(self):
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_MAX)", self.source)
        self.assertIn("static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM)", self.source)
        self.assertIn("std::max_element", self.source)
        self.assertIn("double sum = 0.0", self.source)
        self.assertIn("state->output.assign(1, 0.0f)", self.source)

    def test_contiguous_fp32_primitive_uses_bulk_local_gm_transfers(self):
        read_body = re.search(
            r"bool\s+SFU::readPrimitiveInput\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        write_body = re.search(
            r"bool\s+SFU::writePrimitiveOutput\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(read_body)
        self.assertIsNotNone(write_body)
        self.assertIn("stride == sizeof(float)", read_body.group("body"))
        self.assertIn("static_cast<size_t>(desc.elem_count) * sizeof(float)", read_body.group("body"))
        self.assertIn("std::memcpy(values->data()", read_body.group("body"))
        self.assertIn("stride == sizeof(float)", write_body.group("body"))
        self.assertIn("values.size() * sizeof(float)", write_body.group("body"))
        self.assertIn("std::memcpy(raw.data()", write_body.group("body"))

    def test_unified_issue_job_reads_validates_executes_and_records_tagged_state(self):
        self.assertIn("bool SFU::issueJob(uint64_t descAddr, uint64_t tag)", self.source)
        issue_body = re.search(
            r"bool\s+SFU::issueJob\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(issue_body)
        body = issue_body.group("body")
        self.assertIn("readJobDescriptor", body)
        self.assertIn("validateJobDescriptor", body)
        self.assertIn("executeJob", body)
        self.assertIn("pendingJobOps_[tag]", body)
        self.assertIn("statOpsIssued_->addData(1)", body)

    def test_unified_issue_job_reports_rejected_descriptor_context_when_verbose(self):
        issue_body = re.search(
            r"bool\s+SFU::issueJob\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(issue_body)
        body = issue_body.group("body")
        self.assertIn("SFU job rejected core=", body)
        self.assertIn("state.desc.reserved0", body)
        self.assertIn("activeWorkerCores_", body)

    def test_unified_job_validator_accepts_softmax_row_and_rejects_unknown_ops(self):
        self.assertIn("SFUStatus SFU::validateJobDescriptor", self.source)
        self.assertIn("static_cast<uint32_t>(SFUJobOp::SOFTMAX_ROW)", self.source)
        self.assertIn("return SFUStatus::InvalidDescriptor", self.source)
        self.assertIn("desc.worker_cores == 0", self.source)
        self.assertIn("desc.chunk_elems == 0", self.source)

    def test_unified_job_validator_rejects_invalid_distributed_worker_slot(self):
        self.assertIn("SFU_JOB_FLAG_DISTRIBUTED_COLUMNS", self.source)
        self.assertIn("desc.reserved0 >= desc.worker_cores", self.source)
        self.assertIn("desc.worker_cores > activeWorkerCores_", self.source)
        self.assertIn("desc.owner_core > coreId_", self.source)
        self.assertIn("coreId_ - desc.owner_core != desc.reserved0", self.source)

    def test_unified_job_duplicate_tag_aborts_existing_state_without_overwrite(self):
        issue_body = re.search(
            r"bool\s+SFU::issueJob\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(issue_body)
        body = issue_body.group("body")
        self.assertIn("auto existing = pendingJobOps_.find(tag)", body)
        self.assertIn("abortDistributedSoftmaxJob(&existing->second)", body)
        self.assertIn("existing->second.status = SFUStatus::InvalidDescriptor", body)
        self.assertIn("Duplicate tags poison the shared tag identity", body)
        self.assertLess(
            body.index("auto existing = pendingJobOps_.find(tag)"),
            body.index("pendingJobOps_[tag] = state"),
        )

    def test_unified_softmax_row_executor_is_inside_sfu_not_guest_stage_machine(self):
        self.assertIn("bool SFU::executeSoftmaxRowJob(JobOpState* state)", self.source)
        self.assertIn("std::exp", self.source)
        self.assertIn("globalMax", self.source)
        self.assertIn("globalSum", self.source)
        self.assertIn("state->desc.worker_cores", self.source)
        self.assertIn("state->desc.chunk_elems", self.source)
        self.assertIn("statSoftmaxRows_->addData", self.source)

    def test_unified_softmax_job_streams_rows_in_bands_and_chunks(self):
        self.assertIn("GOLEM_SFU_JOB_SOFTMAX_ROW_BAND_ROWS", self.source)
        self.assertIn("SoftmaxJobRowBandState", self.source)
        self.assertIn("readSoftmaxJobChunk", self.source)
        self.assertIn("writeSoftmaxJobChunk", self.source)
        self.assertIn("for (uint32_t rowBandBegin", self.source)
        self.assertIn("std::vector<float> chunk", self.source)
        self.assertNotIn("std::vector<float> row(desc.cols", self.source)
        self.assertNotIn("std::vector<float> out(desc.cols", self.source)

    def test_unified_softmax_job_reduces_worker_local_max_and_sum_inside_sfu(self):
        self.assertIn("localMax", self.source)
        self.assertIn("localSum", self.source)
        self.assertIn("worker * bandRows", self.source)
        self.assertIn("globalMax[bandRow]", self.source)
        self.assertIn("globalSum[bandRow]", self.source)
        self.assertIn("const uint32_t begin = workerColumnBegin", self.source)
        self.assertIn("const uint32_t end = workerColumnEnd", self.source)

    def test_unified_softmax_job_records_internal_chunk_pass_stats(self):
        for stat_name in (
            "sfu_job_softmax_max_chunks",
            "sfu_job_softmax_sum_chunks",
            "sfu_job_softmax_norm_chunks",
        ):
            self.assertIn(stat_name, self.header)
            self.assertIn(stat_name, self.source)

        for member in (
            "statJobSoftmaxMaxChunks_",
            "statJobSoftmaxSumChunks_",
            "statJobSoftmaxNormChunks_",
        ):
            self.assertIn(member, self.header)
            self.assertIn(member, self.source)

        body = re.search(
            r"bool\s+SFU::executeSoftmaxRowJob\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(body)
        executor = body.group("body")
        self.assertIn("statJobSoftmaxMaxChunks_->addData(1)", executor)
        self.assertIn("statJobSoftmaxSumChunks_->addData(1)", executor)
        self.assertIn("statJobSoftmaxNormChunks_->addData(1)", executor)

    def test_distributed_unified_softmax_records_pending_stage_state(self):
        for declaration in (
            "enum class SoftmaxJobStage",
            "SoftmaxJobStage stage",
            "uint32_t workerSlot",
            "uint32_t colBegin",
            "uint32_t colEnd",
            "std::vector<double> localMax",
            "std::vector<double> localSum",
        ):
            self.assertIn(declaration, self.header)
        self.assertIn("executeDistributedSoftmaxRowJob", self.header)
        self.assertIn("advanceDistributedSoftmaxJob", self.header)

    def test_distributed_unified_softmax_issue_submits_only_local_max(self):
        self.assertIn("bool SFU::executeDistributedSoftmaxRowJob", self.source)
        self.assertIn("readDistributedSoftmaxJobChunk", self.source)
        self.assertIn("submitDistributedSoftmaxMax", self.source)
        self.assertIn("SoftmaxJobStage::MaxSubmitted", self.source)
        self.assertIn("state->status = SFUStatus::Pending", self.source)
        self.assertIn("col - state->colBegin", self.source)

    def test_distributed_unified_softmax_wait_advances_sum_and_normalize(self):
        self.assertIn("bool SFU::advanceDistributedSoftmaxJob", self.source)
        self.assertIn("distributedSoftmaxMaxReady", self.source)
        self.assertIn("submitDistributedSoftmaxSum", self.source)
        self.assertIn("SoftmaxJobStage::SumSubmitted", self.source)
        self.assertIn("distributedSoftmaxSumReady", self.source)
        self.assertIn("markDistributedSoftmaxNormalized", self.source)
        self.assertIn("SoftmaxJobStage::Complete", self.source)

        wait_body = re.search(
            r"bool\s+SFU::wait\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(wait_body)
        self.assertIn("advanceDistributedSoftmaxJob", wait_body.group("body"))

    def test_distributed_softmax_reducer_tracks_worker_slots_and_cleans_rows(self):
        self.assertIn("struct DistributedSoftmaxReducerRowState", self.source)
        self.assertIn("uint32_t expectedRows = 0", self.source)
        self.assertIn("uint32_t expectedCols = 0", self.source)
        self.assertIn("maxSeen", self.source)
        self.assertIn("sumSeen", self.source)
        self.assertIn("normalizeSeen", self.source)
        self.assertIn("distributedSoftmaxReducerRows().erase", self.source)

    def test_distributed_softmax_reduction_transport_can_be_modeled_as_noc_messages(self):
        self.assertIn('{"distributed_reduction_transport"', self.header)
        self.assertIn("enum class DistributedReductionTransport", self.header)
        self.assertIn("DistributedReductionTransport distributedReductionTransport_", self.header)
        for stat_name in (
            "sfu_reduction_max_requests",
            "sfu_reduction_max_responses",
            "sfu_reduction_sum_requests",
            "sfu_reduction_sum_responses",
        ):
            self.assertIn(stat_name, self.header)
            self.assertIn(stat_name, self.source)
        for helper in (
            "recordDistributedReductionRequest",
            "recordDistributedReductionResponse",
        ):
            self.assertIn(helper, self.header)
            self.assertIn(helper, self.source)

        self.assertIn('params.find<std::string>("distributed_reduction_transport"', self.source)
        self.assertIn('transport == "modeled_noc"', self.source)
        self.assertIn('transport == "shared"', self.source)
        self.assertIn('transport == "explicit_noc"', self.source)
        self.assertIn("DistributedReductionTransport::ExplicitNoC", self.source)
        self.assertIn("explicitDistributedReductionEnabled", self.header)
        self.assertIn("explicitDistributedReductionEnabled", self.source)
        self.assertIn('invalid distributed_reduction_transport', self.source)
        self.assertIn("output_.fatal", self.source)
        self.assertIn("recordDistributedReductionRequest(true)", self.source)
        self.assertIn("recordDistributedReductionRequest(false)", self.source)
        self.assertIn("recordDistributedReductionResponse(true)", self.source)
        self.assertIn("recordDistributedReductionResponse(false)", self.source)

    def test_explicit_noc_reduction_uses_global_memory_transport_interfaces(self):
        for token in (
            "handleReductionTransportMessage",
            "sendReductionMessage",
            "distributedReductionResponseInbox_",
            "MaxRequest",
            "MaxResponse",
            "SumRequest",
            "SumResponse",
        ):
            self.assertIn(token, self.source + self.header)
        self.assertIn("setReductionMessageHandler", self.rocc_header)
        self.assertIn("sfu->receiveReductionMessage(message)", self.rocc_header)
        self.assertNotIn("setReductionMessageHandler", self.source)

    def test_explicit_noc_reduction_waits_for_keyed_response_inbox(self):
        advance_start = self.source.index("bool SFU::advanceDistributedSoftmaxJob")
        abort_start = self.source.index("void SFU::abortDistributedSoftmaxJob")
        advance_body = self.source[advance_start:abort_start]
        self.assertIn("distributedReductionResponseInbox_.find", advance_body)
        self.assertIn("distributedReductionResponseInbox_.erase", advance_body)
        self.assertNotIn("drainDistributedReductionMessages", advance_body)

        execute_start = self.source.index("bool SFU::executeDistributedSoftmaxRowJob")
        advance_start = self.source.index("bool SFU::advanceDistributedSoftmaxJob")
        execute_body = self.source[execute_start:advance_start]
        self.assertIn("globalMem_->sendReductionMessage", execute_body)
        self.assertIn("globalMem_->reductionNetworkAvailable", execute_body)

    def test_explicit_noc_rejects_stale_requests_after_abort(self):
        self.assertIn("distributedSoftmaxAbortedRows", self.source)
        self.assertIn("markDistributedSoftmaxRowAborted", self.source)
        self.assertIn("distributedSoftmaxRowAborted", self.source)

        max_submit = re.search(
            r"DistributedReducerResult\s+submitDistributedSoftmaxMax\s*\([^)]*\)"
            r"\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        sum_submit = re.search(
            r"DistributedReducerResult\s+submitDistributedSoftmaxSum\s*\([^)]*\)"
            r"\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(max_submit)
        self.assertIsNotNone(sum_submit)
        self.assertIn("distributedSoftmaxRowAborted", max_submit.group("body"))
        self.assertIn("distributedSoftmaxRowAborted", sum_submit.group("body"))

        advance_start = self.source.index("bool SFU::advanceDistributedSoftmaxJob")
        abort_start = self.source.index("void SFU::abortDistributedSoftmaxJob")
        advance_body = self.source[advance_start:abort_start]
        self.assertGreaterEqual(advance_body.count("distributedSoftmaxRowAborted"), 2)

    def test_explicit_noc_rejects_response_duplicates_after_stage_advance(self):
        handler_start = self.source.index("void SFU::handleReductionTransportMessage")
        record_start = self.source.index("void SFU::recordDistributedReductionRequest")
        handler_body = self.source[handler_start:record_start]

        # The inbox entry is consumed before MaxSubmitted advances to SumSubmitted;
        # therefore inbox-resident deduplication alone cannot reject a delayed max reply.
        advance_start = self.source.index("bool SFU::advanceDistributedSoftmaxJob")
        abort_start = self.source.index("void SFU::abortDistributedSoftmaxJob")
        advance_body = self.source[advance_start:abort_start]
        self.assertLess(
            advance_body.index("distributedReductionResponseInbox_.erase(response)"),
            advance_body.index("state->stage = SoftmaxJobStage::SumSubmitted"),
        )

        # Admission is tied to the receiving worker's active stage and its
        # consumed row marker, while retaining the full response identity check.
        for token in (
            "const SoftmaxJobStage expectedStage",
            "job->second.tag != message.tag",
            "job->second.stage != expectedStage",
            "const std::vector<uint8_t>& responseSeen",
            "message.row >= responseSeen.size()",
            "responseSeen[message.row] != 0",
        ):
            self.assertIn(token, handler_body)

    def test_distributed_issue_rejects_a_retired_abort_tombstone_identity(self):
        issue_start = self.source.index("bool SFU::issueJob")
        wait_start = self.source.index("bool SFU::wait")
        issue_body = self.source[issue_start:wait_start]

        for token in (
            "distributedSoftmaxRowAborted(distributedSoftmaxReducerKey(",
            "state.desc.job_id",
            "state.tag",
            "state.desc.owner_core",
            "state.desc.rows",
            "state.status = SFUStatus::InvalidDescriptor",
        ):
            self.assertIn(token, issue_body)
        self.assertLess(
            issue_body.index("distributedSoftmaxRowAborted(distributedSoftmaxReducerKey("),
            issue_body.index("executeJob(&state)"),
        )

    def test_distributed_reducer_rejects_row_and_column_cohort_mismatch(self):
        initializer = re.search(
            r"bool\s+initializeDistributedSoftmaxRow\s*\([^)]*expectedRows[^)]*expectedCols[^)]*\)"
            r"\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(initializer)
        body = initializer.group("body")
        self.assertIn("row->expectedRows = expectedRows", body)
        self.assertIn("row->expectedCols = expectedCols", body)
        self.assertIn("row->expectedRows == expectedRows", body)
        self.assertIn("row->expectedCols == expectedCols", body)

        submit_call = re.search(
            r"submitDistributedSoftmaxMax\(desc\.job_id,.*?state->localMax\[row\]\)",
            self.source,
            re.S,
        )
        self.assertIsNotNone(submit_call)
        self.assertIn("desc.rows", submit_call.group(0))
        self.assertIn("desc.cols", submit_call.group(0))

    def test_distributed_reducer_keys_include_tag_owner_and_propagate_abort(self):
        self.assertIn("using DistributedSoftmaxReducerKey =", self.source)
        self.assertIn("uint64_t tag", self.source)
        self.assertIn("uint32_t ownerCore", self.source)
        self.assertIn("bool aborted = false", self.source)
        self.assertIn("abortSeen", self.source)
        self.assertIn("abortDistributedSoftmaxJob", self.header)
        self.assertIn("abortDistributedSoftmaxJob", self.source)
        self.assertIn("observeDistributedSoftmaxJobAbort", self.header)
        self.assertIn("observeDistributedSoftmaxJobAbort", self.source)
        self.assertIn("bool distributedAbortObserved", self.header)
        self.assertIn("state->distributedAbortObserved = true", self.source)
        self.assertIn("distributedSoftmaxJobMatchesKey", self.source)
        self.assertIn("collectDistributedSoftmaxJobKeys", self.source)
        self.assertIn("workerSlot < rowState.expectedWorkers", self.source)
        self.assertIn("std::find(existingKeys.begin(), existingKeys.end(), key)", self.source)
        self.assertIn("SFU_JOB_FLAG_DISTRIBUTED_ABORT", self.source)

        wait_body = re.search(
            r"bool\s+SFU::wait\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(wait_body)
        self.assertIn("abortDistributedSoftmaxJob(&state)", wait_body.group("body"))

    def test_wait_retires_unified_job_operations(self):
        wait_body = re.search(
            r"bool\s+SFU::wait\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.source,
            re.S,
        )
        self.assertIsNotNone(wait_body)
        body = wait_body.group("body")
        self.assertIn("pendingJobOps_.find(tag)", body)
        self.assertIn("pendingJobOps_.erase", body)


if __name__ == "__main__":
    unittest.main()
