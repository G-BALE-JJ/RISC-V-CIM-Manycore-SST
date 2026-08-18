#!/usr/bin/env python3

import pathlib
import unittest


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[6]


class ManagerCoordinatorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sfu_header = (REPO_ROOT / "src/sst/elements/golem/sfu/sfu.h").read_text()
        cls.sfu_source = (REPO_ROOT / "src/sst/elements/golem/sfu/sfu.cc").read_text()
        cls.rocc = (REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalog.h").read_text()
        cls.rocc_float = (
            REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalogFloat.h"
        ).read_text()
        cls.rocc_int = (
            REPO_ROOT / "src/sst/elements/golem/rocc/roccAnalogInt.h"
        ).read_text()
        cls.gm_header = (
            REPO_ROOT / "src/sst/elements/golem/globalmemory/globalmemory.h"
        ).read_text()
        cls.guest_instr = (
            REPO_ROOT
            / "src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/ex_instr.h"
        ).read_text()
        cls.wrapper = (
            REPO_ROOT
            / "src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_noc_dma_softmax_sfu_pipeline.sh"
        ).read_text()
        cls.archive_architecture = (
            REPO_ROOT
            / "src/sst/elements/golem/tests/architecture/archive/ncores_selfcom_dma.py"
        ).read_text()
        cls.guest_source = (
            REPO_ROOT
            / "src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_noc_dma_softmax_sfu.cpp"
        ).read_text()

    def test_versioned_topology_map_has_explicit_physical_core_ids(self):
        self.assertIn("SFU_SOFTMAX_JOB_PARAMS_VERSION_MANAGER", self.sfu_header)
        self.assertIn("SFU_WORKER_TOPOLOGY_MAP_VERSION", self.sfu_header)
        self.assertIn("SFUWorkerTopologyMapV1", self.sfu_header)
        self.assertIn("worker_core_ids", self.sfu_header)

    def test_manager_control_state_lives_in_rocc(self):
        self.assertIn("ManagerTensorJobState", self.rocc)
        self.assertIn("managerTensorJobs_", self.rocc)
        self.assertIn("tryIssueManagerTensorJobCommand", self.rocc)
        self.assertIn("tryWaitManagerTensorJobCommand", self.rocc)
        self.assertIn("handleManagerTensorCompletion", self.rocc)
        self.assertIn("desc.dtype == SFU_JOB_DTYPE_FP32", self.rocc)

    def test_manager_dispatch_maps_slot_to_physical_core(self):
        self.assertIn("workerCore", self.gm_header)
        self.assertIn("state.workerCoreIds[workerSlot]", self.rocc)
        self.assertIn("sendReductionMessage(workerCore", self.rocc)
        self.assertIn("message.workerCore != coreId_", self.sfu_source)

    def test_manager_has_separate_issue_wait_opcodes_and_legacy_is_retained(self):
        self.assertIn("GOLEM_ROCC_FUNC7_TENSOR_MANAGER_JOB", self.rocc)
        self.assertIn("GOLEM_ROCC_FUNC7_TENSOR_MANAGER_WAIT", self.rocc)
        self.assertIn("tensor_manager_job", self.guest_instr)
        self.assertIn("tensor_manager_wait", self.guest_instr)
        self.assertIn("tryIssueSfuJobCommand", self.rocc)
        self.assertIn("handleTensorRowComplete", self.sfu_source)

    def test_rocc_routes_transport_without_manager_sfu_compute(self):
        self.assertIn("handleReductionTransportMessage", self.rocc)
        self.assertIn("sfu->receiveReductionMessage(message)", self.rocc)
        self.assertIn("receiveReductionMessage", self.sfu_header)
        manager_issue = self.rocc.split("bool tryIssueManagerTensorJobCommand", 1)[1]
        manager_issue = manager_issue.split("bool tryWaitManagerTensorJobCommand", 1)[0]
        self.assertNotIn("sfu->issueJob", manager_issue)

    def test_manager_runner_uses_no_ctrl_architecture_with_narrow_opt_in(self):
        self.assertIn("GOLEM_SFU_MANAGER_COORDINATOR", self.wrapper)
        self.assertIn(
            'GOLEM_ARCH_SCRIPT="small/mvm_noc_softmax_cpu/ncores_selfcom_dma_softmax_archive.py"',
            self.wrapper,
        )
        self.assertIn("_manager_rocc_only", self.archive_architecture)
        self.assertIn('name == "GOLEM_GROUP_MANAGER_ENABLE"', self.archive_architecture)

    def test_manager_selects_actual_vanadis_manager_core(self):
        body = self.guest_source.split(
            "int resolve_executor_core_from_argv_or_exit", 1
        )[1].split("int run_gemm_for_core", 1)[0]
        self.assertIn("const int actual_core_id = sched_getcpu()", body)
        self.assertIn("executor_core_id = requested_core_id", body)
        self.assertIn("executor_core_id < TOTAL_GROUPS", self.guest_source)

    def test_manager_statistics_are_declared_in_rocc_eli(self):
        for source in (self.rocc_float, self.rocc_int):
            self.assertIn('"tensor_manager_jobs_issued"', source)
            self.assertIn('"tensor_manager_completion_received_tick"', source)
            self.assertIn('"tensor_manager_wait_observed_tick"', source)


if __name__ == "__main__":
    unittest.main()
