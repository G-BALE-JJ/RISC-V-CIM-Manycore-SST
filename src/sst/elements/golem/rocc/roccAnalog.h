// Copyright 2009-2025 NTESS. Under the terms
// of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2025, NTESS
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// of the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#ifndef _H_ANALOG_ROCC
#define _H_ANALOG_ROCC

#include <sst/core/output.h>
#include <sst/core/component.h>
#include <sst/core/subcomponent.h>
#include <sst/core/statapi/statbase.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/elements/golem/array/computeArray.h>
#include <sst/elements/golem/groupctrl/groupctrl.h>
#include <sst/elements/golem/requestscheduler/requestscheduler.h>
#include <sst/elements/golem/sfu/sfu.h>
#include <sst/elements/golem/workercmdproc/workercmdproc.h>
#include <sst/elements/vanadis/rocc/vroccinterface.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>


#include <array>
#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>
#include <iostream>

using namespace SST::Interfaces;
using namespace SST::Golem;

namespace SST {
namespace Golem {

constexpr uint32_t GOLEM_ROCC_FLAG_SYNC_MATRIX = 0x0;
constexpr uint32_t GOLEM_ROCC_FLAG_SYNC_VECTOR = 0x1;
constexpr uint32_t GOLEM_ROCC_FLAG_ASYNC_BASE = 0x80000000u;
constexpr uint32_t GOLEM_ROCC_FLAG_ASYNC_MATRIX = 0x40000000u;
constexpr uint32_t GOLEM_ROCC_FLAG_ASYNC_ARRAY_SHIFT = 16u;
constexpr uint32_t GOLEM_ROCC_FLAG_ASYNC_ARRAY_MASK = 0x00FF0000u;
constexpr uint8_t GOLEM_ROCC_FUNC7_TILE_MVM_BATCH = 0x11;
constexpr uint8_t GOLEM_ROCC_FUNC7_TILE_WAIT_BATCH = 0x12;
constexpr uint8_t GOLEM_ROCC_FUNC7_TILE_GM2IMAT_BCAST = 0x13;
constexpr uint8_t GOLEM_ROCC_FUNC7_TILE_GM2IVEC_BATCH = 0x14;
constexpr uint8_t GOLEM_ROCC_FUNC7_WCP_START = 0x15;
constexpr uint8_t GOLEM_ROCC_FUNC7_WCP_WAIT = 0x16;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE = 0x17;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_WAIT = 0x18;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_PRIMITIVE = 0x19;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_WAIT = 0x1a;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH = 0x1b;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH_WAIT = 0x1c;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_JOB = 0x1d;
constexpr uint8_t GOLEM_ROCC_FUNC7_REMOTE_STORE_WAIT = 0x1e;
constexpr uint8_t GOLEM_ROCC_FUNC7_TENSOR_MANAGER_JOB = 0x1f;
constexpr uint8_t GOLEM_ROCC_FUNC7_TENSOR_MANAGER_WAIT = 0x20;
constexpr uint8_t GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_JOB = 0x21;
constexpr uint8_t GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_WAIT = 0x22;
constexpr uint32_t GOLEM_ATTENTION_DESC_MAGIC = 0x41545431u;
constexpr uint16_t GOLEM_ATTENTION_DESC_VERSION = 1u;
constexpr uint32_t GOLEM_ATTENTION_FLAG_CAUSAL = 0x1u;
constexpr uint64_t ATTENTION_C1_WINDOW_BYTES = 26752;
constexpr uint64_t ATTENTION_D1_WINDOW_BYTES = 43136;
constexpr uint64_t ATTENTION_D3_WINDOW_BYTES = 46208;
constexpr uint64_t ATTENTION_E1_WINDOW_BYTES = ATTENTION_C1_WINDOW_BYTES;
constexpr uint64_t ATTENTION_E3_WINDOW_BYTES = 51328;

struct GolemAttentionDescV1 {
    uint32_t magic;
    uint16_t version;
    uint16_t size_bytes;
    uint64_t job_id;
    uint64_t q_addr;
    uint64_t k_addr;
    uint64_t v_addr;
    uint64_t output_addr;
    uint64_t topology_gm_addr;
    uint32_t queries;
    uint32_t keys;
    uint32_t head_dim;
    uint32_t query_block_rows;
    uint32_t key_block_rows;
    uint32_t worker_count;
    uint32_t flags;
    uint32_t query_row_begin;
    uint32_t kv_rows_per_node;
    uint64_t kv_node_stride_bytes;
    uint32_t tensor_root_core;
    uint32_t tensor_manager_slot;
    uint32_t tensor_manager_count;
    uint32_t reserved0;
    uint64_t reserved[1];
};

static_assert(sizeof(GolemAttentionDescV1) == 128,
              "GolemAttentionDescV1 ABI must remain fixed");

template <typename T>
class RoCCAnalog : public SST::Vanadis::VanadisRoCCInterface {

public:
    SST_ELI_REGISTER_SUBCOMPONENT_DERIVED_API(RoCCAnalog<T>, SST::Vanadis::VanadisRoCCInterface)
  
    RoCCAnalog(ComponentId_t id, Params &params)
        : VanadisRoCCInterface(id, params),
          max_instructions(params.find<size_t>("max_instructions", 8)) {

        stat_cycles_mvm_set = registerStatistic<uint64_t>("cycles_mvm_set");
        stat_cycles_mvm_l   = registerStatistic<uint64_t>("cycles_mvm_l");
        stat_cycles_mvm     = registerStatistic<uint64_t>("cycles_mvm");
        stat_cycles_mvm_s   = registerStatistic<uint64_t>("cycles_mvm_s");
        stat_cycles_mvm_mv  = registerStatistic<uint64_t>("cycles_mvm_mv");
        stat_cycles_mvm_ovec2gm = registerStatistic<uint64_t>("cycles_mvm_ovec2gm");
        stat_cycles_mvm_gm2ivec = registerStatistic<uint64_t>("cycles_mvm_gm2ivec");
        stat_cycles_mvm_gm2imat = registerStatistic<uint64_t>("cycles_mvm_gm2imat");
        stat_cycles_remote_st = registerStatistic<uint64_t>("cycles_remote_st");
        stat_cycles_remote_ld = registerStatistic<uint64_t>("cycles_remote_ld");
        statTensorManagerJobsIssued_ = registerStatistic<uint64_t>("tensor_manager_jobs_issued");
        statTensorManagerWorkersMapped_ = registerStatistic<uint64_t>("tensor_manager_workers_mapped");
        statTensorManagerRowsDispatched_ = registerStatistic<uint64_t>("tensor_manager_rows_dispatched");
        statTensorManagerRowsCompleted_ = registerStatistic<uint64_t>("tensor_manager_rows_completed");
        statTensorManagerJobsCompleted_ = registerStatistic<uint64_t>("tensor_manager_jobs_completed");
        statTensorManagerDescriptorAcceptTick_ = registerStatistic<uint64_t>("tensor_manager_descriptor_accept_tick");
        statTensorManagerBandDispatchTick_ = registerStatistic<uint64_t>("tensor_manager_band_dispatch_tick");
        statTensorManagerCompletionReceivedTick_ = registerStatistic<uint64_t>("tensor_manager_completion_received_tick");
        statTensorManagerCompleteTick_ = registerStatistic<uint64_t>("tensor_manager_complete_tick");
        statTensorManagerWaitObservedTick_ = registerStatistic<uint64_t>("tensor_manager_wait_observed_tick");
        statAttentionManagerJobsIssued_ = registerStatistic<uint64_t>("attention_manager_jobs_issued");
        statAttentionManagerJobsCompleted_ = registerStatistic<uint64_t>("attention_manager_jobs_completed");
        statAttentionManagerBandsCompleted_ = registerStatistic<uint64_t>("attention_manager_bands_completed");
        statAttentionManagerBandCompletionsReceived_ = registerStatistic<uint64_t>("attention_manager_band_completions_received");
        statAttentionTensorJobsCompleted_ = registerStatistic<uint64_t>("attention_tensor_jobs_completed");
        statAttentionManagerDescriptorAcceptTick_ = registerStatistic<uint64_t>("attention_manager_descriptor_accept_tick");
        statAttentionManagerDispatchTick_ = registerStatistic<uint64_t>("attention_manager_dispatch_tick");
        statAttentionManagerLocalCompleteTick_ = registerStatistic<uint64_t>("attention_manager_local_complete_tick");
        statAttentionManagerBandCompletionReceivedTick_ = registerStatistic<uint64_t>("attention_manager_band_completion_received_tick");
        statAttentionTensorCompleteTick_ = registerStatistic<uint64_t>("attention_tensor_complete_tick");
        statAttentionManagerWaitObservedTick_ = registerStatistic<uint64_t>("attention_manager_wait_observed_tick");
        statAttentionWorkerDispatchAcceptTick_ = registerStatistic<uint64_t>("attention_worker_dispatch_accept_tick");
        statAttentionWorkerQkTileCompleteTick_ = registerStatistic<uint64_t>("attention_worker_qk_tile_complete_tick");
        statAttentionWorkerSoftmaxTileCompleteTick_ = registerStatistic<uint64_t>("attention_worker_softmax_tile_complete_tick");
        statAttentionWorkerPvTileCompleteTick_ = registerStatistic<uint64_t>("attention_worker_pv_tile_complete_tick");
        statAttentionWorkerOutputDmaAckTick_ = registerStatistic<uint64_t>("attention_worker_output_dma_ack_tick");
        statAttentionPvMatrixBroadcasts_ = registerStatistic<uint64_t>("attention_pv_matrix_broadcasts");
        statAttentionQkArrayOps_ = registerStatistic<uint64_t>("attention_qk_array_ops");
        statAttentionPvArrayOps_ = registerStatistic<uint64_t>("attention_pv_array_ops");
        statAttentionSpHbmBytes_ = registerStatistic<uint64_t>("attention_sp_hbm_bytes");

        latency_mvm_ovec2gm = params.find<uint64_t>("latency_mvm_ovec2gm", 10);
        latency_mvm_gm2ivec = params.find<uint64_t>("latency_mvm_gm2ivec", 15);
        latency_mvm_gm2imat = params.find<uint64_t>("latency_mvm_gm2imat", 20);
        latency_remote_st = params.find<uint64_t>("latency_remote_st", 20);
        latency_remote_ld = params.find<uint64_t>("latency_remote_ld", 25);
        enable_async_array_load = params.find<int>("enable_async_array_load", 1) != 0;
        progress_heartbeat = params.find<int>("progress_heartbeat", 0) != 0;
        progress_interval_cycles = params.find<uint64_t>("progress_interval_cycles", 50000);
        progress_total_mvm_ops = params.find<uint64_t>("progress_total_mvm_ops", 0);
        if (progress_interval_cycles == 0) {
            progress_interval_cycles = 50000;
        }
        if (progress_total_mvm_ops == 0) {
            progress_heartbeat = false;
        }
        progress_next_cycle = progress_interval_cycles;

        coreID = params.find<uint64_t>("core_id", 0);
        StartTickCycle = 0;
        LastTickCycle = 0;
  
        try {
            UnitAlgebra clock = params.find<UnitAlgebra>("clock", "1GHz");
  
            if (!(clock.hasUnits("Hz") || clock.hasUnits("s")) || 
                clock.getRoundedValue() <= 0) {
                output->fatal(CALL_INFO, -1,
                    "%s, Error - Invalid param: clock.\n"
                    "Must have units of Hz or s and be > 0.\n"
                    "SI prefixes ok. You specified '%s'\n",
                    getName().c_str(), clock.toString().c_str());
            }
        } catch (const UnitAlgebra::UnitAlgebraException& exc) {
            output->fatal(CALL_INFO, -1,
                "%s, Invalid param: Exception while parsing 'clock'.\n"
                "'%s'\n",
                getName().c_str(), exc.what());
        }
  
        mmioStartAddr = params.find<uint64_t>("mmioAddr", 0);
        arrayInputSize = params.find<uint64_t>("arrayInputSize", 2);
        arrayOutputSize = params.find<uint64_t>("arrayOutputSize", 2);
  
        numArrays = params.find<uint64_t>("numArrays", 1);
        inputOperandSize = params.find<uint64_t>("inputOperandSize", 4);
        outputOperandSize = params.find<uint64_t>("outputOperandSize", 4);
        attentionWindowOffset_ = params.find<uint64_t>("attention_window_offset", 0xC0000);
        attentionWindowBytes_ = params.find<uint64_t>("attention_window_bytes", 0x10000);
        attentionPvMatrixBroadcast_ = params.find<bool>("attention_pv_matrix_broadcast", false);

        remoteTransferLength = defaultRemoteLength();
  
        output->verbose(
            CALL_INFO, 1, 0,
            "%s: numArrays: %d, arrayInputSize: %d, arrayOutputSize: %d \n",
            getName().c_str(), numArrays, arrayInputSize, arrayOutputSize);
        //std_mem_handlers 是内存请求处理器，其作用是处理从内存系统
        //返回的响应。它解析内存的响应数据，并执行相应的后续操作
        std_mem_handlers = new StandardMemHandlers(this, output);
  
        busy = false;
        //memInterface 是内存接口，它代表了与内存系统的实际通信渠道,
        //负责发起内存读写请求，并处理这些请求的发送和响应
        memInterface = loadUserSubComponent<Interfaces::StandardMem>(
            "memory_interface",
            ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS,
            getTimeConverter("1ps"),
            new StandardMem::Handler2<RoCCAnalog<T>, &RoCCAnalog<T>::processIncomingDataCacheEvent>(this));

        if ( nullptr == memInterface ) {
            output->fatal(
                CALL_INFO, -1,
                "Error: unable to load memory interface subcomponent for RoCCAnalog.\n");
        }
        //加载 Golem 的计算阵列子组件，用于进行阵列计算
        array = loadUserSubComponent<Golem::ComputeArray>(
            "array", ComponentInfo::SHARE_NONE, getTimeConverter("1ps"),
            new SST::Event::Handler2<RoCCAnalog<T>, &RoCCAnalog<T>::handleArrayEvent>(this));

        if ( nullptr == array ) {
            output->fatal(
                CALL_INFO, -1,
                "Error: Unable to load array model subcomponent for RoCCAnalog.\n");
        }

        // 新增：加载 GlobalMemory 子组件（可选）。
        globalMem = loadUserSubComponent<SST::Golem::GlobalMemoryAPI>(
            "global_memory", ComponentInfo::SHARE_NONE);
        uint64_t globalMemStride = params.find<uint64_t>("globalMemStride", 0x4000);
        uint64_t globalMemBase = params.find<uint64_t>("globalMemBase", 0x0);
        if (nullptr == globalMem) {
            // 如果测试未提供，回退到本地的无网络实现，避免fatal。
            output->verbose(
                CALL_INFO, 1,0,
                "Warning: Unable to load Network globalmemory subcomponent for RoCCAnalog, turn to network free implementation.\n");
            Params gmFallbackParams;
            gmFallbackParams.insert("src_id", std::to_string(coreID));
            gmFallbackParams.insert("size", std::to_string(globalMemStride));
            globalMem = loadAnonymousSubComponent<SST::Golem::GlobalMemoryAPI>(
                "golem.GlobalMemoryLocal", "global_memory", 0,
                ComponentInfo::SHARE_NONE, gmFallbackParams);
        }
        // **配置 GlobalMemory 基地址**：根据核心 ID 计算基地址 
        // 获取当前核ID（默认为0）
        // 每个核的地址空间跨度，默认0x4000
        uint64_t baseAddr = globalMemBase + coreID * globalMemStride;        // 计算该核 GlobalMemory 的基地址
        if (globalMem) {
            globalMem->setBaseAddr(baseAddr);                                // 设置 GlobalMemory 子模块的基地址
        }
        output->verbose(CALL_INFO, 1, 0, 
                        "RoCCAnalog: 核心%" PRIu64 " 的 GlobalMemory 基地址配置为 0x%" PRIx64 "\n", 
                        coreID, baseAddr);

        sfu = nullptr;
        sfuEnable = params.find<int>("sfuEnable", 0) != 0;
        if (sfuEnable) {
            sfu = loadUserSubComponent<SST::Golem::SFUAPI>(
                "sfu", ComponentInfo::SHARE_NONE);
            if (nullptr == sfu) {
                output->fatal(CALL_INFO, -1,
                    "Error: sfuEnable=1 but required user subcomponent 'sfu' is missing for RoCCAnalog.\n");
            }
            sfu->bindGlobalMemory(globalMem);
            sfu->setCoreInfo(
                static_cast<uint32_t>(coreID),
                params.find<uint32_t>("active_worker_cores", 1));
        }
        globalMem->setReductionMessageHandler(
            [this](const ReductionTransportMessage& message) {
                handleReductionTransportMessage(message);
            });

        groupCtrl = nullptr;
        if (params.find<int>("groupCtrlEnable", 0) != 0) {
            groupCtrl = loadUserSubComponent<SST::Golem::GroupCtrlAPI>(
                "group_ctrl", ComponentInfo::SHARE_NONE);

            if (nullptr == groupCtrl) {
                output->fatal(CALL_INFO, -1,
                    "Error: groupCtrlEnable=1 but required user subcomponent 'group_ctrl' is missing for RoCCAnalog.\n"
                    "Please wire RoCC slot 'group_ctrl' in the architecture script (setSubComponent).\n");
            }
        }

        requestScheduler = nullptr;
        if (params.find<int>("requestSchedulerEnable", 0) != 0) {
            requestScheduler = loadUserSubComponent<SST::Golem::RequestSchedulerAPI>(
                "request_scheduler", ComponentInfo::SHARE_NONE);
            if (nullptr == requestScheduler) {
                output->fatal(CALL_INFO, -1,
                    "Error: requestSchedulerEnable=1 but required user subcomponent 'request_scheduler' is missing for RoCCAnalog.\n");
            }
        }

        workerCommandProcessor = nullptr;
        if (params.find<int>("workerCommandProcessorEnable", 0) != 0) {
            workerCommandProcessor = loadUserSubComponent<SST::Golem::WorkerCommandProcessorAPI>(
                "worker_command_processor", ComponentInfo::SHARE_NONE);
            if (nullptr == workerCommandProcessor) {
                output->fatal(CALL_INFO, -1,
                    "Error: workerCommandProcessorEnable=1 but required user subcomponent 'worker_command_processor' is missing for RoCCAnalog.\n");
            }
            workerCommandProcessor->bindResources(static_cast<uint32_t>(coreID), output, globalMem, array, requestScheduler);
        }

    }
  
    virtual ~RoCCAnalog() {
        for (auto roccCmd_q_itr = roccCmd_q.begin(); roccCmd_q_itr != roccCmd_q.end();) {
            delete (*roccCmd_q_itr);
            roccCmd_q_itr = roccCmd_q.erase(roccCmd_q_itr);
        }

        for (auto& inflight : inflight_compute_cmds) {
            if (inflight.cmd != nullptr) {
                delete inflight.cmd;
                inflight.cmd = nullptr;
            }
        }

        while (!resp_q.empty()) {
            delete resp_q.front();
            resp_q.pop_front();
        }

        delete std_mem_handlers;
    }
    //RoCC指令队列满/是否忙/队列当前大小
    bool RoCCFull() override { return roccCmd_q.size() >= max_instructions; }
  
    bool isBusy() override { return busy; }
  
    size_t roccQueueSize() override { return roccCmd_q.size(); }

    //入队一条新的RoCC指令，同时统计数据加1
    void push(SST::Vanadis::RoCCCommand *rocc_me) override {
        stat_rocc_issued->addData(1);
        roccCmd_q.push_back(rocc_me);
    }
  
    //返回一条已完成响应；若无响应则返回 nullptr。
    SST::Vanadis::RoCCResponse *respond() override {
        if (resp_q.empty()) {
            return nullptr;
        }
        SST::Vanadis::RoCCResponse *temp = resp_q.front();
        resp_q.pop_front();
        return temp;
    }
  
    // Initialize subcomponents and parameterizable data structures
    void init(unsigned int phase) override {
  
        // Initialize arrayStates 调整其大小为 numArrays，即有多少阵列就有多少状态记录
        arrayStates.resize(numArrays);
        async_matrix_loads.resize(numArrays);
        async_vector_loads.resize(numArrays);
        inflight_compute_cmds.resize(numArrays);
        async_compute_states.resize(numArrays);
  
        // Set the address delimiters
        //inputOperandSize：每个输入操作数占用多少字节（比如float就是4字节）。
        //arrayInputSize：输入向量的长度
        inputDataSize = inputOperandSize * arrayInputSize;
        inputTotalSize = inputDataSize * numArrays;
        outputDataSize = outputOperandSize * arrayOutputSize;
        outputTotalSize = outputDataSize * numArrays;
        inputStartAddr = mmioStartAddr + numArrays;
        outputStartAddr = inputStartAddr + inputTotalSize;
  
        for (int i = 0; i < numArrays; i++) {
            arrayStates[i] = 0;
        }
        //配置RoCC使用的MMIO区间，把从mmioStartAddr开始、长度为inputTotalSize的区域
        //映射给内存接口，便于后续数据传输
    memInterface->setMemoryMappedAddressRegion(mmioStartAddr, inputTotalSize);
    memInterface->init(phase);
    array->init(phase);
    globalMem->init(phase);
    if (groupCtrl) {
        groupCtrl->init(phase);
    }
    if (requestScheduler) {
        requestScheduler->init(phase);
    }
    if (sfu) {
        sfu->init(phase);
    }
    }

    void setup() override {
        if (memInterface) {
            memInterface->setup();
        }
        if (array) {
            array->setup();
        }
        if (globalMem) {
            globalMem->setup();
        }
        if (groupCtrl) {
            groupCtrl->setup();
        }
        if (requestScheduler) {
            requestScheduler->setup();
        }
        if (sfu) {
            sfu->setup();
        }
    }

    void complete(unsigned int phase) override {
        if (memInterface) {
            memInterface->complete(phase);
        }
        if (array) {
            array->complete(phase);
        }
        if (globalMem) {
            globalMem->complete(phase);
        }
        if (groupCtrl) {
            groupCtrl->complete(phase);
        }
        if (requestScheduler) {
            requestScheduler->complete(phase);
        }
        if (sfu) {
            sfu->complete(phase);
        }
    }

    void finish() override {
        maybeReportMvmProgress(true);
        if (memInterface) {
            memInterface->finish();
        }
        if (array) {
            array->finish();
        }
        if (globalMem) {
            globalMem->finish();
        }
        if (groupCtrl) {
            groupCtrl->finish();
        }
        if (requestScheduler) {
            requestScheduler->finish();
        }
        if (sfu) {
            sfu->finish();
        }
    }
  
    // Main clock cycle tick function
    //每一个时钟周期调用一次tick
    void tick(uint64_t cycle) override {
        output->verbose(CALL_INFO, 16, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
        LastTickCycle = cycle;
        // Keep draining async array-load commands even while a synchronous command is in flight.
        tryIssueAsyncArrayLoadCommand(cycle);
        tryCompleteAsyncArrayLoads(cycle);
        progressManagerTensorJobs();
        progressManagerAttentionJobs();
        progressAttentionWorker();
        if (workerCommandProcessor != nullptr && workerCommandProcessor->isBusy()) {
            workerCommandProcessor->tick(cycle);
        }

        if (roccCmd_q.empty() && !busy) {
            output->verbose(CALL_INFO, 16, 0, "--> nothing to do in RoCC\n");
            return;
        }
        output->verbose(CALL_INFO, 16, 0, "busy? %d\n", busy);
  
        if (!busy) {
            if (roccCmd_q.empty()) {
                return;
            }

            auto* next_cmd = roccCmd_q.front();
            if (next_cmd == nullptr || next_cmd->inst == nullptr) {
                roccCmd_q.pop_front();
                delete next_cmd;
                return;
            }
            if (next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_WAIT &&
                sfuWaitBlocked_ && next_cmd->cmd_id == sfuWaitBlockedCmdId_ &&
                getCurrentSimCycle() < sfuWaitBlockedUntilTick_) {
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == 0x3) {
                const uint64_t array_id = next_cmd->rs1;
                const bool is_async_compute = (next_cmd->inst->rd == 0);
                if (array_id >= static_cast<uint64_t>(numArrays)) {
                    enqueueResponse(new SST::Vanadis::RoCCResponse(next_cmd->inst->rd, 1, next_cmd->cmd_id, next_cmd->hw_thread));
                    roccCmd_q.pop_front();
                    delete next_cmd;
                    return;
                }
                if (hasArrayLoadFailure(static_cast<uint32_t>(array_id))) {
                    output->verbose(CALL_INFO, 0, 0,
                        "[RoCC ERROR] async load failed earlier for array=%" PRIu64 ", reject compute cmd_id=%" PRIu64 "\n",
                        array_id,
                        next_cmd->cmd_id);
                    enqueueResponse(new SST::Vanadis::RoCCResponse(next_cmd->inst->rd, 1, next_cmd->cmd_id, next_cmd->hw_thread));
                    roccCmd_q.pop_front();
                    delete next_cmd;
                    return;
                }
                if (array_id < static_cast<uint64_t>(numArrays) &&
                    isArrayLoadInflight(static_cast<uint32_t>(array_id))) {
                    // Keep queue order stable and retry in next cycle.
                    return;
                }

                const uint32_t array_id_u32 = static_cast<uint32_t>(array_id);
                auto& async_state = async_compute_states[array_id_u32];
                if (async_state.submitted) {
                    if (isArrayComputeInflight(array_id_u32) || !async_state.completed) {
                        return;
                    }

                    if (!is_async_compute) {
                        enqueueResponse(new SST::Vanadis::RoCCResponse(
                            next_cmd->inst->rd,
                            async_state.rd_val,
                            next_cmd->cmd_id,
                            next_cmd->hw_thread));
                        async_state = AsyncComputeState{};
                        roccCmd_q.pop_front();
                        delete next_cmd;
                        return;
                    }

                    // A new async submit on the same array must wait until software
                    // retires the prior async completion via a synchronous wait.
                    enqueueResponse(new SST::Vanadis::RoCCResponse(
                        next_cmd->inst->rd,
                        1,
                        next_cmd->cmd_id,
                        next_cmd->hw_thread));
                    roccCmd_q.pop_front();
                    delete next_cmd;
                    return;
                }

                if (isArrayComputeInflight(array_id_u32)) {
                    return;
                }

                roccCmd_q.pop_front();
                issueArrayCompute(next_cmd, array_id_u32, cycle);
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TILE_MVM_BATCH) {
                roccCmd_q.pop_front();
                if (!tryIssueBatchComputeCommand(next_cmd, cycle)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TILE_WAIT_BATCH) {
                roccCmd_q.pop_front();
                if (!tryWaitBatchComputeCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TILE_GM2IMAT_BCAST) {
                roccCmd_q.pop_front();
                if (!tryIssueBatchArrayLoadCommand(next_cmd, cycle, true)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TILE_GM2IVEC_BATCH) {
                roccCmd_q.pop_front();
                if (!tryIssueBatchArrayLoadCommand(next_cmd, cycle, false)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_WCP_START) {
                roccCmd_q.pop_front();
                if (!tryStartWorkerWindow(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_WCP_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitWorkerWindow(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE) {
                roccCmd_q.pop_front();
                if (!tryIssueSfuSoftmaxTileCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitSfuCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_PRIMITIVE) {
                roccCmd_q.pop_front();
                if (!tryIssueSfuPrimitiveCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitSfuPrimitiveCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH) {
                roccCmd_q.pop_front();
                if (!tryIssueSfuPrimitiveBatchCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitSfuPrimitiveBatchCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_JOB) {
                roccCmd_q.pop_front();
                if (!tryIssueSfuJobCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TENSOR_MANAGER_JOB) {
                roccCmd_q.pop_front();
                if (!tryIssueManagerTensorJobCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TENSOR_MANAGER_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitManagerTensorJobCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_JOB) {
                roccCmd_q.pop_front();
                tryIssueManagerAttentionJobCommand(next_cmd);
                return;
            }
            if (next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitManagerAttentionJobCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }

            busy = true;
            curr_cmd = next_cmd;
            roccCmd_q.pop_front();
            StartTickCycle = cycle;
            //根据当前命令的操作码（func7）选择要执行的功能
            switch (curr_cmd->inst->func7) {
                case 0x1: // Set Matrix
                {   
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                              "the Instruction read: mvm.set (MVM set matrix)\n");
                    setMatrix();
                } break;
                case 0x2: // Load Vector
                {   
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                              "the Instruction read: mvm.l (MVM load vector)\n");
                    loadVector();
                } break;
                case 0x3: // Compute MVM
                {   
                    const uint64_t array_id = curr_cmd->rs1;
                    if (array_id < static_cast<uint64_t>(numArrays) && hasArrayLoadFailure(static_cast<uint32_t>(array_id))) {
                        output->verbose(CALL_INFO, 0, 0,
                            "[RoCC ERROR] async load failed earlier for array=%" PRIu64 ", reject compute cmd_id=%" PRIu64 "\n",
                            array_id,
                            curr_cmd->cmd_id);
                        completeRoCC(1);
                        break;
                    }
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                              "the Instruction read: mvm (MVM compute)\n");
                    computeMVM();
                } break;
                case 0x4: // Store Vector
                {   
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                              "the Instruction read: mvm.s (MVM store vector)\n");
                    storeVector();
                } break;
                case 0x5: // Move Vector
                {   
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                              "the Instruction read: mvm.mv (MVM move vector)\n");
                    moveVector();
                } break;
                case 0x6: //mvm.ovec2gm  
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: mvm.ovec2gm (MVM outputvector store)\n");
                    OutputvectorStore(cycle);
                } break;
                case 0x7: //mvm.gm2ivec 
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: mvm.gm2vec (MVM inputvector load)\n");
                    IntputvectorLoad(cycle);
                } break;
                case 0x8: //mvm.gm2imat
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: mvm.gm2imat (MVM inputmatrix load)\n");
                    InputMatrixLoad(cycle);
                } break;
                case 0x9: //remote_st Remote Store 
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: remote_st (MVM remote store)\n");
                    RemoteStore(cycle);
                } break;
                case 0xA: //remote_ld Remote Load 
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: remote_ld (MVM remote load)\n");
                    RemoteLoad(cycle);
                } break;
                case GOLEM_ROCC_FUNC7_REMOTE_STORE_WAIT:
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: remote_st.wait (waitable HBM store)\n");
                    RemoteStoreWait(cycle);
                } break;
                case 0xB: //mvm.slen Remote transfer length setup
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: mvm.slen (MVM remote length setup)\n");
                    SetRemoteLength();
                } break;
                case 0xC: //mvm.ocfg Output buffer configuration
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: mvm.ocfg (MVM output config)\n");
                    ConfigureOutputMode();
                } break;
                case 0xD: // mm2gm (Main Memory -> Global Memory)
                {
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                            "Instruction read: mm2gm (Main Memory -> Global Memory)\n");
                    MainMem2GlobalMem();
                } break;
                case 0xE: // gm2mm (Global Memory -> Main Memory)
                {
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                            "Instruction read: gm2mm (Global Memory -> Main Memory)\n");
                    GlobalMem2MainMem();
                } break;
                case 0xF: // reg2gm (Register -> Global Memory)
                {
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                            "Instruction read: reg2gm (Register -> Global Memory)\n");
                    Reg2GlobalMem();
                } break;
                case 0x10: // gm2reg (Global Memory -> Register)
                {
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                            "Instruction read: gm2reg (Global Memory -> Register)\n");
                    GlobalMem2Reg();
                } break;
                case GOLEM_ROCC_FUNC7_TILE_MVM_BATCH:
                {
                    if (!tryIssueBatchComputeCommand(curr_cmd, cycle)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                case GOLEM_ROCC_FUNC7_TILE_WAIT_BATCH:
                {
                    if (!tryWaitBatchComputeCommand(curr_cmd)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                case GOLEM_ROCC_FUNC7_TILE_GM2IMAT_BCAST:
                {
                    if (!tryIssueBatchArrayLoadCommand(curr_cmd, cycle, true)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                case GOLEM_ROCC_FUNC7_TILE_GM2IVEC_BATCH:
                {
                    if (!tryIssueBatchArrayLoadCommand(curr_cmd, cycle, false)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                case GOLEM_ROCC_FUNC7_WCP_START:
                {
                    if (!tryStartWorkerWindow(curr_cmd)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                case GOLEM_ROCC_FUNC7_WCP_WAIT:
                {
                    if (!tryWaitWorkerWindow(curr_cmd)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                default: {
                    output->verbose(CALL_INFO, 0, 0, "ERROR: unrecognized RoCC func7\n");
                    completeRoCC(1);
                } break;
            }
        } else {
            if (curr_cmd != nullptr) {
                switch (curr_cmd->inst->func7) {
                    case 0x6:
                        OutputvectorStore(cycle);
                        break;
                    case 0x7:
                        IntputvectorLoad(cycle);
                        break;
                    case 0x8:
                        InputMatrixLoad(cycle);
                        break;
                    case 0x9:
                        RemoteStore(cycle);
                        break;
                    case 0xA:
                        RemoteLoad(cycle);
                        break;
                    case GOLEM_ROCC_FUNC7_REMOTE_STORE_WAIT:
                        RemoteStoreWait(cycle);
                        break;
                    default:
                        break;
                }
            }
        }
    }
  
    // Issues the read request for the matrix that will be set in the analog array
    void setMatrix() {
        //取出当前RoCC命令的rs1寄存器值，通常代表了矩阵数据的物理地址
        uint64_t rs1 = curr_cmd->rs1;
        output->verbose(CALL_INFO, 1, 0, "RoCC setMatrix rs1: 0x%" PRIx64 "\n", rs1);
        uint32_t load_matrix_flag = 0x0;
        //计算本次要加载的矩阵总字节数，等于输入维数 × 输出维数 × 每个元素的字节数
        matrix_total_size = arrayInputSize * arrayOutputSize * inputOperandSize;
        //查询内存子系统缓存行大小
        uint64_t cache_line_size = memInterface->getLineSize();

        matrix_read_offset = 0;

        //直接用rs1作为物理地址 
        uint64_t physAddr = rs1; // Assuming rs1 is physical address
        //计算这个物理地址在当前cache line中的偏移量 比如 cache_line_size=64，physAddr=0x108，那么offset=8
        uint64_t addr_offset = physAddr % cache_line_size;

        // Calculate initial request size
        //本次首个内存读取请求的字节数：如果起始地址没对齐，需要先补齐一个cache line不能超过本次矩阵的总大小
        uint32_t request_size = std::min(static_cast<uint64_t>(cache_line_size - addr_offset), matrix_total_size);

        // Send first cache request
        //构造一个内存读取请求，请求地址为physAddr，请求长度为request_size，标记“这是个矩阵数据”
        //然后调用memInterface->send()发给内存子系统
        auto *load_req = new StandardMem::Read(physAddr, request_size, load_matrix_flag);
        memInterface->send(load_req);
    }
    
    void loadVector() {
        uint64_t rs1 = curr_cmd->rs1;
        output->verbose(CALL_INFO, 1, 0, "RoCC loadVector rs1: 0x%" PRIx64 "\n", rs1);
        uint32_t load_vector_flag = 0x1;
        vector_total_size = arrayInputSize * inputOperandSize;
        uint64_t cache_line_size = memInterface->getLineSize();

        vector_read_offset = 0;

        uint64_t physAddr = rs1; // Assuming rs1 is physical address
        uint64_t addr_offset = physAddr % cache_line_size;

        // Calculate initial request size
        uint32_t request_size = std::min(static_cast<uint64_t>(cache_line_size - addr_offset), vector_total_size);

        // Send first cache request
        auto *load_req = new StandardMem::Read(physAddr, request_size, load_vector_flag);
        memInterface->send(load_req);
    }
    
    void computeMVM() {
        uint64_t rs1 = curr_cmd->rs1;//rs1表示阵列ID，在多阵列场景下，每个阵列都有唯一编号
        arrayStates[rs1] = 1;//标记“这个阵列正在计算中”
        array->beginComputation(static_cast<uint32_t>(rs1));   //调用Golem的ComputeArray子组件的beginComputation方法
    }

    bool tryIssueBatchComputeCommand(SST::Vanadis::RoCCCommand* cmd, uint64_t cycle) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        const uint64_t start_array = cmd->rs1;
        const uint64_t count = cmd->rs2;
        if (count == 0 || start_array >= static_cast<uint64_t>(numArrays) || (start_array + count) > static_cast<uint64_t>(numArrays)) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }

        for (uint64_t idx = 0; idx < count; ++idx) {
            const uint32_t array_id = static_cast<uint32_t>(start_array + idx);
            if (hasArrayLoadFailure(array_id)) {
                enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
                delete cmd;
                return true;
            }
            if (isArrayLoadInflight(array_id) || isArrayComputeInflight(array_id)) {
                return false;
            }
            if (async_compute_states[array_id].submitted) {
                enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
                delete cmd;
                return true;
            }
        }

        for (uint64_t idx = 0; idx < count; ++idx) {
            const uint32_t array_id = static_cast<uint32_t>(start_array + idx);
            auto* array_cmd = new SST::Vanadis::RoCCCommand(cmd->inst, array_id, 0, cmd->cmd_id, cmd->hw_thread);
            auto& inflight = inflight_compute_cmds[array_id];
            inflight.cmd = array_cmd;
            inflight.start_cycle = cycle;
            inflight.async_mode = true;
            auto& async_state = async_compute_states[array_id];
            async_state.submitted = true;
            async_state.completed = false;
            async_state.rd_val = 0;
            arrayStates[array_id] = 1;
            array->beginComputation(array_id);
        }

        enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitBatchComputeCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        const uint64_t start_array = cmd->rs1;
        const uint64_t count = cmd->rs2;
        if (count == 0 || start_array >= static_cast<uint64_t>(numArrays) || (start_array + count) > static_cast<uint64_t>(numArrays)) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }

        uint64_t aggregate_rd_val = 0;
        for (uint64_t idx = 0; idx < count; ++idx) {
            const uint32_t array_id = static_cast<uint32_t>(start_array + idx);
            auto& async_state = async_compute_states[array_id];
            if (!async_state.submitted) {
                enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
                delete cmd;
                return true;
            }
            if (isArrayComputeInflight(array_id) || !async_state.completed) {
                return false;
            }
            aggregate_rd_val |= async_state.rd_val;
        }

        for (uint64_t idx = 0; idx < count; ++idx) {
            async_compute_states[static_cast<uint32_t>(start_array + idx)] = AsyncComputeState{};
        }

        enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, aggregate_rd_val, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryIssueBatchArrayLoadCommand(SST::Vanadis::RoCCCommand* cmd, uint64_t cycle, bool is_matrix) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        const uint64_t base_addr = cmd->rs1;
        const uint64_t count = cmd->rs2;
        if (count == 0 || count > static_cast<uint64_t>(numArrays)) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }

        const uint64_t vector_stride = static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(inputOperandSize);
        for (uint64_t idx = 0; idx < count; ++idx) {
            const uint32_t array_id = static_cast<uint32_t>(idx);
            auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
            if (state.inflight || isArrayComputeInflight(array_id)) {
                return false;
            }
        }

        for (uint64_t idx = 0; idx < count; ++idx) {
            const uint32_t array_id = static_cast<uint32_t>(idx);
            auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
            const uint64_t address =
                is_matrix ? base_addr : (base_addr + idx * vector_stride);
            const uint64_t total_size = is_matrix
                ? static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(arrayOutputSize) * static_cast<uint64_t>(inputOperandSize)
                : vector_stride;
            initializeAsyncArrayLoad(
                state, array_id, address, total_size, false, 0);
            progressAsyncArrayLoad(array_id, is_matrix);
        }

        enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryStartWorkerWindow(SST::Vanadis::RoCCCommand* cmd) {
        if (workerCommandProcessor == nullptr || cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (workerCommandProcessor->isBusy()) {
            return false;
        }
        std::vector<uint8_t> raw;
        globalMem->rd_from_globalmem(cmd->rs1, sizeof(WorkerTaskListHeader), raw);
        if (raw.size() < sizeof(WorkerTaskListHeader)) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        WorkerTaskListHeader header{};
        std::memcpy(&header, raw.data(), sizeof(header));
        if (header.block_n == 0 || header.block_n > numArrays ||
            header.block_k == 0 || (header.block_k % arrayInputSize) != 0 ||
            header.block_m != arrayOutputSize ||
            header.elem_bytes != inputOperandSize ||
            header.elem_bytes != outputOperandSize) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        const bool ok = workerCommandProcessor->startWindow(header);
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, ok ? 0 : 1, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return ok;
    }

    bool tryWaitWorkerWindow(SST::Vanadis::RoCCCommand* cmd) {
        if (workerCommandProcessor == nullptr || cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (workerCommandProcessor->isBusy()) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryIssueSfuSoftmaxTileCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (sfu == nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (!sfu->issueSoftmaxTile(cmd->rs1, cmd->rs2)) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitSfuCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (sfu == nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        uint64_t completionTick = 0;
        if (sfuWaitBlocked_ && cmd->cmd_id == sfuWaitBlockedCmdId_) {
            if (getCurrentSimCycle() < sfuWaitBlockedUntilTick_) {
                return false;
            }
            sfuWaitBlocked_ = false;
        }
        if (sfu->completionTick(cmd->rs1, &completionTick) &&
            getCurrentSimCycle() < completionTick) {
            sfuWaitBlocked_ = true;
            sfuWaitBlockedCmdId_ = cmd->cmd_id;
            sfuWaitBlockedUntilTick_ = completionTick;
            return false;
        }
        uint64_t status = 1;
        if (!sfu->wait(cmd->rs1, &status)) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, status, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryIssueSfuPrimitiveCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (sfu == nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (!sfu->issuePrimitive(cmd->rs1, cmd->rs2)) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitSfuPrimitiveCommand(SST::Vanadis::RoCCCommand* cmd) {
        return tryWaitSfuCommand(cmd);
    }

    bool tryIssueSfuPrimitiveBatchCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (sfu == nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (!sfu->issuePrimitiveBatch(cmd->rs1, cmd->rs2)) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitSfuPrimitiveBatchCommand(SST::Vanadis::RoCCCommand* cmd) {
        return tryWaitSfuCommand(cmd);
    }

    bool tryIssueSfuJobCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (sfu == nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (!sfu->issueJob(cmd->rs1, cmd->rs2)) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    enum class AttentionWorkerPhase : uint8_t {
        Idle,
        LoadingKv,
        LoadingQ,
        QkProgramMatrix,
        QkProgramInputs,
        QkCompute,
        QkReadOutputs,
        Softmax,
        PvProgramMatrix,
        PvProgramInputs,
        PvRestoreOutput,
        PvCompute,
        PvReadOutputs,
        OutputDma,
        Complete,
    };

    struct AttentionWorkerState {
        ReductionTransportMessage dispatch = {};
        AttentionWorkerPhase phase = AttentionWorkerPhase::Idle;
        uint64_t qLocal = 0;
        uint64_t kLocal = 0;
        uint64_t vLocal = 0;
        uint64_t spLocal = 0;
        uint64_t oLocal = 0;
        uint32_t queryBlock = 0;
        uint32_t keyTile = 0;
        uint32_t panel = 0;
        uint32_t index = 0;
        uint32_t lane = 0;
        uint32_t arraysPending = 0;
        std::vector<uint8_t> transferBytes;
        std::vector<double> arrayPayload;
        std::vector<uint8_t> readOutputBytes;
        std::vector<float> outputScales;
        uint64_t localAddr = 0;
        size_t localLength = 0;
        size_t localOffset = 0;
        bool localInflight = false;
        bool localWrite = false;
        std::function<void(bool, const std::vector<uint8_t>&)> localCallback;
    };

    uint64_t attentionTransferTag() { return allocateLocalTransferTag(); }

    bool attentionCausal(const AttentionWorkerState& state) const {
        return (state.dispatch.flags & GOLEM_ATTENTION_FLAG_CAUSAL) != 0;
    }

    uint32_t attentionQueryRows(const AttentionWorkerState& state) const {
        const uint32_t begin = state.queryBlock * state.dispatch.queryBlockRows;
        return std::min(state.dispatch.queryBlockRows,
                        state.dispatch.expectedRows - begin);
    }

    uint32_t attentionKeyCols(const AttentionWorkerState& state) const {
        const uint32_t begin = state.keyTile * state.dispatch.keyBlockRows;
        return std::min(state.dispatch.keyBlockRows,
                        state.dispatch.expectedCols - begin);
    }

    uint32_t attentionQueryBlocks(const AttentionWorkerState& state) const {
        return (state.dispatch.expectedRows + state.dispatch.queryBlockRows - 1) /
            state.dispatch.queryBlockRows;
    }

    uint32_t attentionKeyPanels(const AttentionWorkerState& state) const {
        return (attentionKeyCols(state) + 15) / 16;
    }

    uint32_t attentionDimensionPanels(const AttentionWorkerState& state) const {
        return (state.dispatch.headDim + 15) / 16;
    }

    uint32_t attentionPanelKeys(const AttentionWorkerState& state) const {
        return std::min<uint32_t>(16, attentionKeyCols(state) - state.panel * 16);
    }

    uint32_t attentionKeyTilesForQueryBlock(const AttentionWorkerState& state) const {
        const uint32_t totalKeyTiles =
            (state.dispatch.expectedCols + state.dispatch.keyBlockRows - 1) /
            state.dispatch.keyBlockRows;
        if (!attentionCausal(state)) return totalKeyTiles;
        const uint32_t queryEnd = state.dispatch.row + std::min(
            state.dispatch.expectedRows,
            (state.queryBlock + 1) * state.dispatch.queryBlockRows) - 1;
        return std::min(totalKeyTiles, queryEnd / state.dispatch.keyBlockRows + 1);
    }

    bool attentionStreamKv(const AttentionWorkerState& state) const {
        return state.dispatch.nodeStrideBytes != 0 && state.dispatch.rowsPerBand != 0;
    }

    uint64_t attentionKvHostAddr(const AttentionWorkerState& state,
                                 uint64_t tensorBase) const {
        const uint32_t keyBegin = state.keyTile * state.dispatch.keyBlockRows;
        const uint32_t nodeBand = keyBegin / state.dispatch.rowsPerBand;
        const uint32_t rowInBand = keyBegin % state.dispatch.rowsPerBand;
        return tensorBase + static_cast<uint64_t>(nodeBand) *
            state.dispatch.nodeStrideBytes + static_cast<uint64_t>(rowInBand) *
            state.dispatch.headDim * sizeof(float);
    }

    uint32_t attentionKvLocalKey(const AttentionWorkerState& state,
                                 uint32_t keyInTile) const {
        return attentionStreamKv(state) ? keyInTile :
            state.keyTile * state.dispatch.keyBlockRows + keyInTile;
    }

    void finishAttentionWorker(bool ok) {
        if (!attentionWorker_) return;
        if (!ok) {
            output->output(
                "Attention worker failure core=%" PRIu64 " phase=%u query_block=%u "
                "key_tile=%u panel=%u index=%u\n",
                coreID, static_cast<unsigned>(attentionWorker_->phase),
                attentionWorker_->queryBlock, attentionWorker_->keyTile,
                attentionWorker_->panel, attentionWorker_->index);
        }
        ReductionTransportMessage completion = attentionWorker_->dispatch;
        completion.kind = ReductionTransportMessageKind::AttentionComplete;
        completion.sendCycle = getCurrentSimCycle();
        completion.value = ok ? 1.0 : 0.0;
        attentionWorker_.reset();
        std::fill(attentionArrayPending_.begin(), attentionArrayPending_.end(), 0);
        if (!globalMem->sendReductionMessage(completion.ownerCore, completion)) {
            output->verbose(CALL_INFO, 1, 0, "Attention completion send failed\n");
        }
    }

    void issueAttentionLocalTransferChunk() {
        if (!attentionWorker_ || attentionWorker_->localInflight ||
            !attentionWorker_->localCallback) return;
        AttentionWorkerState& state = *attentionWorker_;
        const size_t remaining = state.localLength - state.localOffset;
        if (remaining == 0) {
            auto callback = std::move(state.localCallback);
            const std::vector<uint8_t> bytes = state.transferBytes;
            callback(true, bytes);
            return;
        }
        const size_t chunk = std::min(remaining, globalMem->localMaxRequestBytes());
        const uint64_t tag = attentionTransferTag();
        bool accepted = false;
        if (!state.localWrite) {
            accepted = globalMem->localReadAsync(
                state.localAddr + state.localOffset, chunk, LocalMemoryClient::RoCC, tag,
                [this, tag, chunk](bool ok, uint64_t callbackTag,
                                   const std::vector<uint8_t>& bytes) {
                    if (!attentionWorker_) return;
                    AttentionWorkerState& callbackState = *attentionWorker_;
                    callbackState.localInflight = false;
                    if (!ok || callbackTag != tag || bytes.size() != chunk) {
                        auto callback = std::move(callbackState.localCallback);
                        callback(false, {});
                        return;
                    }
                    callbackState.transferBytes.insert(
                        callbackState.transferBytes.end(), bytes.begin(), bytes.end());
                    callbackState.localOffset += chunk;
                    issueAttentionLocalTransferChunk();
                });
        } else {
            std::vector<uint8_t> bytes(
                state.transferBytes.begin() + state.localOffset,
                state.transferBytes.begin() + state.localOffset + chunk);
            accepted = globalMem->localWriteAsync(
                state.localAddr + state.localOffset, bytes, LocalMemoryClient::RoCC, tag,
                [this, tag, chunk](bool ok, uint64_t callbackTag) {
                    if (!attentionWorker_) return;
                    AttentionWorkerState& callbackState = *attentionWorker_;
                    callbackState.localInflight = false;
                    if (!ok || callbackTag != tag) {
                        auto callback = std::move(callbackState.localCallback);
                        callback(false, {});
                        return;
                    }
                    callbackState.localOffset += chunk;
                    issueAttentionLocalTransferChunk();
                });
        }
        if (accepted) state.localInflight = true;
    }

    void attentionLocalRead(uint64_t addr, size_t length,
                            std::function<void(bool, const std::vector<uint8_t>&)> callback) {
        AttentionWorkerState& state = *attentionWorker_;
        state.localAddr = addr;
        state.localLength = length;
        state.localOffset = 0;
        state.localInflight = false;
        state.localWrite = false;
        state.transferBytes.clear();
        state.localCallback = std::move(callback);
        issueAttentionLocalTransferChunk();
    }

    void attentionLocalWrite(uint64_t addr, const std::vector<uint8_t>& bytes,
                             std::function<void(bool)> callback) {
        AttentionWorkerState& state = *attentionWorker_;
        state.localAddr = addr;
        state.localLength = bytes.size();
        state.localOffset = 0;
        state.localInflight = false;
        state.localWrite = true;
        state.transferBytes = bytes;
        state.localCallback = [callback = std::move(callback)](
                                  bool ok, const std::vector<uint8_t>&) { callback(ok); };
        issueAttentionLocalTransferChunk();
    }

    std::vector<double> attentionBytesToDoubles(const std::vector<uint8_t>& bytes) const {
        const size_t count = bytes.size() / sizeof(float);
        std::vector<double> values(count, 0.0);
        for (size_t i = 0; i < count; ++i) {
            float value = 0.0f;
            std::memcpy(&value, bytes.data() + i * sizeof(float), sizeof(float));
            values[i] = value;
        }
        return values;
    }

    std::vector<uint8_t> attentionDoublesToBytes(const std::vector<double>& values) const {
        std::vector<uint8_t> bytes(values.size() * sizeof(float));
        for (size_t i = 0; i < values.size(); ++i) {
            const float value = static_cast<float>(values[i]);
            std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(float));
        }
        return bytes;
    }

    void beginAttentionQueryBlock() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::LoadingQ;
        state.panel = 0;
        state.keyTile = 0;
        const uint64_t qBytes = static_cast<uint64_t>(attentionQueryRows(state)) *
            state.dispatch.headDim * sizeof(float);
        globalMem->dma_read_from_host_to_globalmem(
            state.dispatch.qAddr + static_cast<uint64_t>(state.queryBlock) *
                state.dispatch.queryBlockRows * state.dispatch.headDim * sizeof(float),
            qBytes, state.qLocal,
            [this](bool ok) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                loadAttentionKeyTile();
            });
    }

    void beginAttentionKeyTile() {
        if (!attentionWorker_) return;
        attentionLocalRead(attentionWorker_->qLocal,
            static_cast<uint64_t>(attentionQueryRows(*attentionWorker_)) *
                attentionWorker_->dispatch.headDim * sizeof(float),
            [this](bool readOk, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !readOk) { finishAttentionWorker(false); return; }
                const std::vector<double> q = attentionBytesToDoubles(bytes);
                attentionWorker_->arrayPayload.assign(
                    static_cast<size_t>(arrayOutputSize) * arrayInputSize, 0.0);
                std::copy(q.begin(), q.end(), attentionWorker_->arrayPayload.begin());
                attentionWorker_->phase = AttentionWorkerPhase::QkProgramMatrix;
                attentionWorker_->panel = 0;
                attentionWorker_->index = 0;
                beginAttentionQkPanel();
            });
    }

    void loadAttentionKeyTile() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (!attentionStreamKv(state)) {
            beginAttentionKeyTile();
            return;
        }
        const uint64_t tileBytes = static_cast<uint64_t>(attentionKeyCols(state)) *
            state.dispatch.headDim * sizeof(float);
        globalMem->dma_read_from_host_to_globalmem(
            attentionKvHostAddr(state, state.dispatch.kAddr), tileBytes, state.kLocal,
            [this, tileBytes](bool ok) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                globalMem->dma_read_from_host_to_globalmem(
                    attentionKvHostAddr(*attentionWorker_, attentionWorker_->dispatch.vAddr),
                    tileBytes, attentionWorker_->vLocal,
                    [this](bool vOk) {
                        if (!attentionWorker_ || !vOk) {
                            finishAttentionWorker(false); return;
                        }
                        beginAttentionKeyTile();
                    });
            });
    }

    void beginAttentionQkPanel() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.phase == AttentionWorkerPhase::QkProgramMatrix) {
            if (state.index == attentionPanelKeys(state)) {
                state.phase = AttentionWorkerPhase::QkProgramInputs;
                state.index = 0;
                programAttentionQkInput();
                return;
            }
            const uint32_t arrayId = state.index;
            const uint64_t tag = attentionTransferTag();
            if (!array->programMatrixAsync(arrayId, state.arrayPayload, sizeof(float), tag,
                    [this, tag](bool ok, uint64_t callbackTag) {
                        if (!attentionWorker_ || !ok || callbackTag != tag) {
                            finishAttentionWorker(false); return;
                        }
                        attentionWorker_->index += 1;
                        beginAttentionQkPanel();
                    })) finishAttentionWorker(false);
            return;
        }
    }

    void programAttentionQkInput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t activeArrays = attentionPanelKeys(state);
        if (state.index == activeArrays) {
            state.phase = AttentionWorkerPhase::QkCompute;
            state.arraysPending = activeArrays;
            if (attentionArrayPending_.size() < 16) attentionArrayPending_.resize(16, 0);
            for (uint32_t arrayId = 0; arrayId < activeArrays; ++arrayId) {
                array->configureOutputMode(arrayId, 0);
                attentionArrayPending_[arrayId] = 1;
                arrayStates[arrayId] = 1;
                array->beginComputation(arrayId);
                statAttentionQkArrayOps_->addData(1);
            }
            return;
        }
        const uint32_t arrayId = state.index;
        const uint32_t key = attentionKvLocalKey(state, state.panel * 16 + arrayId);
        attentionLocalRead(state.kLocal + static_cast<uint64_t>(key) *
                state.dispatch.headDim * sizeof(float),
            state.dispatch.headDim * sizeof(float),
            [this, arrayId](bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                const std::vector<double> input = attentionBytesToDoubles(bytes);
                const uint64_t tag = attentionTransferTag();
                if (!array->programInputAsync(arrayId, input, sizeof(float), tag,
                        [this, tag](bool programOk, uint64_t callbackTag) {
                            if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                finishAttentionWorker(false); return;
                            }
                            attentionWorker_->index += 1;
                            programAttentionQkInput();
                        })) finishAttentionWorker(false);
            });
    }

    void readAttentionQkOutput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.index == attentionPanelKeys(state)) {
            if (++state.panel < attentionKeyPanels(state)) {
                state.phase = AttentionWorkerPhase::QkProgramInputs;
                state.index = 0;
                programAttentionQkInput();
            } else {
                statAttentionWorkerQkTileCompleteTick_->addData(getCurrentSimCycle());
                beginAttentionSoftmax();
            }
            return;
        }
        const uint32_t arrayId = state.index;
        const uint64_t tag = attentionTransferTag();
        if (!array->readOutputAsync(arrayId, sizeof(float), tag,
                [this, tag, arrayId](bool ok, uint64_t callbackTag,
                                     const std::vector<double>& values) {
                    if (!attentionWorker_ || !ok || callbackTag != tag || values.size() != 16) {
                        finishAttentionWorker(false); return;
                    }
                    attentionWorker_->readOutputBytes = attentionDoublesToBytes(values);
                    attentionWorker_->lane = 0;
                    const auto issueLane = [this, arrayId](auto&& self) -> void {
                        if (!attentionWorker_) return;
                        AttentionWorkerState& laneState = *attentionWorker_;
                        if (laneState.lane == attentionQueryRows(laneState)) {
                            laneState.index += 1;
                            readAttentionQkOutput();
                            return;
                        }
                        const uint32_t query = laneState.lane;
                        const uint32_t key = laneState.panel * 16 + arrayId;
                        std::vector<uint8_t> scalar(
                            laneState.readOutputBytes.begin() + query * sizeof(float),
                            laneState.readOutputBytes.begin() + (query + 1) * sizeof(float));
                        const uint64_t addr = laneState.spLocal +
                            (static_cast<uint64_t>(query) * attentionKeyCols(laneState) + key) *
                                sizeof(float);
                        attentionLocalWrite(addr, scalar, [this, self](bool writeOk) mutable {
                            if (!attentionWorker_ || !writeOk) {
                                finishAttentionWorker(false); return;
                            }
                            attentionWorker_->lane += 1;
                            self(self);
                        });
                    };
                    issueLane(issueLane);
                })) finishAttentionWorker(false);
    }

    void beginAttentionSoftmax() {
        if (!attentionWorker_ || sfu == nullptr) { finishAttentionWorker(false); return; }
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::Softmax;
        AttentionTileRequest request;
        request.tag = state.dispatch.tag +
            state.queryBlock * ((state.dispatch.expectedCols +
                state.dispatch.keyBlockRows - 1) / state.dispatch.keyBlockRows) +
            state.keyTile + 1;
        request.jobId = state.dispatch.jobId;
        request.localScoreAddr = state.spLocal;
        request.globalRowBegin = state.dispatch.row +
            state.queryBlock * state.dispatch.queryBlockRows;
        request.keyBegin = state.keyTile * state.dispatch.keyBlockRows;
        request.rows = attentionQueryRows(state);
        request.cols = attentionKeyCols(state);
        request.headDim = state.dispatch.headDim;
        request.keyTile = state.keyTile;
        request.keyTiles = attentionKeyTilesForQueryBlock(state);
        request.causal = attentionCausal(state);
        request.firstTileForJob = state.queryBlock == 0 && state.keyTile == 0;
        if (!sfu->issueAttentionTile(request, [this](
                bool ok, const AttentionTileResult& result) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                statAttentionWorkerSoftmaxTileCompleteTick_->addData(getCurrentSimCycle());
                attentionWorker_->outputScales.assign(
                    result.oldOutputScale.begin(),
                    result.oldOutputScale.begin() + result.rows);
                attentionWorker_->panel = 0;
                beginAttentionPvPanel();
            })) finishAttentionWorker(false);
    }

    void beginAttentionPvPanel() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::PvProgramMatrix;
        const uint32_t keyCols = attentionKeyCols(state);
        attentionLocalRead(
            state.vLocal + (attentionStreamKv(state) ? 0 :
                static_cast<uint64_t>(state.keyTile) *
                    state.dispatch.keyBlockRows * state.dispatch.headDim * sizeof(float)),
            static_cast<uint64_t>(keyCols) * state.dispatch.headDim * sizeof(float),
            [this](bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                const std::vector<double> v = attentionBytesToDoubles(bytes);
                AttentionWorkerState& callbackState = *attentionWorker_;
                callbackState.arrayPayload.assign(
                    static_cast<size_t>(arrayOutputSize) * arrayInputSize, 0.0);
                for (uint32_t dim = 0; dim < 16; ++dim) {
                    for (uint32_t key = 0; key < attentionKeyCols(callbackState); ++key) {
                        callbackState.arrayPayload[dim * arrayInputSize + key] =
                            v[key * callbackState.dispatch.headDim +
                              callbackState.panel * 16 + dim];
                    }
                }
                if (attentionPvMatrixBroadcast_) {
                    std::vector<uint32_t> arrayIDs(attentionQueryRows(callbackState));
                    std::iota(arrayIDs.begin(), arrayIDs.end(), 0);
                    const uint64_t tag = attentionTransferTag();
                    if (!array->programMatrixGroupAsync(
                            arrayIDs, callbackState.arrayPayload, sizeof(float), tag,
                            [this, tag](bool programOk, uint64_t callbackTag) {
                                if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                    finishAttentionWorker(false); return;
                                }
                                attentionWorker_->phase =
                                    AttentionWorkerPhase::PvProgramInputs;
                                attentionWorker_->index = 0;
                                programAttentionPvInput();
                            })) {
                        finishAttentionWorker(false);
                    } else {
                        statAttentionPvMatrixBroadcasts_->addData(1);
                    }
                    return;
                }
                callbackState.index = 0;
                const auto programMatrix = [this](auto&& self) -> void {
                    if (!attentionWorker_) return;
                    AttentionWorkerState& matrixState = *attentionWorker_;
                    if (matrixState.index == attentionQueryRows(matrixState)) {
                        matrixState.phase = AttentionWorkerPhase::PvProgramInputs;
                        matrixState.index = 0;
                        programAttentionPvInput();
                        return;
                    }
                    const uint32_t arrayId = matrixState.index;
                    const uint64_t tag = attentionTransferTag();
                    if (!array->programMatrixAsync(arrayId, matrixState.arrayPayload,
                            sizeof(float), tag, [this, tag, self](bool programOk,
                                                                 uint64_t callbackTag) mutable {
                                if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                    finishAttentionWorker(false); return;
                                }
                                attentionWorker_->index += 1;
                                self(self);
                            })) finishAttentionWorker(false);
                };
                programMatrix(programMatrix);
            });
    }

    void programAttentionPvInput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.index == attentionQueryRows(state)) {
            state.phase = AttentionWorkerPhase::PvRestoreOutput;
            state.index = 0;
            prepareAttentionPvOutput();
            return;
        }
        const uint32_t arrayId = state.index;
        const uint32_t keyCols = attentionKeyCols(state);
        attentionLocalRead(state.spLocal + static_cast<uint64_t>(arrayId) * keyCols * sizeof(float),
            keyCols * sizeof(float), [this, arrayId](bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                std::vector<double> input(arrayInputSize, 0.0);
                const std::vector<double> p = attentionBytesToDoubles(bytes);
                std::copy(p.begin(), p.end(), input.begin());
                const uint64_t tag = attentionTransferTag();
                if (!array->programInputAsync(arrayId, input, sizeof(float), tag,
                        [this, tag](bool programOk, uint64_t callbackTag) {
                            if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                finishAttentionWorker(false); return;
                            }
                            attentionWorker_->index += 1;
                            programAttentionPvInput();
                        })) finishAttentionWorker(false);
            });
    }

    void startAttentionPvComputation() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::PvCompute;
        const uint32_t activeArrays = attentionQueryRows(state);
        state.arraysPending = activeArrays;
        for (uint32_t arrayId = 0; arrayId < activeArrays; ++arrayId) {
            attentionArrayPending_[arrayId] = 1;
            arrayStates[arrayId] = 1;
            array->beginComputation(arrayId);
            statAttentionPvArrayOps_->addData(1);
        }
    }

    void prepareAttentionPvOutput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.keyTile == 0) {
            for (uint32_t arrayId = 0; arrayId < attentionQueryRows(state); ++arrayId) {
                array->configureOutputMode(arrayId, 0);
            }
            startAttentionPvComputation();
            return;
        }
        if (state.index == attentionQueryRows(state)) {
            startAttentionPvComputation();
            return;
        }
        const uint32_t arrayId = state.index;
        const uint64_t addr = state.oLocal +
            (static_cast<uint64_t>(arrayId) * state.dispatch.headDim +
             state.panel * 16) * sizeof(float);
        attentionLocalRead(addr, 16 * sizeof(float),
            [this, arrayId](bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !ok ||
                    attentionWorker_->outputScales.size() !=
                        attentionQueryRows(*attentionWorker_)) {
                    finishAttentionWorker(false); return;
                }
                std::vector<double> output = attentionBytesToDoubles(bytes);
                const double scale = attentionWorker_->outputScales[arrayId];
                for (double& value : output) value *= scale;
                const uint64_t tag = attentionTransferTag();
                if (!array->writeOutputAsync(arrayId, output, sizeof(float), tag,
                        [this, tag, arrayId](bool writeOk, uint64_t callbackTag) {
                            if (!attentionWorker_ || !writeOk || callbackTag != tag) {
                                finishAttentionWorker(false); return;
                            }
                            array->configureOutputMode(arrayId, 1);
                            attentionWorker_->index += 1;
                            prepareAttentionPvOutput();
                        })) finishAttentionWorker(false);
            });
    }

    void readAttentionPvOutput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.index == attentionQueryRows(state)) {
            if (++state.panel < attentionDimensionPanels(state)) {
                beginAttentionPvPanel();
            } else {
                statAttentionWorkerPvTileCompleteTick_->addData(getCurrentSimCycle());
                const uint32_t keyTiles = attentionKeyTilesForQueryBlock(state);
                if (++state.keyTile < keyTiles) {
                    loadAttentionKeyTile();
                    return;
                }
                state.phase = AttentionWorkerPhase::OutputDma;
                const uint64_t blockBytes = static_cast<uint64_t>(attentionQueryRows(state)) *
                    state.dispatch.headDim * sizeof(float);
                globalMem->dma_write_from_globalmem_to_host(
                    state.oLocal, state.dispatch.oAddr +
                        static_cast<uint64_t>(state.queryBlock) *
                            state.dispatch.queryBlockRows * state.dispatch.headDim * sizeof(float),
                    blockBytes, [this](bool ok) {
                        if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                        statAttentionWorkerOutputDmaAckTick_->addData(getCurrentSimCycle());
                        AttentionWorkerState& state = *attentionWorker_;
                        const uint32_t queryBlocks = attentionQueryBlocks(state);
                        if (++state.queryBlock < queryBlocks) beginAttentionQueryBlock();
                        else finishAttentionWorker(true);
                    });
            }
            return;
        }
        const uint32_t arrayId = state.index;
        const uint64_t tag = attentionTransferTag();
        if (!array->readOutputAsync(arrayId, sizeof(float), tag,
                [this, tag, arrayId](bool ok, uint64_t callbackTag,
                                     const std::vector<double>& values) {
                    if (!attentionWorker_ || !ok || callbackTag != tag || values.size() != 16) {
                        finishAttentionWorker(false); return;
                    }
                    const std::vector<uint8_t> bytes = attentionDoublesToBytes(values);
                    const uint64_t addr = attentionWorker_->oLocal +
                        (static_cast<uint64_t>(arrayId) *
                             attentionWorker_->dispatch.headDim +
                         attentionWorker_->panel * 16) * sizeof(float);
                    attentionLocalWrite(addr, bytes, [this](bool writeOk) {
                        if (!attentionWorker_ || !writeOk) {
                            finishAttentionWorker(false); return;
                        }
                        attentionWorker_->index += 1;
                        readAttentionPvOutput();
                    });
                })) finishAttentionWorker(false);
    }

    bool handleAttentionArrayDone(uint32_t arrayId) {
        if (!attentionWorker_ || arrayId >= attentionArrayPending_.size() ||
            attentionArrayPending_[arrayId] == 0) return false;
        attentionArrayPending_[arrayId] = 0;
        arrayStates[arrayId] = 0;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.arraysPending == 0) { finishAttentionWorker(false); return true; }
        state.arraysPending -= 1;
        if (state.arraysPending == 0) {
            state.index = 0;
            if (state.phase == AttentionWorkerPhase::QkCompute) {
                state.phase = AttentionWorkerPhase::QkReadOutputs;
                readAttentionQkOutput();
            } else if (state.phase == AttentionWorkerPhase::PvCompute) {
                state.phase = AttentionWorkerPhase::PvReadOutputs;
                readAttentionPvOutput();
            } else {
                finishAttentionWorker(false);
            }
        }
        return true;
    }

    void startAttentionWorker(const ReductionTransportMessage& message) {
        const bool c1Shape = message.expectedRows == 32 && message.expectedCols == 32;
        const bool d1Shape = message.expectedRows == 64 && message.expectedCols == 64;
        const bool d3Shape = message.expectedRows == 20 && message.expectedCols == 70;
        const bool e1Shape = message.expectedRows == 16 && message.expectedCols == 256 &&
            message.rowsPerBand == 64 && message.headDim == 64 &&
            message.nodeStrideBytes != 0;
        const bool e3Shape = message.expectedRows == 64 && message.expectedCols == 1024 &&
            message.rowsPerBand == 256 && message.headDim == 128 &&
            message.nodeStrideBytes != 0;
        const bool e4Shape = message.expectedRows == 128 && message.expectedCols == 2048 &&
            message.rowsPerBand == 512 && message.headDim == 128 &&
            message.nodeStrideBytes != 0;
        const bool e5Shape = message.expectedRows == 256 && message.expectedCols == 4096 &&
            message.rowsPerBand == 1024 && message.headDim == 128 &&
            message.nodeStrideBytes != 0;
        const uint64_t requiredWindow = d3Shape ? ATTENTION_D3_WINDOW_BYTES :
            (d1Shape ? ATTENTION_D1_WINDOW_BYTES :
             ((e3Shape || e4Shape || e5Shape) ? ATTENTION_E3_WINDOW_BYTES :
              (e1Shape ? ATTENTION_E1_WINDOW_BYTES : ATTENTION_C1_WINDOW_BYTES)));
        if (attentionWorker_ || globalMem == nullptr || array == nullptr || sfu == nullptr ||
            message.workerCore != coreID ||
            (!c1Shape && !d1Shape && !d3Shape && !e1Shape && !e3Shape && !e4Shape &&
             !e5Shape) ||
            ((!(e3Shape || e4Shape || e5Shape) && message.headDim != 64) ||
             ((e3Shape || e4Shape || e5Shape) && message.headDim != 128)) ||
            message.queryBlockRows != 16 || message.keyBlockRows != 32 ||
            (message.flags & ~GOLEM_ATTENTION_FLAG_CAUSAL) != 0 ||
            numArrays < 16 || arrayInputSize != static_cast<int>(message.headDim) ||
            arrayOutputSize != 16 || attentionWindowBytes_ < requiredWindow ||
            attentionWindowOffset_ + requiredWindow > globalMem->getSize()) {
            ReductionTransportMessage rejected = message;
            rejected.kind = ReductionTransportMessageKind::AttentionComplete;
            rejected.value = 0.0;
            globalMem->sendReductionMessage(message.ownerCore, rejected);
            return;
        }
        attentionWorker_ = std::make_unique<AttentionWorkerState>();
        AttentionWorkerState& state = *attentionWorker_;
        state.dispatch = message;
        statAttentionWorkerDispatchAcceptTick_->addData(getCurrentSimCycle());
        state.phase = AttentionWorkerPhase::LoadingKv;
        const uint64_t base = globalMem->getBaseAddr() + attentionWindowOffset_;
        state.qLocal = base;
        const uint64_t qTileBytes = static_cast<uint64_t>(message.queryBlockRows) *
            message.headDim * sizeof(float);
        state.kLocal = state.qLocal + qTileBytes;
        const uint64_t kvBytes = (e1Shape || e3Shape || e4Shape || e5Shape) ?
            static_cast<uint64_t>(message.keyBlockRows) * message.headDim * sizeof(float) :
            static_cast<uint64_t>(message.expectedCols) * message.headDim * sizeof(float);
        state.vLocal = state.kLocal + kvBytes;
        state.spLocal = state.vLocal + kvBytes;
        state.oLocal = state.spLocal + static_cast<uint64_t>(message.queryBlockRows) *
            message.keyBlockRows * sizeof(float);
        attentionArrayPending_.assign(numArrays, 0);
        if (e1Shape || e3Shape || e4Shape || e5Shape) {
            beginAttentionQueryBlock();
            return;
        }
        globalMem->dma_read_from_host_to_globalmem(
            message.kAddr, kvBytes, state.kLocal,
            [this](bool ok) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                globalMem->dma_read_from_host_to_globalmem(
                    attentionWorker_->dispatch.vAddr,
                    static_cast<uint64_t>(attentionWorker_->dispatch.expectedCols) *
                        attentionWorker_->dispatch.headDim * sizeof(float),
                    attentionWorker_->vLocal, [this](bool vOk) {
                        if (!attentionWorker_ || !vOk) { finishAttentionWorker(false); return; }
                        beginAttentionQueryBlock();
                    });
            });
    }

    void progressAttentionWorker() {
        if (attentionWorker_ && attentionWorker_->localCallback &&
            !attentionWorker_->localInflight &&
            attentionWorker_->localOffset < attentionWorker_->localLength) {
            issueAttentionLocalTransferChunk();
        }
    }

    enum class ManagerAttentionJobPhase : uint8_t {
        ReadDescriptor,
        ReadTopology,
        Dispatch,
        Running,
        Complete,
    };

    struct ManagerAttentionJobState {
        uint64_t tag = 0;
        uint64_t descAddr = 0;
        GolemAttentionDescV1 desc = {};
        std::array<uint32_t, SFU_WORKER_TOPOLOGY_MAX_WORKERS> workerCoreIds = {};
        uint32_t workersDispatched = 0;
        uint32_t workersCompleted = 0;
        uint32_t completionBitmap = 0;
        uint32_t managersCompleted = 0;
        uint32_t managerCompletionBitmap = 0;
        SFUStatus status = SFUStatus::Pending;
        ManagerAttentionJobPhase phase = ManagerAttentionJobPhase::ReadDescriptor;
        bool readInflight = false;
        uint64_t readTag = 0;
    };

    void failManagerAttentionJob(ManagerAttentionJobState& state, SFUStatus status) {
        state.status = status;
        state.readInflight = false;
        state.phase = ManagerAttentionJobPhase::Complete;
    }

    bool tryIssueManagerAttentionJobCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) return true;
        const uint64_t tag = cmd->rs2;
        uint64_t response = 0;
        if (globalMem == nullptr || managerAttentionJobs_.count(tag) != 0) {
            response = static_cast<uint64_t>(SFUStatus::InvalidDescriptor);
        } else {
            ManagerAttentionJobState state;
            state.tag = tag;
            state.descAddr = cmd->rs1;
            managerAttentionJobs_.emplace(tag, std::move(state));
            statAttentionManagerJobsIssued_->addData(1);
            statAttentionManagerDescriptorAcceptTick_->addData(getCurrentSimCycle());
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, response, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitManagerAttentionJobCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) return true;
        auto it = managerAttentionJobs_.find(cmd->rs1);
        if (it == managerAttentionJobs_.end()) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, static_cast<uint64_t>(SFUStatus::InvalidDescriptor),
                cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (it->second.phase != ManagerAttentionJobPhase::Complete) return false;
        const uint64_t status = static_cast<uint64_t>(it->second.status);
        statAttentionManagerWaitObservedTick_->addData(getCurrentSimCycle());
        managerAttentionJobs_.erase(it);
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, status, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool validateManagerAttentionDescriptor(const GolemAttentionDescV1& desc) const {
        const bool c1Shape = desc.queries == 32 && desc.keys == 32;
        const bool d1Shape = desc.queries == 64 && desc.keys == 64;
        const bool d3Shape = desc.queries == 20 && desc.keys == 70;
        const bool e1Shape = desc.queries == 64 && desc.keys == 256 &&
            desc.worker_count == 4 && desc.query_row_begin % 64 == 0 &&
            desc.query_row_begin < 256 && desc.kv_rows_per_node == 64 &&
            desc.kv_node_stride_bytes != 0 && desc.flags == 0 &&
            desc.tensor_root_core == 0 && desc.tensor_manager_count == 4 &&
            desc.tensor_manager_slot < desc.tensor_manager_count &&
            desc.tensor_manager_slot == coreID;
        const bool e3Shape = desc.queries == 256 && desc.keys == 1024 &&
            desc.head_dim == 128 && desc.worker_count == 4 &&
            desc.query_row_begin % 256 == 0 && desc.query_row_begin < 1024 &&
            desc.kv_rows_per_node == 256 && desc.kv_node_stride_bytes != 0 &&
            desc.flags == 0 && desc.tensor_root_core == 0 &&
            desc.tensor_manager_count == 4 &&
            desc.tensor_manager_slot < desc.tensor_manager_count &&
            desc.tensor_manager_slot == coreID;
        const bool e4Shape = desc.queries == 512 && desc.keys == 2048 &&
            desc.head_dim == 128 && desc.worker_count == 4 &&
            desc.query_row_begin % 512 == 0 && desc.query_row_begin < 2048 &&
            desc.kv_rows_per_node == 512 && desc.kv_node_stride_bytes != 0 &&
            desc.flags == 0 && desc.tensor_root_core == 0 &&
            desc.tensor_manager_count == 4 &&
            desc.tensor_manager_slot < desc.tensor_manager_count &&
            desc.tensor_manager_slot == coreID;
        const bool e5Shape = desc.queries == 1024 && desc.keys == 4096 &&
            desc.head_dim == 128 && desc.worker_count == 4 &&
            desc.query_row_begin % 1024 == 0 && desc.query_row_begin < 4096 &&
            desc.kv_rows_per_node == 1024 && desc.kv_node_stride_bytes != 0 &&
            desc.flags == 0 && desc.tensor_root_core == 0 &&
            desc.tensor_manager_count == 4 &&
            desc.tensor_manager_slot < desc.tensor_manager_count &&
            desc.tensor_manager_slot == coreID;
        const uint64_t requiredWindow = d3Shape ? ATTENTION_D3_WINDOW_BYTES :
            (d1Shape ? ATTENTION_D1_WINDOW_BYTES :
             ((e3Shape || e4Shape || e5Shape) ? ATTENTION_E3_WINDOW_BYTES :
              (e1Shape ? ATTENTION_E1_WINDOW_BYTES : ATTENTION_C1_WINDOW_BYTES)));
        return desc.magic == GOLEM_ATTENTION_DESC_MAGIC &&
            desc.version == GOLEM_ATTENTION_DESC_VERSION &&
            desc.size_bytes == sizeof(GolemAttentionDescV1) &&
            (c1Shape || d1Shape || d3Shape || e1Shape || e3Shape || e4Shape || e5Shape) &&
            ((!(e3Shape || e4Shape || e5Shape) && desc.head_dim == 64) ||
             ((e3Shape || e4Shape || e5Shape) && desc.head_dim == 128)) &&
            desc.query_block_rows == 16 && desc.key_block_rows == 32 &&
            (((e1Shape || e3Shape || e4Shape || e5Shape) && desc.worker_count == 4) ||
             (!e1Shape && !e3Shape && !e4Shape && !e5Shape && desc.worker_count == 1)) &&
            (desc.flags & ~GOLEM_ATTENTION_FLAG_CAUSAL) == 0 &&
            desc.q_addr != 0 && desc.k_addr != 0 && desc.v_addr != 0 &&
            desc.output_addr != 0 && desc.topology_gm_addr != 0 &&
            attentionWindowBytes_ >= requiredWindow;
    }

    void progressManagerAttentionJobs() {
        if (globalMem == nullptr) return;
        for (auto& entry : managerAttentionJobs_) {
            ManagerAttentionJobState& state = entry.second;
            if (state.readInflight || state.phase == ManagerAttentionJobPhase::Running ||
                state.phase == ManagerAttentionJobPhase::Complete) continue;
            if (state.phase == ManagerAttentionJobPhase::ReadDescriptor) {
                const uint64_t jobTag = state.tag;
                const uint64_t readTag = allocateLocalTransferTag();
                const bool accepted = globalMem->localReadAsync(
                    state.descAddr, sizeof(GolemAttentionDescV1), LocalMemoryClient::Control,
                    readTag, [this, jobTag, readTag](bool ok, uint64_t tag,
                                                     const std::vector<uint8_t>& bytes) {
                        auto it = managerAttentionJobs_.find(jobTag);
                        if (it == managerAttentionJobs_.end()) return;
                        ManagerAttentionJobState& callbackState = it->second;
                        callbackState.readInflight = false;
                        if (!ok || tag != readTag || bytes.size() != sizeof(GolemAttentionDescV1)) {
                            failManagerAttentionJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        std::memcpy(&callbackState.desc, bytes.data(), sizeof(callbackState.desc));
                        if (!validateManagerAttentionDescriptor(callbackState.desc)) {
                            failManagerAttentionJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        callbackState.phase = ManagerAttentionJobPhase::ReadTopology;
                    });
                if (accepted) {
                    state.readInflight = true;
                    state.readTag = readTag;
                }
            } else if (state.phase == ManagerAttentionJobPhase::ReadTopology) {
                const uint64_t jobTag = state.tag;
                const uint64_t readTag = allocateLocalTransferTag();
                const bool accepted = globalMem->localReadAsync(
                    state.desc.topology_gm_addr, sizeof(SFUWorkerTopologyMapV1),
                    LocalMemoryClient::Control, readTag,
                    [this, jobTag, readTag](bool ok, uint64_t tag,
                                            const std::vector<uint8_t>& bytes) {
                        auto it = managerAttentionJobs_.find(jobTag);
                        if (it == managerAttentionJobs_.end()) return;
                        ManagerAttentionJobState& callbackState = it->second;
                        callbackState.readInflight = false;
                        SFUWorkerTopologyMapV1 topology = {};
                        if (!ok || tag != readTag || bytes.size() != sizeof(topology)) {
                            failManagerAttentionJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        std::memcpy(&topology, bytes.data(), sizeof(topology));
                        if (topology.magic != SFU_WORKER_TOPOLOGY_MAP_MAGIC ||
                            topology.version != SFU_WORKER_TOPOLOGY_MAP_VERSION ||
                            topology.size_bytes != sizeof(topology) ||
                            topology.worker_count != callbackState.desc.worker_count ||
                            topology.worker_count == 0 ||
                            topology.worker_count > SFU_WORKER_TOPOLOGY_MAX_WORKERS) {
                            failManagerAttentionJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        for (uint32_t slot = 0; slot < topology.worker_count; ++slot) {
                            const uint32_t workerCore = topology.worker_core_ids[slot];
                            if (workerCore == coreID) {
                                failManagerAttentionJob(
                                    callbackState, SFUStatus::InvalidDescriptor);
                                return;
                            }
                            for (uint32_t prior = 0; prior < slot; ++prior) {
                                if (topology.worker_core_ids[prior] == workerCore) {
                                    failManagerAttentionJob(
                                        callbackState, SFUStatus::InvalidDescriptor);
                                    return;
                                }
                            }
                            callbackState.workerCoreIds[slot] = workerCore;
                        }
                        callbackState.phase = ManagerAttentionJobPhase::Dispatch;
                    });
                if (accepted) {
                    state.readInflight = true;
                    state.readTag = readTag;
                }
            } else if (state.phase == ManagerAttentionJobPhase::Dispatch) {
                const uint32_t workerSlot = state.workersDispatched;
                if (workerSlot >= state.desc.worker_count) {
                    state.phase = ManagerAttentionJobPhase::Running;
                    continue;
                }
                const uint32_t workerCore = state.workerCoreIds[workerSlot];
                const uint32_t rowsPerWorker = state.desc.worker_count == 1 ?
                    state.desc.queries :
                    (state.desc.queries + state.desc.worker_count - 1) /
                        state.desc.worker_count;
                const uint32_t localQueryBegin = workerSlot * rowsPerWorker;
                const uint32_t workerRows = state.desc.worker_count == 1 ? state.desc.queries :
                    std::min(rowsPerWorker,
                             state.desc.queries - localQueryBegin);
                ReductionTransportMessage message = {};
                message.kind = ReductionTransportMessageKind::AttentionDispatch;
                message.jobId = state.desc.job_id;
                message.tag = state.tag;
                message.ownerCore = static_cast<uint32_t>(coreID);
                message.workerSlot = workerSlot;
                message.workerCore = workerCore;
                message.row = state.desc.query_row_begin + localQueryBegin;
                message.expectedWorkers = state.desc.worker_count;
                message.expectedRows = workerRows;
                message.expectedCols = state.desc.keys;
                message.headDim = state.desc.head_dim;
                message.queryBlockRows = state.desc.query_block_rows;
                message.keyBlockRows = state.desc.key_block_rows;
                message.flags = state.desc.flags;
                message.nodeStrideBytes = state.desc.kv_node_stride_bytes;
                message.rowsPerBand = state.desc.kv_rows_per_node;
                message.qAddr = state.desc.q_addr +
                    static_cast<uint64_t>(localQueryBegin) * state.desc.head_dim * sizeof(float);
                message.kAddr = state.desc.k_addr;
                message.vAddr = state.desc.v_addr;
                message.oAddr = state.desc.output_addr +
                    static_cast<uint64_t>(localQueryBegin) * state.desc.head_dim * sizeof(float);
                message.sendCycle = getCurrentSimCycle();
                if (!globalMem->sendReductionMessage(workerCore, message)) {
                    failManagerAttentionJob(state, SFUStatus::GlobalMemoryUnavailable);
                } else {
                    state.workersDispatched += 1;
                    if (state.workersDispatched == state.desc.worker_count) {
                        state.phase = ManagerAttentionJobPhase::Running;
                        statAttentionManagerDispatchTick_->addData(getCurrentSimCycle());
                    }
                }
            }
        }
    }

    bool handleManagerAttentionCompletion(const ReductionTransportMessage& message) {
        if (message.kind != ReductionTransportMessageKind::AttentionComplete ||
            message.ownerCore != coreID) return false;
        auto it = managerAttentionJobs_.find(message.tag);
        if (it == managerAttentionJobs_.end()) return false;
        ManagerAttentionJobState& state = it->second;
        const uint32_t workerSlot = message.workerSlot;
        const bool validPhase = state.phase == ManagerAttentionJobPhase::Dispatch ||
            state.phase == ManagerAttentionJobPhase::Running;
        if (!validPhase || message.jobId != state.desc.job_id ||
            message.expectedWorkers != state.desc.worker_count ||
            workerSlot >= state.desc.worker_count ||
            message.workerCore != state.workerCoreIds[workerSlot] ||
            (state.completionBitmap & (1u << workerSlot)) != 0 || message.value != 1.0) {
            failManagerAttentionJob(state, SFUStatus::InvalidDescriptor);
            return true;
        }
        state.completionBitmap |= 1u << workerSlot;
        state.workersCompleted += 1;
        if (state.workersCompleted == state.desc.worker_count) {
            statAttentionManagerBandsCompleted_->addData(1);
            statAttentionManagerLocalCompleteTick_->addData(getCurrentSimCycle());
            if (state.desc.tensor_manager_count <= 1) {
                state.status = SFUStatus::Success;
                state.phase = ManagerAttentionJobPhase::Complete;
                statAttentionManagerJobsCompleted_->addData(1);
                statAttentionTensorJobsCompleted_->addData(1);
                statAttentionTensorCompleteTick_->addData(getCurrentSimCycle());
            } else {
                ReductionTransportMessage managerCompletion = {};
                managerCompletion.kind =
                    ReductionTransportMessageKind::AttentionManagerComplete;
                managerCompletion.jobId = state.desc.job_id;
                managerCompletion.tag = state.tag;
                managerCompletion.ownerCore = state.desc.tensor_root_core;
                managerCompletion.workerSlot = state.desc.tensor_manager_slot;
                managerCompletion.workerCore = static_cast<uint32_t>(coreID);
                managerCompletion.expectedWorkers = state.desc.tensor_manager_count;
                managerCompletion.sendCycle = getCurrentSimCycle();
                managerCompletion.value = 1.0;
                if (coreID == state.desc.tensor_root_core) {
                    handleAttentionManagerBandCompletion(managerCompletion);
                } else if (!globalMem->sendReductionMessage(
                               state.desc.tensor_root_core, managerCompletion)) {
                    failManagerAttentionJob(
                        state, SFUStatus::GlobalMemoryUnavailable);
                } else {
                    state.status = SFUStatus::Success;
                    state.phase = ManagerAttentionJobPhase::Complete;
                    statAttentionManagerJobsCompleted_->addData(1);
                }
            }
        }
        return true;
    }

    bool handleAttentionManagerBandCompletion(
        const ReductionTransportMessage& message) {
        if (message.kind != ReductionTransportMessageKind::AttentionManagerComplete) {
            return false;
        }
        auto it = managerAttentionJobs_.find(message.tag);
        if (it == managerAttentionJobs_.end()) return true;
        ManagerAttentionJobState& state = it->second;
        const uint32_t managerSlot = message.workerSlot;
        if (coreID != state.desc.tensor_root_core || message.ownerCore != coreID ||
            message.jobId != state.desc.job_id ||
            message.expectedWorkers != state.desc.tensor_manager_count ||
            managerSlot >= state.desc.tensor_manager_count ||
            message.workerCore != managerSlot ||
            (state.managerCompletionBitmap & (1u << managerSlot)) != 0 ||
            message.value != 1.0) {
            failManagerAttentionJob(state, SFUStatus::InvalidDescriptor);
            return true;
        }
        state.managerCompletionBitmap |= 1u << managerSlot;
        state.managersCompleted += 1;
        statAttentionManagerBandCompletionsReceived_->addData(1);
        statAttentionManagerBandCompletionReceivedTick_->addData(getCurrentSimCycle());
        if (state.managersCompleted == state.desc.tensor_manager_count) {
            state.status = SFUStatus::Success;
            state.phase = ManagerAttentionJobPhase::Complete;
            statAttentionManagerJobsCompleted_->addData(1);
            statAttentionTensorJobsCompleted_->addData(1);
            statAttentionTensorCompleteTick_->addData(getCurrentSimCycle());
        }
        return true;
    }

    enum class ManagerTensorJobPhase : uint8_t {
        ReadDescriptor,
        ReadParams,
        ReadTopology,
        Dispatch,
        Running,
        Complete,
    };

    struct ManagerTensorJobState {
        uint64_t tag = 0;
        uint64_t descAddr = 0;
        SFUJobDesc desc = {};
        SFUSoftmaxJobParamsV1 params = {};
        std::vector<uint32_t> workerCoreIds;
        std::vector<uint8_t> completionSeen;
        uint32_t rowsCompleted = 0;
        SFUStatus status = SFUStatus::Pending;
        ManagerTensorJobPhase phase = ManagerTensorJobPhase::ReadDescriptor;
        bool readInflight = false;
        uint64_t readTag = 0;
    };

    bool tryIssueManagerTensorJobCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        const uint64_t tag = cmd->rs2;
        if (globalMem == nullptr || managerTensorJobs_.find(tag) != managerTensorJobs_.end()) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        ManagerTensorJobState state;
        state.tag = tag;
        state.descAddr = cmd->rs1;
        managerTensorJobs_.emplace(tag, std::move(state));
        statTensorManagerJobsIssued_->addData(1);
        statTensorManagerDescriptorAcceptTick_->addData(getCurrentSimCycle());
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitManagerTensorJobCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        auto it = managerTensorJobs_.find(cmd->rs1);
        if (it == managerTensorJobs_.end()) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, static_cast<uint64_t>(SFUStatus::InvalidDescriptor),
                cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (it->second.phase != ManagerTensorJobPhase::Complete) {
            return false;
        }
        const uint64_t status = static_cast<uint64_t>(it->second.status);
        statTensorManagerWaitObservedTick_->addData(getCurrentSimCycle());
        managerTensorJobs_.erase(it);
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, status, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    void failManagerTensorJob(ManagerTensorJobState& state, SFUStatus status) {
        state.status = status;
        state.readInflight = false;
        state.phase = ManagerTensorJobPhase::Complete;
    }

    template <typename Metadata>
    void issueManagerMetadataRead(ManagerTensorJobState& state,
                                  uint64_t address,
                                  ManagerTensorJobPhase nextPhase,
                                  Metadata ManagerTensorJobState::* destination) {
        if (state.readInflight || sizeof(Metadata) > globalMem->localMaxRequestBytes()) {
            if (sizeof(Metadata) > globalMem->localMaxRequestBytes()) {
                failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
            }
            return;
        }
        const uint64_t tag = allocateLocalTransferTag();
        const uint64_t jobTag = state.tag;
        const bool accepted = globalMem->localReadAsync(
            address, sizeof(Metadata), LocalMemoryClient::Control, tag,
            [this, jobTag, tag, nextPhase, destination](
                bool success, uint64_t callbackTag, const std::vector<uint8_t>& bytes) {
                auto it = managerTensorJobs_.find(jobTag);
                if (it == managerTensorJobs_.end()) {
                    return;
                }
                ManagerTensorJobState& callbackState = it->second;
                if (!callbackState.readInflight || callbackState.readTag != callbackTag ||
                    callbackTag != tag || !success || bytes.size() != sizeof(Metadata)) {
                    failManagerTensorJob(callbackState, SFUStatus::InvalidDescriptor);
                    return;
                }
                std::memcpy(&(callbackState.*destination), bytes.data(), sizeof(Metadata));
                callbackState.readInflight = false;
                callbackState.phase = nextPhase;
            });
        if (accepted) {
            state.readInflight = true;
            state.readTag = tag;
        }
    }

    bool validateManagerDescriptor(const ManagerTensorJobState& state) const {
        const SFUJobDesc& desc = state.desc;
        return desc.owner_core == coreID && desc.input0_addr != 0 && desc.output_addr != 0 &&
            desc.params_addr != 0 && desc.rows != 0 && desc.cols != 0 &&
            desc.chunk_elems != 0 && desc.worker_cores != 0 &&
            desc.worker_cores <= SFU_WORKER_TOPOLOGY_MAX_WORKERS &&
            desc.dtype == SFU_JOB_DTYPE_FP32 &&
            desc.op_type == static_cast<uint32_t>(SFUJobOp::SOFTMAX_ROW) &&
            (desc.flags & (SFU_JOB_FLAG_ROW_ENGINE_MODEL |
                           SFU_JOB_FLAG_TENSOR_ROW_ENGINE)) ==
                (SFU_JOB_FLAG_ROW_ENGINE_MODEL | SFU_JOB_FLAG_TENSOR_ROW_ENGINE);
    }

    bool validateManagerParams(const ManagerTensorJobState& state) const {
        const SFUSoftmaxJobParamsV1& params = state.params;
        return params.magic == SFU_SOFTMAX_JOB_PARAMS_MAGIC &&
            params.version == SFU_SOFTMAX_JOB_PARAMS_VERSION_MANAGER &&
            params.size_bytes == sizeof(SFUSoftmaxJobParamsV1) &&
            params.mapping_policy == SFU_SOFTMAX_MAPPING_EXPLICIT_TOPOLOGY &&
            params.hbm_layout == SFU_SOFTMAX_HBM_LAYOUT_BAND_STRIPED &&
            params.data_node_mask != 0 && params.node_stride_bytes != 0 &&
            params.rows_per_band != 0 && params.coordinator_core == coreID &&
            params.completion_addr != 0;
    }

    bool validateAndInstallManagerTopology(ManagerTensorJobState& state,
                                           const SFUWorkerTopologyMapV1& topology) {
        if (topology.magic != SFU_WORKER_TOPOLOGY_MAP_MAGIC ||
            topology.version != SFU_WORKER_TOPOLOGY_MAP_VERSION ||
            topology.size_bytes != sizeof(SFUWorkerTopologyMapV1) ||
            topology.worker_count != state.desc.worker_cores) {
            return false;
        }
        state.workerCoreIds.clear();
        for (uint32_t slot = 0; slot < topology.worker_count; ++slot) {
            const uint32_t workerCore = topology.worker_core_ids[slot];
            if (workerCore == coreID ||
                std::find(state.workerCoreIds.begin(), state.workerCoreIds.end(), workerCore) !=
                    state.workerCoreIds.end()) {
                return false;
            }
            state.workerCoreIds.push_back(workerCore);
        }
        statTensorManagerWorkersMapped_->addData(state.workerCoreIds.size());
        return true;
    }

    void dispatchManagerTensorJob(ManagerTensorJobState& state) {
        const uint32_t rowsPerBand = state.params.rows_per_band;
        const uint32_t bands = (state.desc.rows + rowsPerBand - 1) / rowsPerBand;
        if (bands == 0 || bands > state.workerCoreIds.size()) {
            failManagerTensorJob(state, SFUStatus::InvalidShape);
            return;
        }
        state.completionSeen.assign(bands, 0);
        for (uint32_t band = 0; band < bands; ++band) {
            const uint32_t workerSlot = band % state.desc.worker_cores;
            const uint32_t workerCore = state.workerCoreIds[workerSlot];
            ReductionTransportMessage dispatch = {};
            dispatch.kind = ReductionTransportMessageKind::TensorRowDispatch;
            dispatch.jobId = state.desc.job_id;
            dispatch.tag = state.tag;
            dispatch.ownerCore = static_cast<uint32_t>(coreID);
            dispatch.workerSlot = workerSlot;
            dispatch.workerCore = workerCore;
            dispatch.row = band * rowsPerBand;
            dispatch.expectedWorkers = state.desc.worker_cores;
            dispatch.expectedRows = std::min(rowsPerBand, state.desc.rows - dispatch.row);
            dispatch.expectedCols = state.desc.cols;
            dispatch.inputAddr = state.desc.input0_addr;
            dispatch.outputAddr = state.desc.output_addr;
            dispatch.nodeStrideBytes = state.params.node_stride_bytes;
            dispatch.dataNodeMask = state.params.data_node_mask;
            dispatch.rowsPerBand = rowsPerBand;
            if ((state.params.flags & SFU_SOFTMAX_PARAMS_FLAG_ATTENTION) != 0) {
                const uint64_t headDim = state.params.reserved0;
                if (headDim == 0) {
                    failManagerTensorJob(state, SFUStatus::InvalidShape);
                    return;
                }
                const double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
                dispatch.value =
                    (state.params.flags & SFU_SOFTMAX_PARAMS_FLAG_CAUSAL) != 0
                        ? -scale : scale;
            }
            if (!globalMem->sendReductionMessage(workerCore, dispatch)) {
                failManagerTensorJob(state, SFUStatus::GlobalMemoryUnavailable);
                return;
            }
            statTensorManagerBandDispatchTick_->addData(getCurrentSimCycle());
        }
        statTensorManagerRowsDispatched_->addData(state.desc.rows);
        state.phase = ManagerTensorJobPhase::Running;
    }

    void progressManagerTensorJobs() {
        for (auto& entry : managerTensorJobs_) {
            ManagerTensorJobState& state = entry.second;
            if (state.phase == ManagerTensorJobPhase::ReadDescriptor) {
                issueManagerMetadataRead(state, state.descAddr,
                    ManagerTensorJobPhase::ReadParams, &ManagerTensorJobState::desc);
            } else if (state.phase == ManagerTensorJobPhase::ReadParams && !state.readInflight) {
                if (!validateManagerDescriptor(state)) {
                    failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
                } else {
                    issueManagerMetadataRead(state, state.desc.params_addr,
                        ManagerTensorJobPhase::ReadTopology, &ManagerTensorJobState::params);
                }
            } else if (state.phase == ManagerTensorJobPhase::ReadTopology &&
                       !state.readInflight) {
                if (!validateManagerParams(state)) {
                    failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
                    continue;
                }
                if (sizeof(SFUWorkerTopologyMapV1) > globalMem->localMaxRequestBytes()) {
                    failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
                    continue;
                }
                const uint64_t jobTag = state.tag;
                const uint64_t readTag = allocateLocalTransferTag();
                const bool accepted = globalMem->localReadAsync(
                    state.params.completion_addr, sizeof(SFUWorkerTopologyMapV1),
                    LocalMemoryClient::Control, readTag,
                    [this, jobTag, readTag](bool success, uint64_t callbackTag,
                                            const std::vector<uint8_t>& bytes) {
                        auto it = managerTensorJobs_.find(jobTag);
                        if (it == managerTensorJobs_.end()) return;
                        ManagerTensorJobState& callbackState = it->second;
                        if (!callbackState.readInflight || callbackTag != readTag ||
                            callbackState.readTag != callbackTag || !success ||
                            bytes.size() != sizeof(SFUWorkerTopologyMapV1)) {
                            failManagerTensorJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        SFUWorkerTopologyMapV1 topology = {};
                        std::memcpy(&topology, bytes.data(), sizeof(topology));
                        callbackState.readInflight = false;
                        if (!validateAndInstallManagerTopology(callbackState, topology)) {
                            failManagerTensorJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        callbackState.phase = ManagerTensorJobPhase::Dispatch;
                    });
                if (accepted) {
                    state.readInflight = true;
                    state.readTag = readTag;
                }
            } else if (state.phase == ManagerTensorJobPhase::Dispatch) {
                dispatchManagerTensorJob(state);
            }
        }
    }

    bool handleManagerTensorCompletion(const ReductionTransportMessage& message) {
        if (message.kind != ReductionTransportMessageKind::TensorRowComplete ||
            message.ownerCore != coreID) {
            return false;
        }
        auto it = managerTensorJobs_.find(message.tag);
        if (it == managerTensorJobs_.end()) {
            return false;
        }
        ManagerTensorJobState& state = it->second;
        const uint32_t rowsPerBand = state.params.rows_per_band;
        if (state.phase != ManagerTensorJobPhase::Running || rowsPerBand == 0 ||
            message.jobId != state.desc.job_id || message.row % rowsPerBand != 0) {
            failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
            return true;
        }
        const uint32_t band = message.row / rowsPerBand;
        const uint32_t expectedSlot = band % state.desc.worker_cores;
        const uint32_t expectedRows = std::min(rowsPerBand, state.desc.rows - message.row);
        if (band >= state.completionSeen.size() || message.workerSlot != expectedSlot ||
            message.workerCore != state.workerCoreIds[expectedSlot] ||
            message.expectedWorkers != state.desc.worker_cores ||
            message.expectedRows != expectedRows || message.expectedCols != state.desc.cols ||
            message.rowsPerBand != rowsPerBand || state.completionSeen[band] != 0 ||
            message.value != 1.0) {
            failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
            return true;
        }
        state.completionSeen[band] = 1;
        statTensorManagerCompletionReceivedTick_->addData(getCurrentSimCycle());
        statTensorManagerRowsCompleted_->addData(expectedRows);
        state.rowsCompleted += expectedRows;
        if (state.rowsCompleted == state.desc.rows &&
            std::all_of(state.completionSeen.begin(), state.completionSeen.end(),
                        [](uint8_t seen) { return seen != 0; })) {
            state.status = SFUStatus::Success;
            state.phase = ManagerTensorJobPhase::Complete;
            statTensorManagerJobsCompleted_->addData(1);
            statTensorManagerCompleteTick_->addData(getCurrentSimCycle());
        }
        return true;
    }

    void handleReductionTransportMessage(const ReductionTransportMessage& message) {
        if (message.kind == ReductionTransportMessageKind::AttentionDispatch) {
            startAttentionWorker(message);
            return;
        }
        if (handleManagerAttentionCompletion(message)) {
            return;
        }
        if (handleAttentionManagerBandCompletion(message)) {
            return;
        }
        if (handleManagerTensorCompletion(message)) {
            return;
        }
        if (sfu != nullptr) {
            sfu->receiveReductionMessage(message);
        }
    }

    bool isArrayComputeInflight(uint32_t array_id) const {
        if (array_id >= inflight_compute_cmds.size()) {
            return false;
        }
        return inflight_compute_cmds[array_id].cmd != nullptr;
    }

    void issueArrayCompute(SST::Vanadis::RoCCCommand* cmd, uint32_t array_id, uint64_t cycle) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return;
        }
        if (array_id >= inflight_compute_cmds.size()) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return;
        }

        auto& inflight = inflight_compute_cmds[array_id];
        if (inflight.cmd != nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return;
        }

        inflight.cmd = cmd;
        inflight.start_cycle = cycle;
        inflight.async_mode = (cmd->inst->rd == 0);
        if (inflight.async_mode) {
            auto& async_state = async_compute_states[array_id];
            async_state.submitted = true;
            async_state.completed = false;
            async_state.rd_val = 0;
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        }
        arrayStates[array_id] = 1;
        array->beginComputation(array_id);
    }

    void completeArrayCompute(uint32_t array_id, uint64_t rd_val) {
        if (array_id >= inflight_compute_cmds.size()) {
            return;
        }

        auto& inflight = inflight_compute_cmds[array_id];
        if (inflight.cmd == nullptr || inflight.cmd->inst == nullptr) {
            output->verbose(CALL_INFO, 0, 0,
                "[RoCC ERROR] array completion without inflight compute array=%" PRIu32 "\n",
                array_id);
            return;
        }

        const uint64_t cycles_spent = (LastTickCycle >= inflight.start_cycle)
            ? (LastTickCycle - inflight.start_cycle + 1)
            : 0;
        stat_cycles_mvm->addData(cycles_spent);
        mvm_ops_completed++;
        maybeReportMvmProgress(false);

        output->verbose(CALL_INFO, 1, 0,
            "Finalize RoCC compute command array=%" PRIu32 " rd=%" PRIu16 " cmd_id=%" PRIu64 " rd_val=%" PRIu64 "\n",
            array_id,
            inflight.cmd->inst->rd,
            inflight.cmd->cmd_id,
            rd_val);

        if (inflight.async_mode) {
            auto& async_state = async_compute_states[array_id];
            async_state.submitted = true;
            async_state.completed = true;
            async_state.rd_val = rd_val;
        } else {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                inflight.cmd->inst->rd,
                rd_val,
                inflight.cmd->cmd_id,
                inflight.cmd->hw_thread));
        }

        delete inflight.cmd;
        inflight.cmd = nullptr;
        inflight.start_cycle = 0;
        inflight.async_mode = false;
    }
  
    void storeVector() {
        uint64_t rs1 = curr_cmd->rs1; // Destination address (physical)
        uint64_t rs2 = curr_cmd->rs2; // Array ID or source vector index
        vector_total_size = arrayOutputSize * outputOperandSize;
        uint64_t cache_line_size = memInterface->getLineSize();

        write_offset = 0;
        uint64_t physAddr = rs1; // Assuming rs1 is physical address
        uint64_t addr_offset = physAddr % cache_line_size;

        // Resize the output payload to hold the entire vector
        outputPayload.resize(vector_total_size);

        // Reference to the output vector we need to store
        auto& outputVector = *static_cast<std::vector<T>*>(array->getOutputVector(rs2));

        // Fill the output payload with the vector data
        for (size_t i = 0; i < static_cast<size_t>(arrayOutputSize); i++) {
            T value = outputVector[i];
            uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(&value);
            for (size_t j = 0; j < static_cast<size_t>(outputOperandSize); j++) {
                outputPayload[i * outputOperandSize + j] = byte_ptr[j];
            }
        }

        // Optional: Output the stored array for debugging purposes
        output->verbose(CALL_INFO, 9, 0, "Stored array %" PRIu64 ":\n", rs2);
        for (size_t i = 0; i < static_cast<size_t>(arrayOutputSize); i++) {
            if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
                output->verbose(CALL_INFO, 9, 0, "%f ", static_cast<double>(outputVector[i]));
            } else {
                output->verbose(CALL_INFO, 9, 0, "%lld ", static_cast<long long>(outputVector[i]));
            }
        }
        output->verbose(CALL_INFO, 9, 0, "\n\n");

        // Calculate the size of the first memory request
        uint32_t request_size = static_cast<uint32_t>(std::min(
            cache_line_size - addr_offset, 
            vector_total_size - write_offset
        ));

        // Prepare the first chunk of data to write
        std::vector<uint8_t> data_chunk(
            outputPayload.begin() + write_offset,
            outputPayload.begin() + write_offset + request_size
        );

        // Create a new write request to send to the memory interface
        auto* store_req = new StandardMem::Write(physAddr + write_offset, request_size, data_chunk, false, 0, rs1, 0, 0);

        // Send the write request
        memInterface->send(store_req);

        // Update the write offset for subsequent writes
        write_offset += request_size;
    }
    
    void moveVector() {
        uint64_t rs1 = curr_cmd->rs1;//RoCC指令传递的源阵列ID
        uint64_t rs2 = curr_cmd->rs2;//RoCC指令传递的目标阵列ID
        //调用array的moveOutputToInput方法，把ID为rs1阵列的输出向量作为ID为rs2阵列的输入向量
        array->moveOutputToInput(rs1, rs2);

        //拿到“目标阵列rs2的输入向量”对象（即move之后的)
        auto& inputVector = *static_cast<std::vector<T>*>(array->getInputVector(rs2));

        output->verbose(CALL_INFO, 9, 0,
                      "Moved array %" PRIu64 " to array %" PRIu64 ". Array %" PRIu64 ":\n", rs1, rs2, rs2);

        for (int i = 0; i < arrayInputSize; i++) {
            if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
                output->verbose(CALL_INFO, 9, 0, "%f ", static_cast<double>(inputVector[i]));
            } else {
                output->verbose(CALL_INFO, 9, 0, "%ld ", static_cast<long>(inputVector[i]));
            }
        }
        output->verbose(CALL_INFO, 9, 0, "\n");

        completeRoCC(0);
    }

    void ConfigureOutputMode() {
        uint64_t command = curr_cmd->rs1; // 输出模式命令
        uint64_t array_id = curr_cmd->rs2; // 阵列编号

        if (array_id >= static_cast<uint64_t>(numArrays)) {
            output->verbose(CALL_INFO, 0, 0,
                            "mvm.ocfg: invalid array id %" PRIu64 " (numArrays=%d)\n",
                            array_id, numArrays);
            completeRoCC(1);
            return;
        }

        array->configureOutputMode(static_cast<uint32_t>(array_id), command);
        completeRoCC(0);
    }

    void OutputvectorStore(uint64_t) {
        if (curr_cmd == nullptr || curr_cmd->rs2 >= static_cast<uint64_t>(numArrays)) {
            completeRoCC(1);
            return;
        }
        if (!legacy_output_store_.active) {
            legacy_output_store_ = LegacyOutputStoreState{};
            legacy_output_store_.active = true;
            legacy_output_store_.command_id = curr_cmd->cmd_id;
            legacy_output_store_.dest_addr = curr_cmd->rs1;
            legacy_output_store_.array_id = static_cast<uint32_t>(curr_cmd->rs2);
        }
        progressLegacyOutputStore();
    }

    void IntputvectorLoad(uint64_t) {
        beginBlockingArrayLoad(false);
    }

    void InputMatrixLoad(uint64_t) {
        beginBlockingArrayLoad(true);
    }
    void SetRemoteLength() {
        size_t fallback = defaultRemoteLength();
        size_t requested = static_cast<size_t>(curr_cmd->rs1);
        if (requested == 0) {
            requested = fallback;
        }
        if (requested == 0) {
            requested = 1; // 确保非零长度
        }

        remoteTransferLength = requested;
        output->verbose(CALL_INFO, 9, 0,
                        "Remote transfer length updated to %zu bytes (rs1=0x%" PRIx64 ")\n",
                        remoteTransferLength, curr_cmd->rs1);
        completeRoCC(0);
    }

    void RemoteStore(uint64_t cycle) {
        uint64_t cycles_elapsed = (cycle >= StartTickCycle) ? (cycle - StartTickCycle + 1) : 0;

        if (cycles_elapsed < latency_remote_st) {
            return;
        }

        output->verbose(CALL_INFO, 9, 0,
                        "RemoteStore: Executing after %" PRIu64 " cycles (at cycle %" PRIu64 ")\n",
                        cycles_elapsed, cycle);
        uint64_t local_addr  = curr_cmd->rs1;  // 本地 GlobalMemory 源地址
        uint64_t remote_addr = curr_cmd->rs2;  // 远端 GlobalMemory 目标地址

    // 传输字节数：以 mvm.slen 设置的 remoteTransferLength 为准；
    // 早期实现曾用 rd 寄存器号作为长度（会被编译器分配为如 x15），导致固定为 15 等错误值。
    // 这里不再使用 rd 覆盖，统一按照配置长度来传输。
    uint16_t rd_reg_index = curr_cmd->inst->rd; // 仅用于调试观测（指令目的寄存器号）
    size_t length = resolveRemoteLength();

        std::vector<uint8_t> data(length);
        globalMem->rd_from_globalmem(local_addr, length, data);   // 先读本地
        globalMem->wr_to_network(remote_addr, length, data);      // 再经 NoC 写对端
        completeRoCC(0);  // 发送后即可返回（若需 ACK，可扩展为等待网络回包）
    }

    void RemoteStoreWait(uint64_t cycle) {
        const uint64_t cycles_elapsed =
            (cycle >= StartTickCycle) ? (cycle - StartTickCycle + 1) : 0;
        if (remoteStoreCompletionToken == 0) {
            if (cycles_elapsed < latency_remote_st) {
                return;
            }
            const uint64_t local_addr = curr_cmd->rs1;
            const uint64_t host_addr = curr_cmd->rs2;
            const size_t length = resolveRemoteLength();
            std::vector<uint8_t> data(length);
            globalMem->rd_from_globalmem(local_addr, length, data);
            remoteStoreCompletionToken =
                globalMem->dma_write_to_host_async(host_addr, length, data);
        }
        if (!globalMem->dma_completion_done(remoteStoreCompletionToken)) {
            return;
        }
        globalMem->dma_completion_retire(remoteStoreCompletionToken);
        remoteStoreCompletionToken = 0;
        completeRoCC(0);
    }

    void RemoteLoad(uint64_t cycle) {
        uint64_t cycles_elapsed = (cycle >= StartTickCycle) ? (cycle - StartTickCycle + 1) : 0;

        if (cycles_elapsed < latency_remote_ld) {
            return;
        }
        uint64_t remote_addr = curr_cmd->rs1; // 远端 GlobalMemory 源地址
        uint64_t local_addr  = curr_cmd->rs2; // 本地 GlobalMemory 目标地址
        (void)local_addr; // 当前实现中不直接在此函数写入，读回后由 GlobalMemory 统一存放

    // 传输字节数：统一以 mvm.slen 设置的 remoteTransferLength 为准
    uint16_t rd_reg_index = curr_cmd->inst->rd; // 仅用于调试观测（指令目的寄存器号）
    size_t length = resolveRemoteLength();

        output->verbose(CALL_INFO, 1, 2,
            "RemoteLoad issued (GM base 0x%" PRIx64 "): remote_addr=0x%" PRIx64 " local_addr=0x%" PRIx64 " length=%zu rd_reg=%u\n",
            globalMem ? globalMem->getBaseAddr() : 0, remote_addr, local_addr, length, rd_reg_index);

        // 发起网络读请求；数据返回后由 GlobalMemory 的回调统一写入本地存储
        globalMem->rd_to_network(remote_addr, length, local_addr);
        completeRoCC(0);
    }


    void MainMem2GlobalMem() {
        uint64_t rs1 = curr_cmd->rs1; // 主存源物理地址 (Main Memory Source)
        uint64_t rs2 = curr_cmd->rs2; // GlobalMemory 目标地址 (Destination)
        
        // 初始化状态
        gm_write_dst_addr = rs2;
        gm_write_offset = 0;
        
        // 确定传输长度：优先使用 mvm.slen 设置的长度，或者使用默认向量长度
        gm_write_total_size = resolveRemoteLength();

        // 定义一个新的 Flag，用于在回调中识别这是 write_gm 的数据
        // 0x0 是 Matrix, 0x1 是 Vector, 我们用 0x2
        uint32_t load_gm_flag = 0x10; 
        
        uint64_t cache_line_size = memInterface->getLineSize();
        
        // 计算地址对齐和第一次请求的大小
        uint64_t physAddr = rs1; 
        uint64_t addr_offset = physAddr % cache_line_size;
        
        uint32_t request_size = std::min(static_cast<uint64_t>(cache_line_size - addr_offset), gm_write_total_size);

        output->verbose(CALL_INFO, 1, 0, 
            "write_gm Start: MainMem Addr: 0x%" PRIx64 ", GlobalMem Addr: 0x%" PRIx64 ", Size: %" PRIu64 "\n",
            physAddr, gm_write_dst_addr, gm_write_total_size);

        // 发起读取请求 (Read Request)
        auto *load_req = new StandardMem::Read(physAddr, request_size, load_gm_flag);
        memInterface->send(load_req);
    }

    // 实现 GlobalMem2MainMem (GM -> MainMem)
    void GlobalMem2MainMem() {
        uint64_t mm_dst_addr = curr_cmd->rs1; // Main Memory 目标地址
        uint64_t gm_src_addr = curr_cmd->rs2; // Global Memory 源地址
        
        // 1. 确定传输长度 (使用 mvm.slen 设置的长度)
        // 复用 vector_total_size 变量，因为它在 WriteResp 中被用于检查结束条件
        vector_total_size = resolveRemoteLength();
        write_offset = 0;

        uint64_t cache_line_size = memInterface->getLineSize();
        uint64_t physAddr = mm_dst_addr; 
        uint64_t addr_offset = physAddr % cache_line_size;

        output->verbose(CALL_INFO, 1, 0, 
            "write_mm Start: GlobalMem Addr: 0x%" PRIx64 ", MainMem Addr: 0x%" PRIx64 ", Size: %" PRIu64 "\n",
            gm_src_addr, mm_dst_addr, vector_total_size);

        // 2. 从 GlobalMemory 读取数据到本地 buffer (outputPayload)
        outputPayload.resize(vector_total_size);
        globalMem->rd_from_globalmem(gm_src_addr, vector_total_size, outputPayload);

        // [优化打印] 打印从 GM 读出的全部数据
        if (!outputPayload.empty()) {
            std::string hex_str;
            char buf[8];
            for (size_t i = 0; i < outputPayload.size(); ++i) {
                snprintf(buf, sizeof(buf), "%02X ", outputPayload[i]);
                hex_str += buf;
            }
            output->verbose(CALL_INFO, 1, 0, "Data Write to MainMemory\n");
            output->verbose(CALL_INFO, 10, 0, "Data: %s\n", hex_str.c_str());
        } else {
            output->verbose(CALL_INFO, 1, 0, "[EMPTY]\n");
        }

        // 3. 发起第一个主存写入请求 (DMA Write)
        uint32_t request_size = static_cast<uint32_t>(std::min(
            cache_line_size - addr_offset, 
            vector_total_size - write_offset
        ));

        std::vector<uint8_t> data_chunk(
            outputPayload.begin() + write_offset,
            outputPayload.begin() + write_offset + request_size
        );

        // 构造写请求
        // 注意：storeVector 使用的是 rs1 作为目标，而 write_mm 使用的是 rs2
        // 这里我们在创建请求时传入物理地址，回调中会根据 func7 区分计算下一个地址
        auto* store_req = new StandardMem::Write(physAddr + write_offset, request_size, data_chunk, false, 0, physAddr, 0, 0);

        memInterface->send(store_req);
        write_offset += request_size;
    }

    // reg2gm: 将 rs1 寄存器的值直接写入到 GlobalMemory 地址 rs2
    void Reg2GlobalMem() {
        uint64_t val = curr_cmd->rs1;       // 数据 (来自 Vanadis 寄存器)
        uint64_t dst_addr = curr_cmd->rs2;  // 目标 GlobalMemory 地址

        // 准备数据包：Vanadis 寄存器是 64 位的，所以写入 8 字节
        size_t data_size = sizeof(uint64_t);
        std::vector<uint8_t> payload(data_size);
        memcpy(payload.data(), &val, data_size);

        output->verbose(CALL_INFO, 1, 0, 
            "reg2gm: Writing register value 0x%" PRIx64 " to GlobalMemory Address 0x%" PRIx64 "\n", 
            val, dst_addr);

        globalMem->wr_to_globalmem(dst_addr, data_size, payload);
        completeRoCC(0);
    }


    // gm2reg: 从 GlobalMemory 地址 rs1 读取数据，写入到目标寄存器 rd
    void GlobalMem2Reg() {
        uint64_t src_addr = curr_cmd->rs1;  // 源 GlobalMemory 地址

        // 准备读取缓冲区
        size_t data_size = sizeof(uint64_t);
        std::vector<uint8_t> read_buffer;
        read_buffer.resize(data_size); 

        // 调用 GlobalMemory 读接口
        globalMem->rd_from_globalmem(src_addr, data_size, read_buffer);
        uint64_t val = 0;
        memcpy(&val, read_buffer.data(), data_size);

        output->verbose(CALL_INFO, 1, 0, 
            "gm2reg: Read value 0x%" PRIx64 " from GlobalMemory Address 0x%" PRIx64 "\n", 
            val, src_addr);

        // 完成指令，并将读取到的值作为结果返回
        completeRoCC(val);
    }


    
    //本质上，它是每条RoCC指令生命周期的结束收尾工作
    void completeRoCC(uint64_t rd_val) {
        uint64_t cycles_spent = (LastTickCycle >= StartTickCycle) ? (LastTickCycle - StartTickCycle + 1) : 0;
        if (curr_cmd == nullptr || curr_cmd->inst == nullptr) {
            output->verbose(CALL_INFO, 0, 0, "[RoCC ERROR] completeRoCC with null command\n");
            busy = false;
            return;
        }
        if (curr_cmd != nullptr) {
            switch (curr_cmd->inst->func7) {
                case 0x1: stat_cycles_mvm_set->addData(cycles_spent); break;
                case 0x2: stat_cycles_mvm_l->addData(cycles_spent); break;
                case 0x3: stat_cycles_mvm->addData(cycles_spent); break;
                case 0x4: stat_cycles_mvm_s->addData(cycles_spent); break;
                case 0x5: stat_cycles_mvm_mv->addData(cycles_spent); break;
                case 0x6: stat_cycles_mvm_ovec2gm->addData(cycles_spent); break;
                case 0x7: stat_cycles_mvm_gm2ivec->addData(cycles_spent); break;
                case 0x8: stat_cycles_mvm_gm2imat->addData(cycles_spent); break;
                case 0x9:
                case GOLEM_ROCC_FUNC7_REMOTE_STORE_WAIT:
                    stat_cycles_remote_st->addData(cycles_spent);
                    break;
                case 0xA: stat_cycles_remote_ld->addData(cycles_spent); break;
                default: break;
            }
            if (curr_cmd->inst->func7 == 0x3) {
                mvm_ops_completed++;
                maybeReportMvmProgress(false);
            }
        }
        output->verbose(CALL_INFO, 1, 0,
            "Finalize RoCC command w/ func7=0x%02" PRIx8 " rd=%" PRIu16 " xd=%u xs1=%u xs2=%u rs1=0x%" PRIx64 " rs2=0x%" PRIx64 " rd_val=%" PRIu64 "\n",
            curr_cmd->inst->func7,
            curr_cmd->inst->rd,
            static_cast<unsigned>(curr_cmd->inst->xd),
            static_cast<unsigned>(curr_cmd->inst->xs1),
            static_cast<unsigned>(curr_cmd->inst->xs2),
            curr_cmd->rs1,
            curr_cmd->rs2,
            rd_val
        );

        busy = false;
        enqueueResponse(new SST::Vanadis::RoCCResponse(curr_cmd->inst->rd, rd_val, curr_cmd->cmd_id, curr_cmd->hw_thread));
        delete curr_cmd;
        curr_cmd = nullptr;
    }

    void maybeReportMvmProgress(bool force) {
        if (!progress_heartbeat || progress_total_mvm_ops == 0) {
            return;
        }
        if (!force && LastTickCycle < progress_next_cycle && mvm_ops_completed < progress_total_mvm_ops) {
            return;
        }

        uint64_t completed = mvm_ops_completed;
        if (completed > progress_total_mvm_ops) {
            completed = progress_total_mvm_ops;
        }
        const uint64_t pct = (completed * 100) / progress_total_mvm_ops;
        if (force || pct != progress_last_percent) {
            output->output("RoCC core=%" PRIu64 " MVM_PROGRESS: completed=%" PRIu64 "/%" PRIu64
                           " (%" PRIu64 "%%) cycle=%" PRIu64 "\n",
                           coreID,
                           completed,
                           progress_total_mvm_ops,
                           pct,
                           LastTickCycle);
            progress_last_percent = pct;
        }

        while (progress_next_cycle <= LastTickCycle) {
            progress_next_cycle += progress_interval_cycles;
        }
    }

    void recordWcpArrayCompletion(uint32_t array_id) {
        if (array_id < arrayStates.size()) {
            arrayStates[array_id] = 0;
        }
        mvm_ops_completed++;
        maybeReportMvmProgress(false);
    }

    //在阵列（Array）计算完成后，被SST模拟框架自动调用的,标记这个阵列空闲
    void handleArrayEvent(Event *ev) {
        Golem::ArrayEvent *aev = static_cast<Golem::ArrayEvent *>(ev);
        uint32_t arrayID = aev->getArrayID();
        if (handleAttentionArrayDone(arrayID)) {
            delete ev;
            return;
        }
        if (workerCommandProcessor != nullptr && workerCommandProcessor->handleArrayDone(arrayID, LastTickCycle)) {
            recordWcpArrayCompletion(arrayID);
            delete ev;
            return;
        }
        if (arrayID >= arrayStates.size()) {
            delete ev;
            return;
        }
        arrayStates[arrayID] = 0;
        completeArrayCompute(arrayID, 0);
        delete ev;
    }
  
    class StandardMemHandlers : public Interfaces::StandardMem::RequestHandler {
    public:
        StandardMemHandlers(RoCCAnalog *rocc, SST::Output *output)
            : Interfaces::StandardMem::RequestHandler(output), rocc(rocc) {}
  
        virtual ~StandardMemHandlers() {}
  
        virtual void handle(StandardMem::ReadResp *ev) {
            out->verbose(CALL_INFO, 9, 0,
                     "-> handle read-response (virt-addr: 0x%" PRI_ADDR ")\n", ev->vAddr);
            const uint32_t flags = ev->getAllFlags();

            SST::Vanadis::RoCCCommand *rocc_cmd = rocc->curr_cmd;
  
            if (ev->getFail()) {
                out->verbose(CALL_INFO, 9, 0, "RoCC load failed\n");
                rocc->completeRoCC(1);
                delete ev;
                return;
            }
            //阵列ID（或向量/矩阵的编号），直接从指令的rs2字段获取，用于确定数据写到哪个计算阵列
            int32_t array_id = rocc_cmd->rs2;  // Array ID is in rs2
            switch (flags) {
                //处理了一个矩阵的读取操作。它从内存中读取矩阵数据，并将这些数据存储到计算阵列中
                case GOLEM_ROCC_FLAG_SYNC_MATRIX: // Read response data is matrix to be set
                {
                    rocc->output->verbose(CALL_INFO, 9, 0,
                                "Set matrix read response detected\n");
  
                    size_t payload_size = ev->size;  //payload_size 获取响应数据的大小
                    unsigned char *payload_data = ev->data.data(); //ev->data.data()返回一个字节数组，包含了实际的矩阵数据

                    // Assign the received data to the matrix
                    for (size_t i = 0; i < payload_size; i += rocc->inputOperandSize) {
                        T value = 0;
                        memcpy(&value, &payload_data[i], rocc->inputOperandSize);
                        int index = (rocc->matrix_read_offset + i) / rocc->inputOperandSize;
                        rocc->array->setMatrixItem(array_id, index, value);
                    }
  
                    rocc->matrix_read_offset += payload_size;
  
                    if (rocc->matrix_read_offset < rocc->matrix_total_size) {

                        // Send the next read request
                        uint64_t cache_line_size = rocc->memInterface->getLineSize();
                        uint32_t request_size = static_cast<uint32_t>(std::min(
                        cache_line_size, rocc->matrix_total_size - rocc->matrix_read_offset));
                        uint64_t next_addr = rocc_cmd->rs1 + rocc->matrix_read_offset;
                        auto *load_req = new StandardMem::Read(next_addr, request_size, GOLEM_ROCC_FLAG_SYNC_MATRIX);
                        rocc->memInterface->send(load_req);
                    } else {
                        // Matrix read complete
                        if (array_id >= 0 && static_cast<uint32_t>(array_id) < rocc->async_matrix_loads.size()) {
                            rocc->markArrayLoadReady(static_cast<uint32_t>(array_id), true);
                        }
                        rocc->completeRoCC(0);
                    }
                } break;
                //处理了一个向量的读取操作。它从内存中读取向量数据，并将这些数据存储到计算阵列中
                case GOLEM_ROCC_FLAG_SYNC_VECTOR: // Read response data is input vector
                {
                    rocc->output->verbose(CALL_INFO, 9, 0,
                                "Input vector read response detected\n");
  
                    size_t payload_size = ev->size;
                    unsigned char *payload_data = ev->data.data();
  
                    // Assign the received data to the input vector
                    for (size_t i = 0; i < payload_size; i += rocc->inputOperandSize) {
                        T value = 0;
                        memcpy(&value, &payload_data[i], rocc->inputOperandSize);
                        int index = (rocc->vector_read_offset + i) / rocc->inputOperandSize;
                        rocc->array->setVectorItem(array_id, index, value);
                    }
  
                    rocc->vector_read_offset += payload_size;
  
                    if (rocc->vector_read_offset < rocc->vector_total_size) {

                        // Send the next read request
                        uint64_t cache_line_size = rocc->memInterface->getLineSize();
                        uint32_t request_size = static_cast<uint32_t>(std::min(
                            cache_line_size, 
                            rocc->vector_total_size - rocc->vector_read_offset
                        ));

                        uint64_t next_addr = rocc_cmd->rs1 + rocc->vector_read_offset;
                        auto *load_req = new StandardMem::Read(next_addr, request_size, GOLEM_ROCC_FLAG_SYNC_VECTOR);
                        rocc->memInterface->send(load_req);
                    } else {
                        if (array_id >= 0 && static_cast<uint32_t>(array_id) < rocc->async_vector_loads.size()) {
                            rocc->markArrayLoadReady(static_cast<uint32_t>(array_id), false);
                        }
                        rocc->completeRoCC(0);
                    }
                } break;
                // 处理 write_gm 的主存读取响应
                case 0x10: // Read response data is for GlobalMemory Write
                {
                    rocc->output->verbose(CALL_INFO, 1, 0, "GlobalMemory write-back data received\n");

                    size_t payload_size = ev->size;
                    // print debug data in hex format
                    rocc->output->verbose(CALL_INFO, 1, 0, "Data received from MainMemory \n");
                    
                    // [优化打印] 将数据拼接成字符串，一次性打印
                    if (!ev->data.empty()) {
                        std::string hex_str;
                        char buf[8]; // 临时缓存
                        for (size_t i = 0; i < payload_size; ++i) {
                            // 将每个字节格式化为 "XX " 并追加到字符串
                            snprintf(buf, sizeof(buf), "%02X ", ev->data[i]);
                            hex_str += buf;
                        }
                        // 只调用一次 verbose，这样前缀只会出现一次
                        rocc->output->verbose(CALL_INFO, 10, 0, "Data: %s\n", hex_str.c_str());
                    } else {
                        rocc->output->verbose(CALL_INFO, 1, 0, "[EMPTY]\n");
                    }
                    
                    // 1. 将读取到的数据写入 GlobalMemory
                    // 注意：ev->data 是 std::vector<uint8_t>，直接传给 GlobalMemory 接口
                    // 目标地址 = 基地址 + 当前偏移
                    uint64_t current_dst_addr = rocc->gm_write_dst_addr + rocc->gm_write_offset;
                    
                    // 调用 GlobalMemory 的写接口 (写入本地 GM)
                    rocc->globalMem->wr_to_globalmem(current_dst_addr, payload_size, ev->data);

                    // 2. 更新偏移量
                    rocc->gm_write_offset += payload_size;

                    // 3. 检查是否还有剩余数据需要读取
                    if (rocc->gm_write_offset < rocc->gm_write_total_size) {
                        // 发起下一次读取请求
                        uint64_t cache_line_size = rocc->memInterface->getLineSize();
                        uint32_t request_size = static_cast<uint32_t>(std::min(
                            cache_line_size, 
                            rocc->gm_write_total_size - rocc->gm_write_offset
                        ));

                        // 下一次读取的主存地址 = 初始源地址 (rs1) + 新偏移
                        uint64_t next_src_addr = rocc->curr_cmd->rs1 + rocc->gm_write_offset;
                        
                        auto *load_req = new StandardMem::Read(next_src_addr, request_size, 0x10); // 保持 flag 为 0x2
                        rocc->memInterface->send(load_req);
                    } else {
                        // 全部传输完成
                        rocc->output->verbose(CALL_INFO, 9, 0, "write_gm completed.\n");
                        rocc->completeRoCC(0);
                    }
                } break;

                default:
                {
                    rocc->output->verbose(CALL_INFO, 9, 0,
                                "ERROR: unrecognized read response flag\n");
                    rocc->completeRoCC(1);
                } break;
            }
  
            delete ev;
        }
  
        virtual void handle(StandardMem::WriteResp *ev) {
            out->verbose(CALL_INFO, 9, 0,
                     "-> handle write-response (virt-addr: 0x%" PRI_ADDR ")\n", ev->vAddr);

            if (ev->getFail()) {
                out->verbose(CALL_INFO, 9, 0,
                       "RoCC store failed, responding with error code 1\n");
                rocc->completeRoCC(1);

            } else {
                
                // Continue sending write requests if there is remaining data
                if (rocc->write_offset < rocc->vector_total_size) {

                    // Calculate the size of the next write request
                    uint64_t cache_line_size = rocc->memInterface->getLineSize();
                    uint32_t request_size = static_cast<uint32_t>(std::min(
                        cache_line_size, 
                        rocc->vector_total_size - rocc->write_offset
                    ));
  
                    // Prepare the next chunk of data to write
                    std::vector<uint8_t> data_chunk(
                        rocc->outputPayload.begin() + rocc->write_offset,
                        rocc->outputPayload.begin() + rocc->write_offset + request_size
                    );
      
                    // Compute the next physical address to write to
                    uint64_t next_addr = rocc->curr_cmd->rs1 + rocc->write_offset;
      
                    // Create a new write request
                    auto* store_req = new StandardMem::Write(
                        next_addr, request_size, data_chunk,
                        false, 0, rocc->curr_cmd->rs1, 0, 0
                    );
      
                    // Send the write request
                    rocc->memInterface->send(store_req);
      
                    // Update the write offset
                    rocc->write_offset += request_size;
                } else {
                    // All data has been written; complete the RoCC command
                    rocc->completeRoCC(0);
                }
            }
            delete ev;
        }
  
    private:
        RoCCAnalog *rocc;
    };
  
    void processIncomingDataCacheEvent(StandardMem::Request *ev) {
        output->verbose(CALL_INFO, 9, 0,
                      "received incoming data cache request -> "
                      "processIncomingDataCacheEvent()\n");
  
        assert(ev != nullptr);
        assert(std_mem_handlers != nullptr);
  
        ev->handle(std_mem_handlers);
        output->verbose(CALL_INFO, 9, 0,
                      "completed pass off to incoming handlers\n");
    }

    size_t defaultRemoteLength() const {
        size_t bytes = static_cast<size_t>(arrayInputSize) * static_cast<size_t>(inputOperandSize);
        return (bytes == 0) ? 1 : bytes;
    }

    size_t resolveRemoteLength() const {
        // 统一不再使用 rd 寄存器覆盖长度，避免目标寄存器号（如 x15）无意间成为传输长度。
        // 优先使用通过 mvm.slen 设置的 remoteTransferLength；未设置则退回到一个“输入向量”大小。
        size_t chosen = (remoteTransferLength != 0) ? remoteTransferLength : defaultRemoteLength();
        return (chosen == 0) ? 1 : chosen;
    }
  
private:
    struct InflightComputeState {
        SST::Vanadis::RoCCCommand* cmd = nullptr;
        uint64_t start_cycle = 0;
        bool async_mode = false;
    };

    struct AsyncComputeState {
        bool submitted = false;
        bool completed = false;
        uint64_t rd_val = 0;
    };

    struct AsyncArrayLoadState {
        bool inflight = false;
        uint64_t base_addr = 0;
        uint64_t total_size = 0;
        uint64_t offset = 0;
        uint64_t request_tag = 0;
        uint64_t command_id = 0;
        uint32_t array_id = 0;
        bool ready = false;
        bool failed = false;
        bool local_request_inflight = false;
        bool array_request_inflight = false;
        bool completes_command = false;
        std::vector<uint8_t> payload;
    };

    struct LegacyOutputStoreState {
        bool active = false;
        bool array_request_inflight = false;
        bool local_request_inflight = false;
        uint64_t command_id = 0;
        uint64_t dest_addr = 0;
        uint64_t offset = 0;
        uint64_t request_tag = 0;
        uint32_t array_id = 0;
        std::vector<uint8_t> payload;
    };

    bool isAsyncArrayLoadCommand(const SST::Vanadis::RoCCCommand* cmd) const {
        if (!enable_async_array_load) {
            return false;
        }
        if (cmd == nullptr || cmd->inst == nullptr) {
            return false;
        }
        const uint8_t op = cmd->inst->func7;
        return (cmd->inst->rd == 0) && (op == 0x7 || op == 0x8);
    }

    bool isArrayLoadInflight(uint32_t array_id) const {
        if (array_id >= async_matrix_loads.size() || array_id >= async_vector_loads.size()) {
            return false;
        }
        return async_matrix_loads[array_id].inflight || async_vector_loads[array_id].inflight;
    }

    bool hasArrayLoadFailure(uint32_t array_id) const {
        if (array_id >= async_matrix_loads.size() || array_id >= async_vector_loads.size()) {
            return false;
        }
        return async_matrix_loads[array_id].failed || async_vector_loads[array_id].failed;
    }

    void markArrayLoadReady(uint32_t array_id, bool is_matrix) {
        if (array_id >= async_matrix_loads.size() || array_id >= async_vector_loads.size()) {
            return;
        }
        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        state = AsyncArrayLoadState{};
        state.ready = true;
    }

    void markAsyncLoadFailed(uint32_t array_id, bool is_matrix) {
        if (array_id >= async_matrix_loads.size() || array_id >= async_vector_loads.size()) {
            return;
        }
        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        const bool completes_command = state.completes_command;
        const uint64_t command_id = state.command_id;
        state = AsyncArrayLoadState{};
        state.failed = true;
        output->verbose(CALL_INFO, 0, 0,
            "[RoCC ERROR] async %s load failed on array=%" PRIu32 "\n",
            is_matrix ? "matrix" : "vector",
            array_id);
        if (completes_command && curr_cmd != nullptr &&
            curr_cmd->cmd_id == command_id) {
            completeRoCC(1);
        }
    }

    uint64_t allocateLocalTransferTag() {
        const uint64_t tag = next_local_transfer_tag_++;
        if (next_local_transfer_tag_ == 0) {
            next_local_transfer_tag_ = 1;
        }
        return tag;
    }

    void initializeAsyncArrayLoad(AsyncArrayLoadState& state,
                                  uint32_t array_id,
                                  uint64_t base_addr,
                                  uint64_t total_size,
                                  bool completes_command,
                                  uint64_t command_id) {
        state = AsyncArrayLoadState{};
        state.inflight = true;
        state.array_id = array_id;
        state.base_addr = base_addr;
        state.total_size = total_size;
        state.completes_command = completes_command;
        state.command_id = command_id;
        state.payload.resize(total_size);
    }

    void finishAsyncArrayLoad(uint32_t array_id, bool is_matrix) {
        if (array_id >= async_matrix_loads.size() ||
            array_id >= async_vector_loads.size()) {
            return;
        }
        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        const bool completes_command = state.completes_command;
        const uint64_t command_id = state.command_id;
        markArrayLoadReady(array_id, is_matrix);
        if (completes_command && curr_cmd != nullptr &&
            curr_cmd->cmd_id == command_id) {
            completeRoCC(0);
        }
    }

    void progressAsyncArrayLoad(uint32_t array_id, bool is_matrix) {
        if (array_id >= async_matrix_loads.size() ||
            array_id >= async_vector_loads.size()) {
            return;
        }
        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        if (!state.inflight || state.local_request_inflight ||
            state.array_request_inflight) {
            return;
        }
        if (inputOperandSize == 0 || state.offset > state.total_size) {
            markAsyncLoadFailed(array_id, is_matrix);
            return;
        }

        if (state.offset < state.total_size) {
            const size_t max_request =
                std::max<size_t>(globalMem->localMaxRequestBytes(), 1);
            const size_t chunk_size = std::min<size_t>(
                max_request, state.total_size - state.offset);
            const uint64_t chunk_offset = state.offset;
            const uint64_t tag = allocateLocalTransferTag();
            const bool accepted = globalMem->localReadAsync(
                state.base_addr + chunk_offset, chunk_size,
                LocalMemoryClient::RoCC, tag,
                [this, array_id, is_matrix, chunk_offset, chunk_size](
                    bool success, uint64_t callback_tag,
                    const std::vector<uint8_t>& data) {
                    auto& callback_state = is_matrix
                        ? async_matrix_loads[array_id]
                        : async_vector_loads[array_id];
                    if (!callback_state.inflight ||
                        callback_state.request_tag != callback_tag) {
                        return;
                    }
                    callback_state.local_request_inflight = false;
                    if (!success || data.size() != chunk_size ||
                        callback_state.offset != chunk_offset) {
                        markAsyncLoadFailed(array_id, is_matrix);
                        return;
                    }
                    std::copy(data.begin(), data.end(),
                              callback_state.payload.begin() + chunk_offset);
                    callback_state.offset += chunk_size;
                    progressAsyncArrayLoad(array_id, is_matrix);
                });
            if (accepted) {
                state.local_request_inflight = true;
                state.request_tag = tag;
            }
            return;
        }

        std::vector<double> values(state.total_size / inputOperandSize, 0.0);
        for (size_t i = 0; i < state.total_size; i += inputOperandSize) {
            T value = 0;
            memcpy(&value, &state.payload[i],
                   std::min<size_t>(sizeof(T), inputOperandSize));
            values[i / inputOperandSize] = static_cast<double>(value);
        }
        const uint64_t tag = allocateLocalTransferTag();
        auto callback = [this, array_id, is_matrix](bool success,
                                                    uint64_t callback_tag) {
            auto& callback_state = is_matrix
                ? async_matrix_loads[array_id]
                : async_vector_loads[array_id];
            if (!callback_state.inflight ||
                callback_state.request_tag != callback_tag) {
                return;
            }
            callback_state.array_request_inflight = false;
            if (!success) {
                markAsyncLoadFailed(array_id, is_matrix);
                return;
            }
            finishAsyncArrayLoad(array_id, is_matrix);
        };
        const bool accepted = is_matrix
            ? array->programMatrixAsync(
                  array_id, values, inputOperandSize, tag, callback)
            : array->programInputAsync(
                  array_id, values, inputOperandSize, tag, callback);
        if (accepted) {
            state.array_request_inflight = true;
            state.request_tag = tag;
        }
    }

    bool tryCompleteAsyncArrayLoads(uint64_t) {
        bool progressed = false;
        for (uint32_t array_id = 0; array_id < async_matrix_loads.size(); ++array_id) {
            auto& mstate = async_matrix_loads[array_id];
            if (mstate.inflight) {
                progressAsyncArrayLoad(array_id, true);
                progressed = true;
            }
            auto& vstate = async_vector_loads[array_id];
            if (vstate.inflight) {
                progressAsyncArrayLoad(array_id, false);
                progressed = true;
            }
        }
        return progressed;
    }

    bool tryIssueAsyncArrayLoadCommand(uint64_t cycle) {
        if (roccCmd_q.empty()) {
            return false;
        }
        auto* cmd = roccCmd_q.front();
        if (!isAsyncArrayLoadCommand(cmd)) {
            return false;
        }

        const bool is_matrix = (cmd->inst->func7 == 0x8);
        const uint32_t array_id = static_cast<uint32_t>(cmd->rs2);
        if (array_id >= static_cast<uint32_t>(numArrays)) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
            roccCmd_q.pop_front();
            delete cmd;
            return true;
        }

        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        if (state.inflight) {
            return false;
        }
        if (isArrayComputeInflight(array_id)) {
            return false;
        }

        const uint64_t total_size = is_matrix
            ? static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(arrayOutputSize) * static_cast<uint64_t>(inputOperandSize)
            : static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(inputOperandSize);
        initializeAsyncArrayLoad(
            state, array_id, cmd->rs1, total_size, false, 0);

        if (total_size == 0) {
            markArrayLoadReady(array_id, is_matrix);
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
            roccCmd_q.pop_front();
            delete cmd;
            return true;
        }

        progressAsyncArrayLoad(array_id, is_matrix);
        enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));

        roccCmd_q.pop_front();
        delete cmd;
        return true;
    }

    void beginBlockingArrayLoad(bool is_matrix) {
        if (curr_cmd == nullptr ||
            curr_cmd->rs2 >= static_cast<uint64_t>(numArrays)) {
            completeRoCC(1);
            return;
        }
        const uint32_t array_id = static_cast<uint32_t>(curr_cmd->rs2);
        auto& state = is_matrix
            ? async_matrix_loads[array_id]
            : async_vector_loads[array_id];
        if (state.inflight) {
            progressAsyncArrayLoad(array_id, is_matrix);
            return;
        }
        const uint64_t total_size = is_matrix
            ? static_cast<uint64_t>(arrayInputSize) *
                  static_cast<uint64_t>(arrayOutputSize) *
                  static_cast<uint64_t>(inputOperandSize)
            : static_cast<uint64_t>(arrayInputSize) *
                  static_cast<uint64_t>(inputOperandSize);
        initializeAsyncArrayLoad(
            state, array_id, curr_cmd->rs1, total_size, true, curr_cmd->cmd_id);
        progressAsyncArrayLoad(array_id, is_matrix);
    }

    void finishLegacyOutputStore(bool success) {
        const uint64_t command_id = legacy_output_store_.command_id;
        legacy_output_store_ = LegacyOutputStoreState{};
        if (curr_cmd != nullptr && curr_cmd->cmd_id == command_id) {
            completeRoCC(success ? 0 : 1);
        }
    }

    void progressLegacyOutputStore() {
        auto& state = legacy_output_store_;
        if (!state.active || state.array_request_inflight ||
            state.local_request_inflight) {
            return;
        }
        if (state.payload.empty()) {
            const uint64_t tag = allocateLocalTransferTag();
            const bool accepted = array->readOutputBytesAsync(
                state.array_id, outputOperandSize, tag,
                [this](bool success, uint64_t callback_tag,
                       const std::vector<uint8_t>& bytes) {
                    auto& callback_state = legacy_output_store_;
                    if (!callback_state.active ||
                        callback_state.request_tag != callback_tag) {
                        return;
                    }
                    callback_state.array_request_inflight = false;
                    if (!success ||
                        bytes.size() != static_cast<size_t>(arrayOutputSize) *
                            outputOperandSize ||
                        outputOperandSize == 0) {
                        finishLegacyOutputStore(false);
                        return;
                    }
                    callback_state.payload = bytes;
                    progressLegacyOutputStore();
                });
            if (accepted) {
                state.array_request_inflight = true;
                state.request_tag = tag;
            }
            return;
        }

        if (state.offset >= state.payload.size()) {
            finishLegacyOutputStore(true);
            return;
        }
        const size_t max_request =
            std::max<size_t>(globalMem->localMaxRequestBytes(), 1);
        const size_t chunk_size = std::min<size_t>(
            max_request, state.payload.size() - state.offset);
        const uint64_t chunk_offset = state.offset;
        std::vector<uint8_t> chunk(
            state.payload.begin() + chunk_offset,
            state.payload.begin() + chunk_offset + chunk_size);
        const uint64_t tag = allocateLocalTransferTag();
        const bool accepted = globalMem->localWriteAsync(
            state.dest_addr + chunk_offset, chunk, LocalMemoryClient::RoCC, tag,
            [this, chunk_offset, chunk_size](bool success,
                                             uint64_t callback_tag) {
                auto& callback_state = legacy_output_store_;
                if (!callback_state.active ||
                    callback_state.request_tag != callback_tag) {
                    return;
                }
                callback_state.local_request_inflight = false;
                if (!success || callback_state.offset != chunk_offset) {
                    finishLegacyOutputStore(false);
                    return;
                }
                callback_state.offset += chunk_size;
                progressLegacyOutputStore();
            });
        if (accepted) {
            state.local_request_inflight = true;
            state.request_tag = tag;
        }
    }

    void enqueueResponse(SST::Vanadis::RoCCResponse *resp) {
        if (resp != nullptr) {
            resp_q.push_back(resp);
        }
    }

    std::deque<SST::Vanadis::RoCCCommand *> roccCmd_q;
    std::deque<SST::Vanadis::RoCCResponse *> resp_q;
    std::vector<AsyncComputeState> async_compute_states;
    bool busy;
    SST::Vanadis::RoCCCommand *curr_cmd;
  
    StandardMemHandlers *std_mem_handlers;
    StandardMem *memInterface;
  
    int max_instructions;
  
    Golem::ComputeArray *array;
    std::vector<char> arrayStates;
    SST::Golem::GlobalMemoryAPI *globalMem; //GlobalMemory子组件指针
    SST::Golem::GroupCtrlAPI *groupCtrl;
    SST::Golem::RequestSchedulerAPI *requestScheduler;
    SST::Golem::WorkerCommandProcessorAPI *workerCommandProcessor;
    SST::Golem::SFUAPI *sfu;
    std::unordered_map<uint64_t, ManagerTensorJobState> managerTensorJobs_;
    std::unordered_map<uint64_t, ManagerAttentionJobState> managerAttentionJobs_;
    std::unique_ptr<AttentionWorkerState> attentionWorker_;
    std::vector<uint8_t> attentionArrayPending_;
    bool sfuWaitBlocked_ = false;
    uint64_t sfuWaitBlockedCmdId_ = 0;
    uint64_t sfuWaitBlockedUntilTick_ = 0;
    uint64_t coreID;
    uint64_t attentionWindowOffset_;
    uint64_t attentionWindowBytes_;
    bool attentionPvMatrixBroadcast_;

  
    // Tile Parameters
    int numArrays;
    int arrayInputSize;
    int arrayOutputSize;
    int inputOperandSize;
    int outputOperandSize;
  
    // MMIO range delimiters
    uint64_t mmioStartAddr;
    uint64_t inputDataSize;
    uint64_t outputDataSize;
    uint64_t inputTotalSize;
    uint64_t outputTotalSize;
    uint64_t inputStartAddr;
    uint64_t outputStartAddr;    
    // virtual void compute(uint32_t arrayID) override {
    //     auto& inputVector = inputVectors[arrayID];
    //     auto& outputVector = outputVectors[arrayID];
    //     auto& matrix = matrixData[arrayID];

    //     // Ensure output vector is correctly sized
    //     outputVector.resize(outputArraySize);

    //     // Initialize output vector to zero
    //     std::fill(outputVector.begin(), outputVector.end(), T());

    //     // Print input vector
    //     out.verbose(CALL_INFO, 2, 0, "MVM for array %u:\n\n", arrayID);
    //     for (uint32_t col = 0; col < inputArraySize; col++) {
    //         printValue(inputVector[col]);
    //     }
    //     out.verbose(CALL_INFO, 2, 0, "\n\n");

    //     // Perform matrix-vector multiplication
    //     for (uint32_t row = 0; row < outputArraySize; row++) {
    //         for (uint32_t col = 0; col < inputArraySize; col++) {
    //             outputVector[row] += matrix[row * inputArraySize + col] * inputVector[col];
    //             printValue(matrix[row * inputArraySize + col]);
    //         }
    //         out.verbose(CALL_INFO, 2, 0, "  ");
    //         printValue(outputVector[row]);
    //         out.verbose(CALL_INFO, 2, 0, "\n");
    //     }
    //     out.verbose(CALL_INFO, 2, 0, "\n\n");
    // }

    // virtual SimTime_t getArrayLatency(uint32_t arrayID) override {
    //     return 1;
    // }
  
    // Variables to keep track of read/write request progress
    uint64_t matrix_read_offset;
    uint64_t matrix_total_size;
    uint64_t vector_read_offset;
    uint64_t vector_total_size;
    uint64_t write_offset;
    std::vector<uint8_t> outputPayload;
    size_t remoteTransferLength;
    uint64_t remoteStoreCompletionToken = 0;
    bool enable_async_array_load = true;
    bool sfuEnable = false;

    // 用于 write_gm 的状态变量
    uint64_t gm_write_dst_addr;   // GlobalMemory 的目标地址 (rs2)
    uint64_t gm_write_offset;     // 当前已处理的字节偏移量
    uint64_t gm_write_total_size; // 总共需要传输的字节数

    Statistics::Statistic<uint64_t>* stat_cycles_mvm_set;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_l;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_s;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_mv;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_ovec2gm;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_gm2ivec;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_gm2imat;
    Statistics::Statistic<uint64_t>* stat_cycles_remote_st;
    Statistics::Statistic<uint64_t>* stat_cycles_remote_ld;
    Statistics::Statistic<uint64_t>* statTensorManagerJobsIssued_;
    Statistics::Statistic<uint64_t>* statTensorManagerWorkersMapped_;
    Statistics::Statistic<uint64_t>* statTensorManagerRowsDispatched_;
    Statistics::Statistic<uint64_t>* statTensorManagerRowsCompleted_;
    Statistics::Statistic<uint64_t>* statTensorManagerJobsCompleted_;
    Statistics::Statistic<uint64_t>* statTensorManagerDescriptorAcceptTick_;
    Statistics::Statistic<uint64_t>* statTensorManagerBandDispatchTick_;
    Statistics::Statistic<uint64_t>* statTensorManagerCompletionReceivedTick_;
    Statistics::Statistic<uint64_t>* statTensorManagerCompleteTick_;
    Statistics::Statistic<uint64_t>* statTensorManagerWaitObservedTick_;
    Statistics::Statistic<uint64_t>* statAttentionManagerJobsIssued_;
    Statistics::Statistic<uint64_t>* statAttentionManagerJobsCompleted_;
    Statistics::Statistic<uint64_t>* statAttentionManagerBandsCompleted_;
    Statistics::Statistic<uint64_t>* statAttentionManagerBandCompletionsReceived_;
    Statistics::Statistic<uint64_t>* statAttentionTensorJobsCompleted_;
    Statistics::Statistic<uint64_t>* statAttentionManagerDescriptorAcceptTick_;
    Statistics::Statistic<uint64_t>* statAttentionManagerDispatchTick_;
    Statistics::Statistic<uint64_t>* statAttentionManagerLocalCompleteTick_;
    Statistics::Statistic<uint64_t>* statAttentionManagerBandCompletionReceivedTick_;
    Statistics::Statistic<uint64_t>* statAttentionTensorCompleteTick_;
    Statistics::Statistic<uint64_t>* statAttentionManagerWaitObservedTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerDispatchAcceptTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerQkTileCompleteTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerSoftmaxTileCompleteTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerPvTileCompleteTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerOutputDmaAckTick_;
    Statistics::Statistic<uint64_t>* statAttentionPvMatrixBroadcasts_;
    Statistics::Statistic<uint64_t>* statAttentionQkArrayOps_;
    Statistics::Statistic<uint64_t>* statAttentionPvArrayOps_;
    Statistics::Statistic<uint64_t>* statAttentionSpHbmBytes_;

    uint64_t StartTickCycle;
    uint64_t LastTickCycle;

    uint64_t latency_mvm_ovec2gm;
    uint64_t latency_mvm_gm2ivec;
    uint64_t latency_mvm_gm2imat;
    uint64_t latency_remote_st;
    uint64_t latency_remote_ld;
    std::vector<InflightComputeState> inflight_compute_cmds;
    std::vector<AsyncArrayLoadState> async_matrix_loads;
    std::vector<AsyncArrayLoadState> async_vector_loads;
    LegacyOutputStoreState legacy_output_store_;
    uint64_t next_local_transfer_tag_ = 1;
    bool progress_heartbeat = false;
    uint64_t progress_interval_cycles = 50000;
    uint64_t progress_total_mvm_ops = 0;
    uint64_t mvm_ops_completed = 0;
    uint64_t progress_next_cycle = 0;
    uint64_t progress_last_percent = 101;
};
  
} // namespace Golem
} // namespace SST
  
#endif
