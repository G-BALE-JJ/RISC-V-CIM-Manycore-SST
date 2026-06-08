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
#include <sst/elements/golem/workercmdproc/workercmdproc.h>
#include <sst/elements/vanadis/rocc/vroccinterface.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>


#include <cinttypes>
#include <cstdint>
#include <limits>
#include <queue>
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
    }
  
    // Main clock cycle tick function
    //每一个时钟周期调用一次tick
    void tick(uint64_t cycle) override {
        output->verbose(CALL_INFO, 16, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
        LastTickCycle = cycle;
        // Keep draining async array-load commands even while a synchronous command is in flight.
        tryIssueAsyncArrayLoadCommand(cycle);
        tryCompleteAsyncArrayLoads(cycle);
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
            state.inflight = true;
            state.ready = false;
            state.failed = false;
            state.array_id = array_id;
            state.base_addr = is_matrix ? base_addr : (base_addr + idx * vector_stride);
            state.total_size = is_matrix
                ? static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(arrayOutputSize) * static_cast<uint64_t>(inputOperandSize)
                : vector_stride;
            state.ready_cycle = cycle + (is_matrix ? latency_mvm_gm2imat : latency_mvm_gm2ivec);
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

    void OutputvectorStore(uint64_t cycle) {
        uint64_t cycles_elapsed = (cycle >= StartTickCycle) ? (cycle - StartTickCycle + 1) : 0;

        if (cycles_elapsed < latency_mvm_ovec2gm) {
            return;
        }
        // 将阵列 rs2 的输出向量写入 GlobalMemory 本地地址 rs1
        uint64_t dest_addr = curr_cmd->rs1;  // 本地GlobalMemory目标物理地址
        uint64_t array_id  = curr_cmd->rs2;  // 阵列ID（源计算阵列编号）
        // 计算输出向量总字节数 = 输出元素个数 × 每个元素字节大小
        size_t vector_length = arrayOutputSize * outputOperandSize;
        outputPayload.resize(vector_length);
        // 获取阵列 rs2 的输出向量引用，并填充到字节缓冲区
        auto& outputVector = *static_cast<std::vector<T>*>(array->getOutputVector(array_id));
        for (size_t i = 0; i < static_cast<size_t>(arrayOutputSize); ++i) {
            T value = outputVector[i];
            uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(&value);
            for (size_t j = 0; j < static_cast<size_t>(outputOperandSize); ++j) {
                outputPayload[i * outputOperandSize + j] = byte_ptr[j];
            }
        }
        // 输出调试信息：打印存储的向量内容
        output->verbose(CALL_INFO, 9, 0, 
                        "Local store vector from array %" PRIu64 " to local address 0x%" PRIx64 ":\n", 
                        array_id, dest_addr);
        for (size_t i = 0; i < static_cast<size_t>(arrayOutputSize); ++i) {
            if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
                output->verbose(CALL_INFO, 9, 0, "%f ", static_cast<double>(outputVector[i]));
            } else {
                output->verbose(CALL_INFO, 9, 0, "%lld ", static_cast<long long>(outputVector[i]));
            }
        }
        output->verbose(CALL_INFO, 9, 0, "\n");
        // 调用 GlobalMemory 接口将字节数据写入本地 GlobalMemory 存储
        globalMem->wr_to_globalmem(dest_addr, vector_length, outputPayload);
        // 完成指令执行（rd 寄存器返回值 0 表示成功）
        completeRoCC(0);
    }

    void IntputvectorLoad(uint64_t cycle) {
        uint64_t cycles_elapsed = (cycle >= StartTickCycle) ? (cycle - StartTickCycle + 1) : 0;

        if (cycles_elapsed < latency_mvm_gm2ivec) {
            return;
        }
        // 从本地 GlobalMemory 地址 rs1 加载数据到阵列 rs2 的输入向量
        uint64_t src_addr = curr_cmd->rs1;   // 本地GlobalMemory源物理地址
        uint64_t array_id = curr_cmd->rs2;   // 阵列ID（目标计算阵列编号）
        // 计算输入向量总字节数 = 输入元素个数 × 每个元素字节大小
        size_t vector_length = arrayInputSize * inputOperandSize;
        std::vector<uint8_t> inputData;
        inputData.reserve(vector_length);
        // 从本地 GlobalMemory 读取指定长度的数据到字节缓冲区
        globalMem->rd_from_globalmem(src_addr, vector_length, inputData);
        // 将读取的字节数据解析为 T 类型元素，设置给阵列 rs2 的输入向量
        for (size_t i = 0; i < vector_length; i += inputOperandSize) {
            T value = 0;
            memcpy(&value, &inputData[i], inputOperandSize);
            size_t index = i / inputOperandSize;
            array->setVectorItem(array_id, static_cast<int>(index), value);
        }
        // 输出调试信息：打印加载的向量内容
        auto& inputVector = *static_cast<std::vector<T>*>(array->getInputVector(array_id));
        output->verbose(CALL_INFO, 9, 0, 
                        "Local load vector to array %" PRIu64 " from local address 0x%" PRIx64 ":\n", 
                        array_id, src_addr);
        for (int i = 0; i < arrayInputSize; ++i) {
            if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
                output->verbose(CALL_INFO, 9, 0, "%f ", static_cast<double>(inputVector[i]));
            } else {
                output->verbose(CALL_INFO, 9, 0, "%ld ", static_cast<long>(inputVector[i]));
            }
        }
        output->verbose(CALL_INFO, 9, 0, "\n");
        if (array_id < async_vector_loads.size()) {
            markArrayLoadReady(static_cast<uint32_t>(array_id), false);
        }
        // 指令执行完成
        completeRoCC(0);
    }

    void InputMatrixLoad(uint64_t cycle) {
        uint64_t cycles_elapsed = (cycle >= StartTickCycle) ? (cycle - StartTickCycle + 1) : 0;
        if (cycles_elapsed < latency_mvm_gm2imat) {
            return;
        }

        // 2. 解析指令参数
        uint64_t src_addr = curr_cmd->rs1;   // GlobalMemory 源地址
        uint64_t array_id = curr_cmd->rs2;   // 目标阵列 ID

        // 3. 计算矩阵总字节数
        // 矩阵大小 = 行数(InputSize) * 列数(OutputSize) * 元素大小
        size_t matrix_size = arrayInputSize * arrayOutputSize * inputOperandSize;

        // 4. 准备接收缓冲区
        std::vector<uint8_t> matrixData;
        matrixData.reserve(matrix_size);

        // 5. 从 GlobalMemory 读取数据 (同步/阻塞式读取，或者假设数据立即可用)
        // 注意：根据你原有的 IntputvectorLoad 实现，这里似乎假设 rd_from_globalmem 是立即完成填充 vector 的
        globalMem->rd_from_globalmem(src_addr, matrix_size, matrixData);

        // 6. 将数据填入计算阵列 (Set Matrix Item)
        // 这里的逻辑参考了 handle() 中 case 0x0 的逻辑
        for (size_t i = 0; i < matrix_size; i += inputOperandSize) {
            T value = 0;
            // 字节转类型 T (int64_t 或 float)
            memcpy(&value, &matrixData[i], inputOperandSize);
            
            // 计算 flat index (扁平化索引)
            size_t index = i / inputOperandSize;
            
            // 写入阵列
            array->setMatrixItem(static_cast<uint32_t>(array_id), static_cast<int>(index), value);
        }

        // 7. 打印调试信息
        output->verbose(CALL_INFO, 9, 0, 
                        "mvm.g2m: Loaded matrix to array %" PRIu64 " from GM addr 0x%" PRIx64 "\n", 
                        array_id, src_addr);
        if (array_id < async_matrix_loads.size()) {
            markArrayLoadReady(static_cast<uint32_t>(array_id), true);
        }
        // 8. 完成指令
        completeRoCC(0);
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
                case 0x9: stat_cycles_remote_st->addData(cycles_spent); break;
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
        uint64_t ready_cycle = 0;
        uint32_t array_id = 0;
        bool ready = false;
        bool failed = false;
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
        state.inflight = false;
        state.ready = true;
        state.failed = false;
    }

    void markAsyncLoadFailed(uint32_t array_id, bool is_matrix) {
        if (array_id >= async_matrix_loads.size() || array_id >= async_vector_loads.size()) {
            return;
        }
        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        state.inflight = false;
        state.ready = false;
        state.failed = true;
        output->verbose(CALL_INFO, 0, 0,
            "[RoCC ERROR] async %s load failed on array=%" PRIu32 "\n",
            is_matrix ? "matrix" : "vector",
            array_id);
    }

    void completeAsyncLocalArrayLoad(uint32_t array_id, bool is_matrix) {
        if (array_id >= async_matrix_loads.size() || array_id >= async_vector_loads.size()) {
            return;
        }

        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        if (!state.inflight) {
            return;
        }

        std::vector<uint8_t> payload;
        payload.reserve(state.total_size);
        globalMem->rd_from_globalmem(state.base_addr, state.total_size, payload);
        if (payload.size() < state.total_size) {
            markAsyncLoadFailed(array_id, is_matrix);
            return;
        }
        for (size_t i = 0; i < state.total_size; i += inputOperandSize) {
            T value = 0;
            memcpy(&value, &payload[i], inputOperandSize);
            int index = static_cast<int>(i / inputOperandSize);
            if (is_matrix) {
                array->setMatrixItem(array_id, index, value);
            } else {
                array->setVectorItem(array_id, index, value);
            }
        }
        markArrayLoadReady(array_id, is_matrix);
    }

    bool tryCompleteAsyncArrayLoads(uint64_t cycle) {
        bool progressed = false;
        for (uint32_t array_id = 0; array_id < async_matrix_loads.size(); ++array_id) {
            auto& mstate = async_matrix_loads[array_id];
            if (mstate.inflight && cycle >= mstate.ready_cycle) {
                completeAsyncLocalArrayLoad(array_id, true);
                progressed = true;
            }
            auto& vstate = async_vector_loads[array_id];
            if (vstate.inflight && cycle >= vstate.ready_cycle) {
                completeAsyncLocalArrayLoad(array_id, false);
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

        state.inflight = true;
        state.ready = false;
        state.failed = false;
        state.array_id = array_id;
        state.base_addr = cmd->rs1;
        state.total_size = is_matrix
            ? static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(arrayOutputSize) * static_cast<uint64_t>(inputOperandSize)
            : static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(inputOperandSize);
        state.ready_cycle = cycle + (is_matrix ? latency_mvm_gm2imat : latency_mvm_gm2ivec);

        if (state.total_size == 0) {
            markArrayLoadReady(array_id, is_matrix);
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
            roccCmd_q.pop_front();
            delete cmd;
            return true;
        }

        // Async load from local GlobalMemory is accepted immediately and will
        // complete after the configured local-load latency.
        enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));

        roccCmd_q.pop_front();
        delete cmd;
        return true;
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
    uint64_t coreID;

  
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
    bool enable_async_array_load = true;

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
