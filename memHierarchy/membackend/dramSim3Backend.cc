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

#include <sst/core/sst_config.h>
#include "sst/elements/memHierarchy/util.h"
#include "membackend/dramSim3Backend.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace SST;
using namespace SST::MemHierarchy;

namespace {
std::vector<uint64_t> g_dramsim_backend_read_latencies;
bool g_dramsim_backend_summary_printed = false;
uint64_t g_dramsim_backend_first_read_arrival_cycle = UINT64_MAX;
uint64_t g_dramsim_backend_last_read_complete_cycle = 0;
uint64_t g_dramsim_backend_completed_reads = 0;
uint64_t g_dramsim_backend_outstanding_reads = 0;
uint64_t g_dramsim_backend_read_active_start_cycle = 0;
uint64_t g_dramsim_backend_read_active_cycles = 0;

uint64_t percentile_of_sorted(const std::vector<uint64_t>& vals, double pct) {
    if (vals.empty()) return 0;
    size_t idx = static_cast<size_t>(pct * static_cast<double>(vals.size() - 1) + 0.5);
    if (idx >= vals.size()) idx = vals.size() - 1;
    return vals[idx];
}
}

DRAMSim3Memory::DRAMSim3Memory(ComponentId_t id, Params &params) : SimpleMemBackend(id, params){
    std::string configIniFilename = params.find<std::string>("config_ini", NO_STRING_DEFINED);
    if(NO_STRING_DEFINED == configIniFilename)
        output->fatal(CALL_INFO, -1, "Model must define a 'config_ini' file parameter\n");
    std::string outputDirname = params.find<std::string>("output_dir", "./");

    readCB = std::bind(&DRAMSim3Memory::dramSimDone, this, 0, std::placeholders::_1, 0);
    writeCB = std::bind(&DRAMSim3Memory::dramSimDone, this, 0, std::placeholders::_1, 0);

    UnitAlgebra ramSize = UnitAlgebra(params.find<std::string>("mem_size", "0B"));
    if (ramSize.getRoundedValue() % (1024*1024) != 0) {
        output->fatal(CALL_INFO, -1, "For DRAMSim3, backend.mem_size must be a multiple of 1MiB. Note: for units in base-10 use 'MB', for base-2 use 'MiB'. You specified '%s'\n", ramSize.toString().c_str());
    }
    unsigned int ramSizeMiB = ramSize.getRoundedValue() / (1024*1024);

    memSystem = new dramsim3::MemorySystem(configIniFilename, outputDirname, readCB, writeCB);
}


bool DRAMSim3Memory::issueRequest(ReqId id, Addr addr, bool isWrite, unsigned ){
    bool ok = memSystem->WillAcceptTransaction(addr, isWrite);
    if (!ok) return false;
    ok = memSystem->AddTransaction(addr, isWrite);
    if (!ok) return false;  // This *SHOULD* always be ok
#ifdef __SST_DEBUG_OUTPUT__
    output->debug(_L10_, "Issued transaction for address %" PRIx64 "\n", (Addr)addr);
#endif
    dramReqs[addr].push_back(id);
    dramReqIssueCycles[addr].push_back(currentCycle_);
    dramReqIsWrite[addr].push_back(isWrite);
    if (!isWrite) {
        g_dramsim_backend_first_read_arrival_cycle =
            std::min(g_dramsim_backend_first_read_arrival_cycle, currentCycle_);
        if (g_dramsim_backend_outstanding_reads == 0) {
            g_dramsim_backend_read_active_start_cycle = currentCycle_;
        }
        g_dramsim_backend_outstanding_reads++;
    }
    return true;
}



bool DRAMSim3Memory::clock(Cycle_t cycle){
    currentCycle_ = static_cast<uint64_t>(cycle);
    memSystem->ClockTick();
    return false;
}



void DRAMSim3Memory::finish(){
    if (!g_dramsim_backend_summary_printed && !g_dramsim_backend_read_latencies.empty()) {
        std::sort(g_dramsim_backend_read_latencies.begin(), g_dramsim_backend_read_latencies.end());
        unsigned long long sum = 0;
        for (auto v : g_dramsim_backend_read_latencies) sum += v;
        const uint64_t avg = static_cast<uint64_t>(sum / g_dramsim_backend_read_latencies.size());
        const uint64_t p95 = percentile_of_sorted(g_dramsim_backend_read_latencies, 0.95);
        const uint64_t p99 = percentile_of_sorted(g_dramsim_backend_read_latencies, 0.99);
        const uint64_t maxv = g_dramsim_backend_read_latencies.back();
        std::cout << "DRAMSIM3_BACKEND_READ_LATENCY_GLOBAL count=" << g_dramsim_backend_read_latencies.size()
                  << " avg_cycles=" << avg
                  << " p95_cycles=" << p95
                  << " p99_cycles=" << p99
                  << " max_cycles=" << maxv << std::endl;
        const uint64_t first = g_dramsim_backend_first_read_arrival_cycle;
        const uint64_t last = g_dramsim_backend_last_read_complete_cycle;
        const uint64_t window = (last > first) ? (last - first) : 1;
        std::cout << "DRAMSIM3_BACKEND_READ_SERVICE_WINDOW_GLOBAL count=" << g_dramsim_backend_completed_reads
                  << " first_arrival_cycle=" << first
                  << " last_complete_cycle=" << last
                  << " window_cycles=" << window << std::endl;
        std::cout << "DRAMSIM3_BACKEND_READ_ACTIVE_WINDOW_GLOBAL count=" << g_dramsim_backend_completed_reads
                  << " active_cycles=" << g_dramsim_backend_read_active_cycles
                  << " outstanding_reads_at_finish=" << g_dramsim_backend_outstanding_reads << std::endl;
        g_dramsim_backend_summary_printed = true;
    }
    memSystem->PrintStats();
}



void DRAMSim3Memory::dramSimDone(unsigned int id, uint64_t addr, uint64_t clockcycle){
    (void)id;
    (void)clockcycle;
    std::deque<ReqId> &reqs = dramReqs[addr];
    std::deque<uint64_t> &issueCycles = dramReqIssueCycles[addr];
    std::deque<bool> &isWrites = dramReqIsWrite[addr];
#ifdef __SST_DEBUG_OUTPUT__
    output->debug(_L10_, "Memory Request for %" PRIx64 " Finished [%zu reqs]\n", (Addr)addr, reqs.size());
#endif
    if (reqs.size() == 0) 
        output->fatal(CALL_INFO, -1, "Error: reqs.size() is 0 at DRAMSim3Memory done\n");
    if (issueCycles.size() == 0)
        output->fatal(CALL_INFO, -1, "Error: issueCycles.size() is 0 at DRAMSim3Memory done\n");
    if (isWrites.size() == 0)
        output->fatal(CALL_INFO, -1, "Error: isWrites.size() is 0 at DRAMSim3Memory done\n");
    ReqId reqId = reqs.front();
    reqs.pop_front();
    const uint64_t issueCycle = issueCycles.front();
    issueCycles.pop_front();
    const bool isWrite = isWrites.front();
    isWrites.pop_front();
    if (!isWrite && currentCycle_ >= issueCycle) {
        g_dramsim_backend_read_latencies.push_back(currentCycle_ - issueCycle);
        g_dramsim_backend_completed_reads++;
        g_dramsim_backend_last_read_complete_cycle =
            std::max(g_dramsim_backend_last_read_complete_cycle, currentCycle_);
        if (g_dramsim_backend_outstanding_reads == 0) {
            output->fatal(CALL_INFO, -1, "Error: outstanding read count underflow in DRAMSim3Memory done\n");
        }
        g_dramsim_backend_outstanding_reads--;
        if (g_dramsim_backend_outstanding_reads == 0) {
            g_dramsim_backend_read_active_cycles += currentCycle_ - g_dramsim_backend_read_active_start_cycle;
        }
    }
    if(0 == reqs.size())
        dramReqs.erase(addr);
    if(0 == issueCycles.size())
        dramReqIssueCycles.erase(addr);
    if(0 == isWrites.size())
        dramReqIsWrite.erase(addr);

    handleMemResponse(reqId);
}
