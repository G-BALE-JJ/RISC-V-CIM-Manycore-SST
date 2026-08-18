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

#ifndef _H_ANALOG_ROCC_INT
#define _H_ANALOG_ROCC_INT

#include <sst/elements/golem/rocc/roccAnalog.h>

namespace SST {
namespace Golem {

class RoCCAnalogInt : public RoCCAnalog<int64_t> {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        RoCCAnalogInt,
        "golem",
        "RoCCAnalogInt",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Implements a RoCC accelerator interface for the analog core (int version)",
        SST::Golem::RoCCAnalog<int64_t>
    )

    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"memory_interface", "Set the interface to memory", "SST::Interfaces::StandardMem"},
        {"array", "Analog array model", "SST::Golem::ComputeArray"},
        {"global_memory", "Local global memory","SST::Golem::GlobalMemoryAPI"},
        {"group_ctrl", "Group control endpoint", "SST::SubComponent"},
        {"request_scheduler", "Request scheduler endpoint", "SST::SubComponent"})

    SST_ELI_DOCUMENT_PORTS(
        {"dcache_link", "Connects the RoCC frontend to the data cache", {}})

    SST_ELI_DOCUMENT_PARAMS(
        {"verbose", "Set the verbosity of output for the RoCC", "0"},
        {"max_instructions",
         "Set the maximum number of RoCC instructions permitted in the queue", "8"},
        {"clock",
         "Clock frequency for component TimeConverter. MMIOTile is Unclocked but "
         "subcomponents use the TimeConverter", "1GHz"},
        {"mmioAddr", "Address MMIO interface"},
        {"numArrays", "Number of distinct arrays in the tile.", "1"},
        {"arrayInputSize", "Length of input vector (implies array rows)."},
        {"arrayOutputSize", "Length of output vector (implies array columns)."},
        {"inputOperandSize", "Size of input operand in bytes."},
        {"outputOperandSize", "Size of output operand in bytes."},
        {"attention_window_offset", "Offset of the fused Attention Local GM window", "0xC0000"},
        {"attention_window_bytes", "Capacity of the fused Attention Local GM window", "0x10000"},
        {"attention_pv_matrix_broadcast", "Broadcast one PV matrix transfer to all query arrays", "0"},
        {"progress_heartbeat", "Enable lightweight RoCC MVM progress heartbeat logs", "0"},
        {"progress_interval_cycles", "Progress heartbeat interval in RoCC cycles", "50000"},
        {"progress_total_mvm_ops", "Expected total MVM ops for this core", "0"})

    SST_ELI_DOCUMENT_STATISTICS(
        {"roccs_issued", "Count number of RoCC instructions that are issued", "operations", 1},
        {"cycles_mvm_set", "Cycles consumed by mvm.set instructions", "cycles", 1},
        {"cycles_mvm_l", "Cycles consumed by mvm.l instructions", "cycles", 1},
        {"cycles_mvm", "Cycles consumed by mvm compute instructions", "cycles", 1},
        {"cycles_mvm_s", "Cycles consumed by mvm.s instructions", "cycles", 1},
        {"cycles_mvm_mv", "Cycles consumed by mvm.mv instructions", "cycles", 1},
        {"cycles_mvm_ovec2gm", "Cycles consumed by mvm.ovec2gm instructions", "cycles", 1},
        {"cycles_mvm_gm2ivec", "Cycles consumed by mvm.gm2ivec instructions", "cycles", 1},
        {"cycles_mvm_gm2imat", "Cycles consumed by mvm.gm2imat instructions", "cycles", 1},
        {"cycles_remote_st", "Cycles consumed by remote_st instructions", "cycles", 1},
        {"cycles_remote_ld", "Cycles consumed by remote_ld instructions", "cycles", 1},
        {"tensor_manager_jobs_issued", "Tensor jobs accepted by the manager RoCC", "jobs", 1},
        {"tensor_manager_workers_mapped", "Physical worker cores mapped by manager jobs", "workers", 1},
        {"tensor_manager_rows_dispatched", "Tensor rows dispatched by the manager RoCC", "rows", 1},
        {"tensor_manager_rows_completed", "Tensor rows completed at the manager RoCC", "rows", 1},
        {"tensor_manager_jobs_completed", "Tensor jobs completed by the manager RoCC", "jobs", 1},
        {"tensor_manager_descriptor_accept_tick", "Simulation tick when a manager descriptor is accepted", "ticks", 1},
        {"tensor_manager_band_dispatch_tick", "Simulation tick for each manager worker-band dispatch", "ticks", 1},
        {"tensor_manager_completion_received_tick", "Simulation tick for each manager worker completion", "ticks", 1},
        {"tensor_manager_complete_tick", "Simulation tick when all manager workers complete", "ticks", 1},
        {"tensor_manager_wait_observed_tick", "Simulation tick when software observes manager completion", "ticks", 1},
        {"attention_manager_jobs_issued", "Fused Attention jobs issued by manager", "jobs", 1},
        {"attention_manager_jobs_completed", "Fused Attention jobs completed by manager", "jobs", 1},
        {"attention_manager_bands_completed", "Fused Attention bands completed locally", "bands", 1},
        {"attention_manager_band_completions_received", "Attention band completions received by the root manager", "bands", 1},
        {"attention_tensor_jobs_completed", "Fused Attention tensors completed after manager aggregation", "jobs", 1},
        {"attention_manager_descriptor_accept_tick", "Tick when an Attention manager job is accepted", "ticks", 1},
        {"attention_manager_dispatch_tick", "Tick when an Attention manager dispatches all workers", "ticks", 1},
        {"attention_manager_local_complete_tick", "Tick when an Attention manager completes its local band", "ticks", 1},
        {"attention_manager_band_completion_received_tick", "Tick for each Attention manager band received by root", "ticks", 1},
        {"attention_tensor_complete_tick", "Tick when root completes the Attention tensor", "ticks", 1},
        {"attention_manager_wait_observed_tick", "Tick when software observes Attention manager completion", "ticks", 1},
        {"attention_worker_dispatch_accept_tick", "Tick when an Attention worker accepts a dispatch", "ticks", 1},
        {"attention_worker_qk_tile_complete_tick", "Tick when an Attention worker completes a QK tile", "ticks", 1},
        {"attention_worker_softmax_tile_complete_tick", "Tick when an Attention worker completes a Softmax tile", "ticks", 1},
        {"attention_worker_pv_tile_complete_tick", "Tick when an Attention worker completes a PV tile", "ticks", 1},
        {"attention_worker_output_dma_ack_tick", "Tick when an Attention worker receives an output DMA acknowledgement", "ticks", 1},
        {"attention_pv_matrix_broadcasts", "PV matrix group broadcasts", "broadcasts", 1},
        {"attention_qk_array_ops", "QK array operations", "operations", 1},
        {"attention_pv_array_ops", "PV array operations", "operations", 1},
        {"attention_sp_hbm_bytes", "Score/probability HBM bytes", "bytes", 1})

    RoCCAnalogInt(ComponentId_t id, Params &params)
        : RoCCAnalog<int64_t>(id, params) {}
};

} // namespace Golem
} // namespace SST

#endif
