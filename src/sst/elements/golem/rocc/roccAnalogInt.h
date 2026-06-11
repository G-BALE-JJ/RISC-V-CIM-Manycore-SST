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
        {"cycles_remote_ld", "Cycles consumed by remote_ld instructions", "cycles", 1})

    RoCCAnalogInt(ComponentId_t id, Params &params)
        : RoCCAnalog<int64_t>(id, params) {}
};

} // namespace Golem
} // namespace SST

#endif
