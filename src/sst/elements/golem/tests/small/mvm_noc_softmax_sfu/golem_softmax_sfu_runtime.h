#pragma once

#include <cstdint>

#include "../mvm_noc_softmax_cpu/golem_softmax_runtime.h"
#include "../mvm_noc_int_array/pipeline_config.h"

struct SFUSoftmaxTileDesc {
    uint64_t job_id;
    uint64_t local_input_gm_addr;
    uint64_t local_output_gm_addr;
    uint32_t global_m;
    uint32_t global_n;
    uint32_t block_m;
    uint32_t block_n;
    uint32_t m_tile;
    uint32_t n_tile;
    uint32_t valid_m;
    uint32_t valid_n;
    uint32_t n_tiles_per_row;
    uint32_t elem_bytes;
    uint32_t flags;
};

static_assert(sizeof(SFUSoftmaxTileDesc) == 72,
              "SFUSoftmaxTileDesc ABI must match golem.SFU");

#ifdef __cplusplus
extern "C" {
#endif

golem_status_t golemRunSoftmaxSfuForCore(
    const golem_softmax_op_desc_t* op_desc,
    int executor_core_id,
    int worker_core_id,
    const MatmulRuntimeConfig* cfg,
    uint64_t job_id);

#ifdef __cplusplus
}
#endif
