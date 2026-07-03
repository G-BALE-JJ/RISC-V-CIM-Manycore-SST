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

enum class SFUPrimitiveOp : uint32_t {
    EXP = 0x01,
    LOG = 0x02,
    RECIPROCAL = 0x03,
    RSQRT = 0x04,
    SQRT = 0x05,
    TANH = 0x06,
    SIGMOID = 0x07,
    REDUCE_MAX = 0x20,
    REDUCE_SUM = 0x21,
    GELU = 0x40,
    LAYERNORM = 0x41,
    FUSED_SOFTMAX = 0x80,
};

struct SFUPrimitiveDesc {
    uint64_t job_id;
    uint64_t input0_gm_addr;
    uint64_t input1_gm_addr;
    uint64_t output_gm_addr;
    uint32_t op;
    uint32_t dtype;
    uint32_t elem_count;
    uint32_t input0_stride_bytes;
    uint32_t input1_stride_bytes;
    uint32_t output_stride_bytes;
    uint32_t flags;
    uint32_t approx_mode;
};

static_assert(sizeof(SFUPrimitiveDesc) == 64,
              "SFUPrimitiveDesc ABI must match golem.SFU");

struct SFUPrimitiveBatchDesc {
    uint64_t job_id;
    uint64_t desc_array_gm_addr;
    uint32_t desc_count;
    uint32_t flags;
    uint64_t reserved0;
};

static_assert(sizeof(SFUPrimitiveBatchDesc) == 32,
              "SFUPrimitiveBatchDesc ABI must match golem.SFU");

#ifdef __cplusplus
extern "C" {
#endif

golem_status_t golemRunSoftmaxSfuForCore(
    const golem_softmax_op_desc_t* op_desc,
    int executor_core_id,
    int worker_core_id,
    const MatmulRuntimeConfig* cfg,
    uint64_t job_id);

golem_status_t golemRunStandaloneSoftmaxSfuForCore(
    const golem_softmax_op_desc_t* op_desc,
    int executor_core_id,
    int worker_core_id,
    const MatmulRuntimeConfig* cfg,
    uint64_t job_id);

golem_status_t golemRunSoftmaxSfuTileFromLocalAccum(
    const golem_softmax_op_desc_t* op_desc,
    int executor_core_id,
    const MatmulRuntimeConfig* cfg,
    const GemmTaskDescriptor* task,
    uint64_t local_accum_gm,
    uint64_t local_output_gm,
    uint64_t desc_gm,
    uint64_t job_id,
    uint64_t tag);

golem_status_t golemWaitSoftmaxSfuTileAndStore(
    uint64_t tag,
    uint64_t local_output_gm,
    uint64_t output_hbm,
    uint64_t bytes);

#ifdef __cplusplus
}
#endif
