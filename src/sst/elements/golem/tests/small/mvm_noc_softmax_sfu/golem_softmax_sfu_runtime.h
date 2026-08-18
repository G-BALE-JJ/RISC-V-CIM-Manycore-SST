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

enum class SFUJobOp : uint32_t {
    ELEMENTWISE = 0x01,
    REDUCE = 0x02,
    SOFTMAX_ROW = 0x10,
    LAYERNORM = 0x11,
    GELU = 0x12,
};

enum class SFUJobSubOp : uint32_t {
    NONE = 0x00,
    EXP = 0x01,
    LOG = 0x02,
    RECIPROCAL = 0x03,
    RSQRT = 0x04,
    TANH = 0x05,
    SIGMOID = 0x06,
    REDUCE_MAX = 0x20,
    REDUCE_SUM = 0x21,
};

constexpr uint32_t SFU_JOB_FLAG_DISTRIBUTED_COLUMNS = 0x1u;
constexpr uint32_t SFU_JOB_FLAG_DISTRIBUTED_ABORT = 0x2u;
constexpr uint32_t SFU_JOB_FLAG_ROW_ENGINE_MODEL = 0x4u;
constexpr uint32_t SFU_JOB_FLAG_TENSOR_ROW_ENGINE = 0x8u;
constexpr uint32_t SFU_SOFTMAX_JOB_PARAMS_MAGIC = 0x53465531u;
constexpr uint16_t SFU_SOFTMAX_JOB_PARAMS_VERSION = 1u;
constexpr uint16_t SFU_SOFTMAX_JOB_PARAMS_VERSION_ATTENTION = 2u;
constexpr uint16_t SFU_SOFTMAX_JOB_PARAMS_VERSION_MANAGER = 3u;
constexpr uint32_t SFU_SOFTMAX_PARAMS_FLAG_ATTENTION = 0x1u;
constexpr uint32_t SFU_SOFTMAX_PARAMS_FLAG_CAUSAL = 0x2u;
constexpr uint32_t SFU_SOFTMAX_HBM_LAYOUT_BAND_STRIPED = 1u;
constexpr uint32_t SFU_SOFTMAX_MAPPING_EXPLICIT_TOPOLOGY = 2u;
constexpr uint32_t SFU_WORKER_TOPOLOGY_MAP_MAGIC = 0x574d4150u;
constexpr uint16_t SFU_WORKER_TOPOLOGY_MAP_VERSION = 1u;
constexpr uint32_t SFU_WORKER_TOPOLOGY_MAX_WORKERS = 16u;

struct SFUSoftmaxJobParamsV1 {
    uint32_t magic;
    uint16_t version;
    uint16_t size_bytes;
    uint32_t mapping_policy;
    uint32_t tiles_per_row;
    uint32_t row_contexts_hint;
    uint32_t hbm_layout;
    uint32_t data_node_mask;
    uint32_t flags;
    uint64_t completion_addr;
    uint64_t node_stride_bytes;
    uint32_t rows_per_band;
    uint32_t coordinator_core;
    uint64_t reserved0;
};

static_assert(sizeof(SFUSoftmaxJobParamsV1) == 64,
              "Tensor softmax parameter ABI must match golem.SFU");

struct SFUWorkerTopologyMapV1 {
    uint32_t magic;
    uint16_t version;
    uint16_t size_bytes;
    uint32_t worker_count;
    uint32_t reserved0;
    uint32_t worker_core_ids[SFU_WORKER_TOPOLOGY_MAX_WORKERS];
};

static_assert(sizeof(SFUWorkerTopologyMapV1) == 80,
              "Worker topology map ABI must match golem.SFU");

struct SFUJobDesc {
    uint64_t job_id;
    uint64_t input0_addr;
    uint64_t input1_addr;
    uint64_t output_addr;
    uint64_t params_addr;
    uint64_t scratch_addr;
    uint32_t op_type;
    uint32_t sub_op;
    uint32_t dtype;
    uint32_t layout;
    uint32_t rows;
    uint32_t cols;
    uint32_t elem_count;
    uint32_t chunk_elems;
    uint32_t worker_cores;
    uint32_t owner_core;
    uint32_t flags;
    uint32_t reserved0;  // With DISTRIBUTED_COLUMNS, reserved0 stores the worker slot.
    uint64_t reserved1;
    uint64_t reserved2;
    uint64_t reserved3;
    uint64_t reserved4;
};

static_assert(sizeof(SFUJobDesc) == 128,
              "SFUJobDesc ABI must match golem.SFU");

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

golem_status_t golemRunStandaloneSoftmaxSfuJobForCore(
    const golem_softmax_op_desc_t* op_desc,
    int executor_core_id,
    const MatmulRuntimeConfig* cfg,
    uint64_t input_gm,
    uint64_t output_gm,
    uint64_t desc_gm,
    uint32_t chunk_elems,
    uint32_t worker_cores,
    uint32_t worker_slot,
    uint32_t owner_core,
    uint32_t flags,
    uint64_t job_id,
    uint64_t tag);

typedef struct {
    uint64_t launch_start_cycle;
    uint64_t descriptors_ready_cycle;
    uint64_t params_write_done_cycle;
    uint64_t desc_write_done_cycle;
    uint64_t issue_return_cycle;
    uint64_t wait_start_cycle;
    uint64_t wait_return_cycle;
} golem_softmax_launch_timeline_t;

golem_status_t golemRunTensorSoftmaxSfuJob(
    const golem_softmax_op_desc_t* op_desc,
    int executor_core_id,
    const MatmulRuntimeConfig* cfg,
    uint64_t input_hbm,
    uint64_t output_hbm,
    uint64_t scratch_gm,
    uint64_t params_gm,
    uint64_t desc_gm,
    uint64_t node_stride_bytes,
    uint32_t rows_per_band,
    uint32_t row_contexts,
    uint32_t physical_engines,
    uint32_t attention_head_dim,
    bool attention_causal,
    uint64_t job_id,
    uint64_t tag,
    golem_softmax_launch_timeline_t* timeline);

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

const char* golemSoftmaxSfuGetLastErrorString(void);

#ifdef __cplusplus
}
#endif
