#ifndef GOLEM_SOFTMAX_CROSS_TILE_H_
#define GOLEM_SOFTMAX_CROSS_TILE_H_

#include <cstdint>
#include "../golem_operator_api.h"

// Reduction buffer layout per row in HBM:
// [max_val (fp32)] [sum_val (fp32)] [barrier_counter (int32)] [pad (int32)]
// Total: 16 bytes per row
constexpr uint64_t REDUCTION_ENTRY_BYTES = 16;
constexpr uint64_t REDUCTION_MAX_OFFSET = 0;
constexpr uint64_t REDUCTION_SUM_OFFSET = 4;
constexpr uint64_t REDUCTION_BARRIER_OFFSET = 8;

struct CrossTileSoftmaxContext {
    uint64_t reduction_buffer_hbm_base; // HBM base for reduction buffers
    int64_t total_m_tiles;
    int64_t total_n_tiles;
    int64_t n_tiles_per_row;
};

// Initialize cross-tile context (called once before softmax)
golem_status_t golemInitCrossTileSoftmaxContext(
    CrossTileSoftmaxContext* ctx,
    int64_t m, int64_t n,
    int64_t block_m, int64_t block_n,
    uint64_t reduction_buffer_base);

// Run cross-tile softmax for one core's assigned tiles
golem_status_t golemRunCrossTileSoftmaxForCore(
    const golem_softmax_op_desc_t* op_desc,
    const CrossTileSoftmaxContext* ctx,
    int core_id,
    int m_tile,
    int n_tile,
    uint64_t c_tile_hbm_addr,
    int64_t tile_stride);

#endif  // GOLEM_SOFTMAX_CROSS_TILE_H_
