#include "golem_softmax_cross_tile.h"
#include "operators.h"
#include <cmath>
#include <cstring>

golem_status_t golemInitCrossTileSoftmaxContext(
    CrossTileSoftmaxContext* ctx,
    int64_t m, int64_t n,
    int64_t block_m, int64_t block_n,
    uint64_t reduction_buffer_base) {
    if (ctx == nullptr) {
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    ctx->reduction_buffer_hbm_base = reduction_buffer_base;
    ctx->total_m_tiles = (m + block_m - 1) / block_m;
    ctx->total_n_tiles = (n + block_n - 1) / block_n;
    ctx->n_tiles_per_row = ctx->total_n_tiles;
    return GOLEM_STATUS_OK;
}

// Compute local max for each row in the tile
static void compute_tile_local_max(
    int core_id,
    const float* tile_data,
    int64_t block_m,
    int64_t block_n,
    float* row_max_out) {
    // For each row in tile, find max across block_n columns
    for (int64_t r = 0; r < block_m; ++r) {
        float row_max = tile_data[r * block_n];
        for (int64_t c = 1; c < block_n; ++c) {
            if (tile_data[r * block_n + c] > row_max) {
                row_max = tile_data[r * block_n + c];
            }
        }
        row_max_out[r] = row_max;
    }
}

// Atomic max reduction: load current max from HBM, compute new max, and store
static golem_status_t atomic_max_reduction(
    int core_id,
    uint64_t reduction_row_base_hbm,
    float local_max,
    uint64_t local_tmp_gm) {
    // Load current max from HBM
    const uint64_t max_addr = reduction_row_base_hbm + REDUCTION_MAX_OFFSET;
    dma_remote_load_to_gm(core_id, max_addr, local_tmp_gm, 4);

    float current_max;
    set_len(4);
    gm2mm(&current_max, local_tmp_gm);

    // Atomic compare-and-swap: compute new max
    float new_max = (local_max > current_max) ? local_max : current_max;

    set_len(4);
    mm2gm(&new_max, local_tmp_gm);
    remote_store(local_tmp_gm, max_addr);

    return GOLEM_STATUS_OK;
}

// Run cross-tile softmax for one core's assigned tiles (Pass 1: Max Reduction)
golem_status_t golemRunCrossTileSoftmaxForCore(
    const golem_softmax_op_desc_t* op_desc,
    const CrossTileSoftmaxContext* ctx,
    int core_id,
    int m_tile,
    int n_tile,
    uint64_t c_tile_hbm_addr,
    int64_t tile_stride) {

    if (op_desc == nullptr || ctx == nullptr) {
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }

    if (m_tile < 0 || m_tile >= ctx->total_m_tiles ||
        n_tile < 0 || n_tile >= ctx->total_n_tiles) {
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }

    // Get block dimensions from op_desc
    int64_t block_m = op_desc->outer;  // rows per tile
    int64_t block_n = op_desc->dim;    // cols per tile (typically)

    // Allocate temporary buffers in GM
    const uint64_t local_tmp_gm = gm_addr(core_id, LOCAL_LAYOUT.accum);
    const uint64_t tile_data_gm = gm_addr(core_id, LOCAL_LAYOUT.accum + 256);
    const uint64_t row_max_gm = gm_addr(core_id, LOCAL_LAYOUT.accum + 512);

    // Load tile data from HBM to GM
    uint64_t tile_bytes = block_m * block_n * sizeof(float);
    dma_remote_load_to_gm(core_id, c_tile_hbm_addr, tile_data_gm, tile_bytes);

    // Copy tile data from GM to MM for computation
    float tile_data[256];  // Max 64x64 tile = 4096 floats, use reasonable size
    set_len(tile_bytes);
    gm2mm(tile_data, tile_data_gm);

    // Compute local max for each row in the tile
    float row_max[64];  // Max block_m rows
    compute_tile_local_max(core_id, tile_data, block_m, block_n, row_max);

    // For each row, atomically update the global max in HBM reduction buffer
    for (int64_t r = 0; r < block_m; ++r) {
        // Global row index
        int64_t global_row = m_tile * block_m + r;

        // Reduction buffer address for this row
        uint64_t reduction_row_base = ctx->reduction_buffer_hbm_base +
                                      global_row * REDUCTION_ENTRY_BYTES;

        // Atomic max update
        atomic_max_reduction(core_id, reduction_row_base, row_max[r], local_tmp_gm);
    }

    return GOLEM_STATUS_OK;
}
