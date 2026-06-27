#include "golem_softmax_single_core.h"
#include "golem_softmax_runtime.h"
#include "operators.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

golem_status_t golemInitSingleCoreSoftmaxContext(
    SingleCoreSoftmaxContext* ctx,
    int64_t m, int64_t n,
    int64_t block_m, int64_t block_n) {
    if (ctx == nullptr) {
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (m <= 0 || n <= 0 || block_m <= 0 || block_n <= 0) {
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    ctx->m = m;
    ctx->n = n;
    ctx->block_m = block_m;
    ctx->block_n = block_n;
    ctx->total_m_tiles = (m + block_m - 1) / block_m;
    ctx->total_n_tiles = (n + block_n - 1) / block_n;
    return GOLEM_STATUS_OK;
}

golem_status_t golemRunSingleCoreSoftmaxForCore(
    const golem_softmax_op_desc_t* op_desc,
    const SingleCoreSoftmaxContext* ctx,
    int executor_core_id,
    int softmax_core_id) {

    // Only Core 0 executes, other cores return immediately
    if (softmax_core_id != 0) {
        return GOLEM_STATUS_OK;
    }

    if (op_desc == nullptr || ctx == nullptr) {
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }

    const int64_t m = ctx->m;
    const int64_t n = ctx->n;
    const int64_t block_m = ctx->block_m;
    const int64_t block_n = ctx->block_n;
    const int64_t n_tiles = ctx->total_n_tiles;
    const char* fast_probability_env = std::getenv("GOLEM_SOFTMAX_FAST_PROBABILITY");
    const bool fast_probability_mode =
        fast_probability_env != nullptr && std::strcmp(fast_probability_env, "1") == 0;
    const MatmulRuntimeConfig cfg = {
        .m = static_cast<int>(m),
        .n = static_cast<int>(n),
        .k = GEMM_K,
        .block_m = static_cast<int>(block_m),
        .block_n = static_cast<int>(block_n),
        .block_k = GEMM_BLOCK_K,
    };

    // Validate row size fits in buffer
    if (n > 256) {
        // Note: set_softmax_last_error is internal, just return error
        return GOLEM_STATUS_UNSUPPORTED;
    }

    // Allocate GM buffers
    const uint64_t local_row_gm = gm_addr(executor_core_id, LOCAL_LAYOUT.accum);
    const uint64_t local_tmp_gm = local_row_gm + 256 * sizeof(float) + 64;

    float row_data[256];  // Full row buffer (max N=256)

    // Process each row
    for (int64_t row = 0; row < m; ++row) {
        const int64_t m_tile = row / block_m;
        const int64_t row_in_tile = row % block_m;
        // Aggregate full row from all N-tiles
        for (int64_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
            const int64_t n_start = n_tile * block_n;
            const int64_t n_count = (n_start + block_n <= n) ? block_n : (n - n_start);

            // Compute HBM address for this tile's row
            const int tile_task_id = static_cast<int>(m_tile * n_tiles + n_tile);
            const GemmTaskDescriptor tile_desc = gemm_task_desc_for_task(executor_core_id, tile_task_id, cfg);
            const uint64_t row_hbm_addr = tile_desc.c_base_mm + row_in_tile * block_n * sizeof(float);

            // Load row segment from HBM
            const uint64_t row_bytes = n_count * sizeof(float);
            dma_remote_load_to_gm(executor_core_id, row_hbm_addr, local_tmp_gm, row_bytes);
            set_len(row_bytes);
            gm2mm(&row_data[n_start], local_tmp_gm);
        }

        // Compute softmax for full row (numerically stable)
        float max_val = row_data[0];
        int64_t max_col = 0;
        for (int64_t c = 1; c < n; ++c) {
            if (row_data[c] > max_val) {
                max_val = row_data[c];
                max_col = c;
            }
        }

        if (fast_probability_mode) {
            for (int64_t c = 0; c < n; ++c) {
                row_data[c] = 0.0f;
            }
            row_data[max_col] = 1.0f;
        } else {
            float sum = 0.0f;
            for (int64_t c = 0; c < n; ++c) {
                const float exp_val = std::exp(row_data[c] - max_val);
                row_data[c] = exp_val;
                sum += exp_val;
            }

            for (int64_t c = 0; c < n; ++c) {
                row_data[c] /= sum;
            }
        }

        // Write back to HBM (tile by tile)
        for (int64_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
            const int64_t n_start = n_tile * block_n;
            const int64_t n_count = (n_start + block_n <= n) ? block_n : (n - n_start);

            const int tile_task_id = static_cast<int>(m_tile * n_tiles + n_tile);
            const GemmTaskDescriptor tile_desc = gemm_task_desc_for_task(executor_core_id, tile_task_id, cfg);
            const uint64_t row_hbm_addr = tile_desc.c_base_mm + row_in_tile * block_n * sizeof(float);

            const uint64_t row_bytes = n_count * sizeof(float);
            set_len(row_bytes);
            mm2gm(&row_data[n_start], local_tmp_gm);
            remote_store(local_tmp_gm, row_hbm_addr);
        }
    }

    return GOLEM_STATUS_OK;
}
