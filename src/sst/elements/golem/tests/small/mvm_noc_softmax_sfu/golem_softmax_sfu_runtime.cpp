#include "golem_softmax_sfu_runtime.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ex_instr.h"
#include "../mvm_noc_int_array/gemm_matmul_op.h"
#include "../mvm_noc_int_array/operators.h"

namespace {

thread_local std::string g_sfu_softmax_last_error;

constexpr uint64_t SFU_DESC_GM_OFFSET = LOCAL_LAYOUT.tmp;
constexpr uint64_t SFU_INPUT_GM_OFFSET = LOCAL_LAYOUT.accum;
constexpr uint64_t SFU_OUTPUT_GM_OFFSET = LOCAL_LAYOUT.out;

enum : uint64_t {
    SFU_STATUS_SUCCESS = 0,
    SFU_STATUS_PENDING = 1,
};

void set_sfu_softmax_last_error(const char* fmt, ...) {
    char buffer[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    g_sfu_softmax_last_error = buffer;
}

bool is_supported_axis(int64_t axis) {
    return axis == -1 || axis == 1;
}

golem_status_t validate_sfu_softmax_request(const golem_softmax_op_desc_t* op_desc,
                                            int executor_core_id,
                                            int worker_core_id,
                                            const MatmulRuntimeConfig* cfg) {
    if (op_desc == nullptr) {
        set_sfu_softmax_last_error("softmax SFU op_desc is null");
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (!is_supported_axis(op_desc->axis)) {
        set_sfu_softmax_last_error("softmax SFU only supports axis=-1/1, got %lld",
                                   static_cast<long long>(op_desc->axis));
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (op_desc->dtype != GOLEM_DTYPE_FP32 || op_desc->layout != GOLEM_LAYOUT_ROW_MAJOR) {
        set_sfu_softmax_last_error("softmax SFU v1 requires fp32 row-major tensors");
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (cfg == nullptr) {
        set_sfu_softmax_last_error("softmax SFU cfg is null");
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (op_desc->outer <= 0 || op_desc->dim <= 0 || cfg->block_m <= 0 || cfg->block_n <= 0) {
        set_sfu_softmax_last_error("softmax SFU dimensions must be positive");
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (executor_core_id < 0 || executor_core_id >= TOTAL_CORES) {
        set_sfu_softmax_last_error("softmax SFU invalid executor core id=%d", executor_core_id);
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (worker_core_id < 0 || worker_core_id >= TOTAL_CORES) {
        set_sfu_softmax_last_error("softmax SFU invalid worker core id=%d", worker_core_id);
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (op_desc->outer != cfg->m || op_desc->dim != cfg->n) {
        set_sfu_softmax_last_error("softmax SFU op/cfg shape mismatch");
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (cfg->block_m > GEMM_BLOCK_M || cfg->block_n > GEMM_BLOCK_N) {
        set_sfu_softmax_last_error("softmax SFU block exceeds compiled GEMM tile shape");
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if ((cfg->m % cfg->block_m) != 0 || (cfg->n % cfg->block_n) != 0) {
        set_sfu_softmax_last_error("softmax SFU v1 requires tile-divisible M/N");
        return GOLEM_STATUS_UNSUPPORTED;
    }
    return GOLEM_STATUS_OK;
}

void write_sfu_desc_to_gm(int core_id, uint64_t desc_gm_addr, const SFUSoftmaxTileDesc& desc) {
    const uint64_t* words = reinterpret_cast<const uint64_t*>(&desc);
    constexpr size_t kWords = sizeof(SFUSoftmaxTileDesc) / sizeof(uint64_t);
    for (size_t i = 0; i < kWords; ++i) {
        reg2gm(words[i], desc_gm_addr + static_cast<uint64_t>(i) * sizeof(uint64_t));
    }
}

int task_id_for_tile(int n_tiles, int m_tile, int n_tile) {
    return m_tile * n_tiles + n_tile;
}

} // namespace

extern "C" golem_status_t golemRunSoftmaxSfuForCore(
    const golem_softmax_op_desc_t* op_desc,
    int executor_core_id,
    int worker_core_id,
    const MatmulRuntimeConfig* cfg,
    uint64_t job_id) {
    golem_status_t status = validate_sfu_softmax_request(op_desc, executor_core_id, worker_core_id, cfg);
    if (status != GOLEM_STATUS_OK) {
        return status;
    }

    const int m_tiles = gemm_m_tiles(*cfg);
    const int n_tiles = gemm_n_tiles(*cfg);
    const int worker_slot = gemm_worker_slot_for_core(worker_core_id);
    if (worker_slot < 0) {
        return GOLEM_STATUS_OK;
    }

    struct PendingTile {
        uint64_t tag;
        uint64_t output_hbm;
        uint64_t local_output_gm;
        uint64_t bytes;
    };
    std::vector<PendingTile> pending;

    const uint64_t desc_gm = gm_addr(executor_core_id, SFU_DESC_GM_OFFSET);
    const uint64_t local_input_gm = gm_addr(executor_core_id, SFU_INPUT_GM_OFFSET);
    const uint64_t local_output_gm = gm_addr(executor_core_id, SFU_OUTPUT_GM_OFFSET);

    for (int m_tile = 0; m_tile < m_tiles; ++m_tile) {
        for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
            const int task_id = task_id_for_tile(n_tiles, m_tile, n_tile);
            if ((task_id % ACTIVE_GEMM_CORES) != worker_slot) {
                continue;
            }
            const GemmTaskDescriptor task = gemm_task_desc_for_task(executor_core_id, task_id, *cfg);

            const uint32_t valid_m = static_cast<uint32_t>(task.block_m);
            const uint32_t valid_n = static_cast<uint32_t>(task.block_n);
            const uint64_t tile_bytes =
                static_cast<uint64_t>(task.block_m) * static_cast<uint64_t>(task.block_n) * sizeof(float);

            dma_remote_load_to_gm(executor_core_id, task.c_base_mm, local_input_gm, tile_bytes);

            const SFUSoftmaxTileDesc desc = {
                .job_id = job_id,
                .local_input_gm_addr = local_input_gm,
                .local_output_gm_addr = local_output_gm,
                .global_m = static_cast<uint32_t>(cfg->m),
                .global_n = static_cast<uint32_t>(cfg->n),
                .block_m = static_cast<uint32_t>(task.block_m),
                .block_n = static_cast<uint32_t>(task.block_n),
                .m_tile = static_cast<uint32_t>(task.m_tile),
                .n_tile = static_cast<uint32_t>(task.n_tile),
                .valid_m = valid_m,
                .valid_n = valid_n,
                .n_tiles_per_row = static_cast<uint32_t>(n_tiles),
                .elem_bytes = sizeof(float),
                .flags = 0,
            };
            write_sfu_desc_to_gm(executor_core_id, desc_gm, desc);

            const uint64_t tag = static_cast<uint64_t>(task_id) + 1;
            sfu_softmax_tile(desc_gm, tag);
            pending.push_back(PendingTile{
                .tag = tag,
                .output_hbm = task.c_base_mm,
                .local_output_gm = local_output_gm,
                .bytes = tile_bytes,
            });
        }
    }

    for (const PendingTile& tile : pending) {
        const uint64_t sfu_status = sfu_wait(tile.tag);
        if (sfu_status != SFU_STATUS_SUCCESS) {
            set_sfu_softmax_last_error("softmax SFU wait failed with status=%llu",
                                       static_cast<unsigned long long>(sfu_status));
            return GOLEM_STATUS_INTERNAL_ERROR;
        }
        set_len(tile.bytes);
        remote_store(tile.local_output_gm, tile.output_hbm);
    }

    return GOLEM_STATUS_OK;
}

extern "C" const char* golemSoftmaxSfuGetLastErrorString(void) {
    return g_sfu_softmax_last_error.c_str();
}
