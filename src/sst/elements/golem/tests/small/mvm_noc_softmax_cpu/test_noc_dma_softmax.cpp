#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core_bind.h"
#include "pipeline_config.h"
#include "golem_matmul_runtime.h"
#include "golem_softmax_runtime.h"
#include "golem_softmax_cross_tile.h"
#include "golem_softmax_single_core.h"

namespace {

bool close_enough(float got, float expected) {
    const float diff = std::fabs(got - expected);
    return diff <= 1.0e-5f;
}

int64_t read_i64_env_or_default(const char* name, int64_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0')) {
        return default_value;
    }
    return static_cast<int64_t>(parsed);
}

golem_dtype_t read_dtype_env_or_default(const char* name, golem_dtype_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    if (std::strcmp(raw, "int32") == 0 || std::strcmp(raw, "i32") == 0) {
        return GOLEM_DTYPE_INT32;
    }
    if (std::strcmp(raw, "fp32") == 0 || std::strcmp(raw, "float32") == 0 || std::strcmp(raw, "float") == 0) {
        return GOLEM_DTYPE_FP32;
    }
    return default_value;
}

golem_matmul_op_desc_t make_matmul_desc_from_env() {
    const golem_dtype_t dtype = read_dtype_env_or_default("GOLEM_MATMUL_DTYPE", GOLEM_DTYPE_FP32);
    return {
        .m = read_i64_env_or_default("GOLEM_MATMUL_M", GEMM_M),
        .n = read_i64_env_or_default("GOLEM_MATMUL_N", GEMM_N),
        .k = read_i64_env_or_default("GOLEM_MATMUL_K", GEMM_K),
        .block_m = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_M", TILE_M),
        .block_n = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_N", TILE_N_MAX),
        .block_k = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_K", TILE_K),
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
        .transpose_a = 0,
        .transpose_b = 0,
    };
}

int read_requested_core_from_argv(int argc, char* argv[]) {
    if (argc < 2) {
        return 0;
    }
    return std::atoi(argv[1]);
}

int run_gemm(const golem_matmul_op_desc_t& op_desc) {
    golem_tensor_desc_t a_desc = {
        .data = nullptr,
        .ndim = 2,
        .shape = {op_desc.m, op_desc.k},
        .stride = {op_desc.k, 1},
        .dtype = op_desc.dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };
    golem_tensor_desc_t b_desc = {
        .data = nullptr,
        .ndim = 2,
        .shape = {op_desc.k, op_desc.n},
        .stride = {op_desc.n, 1},
        .dtype = op_desc.dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };
    golem_tensor_desc_t c_desc = {
        .data = nullptr,
        .ndim = 2,
        .shape = {op_desc.m, op_desc.n},
        .stride = {op_desc.n, 1},
        .dtype = op_desc.dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };

    golem_kernel_handle_t kernel = nullptr;
    golem_status_t status = golemCreateMatmulKernel(&op_desc, &kernel);
    if (status != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemCreateMatmulKernel failed: %s\n", golemGetLastErrorString());
        return 1;
    }

    status = golemRunMatmul(kernel, &a_desc, &b_desc, &c_desc);
    if (status != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemRunMatmul failed: %s\n", golemGetLastErrorString());
        golemDestroyKernel(kernel);
        return 1;
    }

    status = golemDestroyKernel(kernel);
    if (status != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemDestroyKernel failed: %s\n", golemGetLastErrorString());
        return 1;
    }
    return 0;
}

int run_tile_local_softmax_for_core(int executor_core_id, int softmax_core_id, const golem_matmul_op_desc_t& op_desc) {
    if (op_desc.dtype != GOLEM_DTYPE_FP32) {
        if (softmax_core_id == 0) {
            std::fprintf(stderr, "[SOFTMAX] skip: softmax v1 only supports fp32\n");
        }
        return 0;
    }

    MatmulRuntimeConfig cfg = {
        .m = static_cast<int>(op_desc.m),
        .n = static_cast<int>(op_desc.n),
        .k = static_cast<int>(op_desc.k),
        .block_m = static_cast<int>(op_desc.block_m),
        .block_n = static_cast<int>(op_desc.block_n),
        .block_k = static_cast<int>(op_desc.block_k),
    };
    const int total_tasks = gemm_total_tasks(cfg);
    const int worker_slot = gemm_worker_slot_for_core(softmax_core_id);
    if (worker_slot < 0 || worker_slot >= total_tasks) {
        return 0;
    }

    // Choose softmax mode: single-core (default) or cross-tile
    const char* softmax_mode = std::getenv("GOLEM_SOFTMAX_MODE");
    const bool use_cross_tile = (softmax_mode != nullptr && std::strcmp(softmax_mode, "cross-tile") == 0);

    if (use_cross_tile) {
        // Cross-tile mode: all cores participate with barriers
        if (softmax_core_id == 0) {
            std::printf("[SOFTMAX] mode=cross-tile (multi-core with barriers)\n");
        }
    } else {
        // Single-core mode: only Core 0 executes
        if (softmax_core_id == 0) {
            std::printf("[SOFTMAX] mode=single-core (Core 0 post-processing)\n");
        }
    }

    if (use_cross_tile) {
        // Initialize cross-tile softmax context
    const int m_tiles = (cfg.m + cfg.block_m - 1) / cfg.block_m;
    const int n_tiles = (cfg.n + cfg.block_n - 1) / cfg.block_n;
    // Place reduction buffer in HBM at end of data region (identity_base + max_data_size)
    // Assume C matrix uses at most 1MB, so place reduction buffer at 0x8000000 + 0x100000 = 0x8100000
    const uint64_t reduction_buffer_hbm = 0x8100000;

    CrossTileSoftmaxContext ctx;
    golem_status_t status = golemInitCrossTileSoftmaxContext(
        &ctx,
        cfg.m,        // Pass actual dimensions, not tile counts
        cfg.n,
        cfg.block_m,
        cfg.block_n,
        reduction_buffer_hbm);
    if (status != GOLEM_STATUS_OK) {
        std::fprintf(stderr,
                     "[Core %d] [SOFTMAX] failed to initialize cross-tile context: %s\n",
                     softmax_core_id,
                     golemSoftmaxGetLastErrorString());
        return 1;
    }

    // Core 0 initializes the reduction buffer
    if (softmax_core_id == 0) {
        const int64_t total_rows = cfg.m;
        status = golemInitCrossTileReductionBuffer(&ctx, executor_core_id, total_rows);
        if (status != GOLEM_STATUS_OK) {
            std::fprintf(stderr,
                         "[Core 0] [SOFTMAX] failed to initialize reduction buffer: %s\n",
                         golemSoftmaxGetLastErrorString());
            return 1;
        }
        std::printf("[Core 0] [SOFTMAX] reduction buffer initialized (rows=%ld)\n", total_rows);
        std::fflush(stdout);
    }

    // TODO: Add barrier to ensure core 0 finishes initialization before other cores start
    // For now, rely on sequential execution in single-core tests

    std::printf("[Core %d] [SOFTMAX] starting cross-tile loop: worker_slot=%d total_tasks=%d\n",
                softmax_core_id, worker_slot, total_tasks);
    std::fflush(stdout);

    int softmax_tiles = 0;
    for (int task_id = worker_slot; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_task_desc_for_task(executor_core_id, task_id, cfg);

        // Extract tile indices from task_id
        const int m_tile = task_id / n_tiles;
        const int n_tile = task_id % n_tiles;

        std::printf("[Core %d] [SOFTMAX] task=%d m_tile=%d n_tile=%d c_base_mm=0x%lx\n",
                    softmax_core_id, task_id, m_tile, n_tile, desc.c_base_mm);
        std::fflush(stdout);

        golem_softmax_op_desc_t softmax_desc = {
            .outer = desc.block_m,
            .dim = desc.block_n,
            .axis = -1,
            .dtype = GOLEM_DTYPE_FP32,
            .layout = GOLEM_LAYOUT_ROW_MAJOR,
        };

        status = golemRunCrossTileSoftmaxForCore(
            &softmax_desc,
            &ctx,
            executor_core_id,
            m_tile,
            n_tile,
            desc.c_base_mm,
            desc.block_n);

        std::printf("[Core %d] [SOFTMAX] task=%d returned status=%d\n", softmax_core_id, task_id, status);
        std::fflush(stdout);

        if (status != GOLEM_STATUS_OK) {
            std::fprintf(stderr,
                         "[Core %d] [SOFTMAX] cross-tile task=%d (m_tile=%d, n_tile=%d) failed: %s\n",
                         softmax_core_id,
                         task_id,
                         m_tile,
                         n_tile,
                         golemSoftmaxGetLastErrorString());
            return 1;
        }
        softmax_tiles++;
    }

    std::printf("[Core %d] [SOFTMAX] cross-tile softmax complete: tiles=%d\n", softmax_core_id, softmax_tiles);
    std::fflush(stdout);
    } else {
        // Single-core mode: only Core 0 aggregates and computes softmax
        SingleCoreSoftmaxContext ctx;
        golem_status_t status = golemInitSingleCoreSoftmaxContext(
            &ctx,
            cfg.m,
            cfg.n,
            cfg.block_m,
            cfg.block_n);
        if (status != GOLEM_STATUS_OK) {
            if (softmax_core_id == 0) {
                std::fprintf(stderr,
                             "[Core 0] [SOFTMAX] failed to initialize single-core context: %s\n",
                             golemSoftmaxGetLastErrorString());
            }
            return 1;
        }

        golem_softmax_op_desc_t softmax_desc = {
            .outer = cfg.m,
            .dim = cfg.n,
            .axis = -1,
            .dtype = GOLEM_DTYPE_FP32,
            .layout = GOLEM_LAYOUT_ROW_MAJOR,
        };

        if (softmax_core_id == 0) {
            std::printf("[Core 0] [SOFTMAX] starting single-core softmax: m=%d n=%d executor_core=%d\n",
                        cfg.m, cfg.n, executor_core_id);
            std::fflush(stdout);
        }

        status = golemRunSingleCoreSoftmaxForCore(
            &softmax_desc,
            &ctx,
            executor_core_id,
            softmax_core_id);

        if (status != GOLEM_STATUS_OK) {
            if (softmax_core_id == 0) {
                std::fprintf(stderr,
                             "[Core 0] [SOFTMAX] single-core softmax failed: %s\n",
                             golemSoftmaxGetLastErrorString());
            }
            return 1;
        }

        if (softmax_core_id == 0) {
            std::printf("[Core 0] [SOFTMAX] single-core softmax complete\n");
            std::fflush(stdout);
        }
    }

    return 0;
}

int run_riscv_gemm_softmax(int argc, char* argv[]) {
    const int executor_core_id = bind_and_resolve_core_from_argv_or_exit(argc, argv, TOTAL_CORES);
    const int softmax_core_id = read_requested_core_from_argv(argc, argv);
    if (softmax_core_id < 0 || softmax_core_id >= TOTAL_CORES) {
        std::fprintf(stderr, "[ERROR] invalid requested core id=%d, TOTAL_CORES=%d\n", softmax_core_id, TOTAL_CORES);
        return 1;
    }

    const golem_matmul_op_desc_t op_desc = make_matmul_desc_from_env();
    int status = run_gemm(op_desc);
    if (status != 0) {
        return status;
    }
    return run_tile_local_softmax_for_core(executor_core_id, softmax_core_id, op_desc);
}

int run_pointer_softmax_selftest() {
    float input[3] = {1.0f, 2.0f, 3.0f};
    float output[3] = {0.0f, 0.0f, 0.0f};

    golem_softmax_op_desc_t op = {
        .outer = 1,
        .dim = 3,
        .axis = -1,
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };

    golem_tensor_desc_t input_desc = {
        .data = input,
        .ndim = 2,
        .shape = {1, 3},
        .stride = {3, 1},
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };

    golem_tensor_desc_t output_desc = {
        .data = output,
        .ndim = 2,
        .shape = {1, 3},
        .stride = {3, 1},
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };

    const golem_status_t status = golemRunSoftmaxCpu(&op, &input_desc, &output_desc);
    if (status != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[SOFTMAX-SELFTEST] run failed: %s\n", golemSoftmaxGetLastErrorString());
        return 1;
    }

    const float expected[3] = {0.09003057f, 0.24472848f, 0.66524094f};
    for (int i = 0; i < 3; ++i) {
        if (!close_enough(output[i], expected[i])) {
            std::fprintf(stderr,
                         "[SOFTMAX-SELFTEST] mismatch at %d: got %.8f expected %.8f\n",
                         i,
                         static_cast<double>(output[i]),
                         static_cast<double>(expected[i]));
            return 1;
        }
    }

    std::printf("[SOFTMAX-SELFTEST] PASS pointer fp32 row-major\n");
    return 0;
}

int run_gm_softmax_selftest() {
    float input[6] = {1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f};
    float output[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    golem_softmax_op_desc_t op = {
        .outer = 2,
        .dim = 3,
        .axis = -1,
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };

    const golem_status_t status = golemRunSoftmaxCpuGm(
        &op,
        reinterpret_cast<uint64_t>(input),
        reinterpret_cast<uint64_t>(output),
        3,
        3);
    if (status != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[SOFTMAX-GM-SELFTEST] run failed: %s\n", golemSoftmaxGetLastErrorString());
        return 1;
    }

    const float expected[6] = {
        0.09003057f, 0.24472848f, 0.66524094f,
        0.01587624f, 0.11731043f, 0.86681336f,
    };
    for (int i = 0; i < 6; ++i) {
        if (!close_enough(output[i], expected[i])) {
            std::fprintf(stderr,
                         "[SOFTMAX-GM-SELFTEST] mismatch at %d: got %.8f expected %.8f\n",
                         i,
                         static_cast<double>(output[i]),
                         static_cast<double>(expected[i]));
            return 1;
        }
    }

    std::printf("[SOFTMAX-GM-SELFTEST] PASS fp32 row-major\n");
    return 0;
}

int run_gm_softmax_dim64_selftest() {
    float input[64];
    float output[64];
    for (int i = 0; i < 64; ++i) {
        input[i] = static_cast<float>(i % 8);
        output[i] = 0.0f;
    }

    golem_softmax_op_desc_t op = {
        .outer = 1,
        .dim = 64,
        .axis = -1,
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };

    const golem_status_t status = golemRunSoftmaxCpuGm(
        &op,
        reinterpret_cast<uint64_t>(input),
        reinterpret_cast<uint64_t>(output),
        64,
        64);
    if (status != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[SOFTMAX-GM64-SELFTEST] run failed: %s\n", golemSoftmaxGetLastErrorString());
        return 1;
    }

    float sum = 0.0f;
    for (int i = 0; i < 64; ++i) {
        sum += output[i];
        if (!(output[i] > 0.0f && output[i] < 1.0f)) {
            std::fprintf(stderr,
                         "[SOFTMAX-GM64-SELFTEST] invalid probability at %d: got %.8f\n",
                         i,
                         static_cast<double>(output[i]));
            return 1;
        }
    }
    if (!close_enough(sum, 1.0f)) {
        std::fprintf(stderr,
                     "[SOFTMAX-GM64-SELFTEST] probability sum mismatch: got %.8f expected 1.00000000\n",
                     static_cast<double>(sum));
        return 1;
    }

    std::printf("[SOFTMAX-GM64-SELFTEST] PASS fp32 dim64 row-major\n");
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc > 1) {
        return run_riscv_gemm_softmax(argc, argv);
    }
    int status = run_pointer_softmax_selftest();
    if (status != 0) {
        return status;
    }
    status = run_gm_softmax_selftest();
    if (status != 0) {
        return status;
    }
    return run_gm_softmax_dim64_selftest();
}
