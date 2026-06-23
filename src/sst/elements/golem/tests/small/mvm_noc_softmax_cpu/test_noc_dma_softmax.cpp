#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__riscv)
#include "core_bind.h"
#include "pipeline_config.h"
#include "golem_matmul_runtime.h"
#endif
#include "golem_softmax_runtime.h"

namespace {

bool close_enough(float got, float expected) {
    const float diff = std::fabs(got - expected);
    return diff <= 1.0e-5f;
}

#if defined(__riscv)
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

int run_tile_local_softmax_for_core(int core_id, const golem_matmul_op_desc_t& op_desc) {
    if (op_desc.dtype != GOLEM_DTYPE_FP32) {
        if (core_id == 0) {
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
    const int worker_slot = gemm_worker_slot_for_core(core_id);
    if (worker_slot < 0 || worker_slot >= total_tasks) {
        return 0;
    }

    int softmax_tiles = 0;
    for (int task_id = worker_slot; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_task_desc_for_task(core_id, task_id, cfg);
        golem_softmax_op_desc_t softmax_desc = {
            .outer = desc.block_m,
            .dim = desc.block_n,
            .axis = -1,
            .dtype = GOLEM_DTYPE_FP32,
            .layout = GOLEM_LAYOUT_ROW_MAJOR,
        };
        const golem_status_t status = golemRunSoftmaxCpuGmForCore(
            &softmax_desc,
            core_id,
            desc.c_base_mm,
            desc.c_base_mm,
            desc.block_n,
            desc.block_n);
        if (status != GOLEM_STATUS_OK) {
            std::fprintf(stderr,
                         "[Core %d] [SOFTMAX] tile task=%d failed: %s\n",
                         core_id,
                         task_id,
                         golemSoftmaxGetLastErrorString());
            return 1;
        }
        softmax_tiles++;
    }

    std::printf("[Core %d] [SOFTMAX] tile-local softmax complete: tiles=%d\n", core_id, softmax_tiles);
    std::fflush(stdout);
    return 0;
}

int run_riscv_gemm_softmax(int argc, char* argv[]) {
    bind_and_resolve_core_from_argv_or_exit(argc, argv, TOTAL_CORES);
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        std::fprintf(stderr, "[ERROR] invalid runtime core id=%d, TOTAL_CORES=%d\n", core_id, TOTAL_CORES);
        return 1;
    }

    const golem_matmul_op_desc_t op_desc = make_matmul_desc_from_env();
    int status = run_gemm(op_desc);
    if (status != 0) {
        return status;
    }
    return run_tile_local_softmax_for_core(core_id, op_desc);
}
#endif

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
#if defined(__riscv)
    if (argc > 1) {
        return run_riscv_gemm_softmax(argc, argv);
    }
#else
    (void)argc;
    (void)argv;
#endif
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
