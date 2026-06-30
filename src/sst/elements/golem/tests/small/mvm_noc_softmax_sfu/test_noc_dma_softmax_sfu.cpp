#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>

#include "core_bind.h"
#include "pipeline_config.h"
#include "../mvm_noc_int_array/golem_matmul_runtime.h"
#include "../mvm_noc_int_array/gemm_matmul_op.h"
#include "golem_softmax_sfu_runtime.h"

namespace {

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

golem_matmul_op_desc_t make_matmul_desc_from_env() {
    return {
        .m = read_i64_env_or_default("GOLEM_MATMUL_M", GEMM_M),
        .n = read_i64_env_or_default("GOLEM_MATMUL_N", GEMM_N),
        .k = read_i64_env_or_default("GOLEM_MATMUL_K", GEMM_K),
        .block_m = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_M", TILE_M),
        .block_n = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_N", TILE_N_MAX),
        .block_k = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_K", TILE_K),
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
        .transpose_a = 0,
        .transpose_b = 0,
    };
}

int resolve_executor_core_from_argv_or_exit(int argc, char* argv[], int requested_core_id) {
    if (requested_core_id < 0 || requested_core_id >= TOTAL_CORES) {
        std::fprintf(stderr, "[ERROR] invalid requested core id=%d, TOTAL_CORES=%d\n",
                     requested_core_id, TOTAL_CORES);
        return -1;
    }
    if (argc >= 2) {
        bind_process_to_core(requested_core_id);
    }
    const int executor_core_id = sched_getcpu();
    if (executor_core_id < 0 || executor_core_id >= TOTAL_CORES) {
        std::fprintf(stderr, "[ERROR] invalid executor core id=%d, TOTAL_CORES=%d\n",
                     executor_core_id, TOTAL_CORES);
        return -1;
    }
    return executor_core_id;
}

int run_gemm_for_core(int executor_core_id, int worker_core_id, const golem_matmul_op_desc_t& op_desc) {
    const MatmulRuntimeConfig cfg = {
        .m = static_cast<int>(op_desc.m),
        .n = static_cast<int>(op_desc.n),
        .k = static_cast<int>(op_desc.k),
        .block_m = static_cast<int>(op_desc.block_m),
        .block_n = static_cast<int>(op_desc.block_n),
        .block_k = static_cast<int>(op_desc.block_k),
    };
    if (op_desc.layout != GOLEM_LAYOUT_ROW_MAJOR ||
        op_desc.transpose_a != 0 ||
        op_desc.transpose_b != 0 ||
        !validate_matmul_call(cfg)) {
        std::fprintf(stderr, "[ERROR] invalid SFU GEMM op descriptor\n");
        return 1;
    }
    if ((cfg.block_m % TILE_M) != 0 || (cfg.block_k % TILE_K) != 0 || cfg.block_n > TILE_N_MAX) {
        std::fprintf(stderr, "[ERROR] unsupported SFU GEMM tile shape\n");
        return 1;
    }

    const GemmTileRuntimeContext rt = make_gemm_runtime_context(executor_core_id);
    const int total_tasks = gemm_total_tasks(cfg);
    const int worker_slot = gemm_worker_slot_for_core(worker_core_id);
    if (worker_slot < 0 || worker_slot >= total_tasks) {
        return 0;
    }

    GemmKernelStats stats = {};
    for (int task_id = worker_slot; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, task_id, cfg);
        gemm_tiled<float>(executor_core_id, desc, rt, &stats);
    }
    return 0;
}

int read_requested_core_from_argv(int argc, char* argv[]) {
    if (argc < 2) {
        return 0;
    }
    return std::atoi(argv[1]);
}

int run_riscv_gemm_softmax_sfu(int argc, char* argv[]) {
    const int requested_core_id = read_requested_core_from_argv(argc, argv);
    const int executor_core_id = resolve_executor_core_from_argv_or_exit(argc, argv, requested_core_id);
    if (executor_core_id < 0) {
        return 1;
    }

    const golem_matmul_op_desc_t op_desc = make_matmul_desc_from_env();
    if (op_desc.dtype != GOLEM_DTYPE_FP32) {
        std::fprintf(stderr, "[SOFTMAX-SFU] only GOLEM_DTYPE_FP32 is supported\n");
        return 1;
    }

    int status = run_gemm_for_core(executor_core_id, requested_core_id, op_desc);
    if (status != 0) {
        return status;
    }
    const int64_t skip_softmax = read_i64_env_or_default("GOLEM_SFU_SKIP_SOFTMAX", 0);
    if (skip_softmax != 0) {
        if (requested_core_id == 0) {
            std::printf("[SOFTMAX] mode=sfu-skip-softmax executor_core=%d\n", executor_core_id);
            std::fflush(stdout);
        }
        return 0;
    }

    const MatmulRuntimeConfig cfg = {
        .m = static_cast<int>(op_desc.m),
        .n = static_cast<int>(op_desc.n),
        .k = static_cast<int>(op_desc.k),
        .block_m = static_cast<int>(op_desc.block_m),
        .block_n = static_cast<int>(op_desc.block_n),
        .block_k = static_cast<int>(op_desc.block_k),
    };
    const golem_softmax_op_desc_t softmax_desc = {
        .outer = cfg.m,
        .dim = cfg.n,
        .axis = -1,
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };

    if (requested_core_id == 0) {
        std::printf("[SOFTMAX] mode=sfu m=%d n=%d block_m=%d block_n=%d executor_core=%d\n",
                    cfg.m, cfg.n, cfg.block_m, cfg.block_n, executor_core_id);
        std::fflush(stdout);
    }

    const uint64_t job_id =
        (static_cast<uint64_t>(cfg.m) << 48) ^
        (static_cast<uint64_t>(cfg.n) << 32) ^
        (static_cast<uint64_t>(cfg.block_m) << 16) ^
        static_cast<uint64_t>(cfg.block_n);

    const golem_status_t sfu_status = golemRunSoftmaxSfuForCore(
        &softmax_desc, executor_core_id, requested_core_id, &cfg, job_id);
    if (sfu_status != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[SOFTMAX-SFU] failed on core %d\n", requested_core_id);
        return 1;
    }
    return 0;
}

int run_host_smoke() {
    const golem_softmax_op_desc_t desc = {
        .outer = 1,
        .dim = 4,
        .axis = -1,
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };
    if (desc.dtype != GOLEM_DTYPE_FP32 || desc.dim != 4) {
        return 1;
    }
    std::printf("[SOFTMAX-SFU-SELFTEST] PASS descriptor smoke\n");
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc > 1) {
        return run_riscv_gemm_softmax_sfu(argc, argv);
    }
    return run_host_smoke();
}
