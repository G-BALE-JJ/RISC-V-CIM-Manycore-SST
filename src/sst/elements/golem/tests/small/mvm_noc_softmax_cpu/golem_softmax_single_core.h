#ifndef GOLEM_SOFTMAX_SINGLE_CORE_H_
#define GOLEM_SOFTMAX_SINGLE_CORE_H_

#include <cstdint>
#include "golem_softmax_runtime.h"

// Single-core post-processing softmax: Core 0 aggregates full rows then computes softmax
// Advantages: Simple, no synchronization overhead, fast SST simulation
// Disadvantages: Softmax phase is serial (but GEMM phase is still parallel)

struct SingleCoreSoftmaxContext {
    int64_t total_m_tiles;
    int64_t total_n_tiles;
    int64_t m;
    int64_t n;
    int64_t block_m;
    int64_t block_n;
};

// Initialize single-core context
golem_status_t golemInitSingleCoreSoftmaxContext(
    SingleCoreSoftmaxContext* ctx,
    int64_t m, int64_t n,
    int64_t block_m, int64_t block_n);

// Run single-core softmax (only Core 0 executes, other cores return immediately)
// This function aggregates full rows from all N-tiles and computes row-wise softmax
golem_status_t golemRunSingleCoreSoftmaxForCore(
    const golem_softmax_op_desc_t* op_desc,
    const SingleCoreSoftmaxContext* ctx,
    int executor_core_id,
    int softmax_core_id);

#endif  // GOLEM_SOFTMAX_SINGLE_CORE_H_
