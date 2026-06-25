#!/bin/bash
# 16-core large matrix GEMM + single-core softmax test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "16-Core Large Matrix GEMM + Single-Core Softmax"
echo "=========================================="

# Configuration
MATRIX_SIZE=${1:-256}  # Default 256x256, can pass as argument
GEMM_M=$MATRIX_SIZE
GEMM_N=$MATRIX_SIZE
GEMM_K=$MATRIX_SIZE

echo "Matrix dimensions: M=$GEMM_M, N=$GEMM_N, K=$GEMM_K"
echo "GEMM: 16 cores (parallel)"
echo "Softmax: Core 0 only (single-core post-processing)"
echo ""

# Ensure single-core softmax mode (default)
export GOLEM_SOFTMAX_MODE=single-core

# Run pipeline with 16 cores
./run_noc_dma_softmax_pipeline.sh \
  --groups 4 \
  --num-cores 16 \
  --gemm-cores 16 \
  --gemm-m $GEMM_M \
  --gemm-n $GEMM_N \
  --gemm-k $GEMM_K \
  --gemm-block-m 64 \
  --gemm-block-n 64 \
  --gemm-block-k 64 \
  --num-mem-nodes 8 \
  --group-manager-enable 1 \
  --ctrl-link-enable 1 \
  --verify-softmax \
  --softmax-reference probability \
  "$@"

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo ""
echo "Task distribution:"
echo "  GEMM tiles: $(( (GEMM_M/64) * (GEMM_N/64) )) tiles"
echo "  GEMM cores: 16 cores (parallel)"
echo "  Softmax: Core 0 aggregates $(( GEMM_M/64 * GEMM_N/64 )) tiles → computes softmax"
echo ""
echo "To test different sizes:"
echo "  ./run_16core_large_matrix.sh 128   # 128x128"
echo "  ./run_16core_large_matrix.sh 256   # 256x256 (default)"
echo "  ./run_16core_large_matrix.sh 512   # 512x512"
