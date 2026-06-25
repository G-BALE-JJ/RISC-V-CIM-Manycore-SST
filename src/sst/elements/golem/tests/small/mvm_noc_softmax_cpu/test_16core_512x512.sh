#!/bin/bash
# Stress test: 16-core 512x512 matrix (64 tiles, heavy)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stress Test: 16-core 512x512 matrix"
echo "GEMM tiles: 64 (8x8 tiles)"
echo "Each core processes: 4 tiles on average"
echo ""

./run_16core_large_matrix.sh 512
