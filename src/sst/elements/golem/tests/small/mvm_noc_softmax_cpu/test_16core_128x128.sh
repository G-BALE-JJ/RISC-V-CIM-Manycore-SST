#!/bin/bash
# Quick test: 16-core 128x128 matrix (16 tiles, fast)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Quick Test: 16-core 128x128 matrix"
echo "GEMM tiles: 4 (2x2 tiles)"
echo "Each core processes: ~0.25 tiles on average"
echo ""

./run_16core_large_matrix.sh 128
