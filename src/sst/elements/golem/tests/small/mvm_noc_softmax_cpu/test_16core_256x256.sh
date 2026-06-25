#!/bin/bash
# Standard test: 16-core 256x256 matrix (16 tiles, typical)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Standard Test: 16-core 256x256 matrix"
echo "GEMM tiles: 16 (4x4 tiles)"
echo "Each core processes: 1 tile on average"
echo ""

./run_16core_large_matrix.sh 256
