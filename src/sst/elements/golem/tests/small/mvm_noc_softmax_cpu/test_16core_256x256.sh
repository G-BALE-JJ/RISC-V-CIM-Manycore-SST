#!/usr/bin/env bash
# Standard test: 16-core 256x256 matrix GEMM + single-core softmax

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/configs/16core_256x256.env"

echo "=========================================="
echo "16-Core 256x256 Test (Standard)"
echo "=========================================="
echo "Config: configs/16core_256x256.env"
echo "GEMM: 16 cores parallel (16 tiles)"
echo "Softmax: Core 0 single-core post-processing"
echo ""

# Load configuration
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Source config to set environment variables
source "$CONFIG"

# Run pipeline with verification
"$SCRIPT_DIR/run_noc_dma_softmax_pipeline.sh" \
    --verify-softmax \
    --softmax-reference probability \
    "$@"
