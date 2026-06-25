#!/usr/bin/env bash
# Stress test: 20-core (16 worker + 4 manager) 512x512 GEMM + single-core softmax

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/configs/20core_512x512.env"

echo "=========================================="
echo "20-Core 512x512 Test (Stress)"
echo "=========================================="
echo "Config: configs/20core_512x512.env"
echo "Architecture: 16 worker cores + 4 manager cores"
echo "GEMM: 16 worker cores parallel (64 tiles)"
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
