#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_RUNNER="$TESTS_DIR/run_noc_dma_pipeline.sh"

QUERIES=64
KEYS=64
HEAD_DIM=64
TIMEOUT_SECONDS=300
ARTIFACT_ROOT=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: run_muticore_attention.sh [options]

Phase A QK^T smoke test using native K[key, dim] storage and transpose_b=1.

Options:
  --queries N        Number of query rows (default: 64)
  --keys N           Number of key rows (default: 64)
  --head-dim N       Head dimension, 64 or 128 (default: 64)
  --timeout N        Simulation timeout in seconds (default: 300)
  --artifact-root P  Output directory (default: /data4/jjgong/tmp/...)
  --dry-run          Print commands without generating files or running SST
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queries) QUERIES="$2"; shift 2 ;;
    --keys) KEYS="$2"; shift 2 ;;
    --head-dim) HEAD_DIM="$2"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "$QUERIES" "$KEYS" "$HEAD_DIM" "$TIMEOUT_SECONDS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "All dimensions and timeout must be positive integers" >&2
    exit 2
  fi
done
if [[ "$HEAD_DIM" != 64 && "$HEAD_DIM" != 128 ]]; then
  echo "--head-dim must be 64 or 128" >&2
  exit 2
fi

largest_divisor_at_most() {
  local value="$1"
  local limit="$2"
  local candidate
  for ((candidate=limit; candidate>=1; --candidate)); do
    if (( value % candidate == 0 )); then
      echo "$candidate"
      return
    fi
  done
}

ARRAY_OUT="$(largest_divisor_at_most "$QUERIES" 16)"
BLOCK_N=16
NUM_ARRAYS="$BLOCK_N"
PADDED_KEYS=$(( (KEYS + BLOCK_N - 1) / BLOCK_N * BLOCK_N ))

if [[ -z "$ARTIFACT_ROOT" ]]; then
  ARTIFACT_ROOT="/data4/jjgong/tmp/muticore_attention_q${QUERIES}_k${KEYS}_d${HEAD_DIM}"
fi

Q_FILE="$ARTIFACT_ROOT/q_${QUERIES}x${HEAD_DIM}.bin"
K_FILE="$ARTIFACT_ROOT/native_k_${KEYS}x${HEAD_DIM}_padded${PADDED_KEYS}.bin"
OUTPUT_FILE="$ARTIFACT_ROOT/qk_${QUERIES}x${KEYS}.bin"
MANIFEST="$ARTIFACT_ROOT/case.json"
RESULT_JSON="$ARTIFACT_ROOT/verification.json"

GENERATE_CMD=(
  python3 "$SCRIPT_DIR/attention_case.py" generate
  --queries "$QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --storage-keys "$PADDED_KEYS"
  --q-file "$Q_FILE" --k-file "$K_FILE" --manifest "$MANIFEST"
)

RUN_CMD=(
  timeout "$TIMEOUT_SECONDS" env
  "GOLEM_RUN_ID=attention_q${QUERIES}_k${KEYS}_d${HEAD_DIM}"
  "GOLEM_ARTIFACT_ROOT=$ARTIFACT_ROOT"
  GOLEM_GROUP_MANAGER_ENABLE=1 GOLEM_CTRL_LINK_ENABLE=1
  GOLEM_A_REUSE_N_TILES=1 GOLEM_B_REUSE_M_TILES=1
  bash "$BASE_RUNNER"
  --dtype fp32 --tensor-source file
  --tensor-a "$Q_FILE" --tensor-b "$K_FILE"
  --transpose-b 1
  --dump-c "$OUTPUT_FILE" --hbm-dump-output 1
  --gemm-m "$QUERIES" --gemm-n "$PADDED_KEYS" --gemm-k "$HEAD_DIM"
  --orig-m "$QUERIES" --orig-n "$KEYS" --orig-k "$HEAD_DIM"
  --gemm-block-m "$ARRAY_OUT" --gemm-block-n "$BLOCK_N" --gemm-block-k "$HEAD_DIM"
  --array-in "$HEAD_DIM" --array-out "$ARRAY_OUT" --num-arrays "$NUM_ARRAYS"
  --num-cores 16 --gemm-cores 16 --num-mem-nodes 5 --mesh-dim-x 4
  --global-stride-kb 1024 --mem-node-size 134217728
)

VERIFY_CMD=(
  python3 "$SCRIPT_DIR/attention_case.py" verify
  --queries "$QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --storage-keys "$PADDED_KEYS"
  --q-file "$Q_FILE" --k-file "$K_FILE" --output-file "$OUTPUT_FILE"
  --result-json "$RESULT_JSON"
)

if (( DRY_RUN )); then
  printf '%q ' "${GENERATE_CMD[@]}"; printf '\n'
  printf '%q ' "${RUN_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$ARTIFACT_ROOT"
"${GENERATE_CMD[@]}"
"${RUN_CMD[@]}"
"${VERIFY_CMD[@]}"

echo "Attention Phase A PASS: QK^T ${QUERIES}x${KEYS}, D=${HEAD_DIM}"
echo "Artifacts: $ARTIFACT_ROOT"
