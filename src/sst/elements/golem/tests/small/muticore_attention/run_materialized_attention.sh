#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
QKT_RUNNER="$SCRIPT_DIR/run_muticore_attention.sh"
SOFTMAX_RUNNER="$TESTS_DIR/small/muticore_softmax/run_muticore_softmax.sh"
GEMM_RUNNER="$TESTS_DIR/run_noc_dma_pipeline.sh"

QUERIES=64
KEYS=64
HEAD_DIM=64
CAUSAL=0
TIMEOUT_SECONDS=600
ARTIFACT_ROOT=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: run_materialized_attention.sh [options]
  --queries N --keys N --head-dim 64|128 --causal 0|1
  --timeout SEC --artifact-root DIR --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queries) QUERIES="$2"; shift 2 ;;
    --keys) KEYS="$2"; shift 2 ;;
    --head-dim) HEAD_DIM="$2"; shift 2 ;;
    --causal) CAUSAL="$2"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "$QUERIES" "$KEYS" "$HEAD_DIM" "$TIMEOUT_SECONDS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "dimensions and timeout must be positive" >&2; exit 2; }
done
[[ "$HEAD_DIM" == 64 || "$HEAD_DIM" == 128 ]] || { echo "head-dim must be 64 or 128" >&2; exit 2; }
[[ "$CAUSAL" == 0 || "$CAUSAL" == 1 ]] || { echo "causal must be 0 or 1" >&2; exit 2; }
(( QUERIES % 16 == 0 && KEYS % 64 == 0 && KEYS <= 4096 )) || {
  echo "materialized baseline requires queries divisible by 16 and keys divisible by 64, <=4096" >&2
  exit 2
}
if (( CAUSAL == 1 && QUERIES != KEYS )); then
  echo "causal baseline currently requires queries == keys" >&2
  exit 2
fi

if [[ -z "$ARTIFACT_ROOT" ]]; then
  ARTIFACT_ROOT="/data4/jjgong/tmp/materialized_attention_q${QUERIES}_k${KEYS}_d${HEAD_DIM}_c${CAUSAL}"
fi
QKT_ROOT="$ARTIFACT_ROOT/qkt"
SOFTMAX_ROOT="$ARTIFACT_ROOT/softmax"
PV_ROOT="$ARTIFACT_ROOT/pv"
Q_FILE="$QKT_ROOT/q_${QUERIES}x${HEAD_DIM}.bin"
K_FILE="$QKT_ROOT/native_k_${KEYS}x${HEAD_DIM}_padded${KEYS}.bin"
S_FILE="$QKT_ROOT/qk_${QUERIES}x${KEYS}.bin"
V_FILE="$ARTIFACT_ROOT/inputs/v_${KEYS}x${HEAD_DIM}.bin"
P_FILE="$SOFTMAX_ROOT/outputs/softmax.bin"
O_FILE="$PV_ROOT/o_${QUERIES}x${HEAD_DIM}.bin"
RESULT_JSON="$ARTIFACT_ROOT/materialized_attention_result.json"

QKT_CMD=(bash "$QKT_RUNNER" --queries "$QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --timeout "$TIMEOUT_SECONDS" --artifact-root "$QKT_ROOT")
GENERATE_V_CMD=(python3 "$SCRIPT_DIR/attention_case.py" generate
  --queries "$QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE")
SOFTMAX_CMD=(bash "$SOFTMAX_RUNNER" --rows "$QUERIES" --cols "$KEYS"
  --timeout "$TIMEOUT_SECONDS" --artifact-root "$SOFTMAX_ROOT"
  --logits-file "$S_FILE" --attention-head-dim "$HEAD_DIM" --causal "$CAUSAL")
PV_CMD=(timeout "$TIMEOUT_SECONDS" env
  "GOLEM_RUN_ID=attention_pv_q${QUERIES}_k${KEYS}_d${HEAD_DIM}_c${CAUSAL}"
  "GOLEM_ARTIFACT_ROOT=$PV_ROOT"
  GOLEM_GROUP_MANAGER_ENABLE=1 GOLEM_CTRL_LINK_ENABLE=1
  bash "$GEMM_RUNNER" --dtype fp32 --tensor-source file
  --tensor-a "$P_FILE" --tensor-b "$V_FILE" --dump-c "$O_FILE" --hbm-dump-output 1
  --gemm-m "$QUERIES" --gemm-n "$HEAD_DIM" --gemm-k "$KEYS"
  --orig-m "$QUERIES" --orig-n "$HEAD_DIM" --orig-k "$KEYS"
  --gemm-block-m 16 --gemm-block-n 16 --gemm-block-k 64
  --array-in 64 --array-out 16 --num-arrays 16
  --num-cores 16 --gemm-cores 16 --num-mem-nodes 5 --mesh-dim-x 4
  --global-stride-kb 1024 --mem-node-size 134217728)
VERIFY_CMD=(python3 "$SCRIPT_DIR/attention_case.py" verify-attention
  --queries "$QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE" --output-file "$O_FILE"
  --causal "$CAUSAL" --result-json "$RESULT_JSON")

if (( DRY_RUN )); then
  printf '%q ' "${QKT_CMD[@]}"; printf '%s\n' '--dry-run'
  printf '%q ' "${GENERATE_V_CMD[@]}"; printf '\n'
  printf '%q ' "${SOFTMAX_CMD[@]}"; printf '%s\n' '--dry-run'
  printf '%q ' "${PV_CMD[@]}"; printf '%s\n' '--dry-run'
  printf '%q ' "${VERIFY_CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$ARTIFACT_ROOT/inputs"
"${QKT_CMD[@]}"
"${GENERATE_V_CMD[@]}"
"${SOFTMAX_CMD[@]}"
"${PV_CMD[@]}"
"${VERIFY_CMD[@]}"

echo "Materialized Attention PASS: Q=${QUERIES}, K=${KEYS}, D=${HEAD_DIM}, causal=${CAUSAL}"
echo "Artifacts: $ARTIFACT_ROOT"
