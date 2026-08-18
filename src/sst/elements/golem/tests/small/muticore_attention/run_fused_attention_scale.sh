#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_RUNNER="$TESTS_DIR/run_noc_dma_pipeline.sh"
ARTIFACT_ROOT=""
TIMEOUT_SECONDS=7200
TIMEOUT_EXPLICIT=0
DRY_RUN=0
SCALE_POINT=e4
ALLOW_EXPENSIVE=0
PV_MATRIX_BROADCAST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="$2"; TIMEOUT_EXPLICIT=1; shift 2 ;;
    --scale-point) SCALE_POINT="$2"; shift 2 ;;
    --allow-expensive) ALLOW_EXPENSIVE=1; shift ;;
    --pv-matrix-broadcast) PV_MATRIX_BROADCAST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: run_fused_attention_scale.sh [--scale-point e2|e3|e4|e5] [--allow-expensive] [--pv-matrix-broadcast] [--artifact-root DIR] [--timeout SEC] [--dry-run]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$SCALE_POINT" in
  e2)
    RUN_ID=fused_attention_e2_s256_d64
    TOTAL_QUERIES=256
    KEYS=256
    HEAD_DIM=64
    MANAGER_QUERIES=64
    ARRAY_INPUT=64
    BUILD_TARGET=scale-e2
    GUEST_NAME=fused_attention_scale_e2
    ;;
  e3)
    RUN_ID=fused_attention_e3_s1024_d128
    TOTAL_QUERIES=1024
    KEYS=1024
    HEAD_DIM=128
    MANAGER_QUERIES=256
    ARRAY_INPUT=128
    BUILD_TARGET=scale-e3
    GUEST_NAME=fused_attention_scale_e3
    ;;
  e4)
    RUN_ID=fused_attention_e4_s2048_d128
    TOTAL_QUERIES=2048
    KEYS=2048
    HEAD_DIM=128
    MANAGER_QUERIES=512
    ARRAY_INPUT=128
    BUILD_TARGET=scale-e4
    GUEST_NAME=fused_attention_scale_e4
    ;;
  e5)
    RUN_ID=fused_attention_e5_s4096_d128
    TOTAL_QUERIES=4096
    KEYS=4096
    HEAD_DIM=128
    MANAGER_QUERIES=1024
    ARRAY_INPUT=128
    BUILD_TARGET=scale-e5
    GUEST_NAME=fused_attention_scale_e5
    if (( ! TIMEOUT_EXPLICIT )); then TIMEOUT_SECONDS=28800; fi
    ;;
  *) echo "Unknown scale point: $SCALE_POINT" >&2; exit 2 ;;
esac

if [[ "$SCALE_POINT" == e5 ]] && (( ! DRY_RUN && ! ALLOW_EXPENSIVE )); then
  echo "E5 is an expensive full SST run; pass --allow-expensive to execute it" >&2
  exit 2
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data4/jjgong/tmp/$RUN_ID}"
Q_FILE="$ARTIFACT_ROOT/q_${TOTAL_QUERIES}x${HEAD_DIM}.bin"
K_FILE="$ARTIFACT_ROOT/k_${KEYS}x${HEAD_DIM}.bin"
V_FILE="$ARTIFACT_ROOT/v_${KEYS}x${HEAD_DIM}.bin"
RESULT_JSON="$ARTIFACT_ROOT/fused_attention_result.json"
LIFECYCLE_JSON="$ARTIFACT_ROOT/attention_lifecycle.json"
GUEST="$SCRIPT_DIR/riscv64/$GUEST_NAME"
HBM_DIR="$ARTIFACT_ROOT/hbm"
STATS_FILE="$ARTIFACT_ROOT/stats/overlap0/$RUN_ID/stats_selfcom.txt"
MEM_NODE_SIZE=134217728
Q_OFFSET=$((0x02000000))
K_OFFSET=$((0x02100000))
V_OFFSET=$((0x02200000))
O_OFFSET=$((0x02300000))

GENERATE_CMD=(python3 "$SCRIPT_DIR/attention_case.py" generate
  --queries "$TOTAL_QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE")

RUN_CMD=(timeout "$TIMEOUT_SECONDS" env
  "GOLEM_RUN_ID=$RUN_ID"
  "GOLEM_ARTIFACT_ROOT=$ARTIFACT_ROOT"
  "VANADIS_EXE=$GUEST"
  GOLEM_SKIP_DEFAULT_GUEST_BUILD=1
  GOLEM_ARCH_SCRIPT=architecture/archive/ncores_selfcom_dma.py
  GOLEM_ATTENTION_FUSED=1
  GOLEM_ATTENTION_HBM_STRIPED=1
  "GOLEM_ATTENTION_QUERIES=$TOTAL_QUERIES"
  "GOLEM_ATTENTION_KEYS=$KEYS"
  "GOLEM_ATTENTION_HEAD_DIM=$HEAD_DIM"
  "GOLEM_ATTENTION_Q_FILE=$Q_FILE"
  "GOLEM_ATTENTION_K_FILE=$K_FILE"
  "GOLEM_ATTENTION_V_FILE=$V_FILE"
  "GOLEM_ATTENTION_Q_OFFSET=$Q_OFFSET"
  "GOLEM_ATTENTION_K_OFFSET=$K_OFFSET"
  "GOLEM_ATTENTION_V_OFFSET=$V_OFFSET"
  GOLEM_ATTENTION_WINDOW_OFFSET=0xC0000
  GOLEM_ATTENTION_WINDOW_BYTES=0x10000
  "GOLEM_ATTENTION_PV_MATRIX_BROADCAST=$PV_MATRIX_BROADCAST"
  GOLEM_SFU_ROW_CONTEXTS=16
  GOLEM_DMA_READ_RETRY_TICKS=4096
  GOLEM_DMA_READ_MAX_RETRIES=32
  GOLEM_GROUP_MANAGER_ENABLE=1
  GOLEM_SFU_MANAGER_COORDINATOR=1
  GOLEM_CTRL_LINK_ENABLE=0
  GOLEM_REQUEST_SCHEDULER_ENABLE=0
  GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0
  GOLEM_SFU_ENABLE=1
  GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc
  bash "$BASE_RUNNER"
  --dtype fp32 --tensor-source file --tensor-a "$Q_FILE" --tensor-b "$K_FILE"
  --transpose-b 1 --hbm-dump-output 1
  --gemm-m "$TOTAL_QUERIES" --gemm-n "$KEYS" --gemm-k "$HEAD_DIM"
  --orig-m "$TOTAL_QUERIES" --orig-n "$KEYS" --orig-k "$HEAD_DIM"
  --gemm-block-m 16 --gemm-block-n 16 --gemm-block-k "$HEAD_DIM"
  --array-in "$ARRAY_INPUT" --array-out 16 --num-arrays 16
  --groups 4 --num-cores 20 --gemm-cores 20 --num-mem-nodes 5 --mesh-dim-x 4
  --global-stride-kb 1024 --mem-node-size "$MEM_NODE_SIZE")

VERIFY_CMD=(python3 "$SCRIPT_DIR/verify_fused_attention_scale_output.py"
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE"
  --queries "$TOTAL_QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --band-rows "$MANAGER_QUERIES" --hbm-dir "$HBM_DIR"
  --output-offset "$O_OFFSET" --result-json "$RESULT_JSON")
VERIFY_STATS_CMD=(python3 "$SCRIPT_DIR/verify_fused_attention_scale_stats.py"
  --profile "$SCALE_POINT"
  --accelerator-clock "${VANADIS_CPU_CLOCK:-1.0GHz}"
  --timebase-ticks-per-second 1000000000000
  --result-json "$LIFECYCLE_JSON" "$STATS_FILE")
if (( PV_MATRIX_BROADCAST )); then
  VERIFY_STATS_CMD+=(--pv-matrix-broadcast)
fi

if (( DRY_RUN )); then
  printf '%q ' "${GENERATE_CMD[@]}"; printf '\n'
  echo "make -C $SCRIPT_DIR $BUILD_TARGET"
  printf '%q ' "${RUN_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_STATS_CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$ARTIFACT_ROOT"
"${GENERATE_CMD[@]}"
make -C "$SCRIPT_DIR" "$BUILD_TARGET"
"${RUN_CMD[@]}"
"${VERIFY_CMD[@]}"
"${VERIFY_STATS_CMD[@]}"

echo "Fused Attention ${SCALE_POINT^^} PASS: B1,H1,S${TOTAL_QUERIES},D${HEAD_DIM},4 managers,16 workers,1 tensor completion"
echo "Artifacts: $ARTIFACT_ROOT"
