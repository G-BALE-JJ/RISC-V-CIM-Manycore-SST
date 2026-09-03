#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKTREE_ROOT="$(cd "$TESTS_DIR/../../../../.." && pwd -P)"
BASE_RUNNER="$TESTS_DIR/run_noc_dma_pipeline.sh"
LOCAL_ELEMENT_LIB="$WORKTREE_ROOT/install/lib/sst-elements-library"
ARTIFACT_ROOT=""
TIMEOUT_SECONDS=7200
TIMEOUT_EXPLICIT=0
DRY_RUN=0
SCALE_POINT=e4
ALLOW_EXPENSIVE=0
PV_MATRIX_BROADCAST=0
QK_MATRIX_BROADCAST=0
QK_DATAFLOW_TRANSPOSE=0
KV_TILE_ROTATION=0
KV_DOUBLE_BUFFER=0
PV_V_TILE_REUSE=0
PV_INPUT_PIPELINE=0
PV_COMPACT_INPUT=0
PV_RESTORE_PIPELINE=0
PV_OUTPUT_PIPELINE=0
PV_EARLY_COMPUTE=0
PV_MATRIX_SOFTMAX_OVERLAP=0
MPI_RANKS="${GOLEM_MPI_RANKS:-1}"
MPI_PARTITIONER=sst.simple
ATTENTION_SST_ARGS="${GOLEM_ATTENTION_SST_ARGS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="$2"; TIMEOUT_EXPLICIT=1; shift 2 ;;
    --scale-point) SCALE_POINT="$2"; shift 2 ;;
    --allow-expensive) ALLOW_EXPENSIVE=1; shift ;;
    --pv-matrix-broadcast) PV_MATRIX_BROADCAST=1; shift ;;
    --qk-matrix-broadcast) QK_MATRIX_BROADCAST=1; shift ;;
    --qk-dataflow-transpose) QK_DATAFLOW_TRANSPOSE=1; QK_MATRIX_BROADCAST=1; shift ;;
    --kv-tile-rotation) KV_TILE_ROTATION=1; shift ;;
    --kv-double-buffer) KV_DOUBLE_BUFFER=1; shift ;;
    --pv-v-tile-reuse) PV_V_TILE_REUSE=1; shift ;;
    --pv-input-pipeline) PV_INPUT_PIPELINE=1; shift ;;
    --pv-compact-input) PV_COMPACT_INPUT=1; shift ;;
    --pv-restore-pipeline) PV_RESTORE_PIPELINE=1; shift ;;
    --pv-output-pipeline) PV_OUTPUT_PIPELINE=1; shift ;;
    --pv-early-compute) PV_EARLY_COMPUTE=1; shift ;;
    --pv-matrix-softmax-overlap) PV_MATRIX_SOFTMAX_OVERLAP=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: run_fused_attention_scale.sh [--scale-point e2|e3|e4|e5] [--allow-expensive] [--pv-matrix-broadcast] [--qk-matrix-broadcast] [--qk-dataflow-transpose] [--kv-tile-rotation] [--kv-double-buffer] [--pv-v-tile-reuse] [--pv-input-pipeline] [--pv-compact-input] [--pv-restore-pipeline] [--pv-output-pipeline] [--pv-early-compute] [--pv-matrix-softmax-overlap] [--artifact-root DIR] [--timeout SEC] [--dry-run]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! [[ "$MPI_RANKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GOLEM_MPI_RANKS must be a positive integer" >&2
  exit 2
fi
if (( MPI_RANKS > 4 || 4 % MPI_RANKS != 0 )); then
  echo "GOLEM_MPI_RANKS must divide the four query-manager bands (supported: 1, 2, 4)" >&2
  exit 2
fi
if (( MPI_RANKS > 1 )); then
  MPI_PARTITIONER=sst.self
fi
if [[ " $ATTENTION_SST_ARGS " == *" --partitioner"* ||
      " $ATTENTION_SST_ARGS " == *" --lib-path"* ||
      " $ATTENTION_SST_ARGS " == *" --add-lib-path"* ]]; then
  echo "GOLEM_ATTENTION_SST_ARGS cannot override the partitioner or element library path" >&2
  exit 2
fi
if [[ ! -f "$LOCAL_ELEMENT_LIB/libgolem.so" ]]; then
  echo "Missing local element library: $LOCAL_ELEMENT_LIB/libgolem.so" >&2
  echo "Build it first with scripts/build_and_install_local.sh" >&2
  exit 1
fi

case "$SCALE_POINT" in
  e2)
    RUN_ID=fused_attention_e2_s256_d64
    TOTAL_QUERIES=256
    KEYS=256
    HEAD_DIM=64
    MANAGER_QUERIES=64
    ARRAY_INPUT=64
    GUEST_NAME=fused_attention_scale_e2
    ;;
  e3)
    RUN_ID=fused_attention_e3_s1024_d128
    TOTAL_QUERIES=1024
    KEYS=1024
    HEAD_DIM=128
    MANAGER_QUERIES=256
    ARRAY_INPUT=128
    GUEST_NAME=fused_attention_scale_e3
    ;;
  e4)
    RUN_ID=fused_attention_e4_s2048_d128
    TOTAL_QUERIES=2048
    KEYS=2048
    HEAD_DIM=128
    MANAGER_QUERIES=512
    ARRAY_INPUT=128
    GUEST_NAME=fused_attention_scale_e4
    ;;
  e5)
    RUN_ID=fused_attention_e5_s4096_d128
    TOTAL_QUERIES=4096
    KEYS=4096
    HEAD_DIM=128
    MANAGER_QUERIES=1024
    ARRAY_INPUT=128
    GUEST_NAME=fused_attention_scale_e5
    if (( ! TIMEOUT_EXPLICIT )); then TIMEOUT_SECONDS=28800; fi
    ;;
  *) echo "Unknown scale point: $SCALE_POINT" >&2; exit 2 ;;
esac

ATTENTION_WINDOW_BYTES=0x10000
if (( KV_DOUBLE_BUFFER )); then
  ATTENTION_WINDOW_BYTES=0x14880
fi

if [[ "$SCALE_POINT" == e5 ]] && (( ! DRY_RUN && ! ALLOW_EXPENSIVE )); then
  echo "E5 is an expensive full SST run; pass --allow-expensive to execute it" >&2
  exit 2
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${TMPDIR:-/tmp}/$RUN_ID}"
Q_FILE="$ARTIFACT_ROOT/q_${TOTAL_QUERIES}x${HEAD_DIM}.bin"
K_FILE="$ARTIFACT_ROOT/k_${KEYS}x${HEAD_DIM}.bin"
V_FILE="$ARTIFACT_ROOT/v_${KEYS}x${HEAD_DIM}.bin"
RESULT_JSON="$ARTIFACT_ROOT/fused_attention_result.json"
LIFECYCLE_JSON="$ARTIFACT_ROOT/attention_lifecycle.json"
MPI_PARTITION_JSON="$ARTIFACT_ROOT/attention_mpi_partition.json"
MPI_PLACEMENT_JSON="$ARTIFACT_ROOT/attention_mpi_placement.json"
GUEST="$SCRIPT_DIR/riscv64/$GUEST_NAME"
HBM_DIR="$ARTIFACT_ROOT/hbm"
STATS_FILE="$ARTIFACT_ROOT/stats/overlap0/$RUN_ID/stats_selfcom.txt"
MEM_NODE_SIZE=134217728
Q_OFFSET=$((0x02000000))
K_OFFSET=$((0x02100000))
V_OFFSET=$((0x02200000))
O_OFFSET=$((0x02300000))

if [[ ! -x "$GUEST" ]]; then
  echo "Missing FlashAttention guest: $GUEST" >&2
  echo "Build it first with scripts/build_and_install_local.sh" >&2
  exit 1
fi

GENERATE_CMD=(python3 "$SCRIPT_DIR/attention_case.py" generate
  --queries "$TOTAL_QUERIES" --keys "$KEYS" --head-dim "$HEAD_DIM"
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE")

RUN_CMD=(timeout "$TIMEOUT_SECONDS" env
  "SST_LIB_PATH=$LOCAL_ELEMENT_LIB"
  "GOLEM_SST_ARGS=--lib-path=$LOCAL_ELEMENT_LIB${ATTENTION_SST_ARGS:+ $ATTENTION_SST_ARGS}"
  "GOLEM_RUN_ID=$RUN_ID"
  "GOLEM_ARTIFACT_ROOT=$ARTIFACT_ROOT"
  "VANADIS_EXE=$GUEST"
  GOLEM_SKIP_DEFAULT_GUEST_BUILD=1
  GOLEM_ARCH_SCRIPT=architecture/archive/ncores_selfcom_dma.py
  GOLEM_ATTENTION_FUSED=1
  GOLEM_ATTENTION_HBM_STRIPED=1
  "GOLEM_ATTENTION_QUERY_BLOCK_MPI=$((MPI_RANKS > 1 ? 1 : 0))"
  "GOLEM_ATTENTION_PLACEMENT_FILE=$MPI_PLACEMENT_JSON"
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
  "GOLEM_ATTENTION_WINDOW_BYTES=$ATTENTION_WINDOW_BYTES"
  "GOLEM_ATTENTION_QK_DATAFLOW_TRANSPOSE=$QK_DATAFLOW_TRANSPOSE"
  "GOLEM_ATTENTION_QK_MATRIX_BROADCAST=$QK_MATRIX_BROADCAST"
  "GOLEM_ATTENTION_PV_MATRIX_BROADCAST=$PV_MATRIX_BROADCAST"
  "GOLEM_ATTENTION_KV_TILE_ROTATION=$KV_TILE_ROTATION"
  "GOLEM_ATTENTION_KV_DOUBLE_BUFFER=$KV_DOUBLE_BUFFER"
  "GOLEM_ATTENTION_PV_V_TILE_REUSE=$PV_V_TILE_REUSE"
  "GOLEM_ATTENTION_PV_INPUT_PIPELINE=$PV_INPUT_PIPELINE"
  "GOLEM_ATTENTION_PV_COMPACT_INPUT=$PV_COMPACT_INPUT"
  "GOLEM_ATTENTION_PV_RESTORE_PIPELINE=$PV_RESTORE_PIPELINE"
  "GOLEM_ATTENTION_PV_OUTPUT_PIPELINE=$PV_OUTPUT_PIPELINE"
  "GOLEM_ATTENTION_PV_EARLY_COMPUTE=$PV_EARLY_COMPUTE"
  "GOLEM_ATTENTION_PV_MATRIX_SOFTMAX_OVERLAP=$PV_MATRIX_SOFTMAX_OVERLAP"
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
  --global-stride-kb 1024 --mem-node-size "$MEM_NODE_SIZE"
  --mpi-ranks "$MPI_RANKS" --mpi-partitioner "$MPI_PARTITIONER")

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
VERIFY_MPI_CMD=(python3 "$SCRIPT_DIR/verify_attention_mpi_partition.py"
  --stats-file "$STATS_FILE" --mpi-ranks "$MPI_RANKS"
  --placement-file "$MPI_PLACEMENT_JSON"
  --result-json "$MPI_PARTITION_JSON")
if (( PV_MATRIX_BROADCAST )); then
  VERIFY_STATS_CMD+=(--pv-matrix-broadcast)
fi
if (( QK_MATRIX_BROADCAST )); then
  VERIFY_STATS_CMD+=(--qk-matrix-broadcast)
fi
if (( QK_DATAFLOW_TRANSPOSE )); then
  VERIFY_STATS_CMD+=(--qk-dataflow-transpose)
fi
if (( KV_DOUBLE_BUFFER )); then
  VERIFY_STATS_CMD+=(--kv-double-buffer)
fi
if (( PV_V_TILE_REUSE )); then
  VERIFY_STATS_CMD+=(--pv-v-tile-reuse)
fi
if (( PV_INPUT_PIPELINE )); then
  VERIFY_STATS_CMD+=(--pv-input-pipeline)
fi
if (( PV_RESTORE_PIPELINE )); then
  VERIFY_STATS_CMD+=(--pv-restore-pipeline)
fi
if (( PV_OUTPUT_PIPELINE )); then
  VERIFY_STATS_CMD+=(--pv-output-pipeline)
fi
if (( PV_EARLY_COMPUTE )); then
  VERIFY_STATS_CMD+=(--pv-early-compute)
fi
if (( PV_MATRIX_SOFTMAX_OVERLAP )); then
  VERIFY_STATS_CMD+=(--pv-matrix-softmax-overlap)
fi

if (( DRY_RUN )); then
  printf '%q ' "${GENERATE_CMD[@]}"; printf '\n'
  printf '%q ' "${RUN_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_STATS_CMD[@]}"; printf '\n'
  if (( MPI_RANKS > 1 )); then
    printf '%q ' "${VERIFY_MPI_CMD[@]}"; printf '\n'
  fi
  exit 0
fi

mkdir -p "$ARTIFACT_ROOT"
rm -f "$RESULT_JSON" "$LIFECYCLE_JSON" "$MPI_PARTITION_JSON" "$MPI_PLACEMENT_JSON" "$STATS_FILE"
for rank in 0 1 2 3; do
  rm -f "${STATS_FILE%.txt}_${rank}.txt"
done
"${GENERATE_CMD[@]}"
"${RUN_CMD[@]}"
"${VERIFY_CMD[@]}"
"${VERIFY_STATS_CMD[@]}"
if (( MPI_RANKS > 1 )); then
  "${VERIFY_MPI_CMD[@]}"
fi

echo "Fused Attention ${SCALE_POINT^^} PASS: B1,H1,S${TOTAL_QUERIES},D${HEAD_DIM},4 managers,16 workers,${MPI_RANKS} MPI rank(s),1 tensor completion"
echo "Artifacts: $ARTIFACT_ROOT"
