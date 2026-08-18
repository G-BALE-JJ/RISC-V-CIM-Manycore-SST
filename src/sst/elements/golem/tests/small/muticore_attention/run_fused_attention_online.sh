#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_RUNNER="$TESTS_DIR/run_noc_dma_pipeline.sh"
ARTIFACT_ROOT=""
TIMEOUT_SECONDS=600
DRY_RUN=0
CAUSAL=0
PARTIAL=0
EXTREME_LOGITS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --causal) CAUSAL="$2"; shift 2 ;;
    --partial) PARTIAL=1; shift ;;
    --extreme-logits) EXTREME_LOGITS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: run_fused_attention_online.sh [--causal 0|1] [--partial] [--extreme-logits] [--artifact-root DIR] [--timeout SEC] [--dry-run]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$CAUSAL" != 0 && "$CAUSAL" != 1 ]]; then
  echo "--causal must be 0 or 1" >&2
  exit 2
fi
if (( (PARTIAL && CAUSAL) || (EXTREME_LOGITS && (CAUSAL || PARTIAL)) )); then
  echo "--partial and --extreme-logits are separate non-causal acceptance modes" >&2
  exit 2
fi
QUERIES=64
KEYS=64
PHASE="d$((CAUSAL + 1))"
MAKE_TARGET=online
GUEST="$SCRIPT_DIR/riscv64/fused_attention_d1"
if (( CAUSAL )); then
  MAKE_TARGET=causal
  GUEST="$SCRIPT_DIR/riscv64/fused_attention_d2_causal"
elif (( PARTIAL )); then
  QUERIES=20
  KEYS=70
  PHASE=d3
  MAKE_TARGET=partial
  GUEST="$SCRIPT_DIR/riscv64/fused_attention_d3_partial"
elif (( EXTREME_LOGITS )); then
  PHASE=d4
fi
if [[ -z "$ARTIFACT_ROOT" ]]; then
  ARTIFACT_ROOT="/data4/jjgong/tmp/fused_attention_${PHASE}_q${QUERIES}_k${KEYS}_d64"
fi
RUN_ID="fused_attention_${PHASE}_q${QUERIES}_k${KEYS}_d64"

Q_FILE="$ARTIFACT_ROOT/q_${QUERIES}x64.bin"
K_FILE="$ARTIFACT_ROOT/k_${KEYS}x64.bin"
V_FILE="$ARTIFACT_ROOT/v_${KEYS}x64.bin"
RESULT_JSON="$ARTIFACT_ROOT/fused_attention_result.json"
HBM_OUT="$ARTIFACT_ROOT/hbm/hbm_out_node1.bin"
STATS_FILE="$ARTIFACT_ROOT/stats/overlap0/$RUN_ID/stats_selfcom.txt"
MEM_NODE_SIZE=134217728
Q_OFFSET=$((0x02000000))
K_OFFSET=$((0x02010000))
V_OFFSET=$((0x02020000))
O_OFFSET=$((0x02030000))

GENERATE_CMD=(python3 "$SCRIPT_DIR/attention_case.py" generate
  --queries "$QUERIES" --keys "$KEYS" --head-dim 64
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE")
if (( EXTREME_LOGITS )); then GENERATE_CMD+=(--extreme-logits); fi
TENSOR_ARGS=(--tensor-source file --tensor-a "$Q_FILE" --tensor-b "$K_FILE")
if (( PARTIAL )); then
  # The disabled legacy GEMM path still needs a self-consistent placeholder tensor shape.
  TENSOR_ARGS=(--tensor-source sample)
fi

RUN_CMD=(timeout "$TIMEOUT_SECONDS" env
  "GOLEM_RUN_ID=$RUN_ID"
  "GOLEM_ARTIFACT_ROOT=$ARTIFACT_ROOT"
  "VANADIS_EXE=$GUEST"
  GOLEM_SKIP_DEFAULT_GUEST_BUILD=1
  GOLEM_ARCH_SCRIPT=architecture/archive/ncores_selfcom_dma.py
  GOLEM_ATTENTION_FUSED=1
  "GOLEM_ATTENTION_QUERIES=$QUERIES"
  "GOLEM_ATTENTION_KEYS=$KEYS"
  "GOLEM_ATTENTION_Q_FILE=$Q_FILE"
  "GOLEM_ATTENTION_K_FILE=$K_FILE"
  "GOLEM_ATTENTION_V_FILE=$V_FILE"
  "GOLEM_ATTENTION_Q_OFFSET=$Q_OFFSET"
  "GOLEM_ATTENTION_K_OFFSET=$K_OFFSET"
  "GOLEM_ATTENTION_V_OFFSET=$V_OFFSET"
  GOLEM_ATTENTION_WINDOW_OFFSET=0xC0000
  GOLEM_ATTENTION_WINDOW_BYTES=0x10000
  GOLEM_SFU_ROW_CONTEXTS=16
  GOLEM_GROUP_MANAGER_ENABLE=1
  GOLEM_SFU_MANAGER_COORDINATOR=1
  GOLEM_CTRL_LINK_ENABLE=0
  GOLEM_REQUEST_SCHEDULER_ENABLE=0
  GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0
  GOLEM_SFU_ENABLE=1
  GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc
  bash "$BASE_RUNNER"
  --dtype fp32 "${TENSOR_ARGS[@]}"
  --transpose-b 1 --hbm-dump-output 1
  --gemm-m 64 --gemm-n 64 --gemm-k 64
  --orig-m 64 --orig-n 64 --orig-k 64
  --gemm-block-m 16 --gemm-block-n 16 --gemm-block-k 64
  --array-in 64 --array-out 16 --num-arrays 16
  --groups 1 --num-cores 2 --gemm-cores 2 --num-mem-nodes 2 --mesh-dim-x 2
  --global-stride-kb 1024 --mem-node-size "$MEM_NODE_SIZE")

VERIFY_CMD=(python3 "$SCRIPT_DIR/attention_case.py" verify-attention
  --queries "$QUERIES" --keys "$KEYS" --head-dim 64
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE"
  --output-file "$HBM_OUT" --output-offset "$O_OFFSET"
  --causal "$CAUSAL" --fused --result-json "$RESULT_JSON")
if (( EXTREME_LOGITS )); then VERIFY_CMD+=(--extreme-logits); fi
VERIFY_STATS_CMD=(python3 "$SCRIPT_DIR/verify_fused_attention_online_stats.py" "$STATS_FILE")
if (( CAUSAL )); then VERIFY_STATS_CMD+=(--causal); fi
if (( PARTIAL )); then VERIFY_STATS_CMD+=(--partial); fi

if (( DRY_RUN )); then
  printf '%q ' "${GENERATE_CMD[@]}"; printf '\n'
  echo "make -C $SCRIPT_DIR $MAKE_TARGET"
  printf '%q ' "${RUN_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_STATS_CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$ARTIFACT_ROOT"
"${GENERATE_CMD[@]}"
make -C "$SCRIPT_DIR" "$MAKE_TARGET"
"${RUN_CMD[@]}"
"${VERIFY_CMD[@]}"
"${VERIFY_STATS_CMD[@]}"

echo "Fused Attention ${PHASE^^} PASS: B1,H1,Sq${QUERIES},Skv${KEYS},D64,Br16,Bc32,causal=$CAUSAL"
echo "Artifacts: $ARTIFACT_ROOT"
