#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_RUNNER="$TESTS_DIR/run_noc_dma_pipeline.sh"
ARTIFACT_ROOT="/data4/jjgong/tmp/fused_attention_c1_s32_d64"
TIMEOUT_SECONDS=600
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: run_fused_attention.sh [--artifact-root DIR] [--timeout SEC] [--dry-run]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

Q_FILE="$ARTIFACT_ROOT/q_32x64.bin"
K_FILE="$ARTIFACT_ROOT/k_32x64.bin"
V_FILE="$ARTIFACT_ROOT/v_32x64.bin"
RESULT_JSON="$ARTIFACT_ROOT/fused_attention_result.json"
GUEST="$SCRIPT_DIR/riscv64/fused_attention_c1"
HBM_OUT="$ARTIFACT_ROOT/hbm/hbm_out_node1.bin"
STATS_FILE="$ARTIFACT_ROOT/stats/overlap0/fused_attention_c1_s32_d64/stats_selfcom.txt"
MEM_NODE_SIZE=134217728
Q_OFFSET=$((0x02000000))
K_OFFSET=$((0x02010000))
V_OFFSET=$((0x02020000))
O_OFFSET=$((0x02030000))

GENERATE_CMD=(python3 "$SCRIPT_DIR/attention_case.py" generate
  --queries 32 --keys 32 --head-dim 64
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE")

RUN_CMD=(timeout "$TIMEOUT_SECONDS" env
  "GOLEM_RUN_ID=fused_attention_c1_s32_d64"
  "GOLEM_ARTIFACT_ROOT=$ARTIFACT_ROOT"
  "VANADIS_EXE=$GUEST"
  GOLEM_SKIP_DEFAULT_GUEST_BUILD=1
  GOLEM_ARCH_SCRIPT=architecture/archive/ncores_selfcom_dma.py
  GOLEM_ATTENTION_FUSED=1
  "GOLEM_ATTENTION_Q_FILE=$Q_FILE"
  "GOLEM_ATTENTION_K_FILE=$K_FILE"
  "GOLEM_ATTENTION_V_FILE=$V_FILE"
  "GOLEM_ATTENTION_Q_OFFSET=$Q_OFFSET"
  "GOLEM_ATTENTION_K_OFFSET=$K_OFFSET"
  "GOLEM_ATTENTION_V_OFFSET=$V_OFFSET"
  GOLEM_ATTENTION_WINDOW_OFFSET=0xC0000
  GOLEM_ATTENTION_WINDOW_BYTES=0x10000
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
  --gemm-m 32 --gemm-n 32 --gemm-k 64
  --orig-m 32 --orig-n 32 --orig-k 64
  --gemm-block-m 16 --gemm-block-n 16 --gemm-block-k 64
  --array-in 64 --array-out 16 --num-arrays 16
  --groups 1 --num-cores 2 --gemm-cores 2 --num-mem-nodes 2 --mesh-dim-x 2
  --global-stride-kb 1024 --mem-node-size "$MEM_NODE_SIZE")

VERIFY_CMD=(python3 "$SCRIPT_DIR/attention_case.py" verify-attention
  --queries 32 --keys 32 --head-dim 64
  --q-file "$Q_FILE" --k-file "$K_FILE" --v-file "$V_FILE"
  --output-file "$HBM_OUT" --output-offset "$O_OFFSET"
  --causal 0 --fused --result-json "$RESULT_JSON")
VERIFY_STATS_CMD=(python3 "$SCRIPT_DIR/verify_fused_attention_stats.py" "$STATS_FILE")

if (( DRY_RUN )); then
  printf '%q ' "${GENERATE_CMD[@]}"; printf '\n'
  echo "make -C $SCRIPT_DIR fused"
  printf '%q ' "${RUN_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_CMD[@]}"; printf '\n'
  printf '%q ' "${VERIFY_STATS_CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$ARTIFACT_ROOT"
"${GENERATE_CMD[@]}"
make -C "$SCRIPT_DIR" fused
"${RUN_CMD[@]}"
"${VERIFY_CMD[@]}"
"${VERIFY_STATS_CMD[@]}"

echo "Fused Attention C1 PASS: B1,H1,S32,D64,Br16,Bc32,non-causal"
echo "Artifacts: $ARTIFACT_ROOT"
