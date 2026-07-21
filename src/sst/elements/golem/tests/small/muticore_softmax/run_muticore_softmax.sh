#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$TESTS_DIR/../../../../.." && pwd)"
WRAPPER="$TESTS_DIR/small/mvm_noc_softmax_sfu/run_noc_dma_softmax_sfu_pipeline.sh"

rows=1024
cols=4096
timeout_seconds=1800
artifact_root=""
dry_run=0

usage() {
	printf 'Usage: %s [--rows N] [--cols N] [--timeout SEC] [--artifact-root DIR] [--dry-run]\n' "$0"
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--rows) rows="$2"; shift 2 ;;
		--cols) cols="$2"; shift 2 ;;
		--timeout) timeout_seconds="$2"; shift 2 ;;
		--artifact-root) artifact_root="$2"; shift 2 ;;
		--dry-run) dry_run=1; shift ;;
		-h|--help) usage; exit 0 ;;
		*) echo "[ERROR] unknown option: $1" >&2; usage >&2; exit 2 ;;
	esac
done

for value in "$rows" "$cols" "$timeout_seconds"; do
	if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
		echo "[ERROR] rows, cols, and timeout must be positive integers" >&2
		exit 2
	fi
done
if (( rows % 16 != 0 )); then
	echo "[ERROR] the first Row Engine profile requires rows divisible by 16" >&2
	exit 2
fi
if (( cols % 64 != 0 || cols > 4096 )); then
	echo "[ERROR] the row-local profile requires cols divisible by 64 and <= 4096" >&2
	exit 2
fi

rows_per_tile=$(( rows / 16 ))
block_m=1
if (( rows_per_tile >= 4 && rows_per_tile % 4 == 0 )); then
	block_m=4
fi
worker_staging_bytes=$(( rows_per_tile * cols * 8 ))
tensor_scratch_rows=$rows
if (( tensor_scratch_rows > 64 )); then
	tensor_scratch_rows=64
fi
tensor_scratch_bytes=$(( tensor_scratch_rows * cols * 4 ))
required_staging_bytes=$worker_staging_bytes
if (( tensor_scratch_bytes > required_staging_bytes )); then
	required_staging_bytes=$tensor_scratch_bytes
fi
global_stride_kb=$(( (required_staging_bytes + 262144 + 262143) / 262144 * 256 ))
if (( global_stride_kb < 512 )); then
	global_stride_kb=512
fi

run_id="muticore_softmax_r${rows}_d${cols}_row_engine"
if [[ -z "$artifact_root" ]]; then
	artifact_root="$TESTS_DIR/artifacts/muticore_softmax/$run_id"
fi
attempt_id="${run_id}_$(date +%Y%m%d_%H%M%S)_$$"
attempt_stats="$artifact_root/stats/$attempt_id"
attempt_stdout="$artifact_root/stdout/$attempt_id"
mkdir -p "$artifact_root/inputs" "$artifact_root/outputs" "$attempt_stats" "$attempt_stdout"

export TMPDIR="${TMPDIR:-/data4/jjgong/tmp}"
export GOLEM_RUN_ID="$attempt_id"
export GOLEM_ARTIFACT_ROOT="$artifact_root"
export GOLEM_HBM_DIR="$artifact_root/hbm"
export GOLEM_RUN_SUMMARY_CSV="$artifact_root/stats/run_summary.csv"
export GOLEM_SFU_GUEST_SNAPSHOT="$artifact_root/guest/$attempt_id/test_noc_dma_softmax_sfu"
export GOLEM_STATS_DIR="$attempt_stats"
export GOLEM_STATS_FILE="$attempt_stats/stats_selfcom.txt"
export GOLEM_STDOUT_DIR="$attempt_stdout"
export GOLEM_TENSOR_DIR="$artifact_root/inputs"
export GOLEM_TENSOR_SOURCE=sample
export GOLEM_HBM_DUMP_OUTPUT=1
export GOLEM_SFU_ENABLE=1
export GOLEM_SFU_STANDALONE_SOFTMAX=1
export GOLEM_SFU_JOB_SOFTMAX=1
export GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM=1
export GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS=0
export GOLEM_SFU_JOB_SOFTMAX_ROW_ENGINE=1
export GOLEM_SFU_JOB_SOFTMAX_TENSOR_CONTROLLER="${GOLEM_SFU_JOB_SOFTMAX_TENSOR_CONTROLLER:-1}"
export GOLEM_SFU_SOFTMAX_HBM_LAYOUT=band_striped
export GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES=1
export GOLEM_SFU_JOB_SOFTMAX_BAND_CORES=16
export GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS="$rows_per_tile"
export GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS="$rows_per_tile"
export GOLEM_SFU_JOB_SOFTMAX_CHUNK_ELEMS=256
export GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc
export GOLEM_SFU_REDUCTION_VN=0
export GOLEM_DMA_RESPONSE_VN=0
export GOLEM_DMA_READ_RETRY_TICKS=65536
export GOLEM_DMA_READ_MAX_RETRIES=8
export GOLEM_DMA_BURST_BYTES="${GOLEM_DMA_BURST_BYTES:-262144}"
export GOLEM_A_REUSE_N_TILES=1
export GOLEM_B_REUSE_M_TILES=1
export GOLEM_SST_ENABLE_ALL_STATS=1
export GOLEM_SST_STAT_LOAD_LEVEL=1
export GOLEM_BENCH_DISABLE_SST_STATS=0
export GOLEM_BENCH_QUIET_LOGS=0
noc_link_bw="${GOLEM_SOFTMAX_NOC_LINK_BW:-1200GB/s}"
noc_xbar_bw="${GOLEM_SOFTMAX_NOC_XBAR_BW:-$noc_link_bw}"
export GOLEM_SOFTMAX_NOC_LINK_BW="$noc_link_bw"
export GOLEM_SOFTMAX_NOC_XBAR_BW="$noc_xbar_bw"
export GOLEM_DIRCTRL_HIGHLINK_BW="${GOLEM_SOFTMAX_DIRCTRL_HIGHLINK_BW:-$noc_link_bw}"
export VANADIS_CPU_CLOCK=2.3GHz
export GOLEM_ARRAY_CLOCK=2.3GHz
export GOLEM_MEMCTRL_CLOCK=2.3GHz

printf '[ROW-ENGINE] rows=%d cols=%d rows_per_tile=%d global_stride_kb=%d artifact_root=%s\n' \
	"$rows" "$cols" "$rows_per_tile" "$global_stride_kb" "$artifact_root"
for name in \
	GOLEM_SFU_JOB_SOFTMAX_ROW_ENGINE \
	GOLEM_SFU_JOB_SOFTMAX_TENSOR_CONTROLLER \
	GOLEM_SFU_SOFTMAX_HBM_LAYOUT \
	GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS \
	GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES \
	GOLEM_SFU_JOB_SOFTMAX_BAND_CORES \
	GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS \
	GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS \
	GOLEM_DMA_BURST_BYTES \
	GOLEM_SOFTMAX_NOC_LINK_BW \
	VANADIS_CPU_CLOCK; do
	printf '[ROW-ENGINE] %s=%s\n' "$name" "${!name}"
done

wrapper_args=(
	--groups 4 --array-in 64 --array-out "$block_m" --num-arrays 64
	--num-cores 16 --gemm-cores 16 --num-mem-nodes 5 --mesh-dim-x 4
	--mem-node-size 134217728 --global-stride-kb "$global_stride_kb"
	--gemm-m "$rows" --gemm-n "$cols" --gemm-k "$cols"
	--gemm-block-m "$block_m" --gemm-block-n 64 --gemm-block-k 64
	--dtype fp32 --group-manager-enable 0 --ctrl-link-enable 0
	--noc-in-buf 512KB --noc-out-buf 512KB
	--noc-link-bw "$noc_link_bw" --noc-xbar-bw "$noc_xbar_bw" --noc-flit-size 128B
	--gm-buf 1024KB --verify-softmax --softmax-reference logits
	--softmax-logits-file "$artifact_root/inputs/softmax_logits_${rows}x${cols}.bin"
	--softmax-c-file "$artifact_root/outputs/softmax.bin"
)

if [[ "$dry_run" -eq 1 ]]; then
	printf '[ROW-ENGINE] command timeout %s bash %s' "$timeout_seconds" "$WRAPPER"
	printf ' %q' "${wrapper_args[@]}"
	printf ' --dry-run\n'
	bash "$WRAPPER" "${wrapper_args[@]}" --dry-run
	exit 0
fi

golem_library="$REPO_ROOT/build/sst-elements/src/sst/elements/golem/.libs/libgolem.so"
if [[ ! -f "$golem_library" ]]; then
	echo "[ERROR] missing production library: $golem_library" >&2
	exit 1
fi
library_sha_before="$(sha256sum "$golem_library" | awk '{print $1}')"

timeout "$timeout_seconds" bash "$WRAPPER" "${wrapper_args[@]}"

library_sha_after="$(sha256sum "$golem_library" | awk '{print $1}')"
if [[ "$library_sha_before" != "$library_sha_after" ]]; then
	echo "[ERROR] production library changed during SST run" >&2
	exit 1
fi

signature_file="$attempt_stats/run_signature.sha256"
sha256sum \
	"$GOLEM_SFU_GUEST_SNAPSHOT" \
	"$golem_library" \
	"$SCRIPT_DIR/run_muticore_softmax.sh" \
	"$SCRIPT_DIR/parse_muticore_softmax.py" \
	"$artifact_root/inputs/softmax_logits_${rows}x${cols}.bin" \
	"$artifact_root/outputs/softmax.bin" > "$signature_file"

python3 "$SCRIPT_DIR/parse_muticore_softmax.py" \
	--stats "$GOLEM_STATS_FILE" \
	--stdout-dir "$GOLEM_STDOUT_DIR" \
	--rows "$rows" --cols "$cols" \
	--attempt-id "$attempt_id" \
	--output "$artifact_root/row_engine_result.json" \
	$(if [[ "$GOLEM_SFU_JOB_SOFTMAX_TENSOR_CONTROLLER" -eq 1 ]]; then printf '%s' '--tensor-controller'; fi) \
	--require-contract
