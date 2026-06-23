#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOFTMAX_BIN="$SCRIPT_DIR/riscv64/test_noc_dma_softmax"
SOFTMAX_BUILD_ENV="$SCRIPT_DIR/riscv64/test_noc_dma_softmax.build.env"
BASELINE_BIN="$TESTS_DIR/small/mvm_noc_int_array/riscv64/test_noc_dma"
BASELINE_BUILD_ENV="$TESTS_DIR/small/mvm_noc_int_array/riscv64/test_noc_dma.build.env"
RISCV_MUSL_TOOLCHAIN_BIN="${RISCV_MUSL_TOOLCHAIN_BIN:-/data/lzq/packages/install/riscv64_musl_toolchain/bin}"
SST_CORE_HOME="${SST_CORE_HOME:-/data4/jjgong/local/sstcore}"
SST_ELEMENTS_HOME="${SST_ELEMENTS_HOME:-/data4/jjgong/local/sstelements}"
SST_LIB_PATH="${SST_LIB_PATH:-/data4/lishun/pkg/sst_install/lib/sst-elements-library}"
CONDA_LIB_DIR="${CONDA_LIB_DIR:-/data4/jjgong/miniconda3/lib}"
export SST_CORE_HOME
export SST_ELEMENTS_HOME
export SST_LIB_PATH
export REAL_SST_BIN="${REAL_SST_BIN:-$SST_CORE_HOME/bin/sst}"
export SST_SOFTMAX_LD_LIBRARY_PATH="$CONDA_LIB_DIR:$SST_CORE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$SCRIPT_DIR/bin:$RISCV_MUSL_TOOLCHAIN_BIN:$SST_CORE_HOME/bin:$SST_ELEMENTS_HOME/bin:$PATH"

HAS_DRY_RUN=0
USER_SET_GOLEM_SKIP_BUILD=0
if [[ -n "${GOLEM_SKIP_BUILD+x}" ]]; then
	USER_SET_GOLEM_SKIP_BUILD=1
fi
USER_SET_GOLEM_VERIFY_C=0
if [[ -n "${GOLEM_VERIFY_C+x}" ]]; then
	USER_SET_GOLEM_VERIFY_C=1
fi

if [[ "$USER_SET_GOLEM_VERIFY_C" -eq 0 ]]; then
	export GOLEM_VERIFY_C=0
fi
GOLEM_VERIFY_SOFTMAX="${GOLEM_VERIFY_SOFTMAX:-0}"
GOLEM_SOFTMAX_C_FILE="${GOLEM_SOFTMAX_C_FILE:-}"
GOLEM_SOFTMAX_VERIFY_REFERENCE="${GOLEM_SOFTMAX_VERIFY_REFERENCE:-probability}"
SOFTMAX_TENSOR_DIR="${GOLEM_TENSOR_DIR:-$SCRIPT_DIR/data}"
SOFTMAX_TENSOR_A_FILE="${GOLEM_TENSOR_A_FILE:-$SOFTMAX_TENSOR_DIR/a.bin}"
SOFTMAX_TENSOR_B_FILE="${GOLEM_TENSOR_B_FILE:-$SOFTMAX_TENSOR_DIR/b.bin}"

load_default_preset() {
	local preset="$TESTS_DIR/configs/default.env"
	if [[ -f "$preset" ]]; then
		# shellcheck source=/dev/null
		source "$preset"
	fi
}

align_up_int() {
	local value="$1"
	local align="$2"
	echo $(( ((value + align - 1) / align) * align ))
}

metadata_get_value() {
	local file="$1"
	local key="$2"
	awk -F= -v k="$key" '$1 == k { print substr($0, index($0, "=") + 1) }' "$file" | tail -n 1
}

write_metadata_file() {
	local file="$1"
	shift
	mkdir -p "$(dirname "$file")"
	: > "$file"
	for key in "$@"; do
		printf '%s=%s\n' "$key" "${!key}" >> "$file"
	done
}

metadata_matches() {
	local file="$1"
	shift
	if [[ ! -f "$file" ]]; then
		return 1
	fi
	for key in "$@"; do
		if [[ "$(metadata_get_value "$file" "$key")" != "${!key}" ]]; then
			return 1
		fi
	done
	return 0
}

softmax_binary_is_fresh() {
	if [[ ! -x "$SOFTMAX_BIN" ]]; then
		return 1
	fi
	local src
	for src in \
		"$SCRIPT_DIR/test_noc_dma_softmax.cpp" \
		"$SCRIPT_DIR/golem_softmax_runtime.cpp" \
		"$SCRIPT_DIR/golem_softmax_runtime.h" \
		"$SCRIPT_DIR/softmax_config.h" \
		"$SCRIPT_DIR/Makefile" \
		"$BASE_DIR/golem_matmul_runtime.cpp" \
		"$BASE_DIR/golem_matmul_runtime.h" \
		"$BASE_DIR/pipeline_config.h" \
		"$BASE_DIR/operators.h"; do
		if [[ "$src" -nt "$SOFTMAX_BIN" ]]; then
			return 1
		fi
	done
	return 0
}

load_default_preset

GOLEM_TOTAL_GROUPS="${GOLEM_TOTAL_GROUPS:-4}"
GOLEM_ARRAY_INPUT_SIZE="${GOLEM_ARRAY_INPUT_SIZE:-4}"
GOLEM_ARRAY_OUTPUT_SIZE="${GOLEM_ARRAY_OUTPUT_SIZE:-4}"
GOLEM_NUM_ARRAYS="${GOLEM_NUM_ARRAYS:-1}"
GOLEM_TOTAL_CORES="${GOLEM_TOTAL_CORES:-${VANADIS_NUM_CORES:-16}}"
GOLEM_TOTAL_GEMM_CORES="${GOLEM_TOTAL_GEMM_CORES:-16}"
GOLEM_NUM_MEMORY_NODES="${GOLEM_NUM_MEMORY_NODES:-5}"
GOLEM_MEM_NODE_SIZE_BYTES="${GOLEM_MEM_NODE_SIZE_BYTES:-67108864}"
GOLEM_GLOBAL_STRIDE_KB="${GOLEM_GLOBAL_STRIDE_KB:-64}"
GOLEM_DMA_STAGGER_CYCLES="${GOLEM_DMA_STAGGER_CYCLES:-0}"
GOLEM_DMA_OVERLAP="${GOLEM_DMA_OVERLAP:-0}"
GOLEM_CTRL_OVERLAP_AB="${GOLEM_CTRL_OVERLAP_AB:-1}"
GOLEM_GROUP_MANAGER_ENABLE="${GOLEM_GROUP_MANAGER_ENABLE:-1}"
GOLEM_CTRL_LINK_ENABLE="${GOLEM_CTRL_LINK_ENABLE:-0}"
GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE="${GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE:-0}"
GOLEM_A_REUSE_N_TILES="${GOLEM_A_REUSE_N_TILES:-1}"
GOLEM_B_REUSE_M_TILES="${GOLEM_B_REUSE_M_TILES:-1}"
GOLEM_DMA_SLOT_COUNT="${GOLEM_DMA_SLOT_COUNT:-4}"
GOLEM_GEMM_M="${GOLEM_GEMM_M:-$GOLEM_ARRAY_OUTPUT_SIZE}"
GOLEM_GEMM_N="${GOLEM_GEMM_N:-$GOLEM_NUM_ARRAYS}"
GOLEM_GEMM_K="${GOLEM_GEMM_K:-$GOLEM_ARRAY_INPUT_SIZE}"
GOLEM_GEMM_BLOCK_M="${GOLEM_GEMM_BLOCK_M:-$GOLEM_ARRAY_OUTPUT_SIZE}"
GOLEM_GEMM_BLOCK_N="${GOLEM_GEMM_BLOCK_N:-$GOLEM_NUM_ARRAYS}"
GOLEM_GEMM_BLOCK_K="${GOLEM_GEMM_BLOCK_K:-$GOLEM_ARRAY_INPUT_SIZE}"
GOLEM_BIAS_ENABLE="${GOLEM_BIAS_ENABLE:-0}"
GOLEM_BIAS_VALUE="${GOLEM_BIAS_VALUE:-0}"

args=("$@")
PIPELINE_ARGS=()
i=0
while [[ "$i" -lt "${#args[@]}" ]]; do
	case "${args[$i]}" in
		--dry-run)
			HAS_DRY_RUN=1
			PIPELINE_ARGS+=("${args[$i]}")
			i=$((i + 1)) ;;
		--groups)
			GOLEM_TOTAL_GROUPS="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--array-in)
			GOLEM_ARRAY_INPUT_SIZE="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--array-out)
			GOLEM_ARRAY_OUTPUT_SIZE="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--num-arrays)
			GOLEM_NUM_ARRAYS="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--gemm-cores)
			GOLEM_TOTAL_GEMM_CORES="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--num-cores)
			GOLEM_TOTAL_CORES="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--num-mem-nodes)
			GOLEM_NUM_MEMORY_NODES="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--mem-node-size)
			GOLEM_MEM_NODE_SIZE_BYTES="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--global-stride-kb)
			GOLEM_GLOBAL_STRIDE_KB="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--gemm-m)
			GOLEM_GEMM_M="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--gemm-n)
			GOLEM_GEMM_N="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--gemm-k)
			GOLEM_GEMM_K="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--gemm-block-m)
			GOLEM_GEMM_BLOCK_M="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--gemm-block-n)
			GOLEM_GEMM_BLOCK_N="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--gemm-block-k)
			GOLEM_GEMM_BLOCK_K="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--dma-stagger-cycles)
			GOLEM_DMA_STAGGER_CYCLES="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--dma-overlap)
			GOLEM_DMA_OVERLAP="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--ctrl-overlap-ab)
			GOLEM_CTRL_OVERLAP_AB="${args[$((i + 1))]}"; i=$((i + 2)) ;;
		--group-manager-enable)
			GOLEM_GROUP_MANAGER_ENABLE="${args[$((i + 1))]}"; i=$((i + 2)) ;;
		--ctrl-link-enable)
			GOLEM_CTRL_LINK_ENABLE="${args[$((i + 1))]}"; i=$((i + 2)) ;;
		--verify-softmax)
			GOLEM_VERIFY_SOFTMAX=1; i=$((i + 1)) ;;
		--softmax-c-file)
			GOLEM_SOFTMAX_C_FILE="${args[$((i + 1))]}"; i=$((i + 2)) ;;
		--softmax-reference)
			GOLEM_SOFTMAX_VERIFY_REFERENCE="${args[$((i + 1))]}"; i=$((i + 2)) ;;
		--tensor-dir)
			SOFTMAX_TENSOR_DIR="${args[$((i + 1))]}"
			SOFTMAX_TENSOR_A_FILE="$SOFTMAX_TENSOR_DIR/a.bin"
			SOFTMAX_TENSOR_B_FILE="$SOFTMAX_TENSOR_DIR/b.bin"
			PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}")
			i=$((i + 2)) ;;
		--tensor-a)
			SOFTMAX_TENSOR_A_FILE="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--tensor-b)
			SOFTMAX_TENSOR_B_FILE="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--bias-enable)
			GOLEM_BIAS_ENABLE="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--bias-value)
			GOLEM_BIAS_VALUE="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		*)
			if [[ "${args[$i]}" == --* && "$((i + 1))" -lt "${#args[@]}" && "${args[$((i + 1))]}" != --* ]]; then
				PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}")
				i=$((i + 2))
			else
				PIPELINE_ARGS+=("${args[$i]}")
				i=$((i + 1))
			fi ;;
	esac
done

GOLEM_GLOBAL_STRIDE_BYTES=$(( GOLEM_GLOBAL_STRIDE_KB * 1024 ))
mat_slot_bytes=$(align_up_int $(( GOLEM_GEMM_BLOCK_M * GOLEM_GEMM_BLOCK_K * 4 )) 256)
vec_slot_bytes=$(align_up_int $(( GOLEM_GEMM_BLOCK_N * GOLEM_GEMM_BLOCK_K * 4 )) 256)
out_scratch_bytes=$(align_up_int $(( GOLEM_ARRAY_OUTPUT_SIZE * 4 )) 256)
out_tile_bytes=$(align_up_int $(( GOLEM_GEMM_BLOCK_M * GOLEM_GEMM_BLOCK_N * 4 )) 256)
required_global_stride_bytes=$(( 0x2000 + GOLEM_DMA_SLOT_COUNT * mat_slot_bytes + GOLEM_DMA_SLOT_COUNT * vec_slot_bytes + out_scratch_bytes + out_tile_bytes + 0x40 + 256 ))
if (( GOLEM_GLOBAL_STRIDE_BYTES < required_global_stride_bytes )); then
	GOLEM_GLOBAL_STRIDE_BYTES=$required_global_stride_bytes
	GOLEM_GLOBAL_STRIDE_KB=$(( (GOLEM_GLOBAL_STRIDE_BYTES + 1023) / 1024 ))
	GOLEM_GLOBAL_STRIDE_BYTES=$(( GOLEM_GLOBAL_STRIDE_KB * 1024 ))
fi

BUILD_KEYS=(
	GOLEM_ARRAY_INPUT_SIZE
	GOLEM_ARRAY_OUTPUT_SIZE
	GOLEM_TOTAL_GROUPS
	GOLEM_TOTAL_CORES
	GOLEM_TOTAL_GEMM_CORES
	GOLEM_NUM_ARRAYS
	GOLEM_NUM_MEMORY_NODES
	GOLEM_MEM_NODE_SIZE_BYTES
	GOLEM_GLOBAL_STRIDE_BYTES
	GOLEM_GEMM_M
	GOLEM_GEMM_N
	GOLEM_GEMM_K
	GOLEM_GEMM_BLOCK_M
	GOLEM_GEMM_BLOCK_N
	GOLEM_GEMM_BLOCK_K
	GOLEM_DMA_STAGGER_CYCLES
	GOLEM_DMA_OVERLAP
	GOLEM_CTRL_OVERLAP_AB
	GOLEM_GROUP_MANAGER_ENABLE
	GOLEM_CTRL_LINK_ENABLE
	GOLEM_A_REUSE_N_TILES
	GOLEM_B_REUSE_M_TILES
	GOLEM_DMA_SLOT_COUNT
	GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE
	GOLEM_BIAS_ENABLE
	GOLEM_BIAS_VALUE
)

SOFTMAX_CFLAGS=""
for key in "${BUILD_KEYS[@]}"; do
	SOFTMAX_CFLAGS+=" -D${key}=${!key}"
done

BASE_DIR="$SCRIPT_DIR/../mvm_noc_int_array"
pushd "$SCRIPT_DIR" >/dev/null
if metadata_matches "$SOFTMAX_BUILD_ENV" "${BUILD_KEYS[@]}" && softmax_binary_is_fresh; then
	echo "[SOFTMAX] Reusing existing softmax binary: riscv64/test_noc_dma_softmax"
else
	make clean ARCH=riscv64
	make ARCH=riscv64 CFLAGS="$SOFTMAX_CFLAGS"
	write_metadata_file "$SOFTMAX_BUILD_ENV" "${BUILD_KEYS[@]}"
fi
popd >/dev/null

export VANADIS_EXE="$SOFTMAX_BIN"
if [[ "$USER_SET_GOLEM_SKIP_BUILD" -eq 0 ]]; then
	if metadata_matches "$BASELINE_BUILD_ENV" "${BUILD_KEYS[@]}" && [[ -x "$BASELINE_BIN" ]]; then
		export GOLEM_SKIP_BUILD=1
	else
		export GOLEM_SKIP_BUILD=0
	fi
else
	export GOLEM_SKIP_BUILD
fi
export GOLEM_MATMUL_DTYPE="${GOLEM_MATMUL_DTYPE:-fp32}"
export GOLEM_VERIFY_C
export GOLEM_VERIFY_SOFTMAX
export GOLEM_GROUP_MANAGER_ENABLE
export GOLEM_CTRL_LINK_ENABLE
export GOLEM_CTRL_OVERLAP_AB

echo "[SOFTMAX] VANADIS_EXE=$VANADIS_EXE"
echo "[SOFTMAX] GOLEM_SKIP_BUILD=$GOLEM_SKIP_BUILD"
echo "[SOFTMAX] GOLEM_VERIFY_C=$GOLEM_VERIFY_C"
echo "[SOFTMAX] GOLEM_VERIFY_SOFTMAX=$GOLEM_VERIFY_SOFTMAX"
echo "[SOFTMAX] GOLEM_SOFTMAX_VERIFY_REFERENCE=$GOLEM_SOFTMAX_VERIFY_REFERENCE"
echo "[SOFTMAX] GOLEM_GROUP_MANAGER_ENABLE=$GOLEM_GROUP_MANAGER_ENABLE"

if [[ "$GOLEM_SKIP_BUILD" -eq 1 ]]; then
	if [[ ! -x "$BASELINE_BIN" || ! -f "$BASELINE_BUILD_ENV" ]]; then
		cat >&2 <<EOF
[SOFTMAX][ERROR] 原 pipeline 的 GOLEM_SKIP_BUILD=1 分支仍会校验 GEMM-only 基线产物。
缺少以下文件之一：
  $BASELINE_BIN
  $BASELINE_BUILD_ENV

处理方式：
  1. 已有基线产物时，保持默认 GOLEM_SKIP_BUILD=1。
  2. 需要重新生成基线 build metadata 时，运行：
     GOLEM_SKIP_BUILD=0 $SCRIPT_DIR/run_noc_dma_softmax_pipeline.sh <原参数>

注意：即使 GOLEM_SKIP_BUILD=0 触发原 GEMM-only binary 构建，SST 实际应用仍由
VANADIS_EXE 指向本目录的 softmax binary。
EOF
		exit 1
	fi
fi
"$TESTS_DIR/run_noc_dma_pipeline.sh" "${PIPELINE_ARGS[@]}" || exit $?

if [[ "$GOLEM_VERIFY_SOFTMAX" -eq 1 && "$HAS_DRY_RUN" -eq 0 ]]; then
	if [[ "$GOLEM_MATMUL_DTYPE" != "fp32" ]]; then
		echo "[SOFTMAX][ERROR] softmax verifier only supports GOLEM_MATMUL_DTYPE=fp32, got $GOLEM_MATMUL_DTYPE" >&2
		exit 1
	fi
	if [[ -z "$GOLEM_SOFTMAX_C_FILE" ]]; then
		GOLEM_SOFTMAX_C_FILE="$TESTS_DIR/artifacts/stats/softmax_c_out.bin"
	fi
	echo "[SOFTMAX] Unpacking softmax C tensor from HBM output..."
	python3 "$TESTS_DIR/tools/unpack_c_from_hbm.py" --out-file "$GOLEM_SOFTMAX_C_FILE"
	echo "[SOFTMAX] Verifying tile-local softmax against A@B golden..."
	python3 "$SCRIPT_DIR/verify_softmax_tile_against_golden.py" \
		--dtype "$GOLEM_MATMUL_DTYPE" \
		--a-file "$SOFTMAX_TENSOR_A_FILE" \
		--b-file "$SOFTMAX_TENSOR_B_FILE" \
		--c-file "$GOLEM_SOFTMAX_C_FILE" \
		--m "$GOLEM_GEMM_M" \
		--n "$GOLEM_GEMM_N" \
		--k "$GOLEM_GEMM_K" \
		--block-m "$GOLEM_GEMM_BLOCK_M" \
		--block-n "$GOLEM_GEMM_BLOCK_N" \
		--reference "$GOLEM_SOFTMAX_VERIFY_REFERENCE" \
		--bias-enable "$GOLEM_BIAS_ENABLE" \
		--bias-value "$GOLEM_BIAS_VALUE"
fi
