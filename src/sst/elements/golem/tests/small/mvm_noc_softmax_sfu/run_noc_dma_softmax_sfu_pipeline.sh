#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_DIR="$TESTS_DIR/small/mvm_noc_int_array"
SFU_BIN="$SCRIPT_DIR/riscv64/test_noc_dma_softmax_sfu"
SFU_BUILD_ENV="$SCRIPT_DIR/riscv64/test_noc_dma_softmax_sfu.build.env"
RISCV_MUSL_TOOLCHAIN_BIN="${RISCV_MUSL_TOOLCHAIN_BIN:-/data/lzq/packages/install/riscv64_musl_toolchain/bin}"
SST_CORE_HOME="${SST_CORE_HOME:-/data4/jjgong/local/sstcore}"
SST_ELEMENTS_HOME="${SST_ELEMENTS_HOME:-/data4/jjgong/RISC-V-CIM-Manycore-SST/install}"
SST_BUILD_LIB_PATH="${SST_BUILD_LIB_PATH:-/data4/jjgong/RISC-V-CIM-Manycore-SST/build/sst-elements/src/sst/elements/golem/.libs}"
SST_INSTALL_LIB_PATH="${SST_INSTALL_LIB_PATH:-/data4/jjgong/RISC-V-CIM-Manycore-SST/install/lib/sst-elements-library}"
if [[ -z "${SST_LIB_PATH+x}" ]]; then
	if [[ -f "$SST_BUILD_LIB_PATH/libgolem.so" ]]; then
		SST_LIB_PATH="$SST_BUILD_LIB_PATH"
	else
		SST_LIB_PATH="$SST_INSTALL_LIB_PATH"
	fi
fi
CONDA_LIB_DIR="${CONDA_LIB_DIR:-/data4/jjgong/miniconda3/lib}"
export SST_CORE_HOME
export SST_ELEMENTS_HOME
export SST_LIB_PATH
export REAL_SST_BIN="${REAL_SST_BIN:-$SST_CORE_HOME/bin/sst}"
export SST_SOFTMAX_LD_LIBRARY_PATH="${SST_SOFTMAX_LD_LIBRARY_PATH:-$CONDA_LIB_DIR:$SST_LIB_PATH:$SST_INSTALL_LIB_PATH:$SST_CORE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH="$SST_SOFTMAX_LD_LIBRARY_PATH"
export PATH="$SCRIPT_DIR/bin:$RISCV_MUSL_TOOLCHAIN_BIN:$SST_CORE_HOME/bin:$SST_ELEMENTS_HOME/bin:$PATH"

HAS_DRY_RUN=0
USER_SET_GOLEM_DMA_READ_RETRY_TICKS=0
if [[ -n "${GOLEM_DMA_READ_RETRY_TICKS+x}" ]]; then
	USER_SET_GOLEM_DMA_READ_RETRY_TICKS=1
fi

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

normalize_path_under_script_dir() {
	local path="$1"
	if [[ -z "$path" || "$path" = /* ]]; then
		echo "$path"
	else
		echo "$SCRIPT_DIR/$path"
	fi
}

derive_memory_routers() {
	local mesh_dim_x="$1"
	local num_memory_nodes="$3"
	local memory_layout="$4"
	local cpu_rows data_memory_node_count data_memory_row_index os_memory_row_index
	local data_row_start os_row_start

	cpu_rows=$(( ($2 + mesh_dim_x - 1) / mesh_dim_x ))
	data_memory_node_count=$(( num_memory_nodes - 1 ))
	if (( data_memory_node_count < 1 )); then
		echo "[ERROR] GOLEM_NUM_MEMORY_NODES must include at least one OS node and one data node" >&2
		return 1
	fi
	if (( data_memory_node_count > mesh_dim_x )); then
		echo "[ERROR] data memory nodes($data_memory_node_count) cannot exceed GOLEM_MESH_DIM_X($mesh_dim_x)" >&2
		return 1
	fi

	case "$memory_layout" in
		top_hbm)
			data_memory_row_index=0
			os_memory_row_index=$(( cpu_rows + 1 ))
			;;
		bottom_hbm)
			data_memory_row_index=$cpu_rows
			os_memory_row_index=$(( cpu_rows + 1 ))
			;;
		*)
			echo "[ERROR] Unsupported GOLEM_MEMORY_LAYOUT=$memory_layout" >&2
			return 1
			;;
	esac

	data_row_start=$(( data_memory_row_index * mesh_dim_x ))
	os_row_start=$(( os_memory_row_index * mesh_dim_x ))
	python3 - "$mesh_dim_x" "$data_memory_node_count" "$data_row_start" "$os_row_start" <<'PY'
import sys

mesh_dim_x = int(sys.argv[1])
data_memory_node_count = int(sys.argv[2])
data_row_start = int(sys.argv[3])
os_row_start = int(sys.argv[4])

if data_memory_node_count == 1:
    columns = [0]
else:
    columns = [
        int(round(i * (mesh_dim_x - 1) / (data_memory_node_count - 1)))
        for i in range(data_memory_node_count)
    ]
routers = [os_row_start] + [data_row_start + col for col in columns]
print(",".join(str(router) for router in routers))
PY
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

sfu_binary_is_fresh() {
	if [[ ! -x "$SFU_BIN" ]]; then
		return 1
	fi
	local src
	for src in \
		"$SCRIPT_DIR/test_noc_dma_softmax_sfu.cpp" \
		"$SCRIPT_DIR/golem_softmax_sfu_runtime.cpp" \
		"$SCRIPT_DIR/golem_softmax_sfu_runtime.h" \
		"$SCRIPT_DIR/ex_instr.h" \
		"$SCRIPT_DIR/Makefile" \
		"$BASE_DIR/golem_matmul_runtime.cpp" \
		"$BASE_DIR/golem_matmul_runtime.h" \
		"$BASE_DIR/gemm_matmul_op.h" \
		"$BASE_DIR/pipeline_config.h" \
		"$BASE_DIR/operators.h"; do
		if [[ "$src" -nt "$SFU_BIN" ]]; then
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
GOLEM_MEMORY_LAYOUT="${GOLEM_MEMORY_LAYOUT:-top_hbm}"
GOLEM_MESH_DIM_X="${GOLEM_MESH_DIM_X:-4}"
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
GOLEM_VERIFY_SOFTMAX="${GOLEM_VERIFY_SOFTMAX:-0}"
GOLEM_SOFTMAX_VERIFY_REFERENCE="${GOLEM_SOFTMAX_VERIFY_REFERENCE:-a_b}"
GOLEM_SOFTMAX_C_FILE="${GOLEM_SOFTMAX_C_FILE:-}"
GOLEM_SOFTMAX_LOGITS_FILE="${GOLEM_SOFTMAX_LOGITS_FILE:-}"
GOLEM_TENSOR_DIR="${GOLEM_TENSOR_DIR:-$SCRIPT_DIR/data}"
GOLEM_TENSOR_A_FILE="${GOLEM_TENSOR_A_FILE:-$GOLEM_TENSOR_DIR/a.bin}"
GOLEM_TENSOR_B_FILE="${GOLEM_TENSOR_B_FILE:-$GOLEM_TENSOR_DIR/b.bin}"
GOLEM_SST_ENABLE_ALL_STATS="${GOLEM_SST_ENABLE_ALL_STATS:-0}"
GOLEM_SST_STAT_LOAD_LEVEL="${GOLEM_SST_STAT_LOAD_LEVEL:-0}"
GOLEM_BENCH_DISABLE_SST_STATS="${GOLEM_BENCH_DISABLE_SST_STATS:-1}"
GOLEM_SFU_ENABLE="${GOLEM_SFU_ENABLE:-1}"
GOLEM_SFU_MAX_INFLIGHT="${GOLEM_SFU_MAX_INFLIGHT:-8}"
GOLEM_SFU_INTERLEAVE_GEMM="${GOLEM_SFU_INTERLEAVE_GEMM:-0}"
GOLEM_SFU_STANDALONE_SOFTMAX="${GOLEM_SFU_STANDALONE_SOFTMAX:-0}"
GOLEM_GEMM_OUT_LAYOUT="${GOLEM_GEMM_OUT_LAYOUT:-colmajor_tile}"

args=("$@")
PIPELINE_ARGS=()
i=0
while [[ "$i" -lt "${#args[@]}" ]]; do
	case "${args[$i]}" in
		--dry-run)
			HAS_DRY_RUN=1; PIPELINE_ARGS+=("${args[$i]}"); i=$((i + 1)) ;;
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
		--mesh-dim-x)
			GOLEM_MESH_DIM_X="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
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
			GOLEM_CTRL_OVERLAP_AB="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
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
		--softmax-logits-file)
			GOLEM_SOFTMAX_LOGITS_FILE="${args[$((i + 1))]}"; i=$((i + 2)) ;;
		--tensor-dir)
			GOLEM_TENSOR_DIR="${args[$((i + 1))]}"
			GOLEM_TENSOR_A_FILE="$GOLEM_TENSOR_DIR/a.bin"
			GOLEM_TENSOR_B_FILE="$GOLEM_TENSOR_DIR/b.bin"
			PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}")
			i=$((i + 2)) ;;
		--tensor-a)
			GOLEM_TENSOR_A_FILE="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
		--tensor-b)
			GOLEM_TENSOR_B_FILE="${args[$((i + 1))]}"; PIPELINE_ARGS+=("${args[$i]}" "${args[$((i + 1))]}"); i=$((i + 2)) ;;
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

if [[ "$GOLEM_SFU_STANDALONE_SOFTMAX" != "0" ]]; then
	if [[ -z "$GOLEM_SOFTMAX_LOGITS_FILE" ]]; then
		GOLEM_SOFTMAX_LOGITS_FILE="$GOLEM_TENSOR_DIR/softmax_logits.bin"
	fi
	GOLEM_SOFTMAX_LOGITS_FILE="$(normalize_path_under_script_dir "$GOLEM_SOFTMAX_LOGITS_FILE")"
	if [[ "$GOLEM_SOFTMAX_VERIFY_REFERENCE" == "a_b" ]]; then
		GOLEM_SOFTMAX_VERIFY_REFERENCE="logits"
	fi
fi

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

if [[ "$GOLEM_CTRL_LINK_ENABLE" == "0" ]]; then
	GOLEM_ARCH_SCRIPT="small/mvm_noc_softmax_cpu/ncores_selfcom_dma_softmax_archive.py"
	GOLEM_REQUEST_SCHEDULER_ENABLE=0
	GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0
	GOLEM_MEMORY_ROUTERS=$(derive_memory_routers "$GOLEM_MESH_DIM_X" "$GOLEM_TOTAL_CORES" "$GOLEM_NUM_MEMORY_NODES" "$GOLEM_MEMORY_LAYOUT")
	if [[ "$USER_SET_GOLEM_DMA_READ_RETRY_TICKS" -eq 0 ]]; then
		GOLEM_DMA_READ_RETRY_TICKS=256
	fi
else
	GOLEM_ARCH_SCRIPT="${GOLEM_ARCH_SCRIPT:-architecture/ncores_selfcom_dma_ctrl.py}"
	GOLEM_REQUEST_SCHEDULER_ENABLE="${GOLEM_REQUEST_SCHEDULER_ENABLE:-1}"
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

SFU_CFLAGS=""
for key in "${BUILD_KEYS[@]}"; do
	SFU_CFLAGS+=" -D${key}=${!key}"
done

pushd "$SCRIPT_DIR" >/dev/null
if metadata_matches "$SFU_BUILD_ENV" "${BUILD_KEYS[@]}" && sfu_binary_is_fresh; then
	echo "[SFU] Reusing existing SFU softmax binary: riscv64/test_noc_dma_softmax_sfu"
else
	make clean ARCH=riscv64
	make ARCH=riscv64 CFLAGS="$SFU_CFLAGS"
	write_metadata_file "$SFU_BUILD_ENV" "${BUILD_KEYS[@]}"
fi
popd >/dev/null

export VANADIS_EXE="$SFU_BIN"
export GOLEM_SKIP_BUILD=0
export GOLEM_SFU_ENABLE
export GOLEM_SFU_MAX_INFLIGHT
export GOLEM_SFU_INTERLEAVE_GEMM
export GOLEM_SFU_STANDALONE_SOFTMAX
export GOLEM_SOFTMAX_LOGITS_FILE
export GOLEM_MATMUL_DTYPE="${GOLEM_MATMUL_DTYPE:-fp32}"
export GOLEM_MATMUL_M="$GOLEM_GEMM_M"
export GOLEM_MATMUL_N="$GOLEM_GEMM_N"
export GOLEM_MATMUL_K="$GOLEM_GEMM_K"
export GOLEM_MATMUL_BLOCK_M="$GOLEM_GEMM_BLOCK_M"
export GOLEM_MATMUL_BLOCK_N="$GOLEM_GEMM_BLOCK_N"
export GOLEM_MATMUL_BLOCK_K="$GOLEM_GEMM_BLOCK_K"
export GOLEM_GEMM_OUT_LAYOUT
export GOLEM_VERIFY_C=0
export GOLEM_GROUP_MANAGER_ENABLE
export GOLEM_CTRL_LINK_ENABLE
export GOLEM_CTRL_OVERLAP_AB
export GOLEM_ARCH_SCRIPT
export GOLEM_REQUEST_SCHEDULER_ENABLE
export GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE
export GOLEM_MEMORY_LAYOUT
export GOLEM_MESH_DIM_X
export GOLEM_MEMORY_ROUTERS
export GOLEM_DMA_READ_RETRY_TICKS
export GOLEM_SST_ENABLE_ALL_STATS
export GOLEM_SST_STAT_LOAD_LEVEL
export GOLEM_BENCH_DISABLE_SST_STATS

echo "[SFU] VANADIS_EXE=$VANADIS_EXE"
echo "[SFU] GOLEM_SKIP_BUILD=$GOLEM_SKIP_BUILD"
echo "[SFU] GOLEM_SFU_ENABLE=$GOLEM_SFU_ENABLE"
echo "[SFU] GOLEM_SFU_MAX_INFLIGHT=$GOLEM_SFU_MAX_INFLIGHT"
echo "[SFU] GOLEM_SFU_INTERLEAVE_GEMM=$GOLEM_SFU_INTERLEAVE_GEMM"
echo "[SFU] GOLEM_SFU_STANDALONE_SOFTMAX=$GOLEM_SFU_STANDALONE_SOFTMAX"
echo "[SFU] GOLEM_SOFTMAX_LOGITS_FILE=${GOLEM_SOFTMAX_LOGITS_FILE:-}"
echo "[SFU] GOLEM_MATMUL_DTYPE=$GOLEM_MATMUL_DTYPE"
echo "[SFU] GOLEM_VERIFY_SOFTMAX=$GOLEM_VERIFY_SOFTMAX"
echo "[SFU] GOLEM_SOFTMAX_VERIFY_REFERENCE=$GOLEM_SOFTMAX_VERIFY_REFERENCE"
echo "[SFU] GOLEM_GROUP_MANAGER_ENABLE=$GOLEM_GROUP_MANAGER_ENABLE"
echo "[SFU] GOLEM_CTRL_LINK_ENABLE=$GOLEM_CTRL_LINK_ENABLE"
echo "[SFU] GOLEM_REQUEST_SCHEDULER_ENABLE=$GOLEM_REQUEST_SCHEDULER_ENABLE"
echo "[SFU] GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=$GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE"
echo "[SFU] GOLEM_ARCH_SCRIPT=$GOLEM_ARCH_SCRIPT"
echo "[SFU] GOLEM_MEMORY_LAYOUT=$GOLEM_MEMORY_LAYOUT"
echo "[SFU] GOLEM_MESH_DIM_X=$GOLEM_MESH_DIM_X"
echo "[SFU] GOLEM_MEMORY_ROUTERS=${GOLEM_MEMORY_ROUTERS:-}"
echo "[SFU] GOLEM_DMA_READ_RETRY_TICKS=${GOLEM_DMA_READ_RETRY_TICKS:-}"
echo "[SFU] GOLEM_SST_ENABLE_ALL_STATS=$GOLEM_SST_ENABLE_ALL_STATS"
echo "[SFU] GOLEM_SST_STAT_LOAD_LEVEL=$GOLEM_SST_STAT_LOAD_LEVEL"
echo "[SFU] GOLEM_BENCH_DISABLE_SST_STATS=$GOLEM_BENCH_DISABLE_SST_STATS"

"$TESTS_DIR/run_noc_dma_pipeline.sh" "${PIPELINE_ARGS[@]}" || exit $?

if [[ "$GOLEM_VERIFY_SOFTMAX" -eq 1 && "$HAS_DRY_RUN" -eq 0 ]]; then
	if [[ "$GOLEM_MATMUL_DTYPE" != "fp32" ]]; then
		echo "[SFU][ERROR] softmax verifier only supports GOLEM_MATMUL_DTYPE=fp32, got $GOLEM_MATMUL_DTYPE" >&2
		exit 1
	fi
	if [[ -z "$GOLEM_SOFTMAX_C_FILE" ]]; then
		GOLEM_SOFTMAX_C_FILE="$TESTS_DIR/artifacts/stats/softmax_sfu_c_out.bin"
	fi
	echo "[SFU] Unpacking SFU softmax C tensor from HBM output..."
	python3 "$TESTS_DIR/tools/unpack_c_from_hbm.py" --out-file "$GOLEM_SOFTMAX_C_FILE"
	echo "[SFU] Verifying full row-wise softmax against A@B golden..."
	python3 "$SCRIPT_DIR/verify_softmax_sfu_against_golden.py" \
		--dtype "$GOLEM_MATMUL_DTYPE" \
		--a-file "$GOLEM_TENSOR_A_FILE" \
		--b-file "$GOLEM_TENSOR_B_FILE" \
		--c-file "$GOLEM_SOFTMAX_C_FILE" \
		--m "$GOLEM_GEMM_M" \
		--n "$GOLEM_GEMM_N" \
		--k "$GOLEM_GEMM_K" \
		--block-m "$GOLEM_GEMM_BLOCK_M" \
		--block-n "$GOLEM_GEMM_BLOCK_N" \
		--reference "$GOLEM_SOFTMAX_VERIFY_REFERENCE" \
		--logits-file "$GOLEM_SOFTMAX_LOGITS_FILE" \
		--bias-enable "$GOLEM_BIAS_ENABLE" \
		--bias-value "$GOLEM_BIAS_VALUE"
fi
