#!/usr/bin/env bash
set -euo pipefail

# 统一运行脚本：
# 1) 设置阵列硬件参数 / RoCC 阵列参数
# 2) 生成 HBM 初始化文件
# 3) 编译 test_noc_dma
# 4) 运行 SST 架构配置脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
RUN_START_EPOCH="$(date +%s)"
RUN_ID="${GOLEM_RUN_ID:-run_$(date +%Y%m%d_%H%M%S)_$$}"
DRAMSIM3_LIB_DIR="${DRAMSIM3_LIB_DIR:-/data4/lishun/pkg/DRAMsim3}"

if [[ -d "$DRAMSIM3_LIB_DIR" ]]; then
	export LD_LIBRARY_PATH="$DRAMSIM3_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

DEFAULT_PRESET_FILE="$SCRIPT_DIR/configs/default.env"
AUTO_PRESET_FILE=""
if [[ -f "$DEFAULT_PRESET_FILE" ]]; then
	# shellcheck source=/dev/null
	source "$DEFAULT_PRESET_FILE"
	AUTO_PRESET_FILE="$DEFAULT_PRESET_FILE"
fi

STATS_DIR_FROM_ENV=0
STATS_FILE_FROM_ENV=0
CORE_MAP_FILE_FROM_ENV=0
MVM_VERIFY_SUMMARY_FROM_ENV=0
MVM_DUMP_DIR_FROM_ENV=0
STDOUT_DIR_FROM_ENV=0

if [[ -n "${GOLEM_STATS_DIR+x}" ]]; then
	STATS_DIR_FROM_ENV=1
fi
if [[ -n "${GOLEM_STATS_FILE+x}" ]]; then
	STATS_FILE_FROM_ENV=1
fi
if [[ -n "${GOLEM_CORE_MAP_FILE+x}" ]]; then
	CORE_MAP_FILE_FROM_ENV=1
fi
if [[ -n "${GOLEM_MVM_VERIFY_SUMMARY_FILE+x}" ]]; then
	MVM_VERIFY_SUMMARY_FROM_ENV=1
fi
if [[ -n "${GOLEM_MVM_DUMP_DIR+x}" ]]; then
	MVM_DUMP_DIR_FROM_ENV=1
fi
if [[ -n "${GOLEM_STDOUT_DIR+x}" ]]; then
	STDOUT_DIR_FROM_ENV=1
fi

ARTIFACT_ROOT="${GOLEM_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
LOG_DIR="${GOLEM_LOG_DIR:-$ARTIFACT_ROOT/logs}"
HBM_DIR="${GOLEM_HBM_DIR:-$ARTIFACT_ROOT/hbm}"
STDOUT_DIR="${GOLEM_STDOUT_DIR:-$ARTIFACT_ROOT/stdout}"
STATS_DIR="${GOLEM_STATS_DIR:-$ARTIFACT_ROOT/stats}"
GOLEM_MVM_DUMP_DIR="${GOLEM_MVM_DUMP_DIR:-$ARTIFACT_ROOT/mvm_dumps}"
STATS_FILE="${GOLEM_STATS_FILE:-$STATS_DIR/stats_selfcom.txt}"
CORE_MAP_FILE="${GOLEM_CORE_MAP_FILE:-$STATS_DIR/core_memory_map.csv}"
MVM_VERIFY_SUMMARY_FILE="${GOLEM_MVM_VERIFY_SUMMARY_FILE:-$STATS_DIR/mvm_verify_summary.csv}"
RUN_SUMMARY_CSV="${GOLEM_RUN_SUMMARY_CSV:-$ARTIFACT_ROOT/stats/run_summary.csv}"
DRAMSIM_STATS_DIR="$STATS_DIR/dramsim3"
EXEC_SUMMARY_FILE="$STATS_DIR/execution_summary.csv"
EXEC_DEBUG_SUMMARY_FILE="$STATS_DIR/execution_debug_summary.csv"
DMA_SUMMARY_FILE="$STATS_DIR/dma_summary.csv"
NOC_SUMMARY_FILE="$STATS_DIR/noc_summary.csv"
MEMORY_SUMMARY_FILE="$STATS_DIR/memory_summary.csv"
NOC_LATENCY_SUMMARY_FILE="$STATS_DIR/noc_latency_summary.csv"
MEMORY_QUEUE_SUMMARY_FILE="$STATS_DIR/memory_queue_summary.csv"
CAUSAL_SUMMARY_FILE="$STATS_DIR/submit_ready_causal_summary.csv"
CAUSAL_TABLE_FILE="$STATS_DIR/submit_ready_causal_table.csv"
SCHED_PRESSURE_SUMMARY_FILE="$STATS_DIR/sched_pressure_summary.csv"
SCHED_PRESSURE_TABLE_FILE="$STATS_DIR/sched_pressure_table.csv"
NOC_HOTSPOT_SUMMARY_FILE="$STATS_DIR/noc_hotspot_summary.csv"
NOC_HOTSPOT_ROUTER_FILE="$STATS_DIR/noc_hotspot_router_table.csv"
NOC_HOTSPOT_PORT_FILE="$STATS_DIR/noc_hotspot_port_table.csv"
EXEC_SUMMARY_FILE="$STATS_DIR/execution_summary.csv"
DMA_SUMMARY_FILE="$STATS_DIR/dma_summary.csv"
NOC_SUMMARY_FILE="$STATS_DIR/noc_summary.csv"
MEMORY_SUMMARY_FILE="$STATS_DIR/memory_summary.csv"

# ===== 默认值（可被环境变量或命令行覆盖） =====
GOLEM_TOTAL_GROUPS="${GOLEM_TOTAL_GROUPS:-4}"

GOLEM_ARRAY_INPUT_SIZE="${GOLEM_ARRAY_INPUT_SIZE:-4}"
GOLEM_ARRAY_OUTPUT_SIZE="${GOLEM_ARRAY_OUTPUT_SIZE:-4}"
GOLEM_NUM_ARRAYS="${GOLEM_NUM_ARRAYS:-1}"
GOLEM_TOTAL_CORES="${GOLEM_TOTAL_CORES:-${VANADIS_NUM_CORES:-16}}"
GOLEM_TOTAL_GEMM_CORES="${GOLEM_TOTAL_GEMM_CORES:-16}"
GOLEM_NUM_MEMORY_NODES="${GOLEM_NUM_MEMORY_NODES:-5}"
GOLEM_MEMORY_LAYOUT="${GOLEM_MEMORY_LAYOUT:-top_hbm}"
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
GOLEM_ARCH_SCRIPT="${GOLEM_ARCH_SCRIPT:-architecture/ncores_selfcom_dma_ctrl.py}"
GOLEM_DRAMSIM3_CONFIG="${GOLEM_DRAMSIM3_CONFIG:-$SCRIPT_DIR/architecture/dram/HBM_4Gb_x128.ini}"
GOLEM_DMA_NODE_CREDITS="${GOLEM_DMA_NODE_CREDITS:-4}"
GOLEM_DMA_NODE_CHUNK_CREDITS="${GOLEM_DMA_NODE_CHUNK_CREDITS:-}"
GOLEM_WCP_PREFETCH_WINDOWS="${GOLEM_WCP_PREFETCH_WINDOWS:-2}"
GOLEM_DMA_WINDOW_K_TILES="${GOLEM_DMA_WINDOW_K_TILES:-4}"
GOLEM_SCHED_SUBMIT_BATCH_SIZE="${GOLEM_SCHED_SUBMIT_BATCH_SIZE:-4}"
GOLEM_SCHED_DONE_BATCH_SIZE="${GOLEM_SCHED_DONE_BATCH_SIZE:-8}"
GOLEM_GEMM_M="${GOLEM_GEMM_M:-$GOLEM_ARRAY_OUTPUT_SIZE}"
GOLEM_GEMM_N="${GOLEM_GEMM_N:-$GOLEM_NUM_ARRAYS}"
GOLEM_GEMM_K="${GOLEM_GEMM_K:-$GOLEM_ARRAY_INPUT_SIZE}"
GOLEM_ORIG_M="${GOLEM_ORIG_M:-}"
GOLEM_ORIG_N="${GOLEM_ORIG_N:-}"
GOLEM_ORIG_K="${GOLEM_ORIG_K:-}"
GOLEM_GEMM_BLOCK_M="${GOLEM_GEMM_BLOCK_M:-$GOLEM_ARRAY_OUTPUT_SIZE}"
GOLEM_GEMM_BLOCK_N="${GOLEM_GEMM_BLOCK_N:-$GOLEM_NUM_ARRAYS}"
GOLEM_GEMM_BLOCK_K="${GOLEM_GEMM_BLOCK_K:-$GOLEM_ARRAY_INPUT_SIZE}"
GOLEM_MATMUL_DTYPE="${GOLEM_MATMUL_DTYPE:-fp32}"
GOLEM_BIAS_ENABLE="${GOLEM_BIAS_ENABLE:-0}"
GOLEM_BIAS_VALUE="${GOLEM_BIAS_VALUE:-0}"
GOLEM_DMA_READ_RETRY_TICKS="${GOLEM_DMA_READ_RETRY_TICKS:-256}"
GOLEM_DMA_READ_MAX_RETRIES="${GOLEM_DMA_READ_MAX_RETRIES:-8}"
GOLEM_DMA_BURST_BYTES="${GOLEM_DMA_BURST_BYTES:-16384}"
GOLEM_DMA_PANEL_CHUNK_BYTES="${GOLEM_DMA_PANEL_CHUNK_BYTES:-16384}"
GOLEM_DMA_CREDIT_CHUNK_BYTES="${GOLEM_DMA_CREDIT_CHUNK_BYTES:-8192}"
GOLEM_DMA_RESPONSE_DRAIN_LIMIT="${GOLEM_DMA_RESPONSE_DRAIN_LIMIT:-0}"
GOLEM_SCHED_ISSUE_BUDGET_PER_TICK="${GOLEM_SCHED_ISSUE_BUDGET_PER_TICK:-2}"
GOLEM_LATENCY_MVM_GM2IMAT="${GOLEM_LATENCY_MVM_GM2IMAT:-10}"
GOLEM_LATENCY_MVM_GM2IVEC="${GOLEM_LATENCY_MVM_GM2IVEC:-10}"
GOLEM_LATENCY_MVM_OVEC2GM="${GOLEM_LATENCY_MVM_OVEC2GM:-10}"
GOLEM_ARRAY_NUM_CU="${GOLEM_ARRAY_NUM_CU:-$GOLEM_ARRAY_OUTPUT_SIZE}"
GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE="${GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE:-1}"
GOLEM_ARRAY_PIPELINE_DEPTH="${GOLEM_ARRAY_PIPELINE_DEPTH:-0}"
GOLEM_ARRAY_CLOCK="${GOLEM_ARRAY_CLOCK:-${VANADIS_CPU_CLOCK:-2.3GHz}}"
GOLEM_MEMCTRL_CLOCK="${GOLEM_MEMCTRL_CLOCK:-${VANADIS_CPU_CLOCK:-2.3GHz}}"

# 可选：RoCC/Array 类型
GOLEM_ROCC_TYPE="${GOLEM_ROCC_TYPE:-golem.RoCCAnalogInt}"
GOLEM_ARRAY_TYPE="${GOLEM_ARRAY_TYPE:-golem.MVMIntArray}"

GOLEM_NOC_INPUT_BUF_SIZE="${GOLEM_NOC_INPUT_BUF_SIZE:-8KB}"
GOLEM_NOC_OUTPUT_BUF_SIZE="${GOLEM_NOC_OUTPUT_BUF_SIZE:-8KB}"
GOLEM_NOC_LINK_BW="${GOLEM_NOC_LINK_BW:-25GB/s}"
GOLEM_NOC_XBAR_BW="${GOLEM_NOC_XBAR_BW:-25GB/s}"
GOLEM_NOC_FLIT_SIZE="${GOLEM_NOC_FLIT_SIZE:-128B}"
GOLEM_MESH_DIM_X="${GOLEM_MESH_DIM_X:-4}"
GOLEM_GM_BUFFER_LENGTH="${GOLEM_GM_BUFFER_LENGTH:-64KB}"
GOLEM_ROCC_VERBOSE="${GOLEM_ROCC_VERBOSE:-0}"
GOLEM_GM_VERBOSE="${GOLEM_GM_VERBOSE:-0}"
GOLEM_GM_DUMP_DATA="${GOLEM_GM_DUMP_DATA:-0}"
GOLEM_DMA_TRACE="${GOLEM_DMA_TRACE:-0}"
GOLEM_REQUEST_SCHEDULER_TRACE="${GOLEM_REQUEST_SCHEDULER_TRACE:-0}"
GOLEM_LLSC_TRACE="${GOLEM_LLSC_TRACE:-0}"
GOLEM_MVM_DUMP_ENABLE="${GOLEM_MVM_DUMP_ENABLE:-0}"
GOLEM_MVM_DUMP_MODE="${GOLEM_MVM_DUMP_MODE:-overwrite}"
GOLEM_PROGRESS_HEARTBEAT="${GOLEM_PROGRESS_HEARTBEAT:-0}"
GOLEM_PROGRESS_INTERVAL_CYCLES="${GOLEM_PROGRESS_INTERVAL_CYCLES:-50000}"
GOLEM_SST_ARGS="${GOLEM_SST_ARGS:-}"
GOLEM_SST_STAT_LOAD_LEVEL="${GOLEM_SST_STAT_LOAD_LEVEL:-16}"
GOLEM_SST_ENABLE_ALL_STATS="${GOLEM_SST_ENABLE_ALL_STATS:-1}"
GOLEM_EXPORT_NOC_HEATMAPS="${GOLEM_EXPORT_NOC_HEATMAPS:-0}"
GOLEM_SKIP_TENSOR_GEN="${GOLEM_SKIP_TENSOR_GEN:-0}"
GOLEM_SKIP_HBM_GEN="${GOLEM_SKIP_HBM_GEN:-0}"
GOLEM_SKIP_BUILD="${GOLEM_SKIP_BUILD:-0}"
GOLEM_HBM_DUMP_OUTPUT="${GOLEM_HBM_DUMP_OUTPUT:-1}"
GOLEM_BENCH_QUIET_LOGS="${GOLEM_BENCH_QUIET_LOGS:-0}"
GOLEM_BENCH_DISABLE_SST_STATS="${GOLEM_BENCH_DISABLE_SST_STATS:-0}"

LOG_FILE="${LOG_FILE:-${GOLEM_PRESET_LOG:-test.log}}"
TENSOR_A_FILE="${GOLEM_TENSOR_A_FILE:-}"
TENSOR_B_FILE="${GOLEM_TENSOR_B_FILE:-}"
DUMP_C_FILE="${GOLEM_DUMP_C_FILE:-}"
TENSOR_SOURCE="${GOLEM_TENSOR_SOURCE:-synthetic}"
TENSOR_DIR="${GOLEM_TENSOR_DIR:-$SCRIPT_DIR/data}"

ARRAY_IN_SET=0
ARRAY_OUT_SET=0
DRY_RUN=0
PRINT_CORE_MAP=0
VERIFY_MVM="${GOLEM_VERIFY_MVM:-0}"
VERIFY_C="${GOLEM_VERIFY_C:-0}"
TOTAL_STAGES=4
PROGRESS_WIDTH=32

supports_fancy_output() {
	[[ -t 1 ]]
}

dtype_nbytes() {
	case "$1" in
		int32|fp32)
			printf '4\n' ;;
		*)
			printf '0\n' ;;
	esac
}

file_size_bytes() {
	local path="$1"
	stat -c '%s' "$path"
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

validate_metadata_file() {
	local file="$1"
	local label="$2"
	shift 2
	if [[ ! -f "$file" ]]; then
		echo "[ERROR] $label metadata 不存在: $file" >&2
		echo "        关闭对应 skip 重新生成一次，之后才能安全复用。" >&2
		return 1
	fi
	local mismatch=0
	for key in "$@"; do
		local expected="${!key}"
		local actual
		actual="$(metadata_get_value "$file" "$key")"
		if [[ -z "$actual" && "$expected" != "" ]]; then
			echo "[ERROR] $label metadata 缺少 $key，期望 $expected" >&2
			mismatch=1
		elif [[ "$actual" != "$expected" ]]; then
			echo "[ERROR] $label metadata 不匹配: $key 当前=$expected 复用文件=$actual" >&2
			mismatch=1
		fi
	done
	if [[ "$mismatch" -ne 0 ]]; then
		echo "        当前配置已经变化，请关闭对应 skip 重新生成。" >&2
		return 1
	fi
	return 0
}

validate_tensor_file_size() {
	local path="$1"
	local label="$2"
	local expected="$3"
	if [[ ! -f "$path" ]]; then
		echo "[ERROR] 缺少 $label tensor: $path" >&2
		return 1
	fi
	local actual
	actual="$(file_size_bytes "$path")"
	if [[ "$actual" != "$expected" ]]; then
		echo "[ERROR] $label tensor 大小不匹配: $path 当前文件=${actual}B 期望=${expected}B" >&2
		echo "        维度或 dtype 已变化，请关闭 GOLEM_SKIP_TENSOR_GEN 重新生成 tensor。" >&2
		return 1
	fi
	return 0
}

validate_hbm_contract_fallback() {
	local file="$1"
	if [[ ! -f "$file" ]]; then
		echo "[ERROR] HBM metadata 不存在，且找不到兼容检查文件: $file" >&2
		echo "        请关闭 GOLEM_SKIP_HBM_GEN 重新生成 HBM。" >&2
		return 1
	fi
	python3 - "$file" \
		"$GOLEM_GEMM_M" "$GOLEM_GEMM_N" "$GOLEM_GEMM_K" \
		"$GOLEM_GEMM_BLOCK_M" "$GOLEM_GEMM_BLOCK_N" "$GOLEM_GEMM_BLOCK_K" \
		"$GOLEM_MATMUL_DTYPE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = {
    "m": int(sys.argv[2]),
    "n": int(sys.argv[3]),
    "k": int(sys.argv[4]),
    "block_m": int(sys.argv[5]),
    "block_n": int(sys.argv[6]),
    "block_k": int(sys.argv[7]),
    "dtype": sys.argv[8],
}
try:
    actual = json.loads(path.read_text())
except Exception as exc:
    print(f"[ERROR] 读取 HBM 兼容 metadata 失败: {path}: {exc}", file=sys.stderr)
    sys.exit(1)

bad = []
for key, want in expected.items():
    got = actual.get(key)
    if got != want:
        bad.append(f"{key} 当前={want} 复用文件={got}")
if bad:
    for item in bad:
        print(f"[ERROR] HBM contract 不匹配: {item}", file=sys.stderr)
    print("        当前矩阵配置已变化，请关闭 GOLEM_SKIP_HBM_GEN 重新生成 HBM。", file=sys.stderr)
    sys.exit(1)
print(f"[WARN] {path} 只覆盖 M/N/K/block/dtype；建议下次关闭 GOLEM_SKIP_HBM_GEN 生成完整 hbm_config.env。")
PY
}

align_up_int() {
	local value="$1"
	local align="$2"
	if (( align <= 0 )); then
		printf '%s\n' "$value"
		return
	fi
	printf '%s\n' $(( ((value + align - 1) / align) * align ))
}

derive_auto_mem_node_size() {
	python3 - "$@" <<'PY'
import sys

(
    gemm_m,
    gemm_n,
    gemm_k,
    block_m,
    block_n,
    block_k,
    elem_bytes,
    num_memory_nodes,
    total_groups,
    total_gemm_cores,
    group_manager_enable,
    a_reuse_n,
    b_reuse_m,
) = [int(x) for x in sys.argv[1:]]


def align_up(value, align):
    return ((value + align - 1) // align) * align


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def next_power_of_two(value):
    return 1 << (value - 1).bit_length()


mm_align = 0x100
m_tiles = gemm_m // block_m
n_tiles = gemm_n // block_n
k_tiles = gemm_k // block_k
m_groups = ceil_div(m_tiles, b_reuse_m)
n_groups = ceil_div(n_tiles, a_reuse_n)
total_macro_tasks = m_groups * n_groups
data_nodes = list(range(1, num_memory_nodes))
dedicated_manager_cores = total_groups if group_manager_enable else 0
active_gemm_cores = total_gemm_cores - dedicated_manager_cores
first_worker_core = dedicated_manager_cores


def owner_core_for_task(task_id):
    if active_gemm_cores <= 0:
        return first_worker_core
    return first_worker_core + (task_id % active_gemm_cores)


def group_id_for_core(core_id):
    return core_id % total_groups if total_groups > 0 else 0


def data_node_for_task(task_id):
    if not data_nodes:
        return 1
    group_id = group_id_for_core(owner_core_for_task(task_id))
    return data_nodes[group_id % len(data_nodes)]


def a_data_node_for_m_tile(m_tile):
    return data_node_for_task(m_tile // b_reuse_m)


def b_data_node_for_n_tile(n_tile):
    return data_node_for_task(n_tile // a_reuse_n)


def max_count(items, predicate):
    out = 0
    for node_idx in data_nodes:
        count = sum(1 for item in items if predicate(node_idx, item))
        out = max(out, count)
    return out or 1


max_a_m_tiles = max_count(range(m_tiles), lambda node, tile: a_data_node_for_m_tile(tile) == node)
max_b_n_tiles = max_count(range(n_tiles), lambda node, tile: b_data_node_for_n_tile(tile) == node)
max_macro_tasks = max_count(range(total_macro_tasks), lambda node, task: data_node_for_task(task) == node)

mat_stride = align_up(block_m * block_k * elem_bytes, mm_align)
vec_stride = align_up(block_k * elem_bytes, mm_align)
out_stride = align_up(block_m * block_n * elem_bytes, mm_align)
bias_stride = align_up(gemm_n * elem_bytes, mm_align)
mat_region_end = max_a_m_tiles * k_tiles * mat_stride
vec_region_end = mat_region_end + max_b_n_tiles * k_tiles * block_n * vec_stride
out_reuse_slots = max(1, a_reuse_n) * max(1, b_reuse_m)
data_region_end = vec_region_end + max_macro_tasks * out_reuse_slots * out_stride
fixed_aux_region_end = 0x01300000
required = max(data_region_end + bias_stride, fixed_aux_region_end, 64 * 1024**2)
chosen = next_power_of_two(required)
print(chosen, required, data_region_end, bias_stride)
PY
}

sst_log_has_fatal() {
	local log_file="$1"
	[[ -f "$log_file" ]] || return 1
	python3 - "$log_file" <<'PY'
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
try:
    text = log_path.read_text(errors="ignore")
except OSError:
    sys.exit(1)

patterns = [
    r"corrupted size vs\. prev_size",
    r"Segmentation fault",
    r"Signal:\s+Aborted",
    r"Signal:\s+Segmentation fault",
    r"Assertion .* failed",
    r"\bfatal\b",
    r"\bpanic\b",
]

for pat in patterns:
    if re.search(pat, text, re.IGNORECASE):
        sys.exit(0)
sys.exit(1)
PY
}

sst_log_has_complete() {
	local log_file="$1"
	[[ -f "$log_file" ]] || return 1
	python3 - "$log_file" <<'PY'
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
try:
    text = log_path.read_text(errors="ignore")
except OSError:
    sys.exit(1)

if re.search(r"Simulation is complete, simulated time:", text):
    sys.exit(0)
sys.exit(1)
PY
}

print_sst_failure_context() {
	local log_file="$1"
	echo "[ERROR] SST failed. Recent log tail:"
	python3 - "$log_file" <<'PY'
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
try:
    lines = log_path.read_text(errors="ignore").splitlines()
except OSError as exc:
    print(f"[WARN] Unable to read log: {exc}")
    sys.exit(0)

for line in lines[-40:]:
    print(line)
PY
}

terminate_sst_process_tree() {
	local pid="$1"
	[[ -n "$pid" ]] || return 0
	local pgid
	pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
	if [[ -n "$pgid" ]]; then
		kill -TERM -- "-$pgid" 2>/dev/null || true
		sleep 0.2
		kill -KILL -- "-$pgid" 2>/dev/null || true
	else
		kill -TERM "$pid" 2>/dev/null || true
		sleep 0.2
		kill -KILL "$pid" 2>/dev/null || true
	fi
}

SST_PID=""

cleanup_sst_on_exit() {
	local status="$?"
	if [[ -n "$SST_PID" ]] && sst_pid_is_running "$SST_PID"; then
		echo "[WARN] Script interrupted or exiting; terminating SST pid=$SST_PID"
		terminate_sst_process_tree "$SST_PID"
		wait "$SST_PID" 2>/dev/null || true
	fi
	return "$status"
}

trap cleanup_sst_on_exit INT TERM EXIT

wait_for_sst_exit() {
	local pid="$1"
	local timeout_sec="${2:-5}"
	local waited=0
	while sst_pid_is_running "$pid"; do
		if (( waited >= timeout_sec * 10 )); then
			return 1
		fi
		sleep 0.1
		((waited++))
	done
	wait "$pid" 2>/dev/null || true
	return 0
}

sst_pid_is_running() {
	local pid="$1"
	[[ -n "$pid" ]] || return 1
	local stat
	stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
	[[ -n "$stat" && "${stat:0:1}" != "Z" ]]
}

estimate_sst_progress_info() {
	local log_file="$1"
	if [[ ! -f "$log_file" ]]; then
		echo "50|架构初始化"
		return
	fi

	# 优先使用里程碑日志（lenet_conv12）：
	# [MILESTONE] stage=conv1|conv2|fc1|fc2|fc3 status=start|done|fail cycle=...
	local ms
	ms="$(tail -n 6000 "$log_file" 2>/dev/null | awk '
		/\[MILESTONE\]/ {
			if (match($0, /stage=([a-zA-Z0-9_]+)/, s) && match($0, /status=([a-zA-Z0-9_]+)/, t)) {
				stage=s[1]; st=t[1];
				seen=1;
				if (st == "start") start[stage]=1;
				if (st == "done") done[stage]=1;
				if (st == "fail") fail=stage;
			}
		}
		END {
			if (seen != 1) {
				print "none";
				exit;
			}
			if (fail != "") {
				printf("99|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 失败:%s", fail);
				exit;
			}
			if (done["fc3"] == 1) {
				print "99|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:完成收尾";
				exit;
			}
			if (start["fc3"] == 1) {
				print "97|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:fc3";
				exit;
			}
			if (done["fc2"] == 1) {
				print "94|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:fc2完成";
				exit;
			}
			if (start["fc2"] == 1) {
				print "89|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:fc2";
				exit;
			}
			if (done["fc1"] == 1) {
				print "84|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:fc1完成";
				exit;
			}
			if (start["fc1"] == 1) {
				print "79|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:fc1";
				exit;
			}
			if (done["conv2"] == 1) {
				print "74|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:conv2完成";
				exit;
			}
			if (start["conv2"] == 1) {
				print "68|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:conv2";
				exit;
			}
			if (done["conv1"] == 1) {
				print "64|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:conv1完成";
				exit;
			}
			if (start["conv1"] == 1) {
				print "58|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:conv1";
				exit;
			}
			print "52|SST阶段: conv1 -> conv2 -> fc1 -> fc2 -> fc3 | 当前:初始化";
		}
	')"
	if [[ "$ms" != "none" ]]; then
		echo "$ms"
		return
	fi

	# 优先使用轻量心跳日志估算真实进度（需要 --progress-heartbeat 1）
	local hb
	hb="$(tail -n 4000 "$log_file" 2>/dev/null | awk '
		BEGIN {
			dma_sum_c = 0; dma_sum_t = 0;
			mvm_sum_c = 0; mvm_sum_t = 0;
		}
		/GlobalMemory core=[0-9]+ DMA_PROGRESS(_FINAL)?:/ {
			if (match($0, /core=([0-9]+)/, c) && match($0, /completed=([0-9]+)\/([0-9]+)/, p)) {
				dma_c[c[1]] = p[1];
				dma_t[c[1]] = p[2];
			}
		}
		/RoCC core=[0-9]+ MVM_PROGRESS:/ {
			if (match($0, /core=([0-9]+)/, c) && match($0, /completed=([0-9]+)\/([0-9]+)/, p)) {
				mvm_c[c[1]] = p[1];
				mvm_t[c[1]] = p[2];
			}
		}
		END {
			for (k in dma_t) { dma_sum_c += dma_c[k]; dma_sum_t += dma_t[k]; }
			for (k in mvm_t) { mvm_sum_c += mvm_c[k]; mvm_sum_t += mvm_t[k]; }
			dma_pct = (dma_sum_t > 0) ? int((dma_sum_c * 100) / dma_sum_t) : -1;
			mvm_pct = (mvm_sum_t > 0) ? int((mvm_sum_c * 100) / mvm_sum_t) : -1;
			printf("%d|%d", dma_pct, mvm_pct);
		}
	')"

	local dma_pct="${hb%%|*}"
	local mvm_pct="${hb##*|}"
	if [[ "$dma_pct" =~ ^-?[0-9]+$ && "$mvm_pct" =~ ^-?[0-9]+$ ]]; then
		if [[ "$mvm_pct" -ge 0 ]]; then
			# stage 3 映射到全局 75~99，完成后由 wait 收敛到 100
			local p=$(( 75 + (mvm_pct * 24 / 100) ))
			if [[ "$dma_pct" -ge 0 ]]; then
				echo "$p|计算阶段(DMA:${dma_pct}% MVM:${mvm_pct}%)"
			else
				echo "$p|计算阶段(MVM:${mvm_pct}%)"
			fi
			return
		fi
		if [[ "$dma_pct" -ge 0 ]]; then
			# stage 3 前半段映射到全局 55~75
			local p=$(( 55 + (dma_pct * 20 / 100) ))
			echo "$p|DMA阶段(DMA:${dma_pct}%)"
			return
		fi
	fi

	if grep -q "mvm.ovec2gm\|MVM compute" "$log_file" 2>/dev/null; then
		echo "90|计算阶段"
	elif grep -q "DMA_READ_COMPLETE\|RemoteLoad issued\|remote_ld" "$log_file" 2>/dev/null; then
		echo "75|DMA阶段"
	elif grep -q "readBinaryELFInfo\|ELF Information" "$log_file" 2>/dev/null; then
		echo "62|ELF装载"
	elif grep -q "init phase=\|Creating CPU core|Configuring for .* core links" "$log_file" 2>/dev/null; then
		echo "52|架构初始化"
	else
		echo "50|架构初始化"
	fi
}

print_progress_bar() {
	local stage="$1"
	local label="$2"
	local filled=$(( stage * PROGRESS_WIDTH / TOTAL_STAGES ))
	local empty=$(( PROGRESS_WIDTH - filled ))
	local bar
	bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
	local pct=$(( stage * 100 / TOTAL_STAGES ))
	echo "[Progress] [${bar}] ${pct}% (${stage}/${TOTAL_STAGES}) ${label}"
}

render_inline_progress_bar() {
	local pct="$1"
	local width="$2"
	local filled=$(( pct * width / 100 ))
	local empty=$(( width - filled ))
	local filled_seg=""
	local empty_seg=""
	local i
	for ((i=0; i<filled; i++)); do
		filled_seg+=$'\033[47m \033[0m'
	done
	for ((i=0; i<empty; i++)); do
		empty_seg+="-"
	done
	printf "[%s%s]" "$filled_seg" "$empty_seg"
}

resolve_stage_clock_ghz() {
	if [[ -n "${GOLEM_STAGE_CLOCK_GHZ:-}" ]]; then
		echo "$GOLEM_STAGE_CLOCK_GHZ"
		return
	fi
	local raw="${VANADIS_CPU_CLOCK:-2.3GHz}"
	if [[ "$raw" =~ ^([0-9]+([.][0-9]+)?)GHz$ ]]; then
		echo "${BASH_REMATCH[1]}"
		return
	fi
	if [[ "$raw" =~ ^([0-9]+([.][0-9]+)?)MHz$ ]]; then
		python3 - <<PY
v=float("${BASH_REMATCH[1]}")
print(v/1000.0)
PY
		return
	fi
	if [[ "$raw" =~ ^([0-9]+([.][0-9]+)?)kHz$ ]]; then
		python3 - <<PY
v=float("${BASH_REMATCH[1]}")
print(v/1000000.0)
PY
		return
	fi
	if [[ "$raw" =~ ^([0-9]+([.][0-9]+)?)Hz$ ]]; then
		python3 - <<PY
v=float("${BASH_REMATCH[1]}")
print(v/1000000000.0)
PY
		return
	fi
	echo "2.3"
}

show_help() {
	cat <<'EOF'
Usage:
	./run_noc_dma_pipeline.sh [options]

Options:
	(自动加载) 若存在 configs/default.env，会在启动时自动 source；
	           default.env 会继续加载分类文件：
	           configs/10_core_gemm.env
	           configs/20_dma.env
	           configs/30_network.env
	           configs/40_debug_io.env
	           configs/50_tensor_verify.env
	--groups N           组数量（默认: 4）
	--array-in N         RoCC 阵列输入长度（默认: 4）
	--array-out N        RoCC 阵列输出长度（默认: 4）
	--num-arrays N       RoCC 阵列实例数（默认: 1）
	--gemm-cores N       GEMM 并发核心数（默认: 16）
	--num-cores N        总核心数（同时用于 SST 与 C++ TOTAL_CORES，默认: 16）
	--num-mem-nodes N    内存节点总数（首节点挂 OS，默认: 4）
	--mem-node-size N|auto 单个内存节点大小（字节），auto 按当前 GEMM/HBM 布局选 2 的幂
	--global-stride-kb N  每核 GM 窗口大小（KB，默认: 64）
	--gemm-m N           GEMM 的 M 维（默认: 跟 --array-out）
	--gemm-n N           GEMM 的 N 维（默认: 跟 --num-arrays）
	--gemm-k N           GEMM 的 K 维（默认: 跟 --array-in）
	--orig-m N           原始请求 M（仅记录/元数据，不参与内核执行）
	--orig-n N           原始请求 N（仅记录/元数据，不参与内核执行）
	--orig-k N           原始请求 K（仅记录/元数据，不参与内核执行）
	--gemm-block-m N     GEMM block_M（默认: 跟 --array-out，phase-1 要求等于 --array-out）
	--gemm-block-n N     GEMM block_N（默认: 跟 --num-arrays，phase-1 要求 <= --num-arrays）
	--gemm-block-k N     GEMM block_K（默认: 跟 --array-in，phase-1 要求等于 --array-in）
	--dtype TYPE         matmul 数据类型：int32|fp32（默认: int32）
	--bias-enable N      可选后处理bias开关（0:关闭,1:开启，默认: 0）
	--bias-value N       bias常量值（int32/fp32，默认: 0）
	--bias-file FILE     bias向量文件（.bin/.csv/.npy），透传给 gen_hbm_init.py
	--pool1-file FILE    预置 pool1 张量文件（6x12x12 fp32 .bin），透传给 gen_hbm_init.py
	--conv2-bpack-file FILE  conv2 B打包文件（3x16x64 fp32 .bin），透传给 gen_hbm_init.py
	--conv2-bias-file FILE   conv2 bias文件（16 fp32 .bin），透传给 gen_hbm_init.py
	--fc1-weight-file FILE   fc1分片权重文件（4x2x64x64 fp32 .bin），透传给 gen_hbm_init.py
	--fc1-bias-file FILE     fc1 bias文件（120 fp32 .bin），透传给 gen_hbm_init.py
	--fc2-weight-file FILE   fc2权重文件（2x2x64x64 fp32 .bin），透传给 gen_hbm_init.py
	--fc2-bias-file FILE     fc2 bias文件（84 fp32 .bin），透传给 gen_hbm_init.py
	--fc3-weight-file FILE   fc3权重文件（2x2x64x64 fp32 .bin），透传给 gen_hbm_init.py
	--fc3-bias-file FILE     fc3 bias文件（10 fp32 .bin），透传给 gen_hbm_init.py
	--dma-stagger-cycles N  每核启动DMA错峰周期（默认: 0，建议dim16从2000起试）
	--dma-overlap N      DMA/计算重叠开关（0:关闭, 1:开启，默认: 0）
	--dma-max-inflight N 每核DMA读请求在途上限（默认: 8）
	--dma-read-retry-ticks N DMA读chunk重试超时tick（默认: 96）
	--dma-read-max-retries N DMA读chunk最大重试次数（默认: 8）
	--dma-burst-bytes N  DMA分块大小（字节，默认: 64）
	--progress-heartbeat N   轻量进度心跳（0关闭,1开启，默认: 0）
	--progress-interval-cycles N 进度心跳周期（cycles，默认: 50000）
	--rocc-type TYPE     RoCC 类型（默认: golem.RoCCAnalogInt）
	--array-type TYPE    阵列类型（默认: golem.MVMIntArray）
	--noc-buf SIZE       同时设置 NoC input/output buffer 大小（默认: 8KB）
	--noc-in-buf SIZE    仅设置 NoC input buffer 大小（默认: 8KB）
	--noc-out-buf SIZE   仅设置 NoC output buffer 大小（默认: 8KB）
	--noc-link-bw BW     NoC 链路带宽（默认: 25GB/s）
	--noc-xbar-bw BW     NoC 路由器 xbar 带宽（默认: 25GB/s）
	--noc-flit-size SIZE NoC flit 大小（默认: 128B）
	--mesh-dim-x N       Mesh 列数 GOLEM_MESH_DIM_X（默认: 4）
	--gm-buf SIZE        GlobalMemory networkIF buffer_length（默认: 64KB）
	--rocc-verbose N     RoCC 日志级别（默认: -1，静默）
	--gm-verbose N       GlobalMemory 日志级别（默认: 0，关键 DMA 仍保留）
	--gm-dump-data N     GlobalMemory 数据hex打印（0关闭，1开启，默认: 0）
	--mvm-dump           打开 MVM 结果按核落盘（等价 GOLEM_MVM_DUMP_ENABLE=1）
	--mvm-dump-dir DIR   MVM 结果输出根目录（默认: mvm_dumps）
	--print-core-map     生成 core->memory-node 映射 CSV（归档到 artifacts/stats）
	--verify-mvm         运行 Python 离线矩阵结果校验（依赖 mvm dump）
	--verify-c           运行 C=AxB 端到端校验（需要 --tensor-a/--tensor-b）
	--tensor-source MODE 输入来源: synthetic|file|sample（默认: synthetic）
	--tensor-dir DIR     sample 模式输出目录（默认: tests/data）
	--tensor-a FILE      外部输入 A 矩阵文件（.bin/.csv/.npy，按 --dtype 解释）传给 gen_hbm_init.py
	--tensor-b FILE      外部输入 B 矩阵文件（.bin/.csv/.npy，按 --dtype 解释）传给 gen_hbm_init.py
	--dump-c FILE        从 hbm_out_node*.bin 反解输出 C 到 FILE（.bin 或 .csv）
	--hbm-dump-output N  是否生成 hbm_out_node*.bin（0/1，默认: 1）
	--no-hbm-dump-output 等价 --hbm-dump-output 0
	--log FILE           SST 输出日志文件名或绝对路径（默认: test.log，存放到 artifacts/logs）
	--dry-run            仅打印配置与命令，不实际执行
	环境变量 GOLEM_SKIP_TENSOR_GEN=1 可跳过 sample tensor 生成
	环境变量 GOLEM_SKIP_HBM_GEN=1 可复用现有 hbm_init/out_node*.bin
	环境变量 GOLEM_SKIP_BUILD=1 可复用现有 riscv64/test_noc_dma
	-h, --help           显示帮助

Output Layout (默认):
	artifacts/logs       主日志与 trace
	artifacts/stdout/overlap0 or overlap1   stdout-* / stderr-* 分片输出
	artifacts/hbm        hbm_init_node*.bin / hbm_out_node*.bin
	artifacts/mvm_dumps/overlap0 or overlap1   MVM dump snapshots
	artifacts/stats/overlap0 or overlap1   stats_selfcom.txt / execution_summary.csv / dma_summary.csv / noc_summary.csv / memory_summary.csv / per-run NoC heatmaps

Priority:
	命令行参数 > 环境变量 > 脚本默认值

Examples:
	./run_noc_dma_pipeline.sh --array-in 8 --array-out 8
	./run_noc_dma_pipeline.sh --array-in 16 --array-out 16 --log test_dim16.log
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--groups)
			GOLEM_TOTAL_GROUPS="$2"; shift 2 ;;
		--array-in)
			GOLEM_ARRAY_INPUT_SIZE="$2"; ARRAY_IN_SET=1; shift 2 ;;
		--array-out)
			GOLEM_ARRAY_OUTPUT_SIZE="$2"; ARRAY_OUT_SET=1; shift 2 ;;
		--num-arrays)
			GOLEM_NUM_ARRAYS="$2"; shift 2 ;;
		--gemm-cores)
			GOLEM_TOTAL_GEMM_CORES="$2"; shift 2 ;;
		--num-cores)
			GOLEM_TOTAL_CORES="$2"; shift 2 ;;
		--num-mem-nodes)
			GOLEM_NUM_MEMORY_NODES="$2"; shift 2 ;;
		--mem-node-size)
			GOLEM_MEM_NODE_SIZE_BYTES="$2"; shift 2 ;;
		--global-stride-kb)
			GOLEM_GLOBAL_STRIDE_KB="$2"; shift 2 ;;
		--gemm-m)
			GOLEM_GEMM_M="$2"; shift 2 ;;
		--gemm-n)
			GOLEM_GEMM_N="$2"; shift 2 ;;
		--gemm-k)
			GOLEM_GEMM_K="$2"; shift 2 ;;
		--orig-m)
			GOLEM_ORIG_M="$2"; shift 2 ;;
		--orig-n)
			GOLEM_ORIG_N="$2"; shift 2 ;;
		--orig-k)
			GOLEM_ORIG_K="$2"; shift 2 ;;
		--gemm-block-m)
			GOLEM_GEMM_BLOCK_M="$2"; shift 2 ;;
		--gemm-block-n)
			GOLEM_GEMM_BLOCK_N="$2"; shift 2 ;;
		--gemm-block-k)
			GOLEM_GEMM_BLOCK_K="$2"; shift 2 ;;
		--dtype)
			GOLEM_MATMUL_DTYPE="$2"; shift 2 ;;
		--bias-enable)
			GOLEM_BIAS_ENABLE="$2"; shift 2 ;;
		--bias-value)
			GOLEM_BIAS_VALUE="$2"; shift 2 ;;
		--bias-file)
			GOLEM_BIAS_FILE="$2"; shift 2 ;;
		--pool1-file)
			GOLEM_POOL1_FILE="$2"; shift 2 ;;
		--conv2-bpack-file)
			GOLEM_CONV2_BPACK_FILE="$2"; shift 2 ;;
		--conv2-bias-file)
			GOLEM_CONV2_BIAS_FILE="$2"; shift 2 ;;
		--fc1-weight-file)
			GOLEM_FC1_WEIGHT_FILE="$2"; shift 2 ;;
		--fc1-bias-file)
			GOLEM_FC1_BIAS_FILE="$2"; shift 2 ;;
		--fc2-weight-file)
			GOLEM_FC2_WEIGHT_FILE="$2"; shift 2 ;;
		--fc2-bias-file)
			GOLEM_FC2_BIAS_FILE="$2"; shift 2 ;;
		--fc3-weight-file)
			GOLEM_FC3_WEIGHT_FILE="$2"; shift 2 ;;
		--fc3-bias-file)
			GOLEM_FC3_BIAS_FILE="$2"; shift 2 ;;
		--dma-stagger-cycles)
			GOLEM_DMA_STAGGER_CYCLES="$2"; shift 2 ;;
		--dma-overlap)
			GOLEM_DMA_OVERLAP="$2"; shift 2 ;;
		--dma-max-inflight)
			GOLEM_DMA_MAX_INFLIGHT="$2"; shift 2 ;;
		--dma-read-retry-ticks)
			GOLEM_DMA_READ_RETRY_TICKS="$2"; shift 2 ;;
		--dma-read-max-retries)
			GOLEM_DMA_READ_MAX_RETRIES="$2"; shift 2 ;;
		--dma-burst-bytes)
			GOLEM_DMA_BURST_BYTES="$2"; shift 2 ;;
		--dma-panel-chunk-bytes)
			GOLEM_DMA_PANEL_CHUNK_BYTES="$2"; shift 2 ;;
		--dma-response-drain-limit)
			GOLEM_DMA_RESPONSE_DRAIN_LIMIT="$2"; shift 2 ;;
		--progress-heartbeat)
			GOLEM_PROGRESS_HEARTBEAT="$2"; shift 2 ;;
		--progress-interval-cycles)
			GOLEM_PROGRESS_INTERVAL_CYCLES="$2"; shift 2 ;;
		--rocc-type)
			GOLEM_ROCC_TYPE="$2"; shift 2 ;;
		--array-type)
			GOLEM_ARRAY_TYPE="$2"; shift 2 ;;
		--noc-buf)
			GOLEM_NOC_INPUT_BUF_SIZE="$2"; GOLEM_NOC_OUTPUT_BUF_SIZE="$2"; shift 2 ;;
		--noc-in-buf)
			GOLEM_NOC_INPUT_BUF_SIZE="$2"; shift 2 ;;
		--noc-out-buf)
			GOLEM_NOC_OUTPUT_BUF_SIZE="$2"; shift 2 ;;
		--noc-link-bw)
			GOLEM_NOC_LINK_BW="$2"; shift 2 ;;
		--noc-xbar-bw)
			GOLEM_NOC_XBAR_BW="$2"; shift 2 ;;
		--noc-flit-size)
			GOLEM_NOC_FLIT_SIZE="$2"; shift 2 ;;
		--mesh-dim-x)
			GOLEM_MESH_DIM_X="$2"; shift 2 ;;
		--gm-buf)
			GOLEM_GM_BUFFER_LENGTH="$2"; shift 2 ;;
		--rocc-verbose)
			GOLEM_ROCC_VERBOSE="$2"; shift 2 ;;
		--gm-verbose)
			GOLEM_GM_VERBOSE="$2"; shift 2 ;;
		--gm-dump-data)
			GOLEM_GM_DUMP_DATA="$2"; shift 2 ;;
		--mvm-dump)
			GOLEM_MVM_DUMP_ENABLE=1; shift ;;
		--mvm-dump-dir)
			GOLEM_MVM_DUMP_DIR="$2"; shift 2 ;;
		--print-core-map)
			PRINT_CORE_MAP=1; shift ;;
		--verify-mvm)
			VERIFY_MVM=1; shift ;;
		--verify-c)
			VERIFY_C=1; shift ;;
		--tensor-source)
			TENSOR_SOURCE="$2"; shift 2 ;;
		--tensor-dir)
			TENSOR_DIR="$2"; shift 2 ;;
		--tensor-a)
			TENSOR_A_FILE="$2"; shift 2 ;;
		--tensor-b)
			TENSOR_B_FILE="$2"; shift 2 ;;
		--dump-c)
			DUMP_C_FILE="$2"; shift 2 ;;
		--hbm-dump-output)
			GOLEM_HBM_DUMP_OUTPUT="$2"; shift 2 ;;
		--no-hbm-dump-output)
			GOLEM_HBM_DUMP_OUTPUT=0; shift ;;
		--log)
			LOG_FILE="$2"; shift 2 ;;
		--dry-run)
			DRY_RUN=1; shift ;;
		-h|--help)
			show_help; exit 0 ;;
		*)
			echo "[ERROR] Unknown option: $1" >&2
			show_help
			exit 1 ;;
	esac
done

if [[ "$ARRAY_IN_SET" -eq 0 ]]; then
	GOLEM_ARRAY_INPUT_SIZE="${GOLEM_ARRAY_INPUT_SIZE:-4}"
fi
if [[ "$ARRAY_OUT_SET" -eq 0 ]]; then
	GOLEM_ARRAY_OUTPUT_SIZE="${GOLEM_ARRAY_OUTPUT_SIZE:-4}"
fi

for n in "$GOLEM_TOTAL_GROUPS" "$GOLEM_ARRAY_INPUT_SIZE" "$GOLEM_ARRAY_OUTPUT_SIZE" "$GOLEM_NUM_ARRAYS" "$GOLEM_TOTAL_CORES" "$GOLEM_TOTAL_GEMM_CORES" "$GOLEM_NUM_MEMORY_NODES" "$GOLEM_GLOBAL_STRIDE_KB"; do
	if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; then
		echo "[ERROR] 参数必须为正整数，收到: $n" >&2
		exit 1
	fi
done

if [[ "$GOLEM_MEM_NODE_SIZE_BYTES" != "auto" ]]; then
	if ! [[ "$GOLEM_MEM_NODE_SIZE_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_MEM_NODE_SIZE_BYTES" -le 0 ]]; then
		echo "[ERROR] GOLEM_MEM_NODE_SIZE_BYTES 必须为正整数或 auto，收到: $GOLEM_MEM_NODE_SIZE_BYTES" >&2
		exit 1
	fi
fi

# Backward compatibility: allow GOLEM_GLOBAL_STRIDE_BYTES override.
if [[ -n "${GOLEM_GLOBAL_STRIDE_BYTES+x}" ]]; then
	if ! [[ "$GOLEM_GLOBAL_STRIDE_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_GLOBAL_STRIDE_BYTES" -le 0 ]]; then
		echo "[ERROR] GOLEM_GLOBAL_STRIDE_BYTES 必须为正整数，收到: $GOLEM_GLOBAL_STRIDE_BYTES" >&2
		exit 1
	fi
	if (( GOLEM_GLOBAL_STRIDE_BYTES % 1024 != 0 )); then
		echo "[ERROR] GOLEM_GLOBAL_STRIDE_BYTES 必须是 1024 的整数倍，收到: $GOLEM_GLOBAL_STRIDE_BYTES" >&2
		exit 1
	fi
	GOLEM_GLOBAL_STRIDE_KB=$(( GOLEM_GLOBAL_STRIDE_BYTES / 1024 ))
fi

GOLEM_GLOBAL_STRIDE_BYTES=$(( GOLEM_GLOBAL_STRIDE_KB * 1024 ))

for n in "$GOLEM_GEMM_M" "$GOLEM_GEMM_N" "$GOLEM_GEMM_K"; do
	if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; then
		echo "[ERROR] GEMM 维度必须为正整数，收到: $n" >&2
		exit 1
	fi
done

for n in "$GOLEM_ORIG_M" "$GOLEM_ORIG_N" "$GOLEM_ORIG_K"; do
	if [[ -n "$n" ]] && { ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; }; then
		echo "[ERROR] 原始维度 --orig-m/--orig-n/--orig-k 必须为正整数，收到: $n" >&2
		exit 1
	fi
done

if [[ "$TENSOR_SOURCE" != "synthetic" && "$TENSOR_SOURCE" != "file" && "$TENSOR_SOURCE" != "sample" ]]; then
	echo "[ERROR] --tensor-source 只能是 synthetic|file|sample，收到: $TENSOR_SOURCE" >&2
	exit 1
fi

if [[ "$TENSOR_SOURCE" == "synthetic" && ( -n "$TENSOR_A_FILE" || -n "$TENSOR_B_FILE" ) ]]; then
	TENSOR_SOURCE="file"
fi

if [[ "$TENSOR_SOURCE" == "sample" ]]; then
	if [[ -z "$TENSOR_A_FILE" ]]; then
		TENSOR_A_FILE="$TENSOR_DIR/a.bin"
	fi
	if [[ -z "$TENSOR_B_FILE" ]]; then
		TENSOR_B_FILE="$TENSOR_DIR/b.bin"
	fi
fi

if [[ "$TENSOR_SOURCE" == "file" ]] && { [[ -n "$TENSOR_A_FILE" && -z "$TENSOR_B_FILE" ]] || [[ -z "$TENSOR_A_FILE" && -n "$TENSOR_B_FILE" ]]; }; then
	echo "[ERROR] --tensor-a 和 --tensor-b 必须同时提供" >&2
	exit 1
fi

if [[ "$VERIFY_C" -eq 1 ]]; then
	if [[ "$TENSOR_SOURCE" == "synthetic" ]]; then
		echo "[ERROR] --verify-c 需要 --tensor-source file 或 sample（synthetic 模式无输入文件可对齐）" >&2
		exit 1
	fi
	if [[ -z "$TENSOR_A_FILE" || -z "$TENSOR_B_FILE" ]]; then
		echo "[ERROR] --verify-c 需要同时提供 --tensor-a 和 --tensor-b" >&2
		exit 1
	fi
	if [[ -z "$DUMP_C_FILE" ]]; then
		DUMP_C_FILE="$STATS_DIR/c_out.csv"
	fi
fi

if [[ "$GOLEM_HBM_DUMP_OUTPUT" != "0" && "$GOLEM_HBM_DUMP_OUTPUT" != "1" ]]; then
	echo "[ERROR] GOLEM_HBM_DUMP_OUTPUT 必须为 0 或 1，收到: $GOLEM_HBM_DUMP_OUTPUT" >&2
	exit 1
fi

if [[ "$GOLEM_HBM_DUMP_OUTPUT" -eq 0 && -n "$DUMP_C_FILE" ]]; then
	echo "[ERROR] GOLEM_HBM_DUMP_OUTPUT=0 时不会生成 hbm_out_node*.bin，不能使用 --dump-c/--verify-c" >&2
	exit 1
fi

for n in "$GOLEM_GEMM_BLOCK_M" "$GOLEM_GEMM_BLOCK_N" "$GOLEM_GEMM_BLOCK_K"; do
	if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; then
		echo "[ERROR] GEMM block 维度必须为正整数，收到: $n" >&2
		exit 1
	fi
done

if (( GOLEM_GEMM_M % GOLEM_GEMM_BLOCK_M != 0 || GOLEM_GEMM_N % GOLEM_GEMM_BLOCK_N != 0 || GOLEM_GEMM_K % GOLEM_GEMM_BLOCK_K != 0 )); then
	echo "[ERROR] GEMM M/N/K 必须可被 block_M/N/K 整除。收到 M/N/K=${GOLEM_GEMM_M}/${GOLEM_GEMM_N}/${GOLEM_GEMM_K}, block=${GOLEM_GEMM_BLOCK_M}/${GOLEM_GEMM_BLOCK_N}/${GOLEM_GEMM_BLOCK_K}" >&2
	exit 1
fi

GOLEM_ELEM_BYTES="$(dtype_nbytes "$GOLEM_MATMUL_DTYPE")"
if (( GOLEM_ELEM_BYTES <= 0 )); then
	echo "[ERROR] 不支持的 dtype: $GOLEM_MATMUL_DTYPE" >&2
	exit 1
fi

if [[ -z "$GOLEM_ORIG_M" ]]; then
	GOLEM_ORIG_M="$GOLEM_GEMM_M"
fi
if [[ -z "$GOLEM_ORIG_N" ]]; then
	GOLEM_ORIG_N="$GOLEM_GEMM_N"
fi
if [[ -z "$GOLEM_ORIG_K" ]]; then
	GOLEM_ORIG_K="$GOLEM_GEMM_K"
fi

if (( GOLEM_GEMM_BLOCK_M % GOLEM_ARRAY_OUTPUT_SIZE != 0 || GOLEM_GEMM_BLOCK_K % GOLEM_ARRAY_INPUT_SIZE != 0 )); then
	echo "[ERROR] 当前运行要求 block_M/block_K 是 ARRAY_OUTPUT/INPUT 的整数倍，收到 block_M=$GOLEM_GEMM_BLOCK_M block_K=$GOLEM_GEMM_BLOCK_K, ARRAY_OUTPUT/INPUT=$GOLEM_ARRAY_OUTPUT_SIZE/$GOLEM_ARRAY_INPUT_SIZE" >&2
	exit 1
fi

if [[ "$GOLEM_GEMM_BLOCK_N" -gt "$GOLEM_NUM_ARRAYS" ]]; then
	echo "[ERROR] 当前 phase-1 要求 block_N <= GOLEM_NUM_ARRAYS($GOLEM_NUM_ARRAYS)，收到 block_N=$GOLEM_GEMM_BLOCK_N" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_SLOT_COUNT" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_SLOT_COUNT" -lt 4 ]]; then
	echo "[ERROR] GOLEM_DMA_SLOT_COUNT 必须为 >=4 的正整数，收到: $GOLEM_DMA_SLOT_COUNT" >&2
	exit 1
fi

# Keep per-core local GM layout within the reserved global stride.
ELEM_BYTES=4
LOCAL_ALIGN=256
LOCAL_DATA_BASE_DEC=$((0x2000))
mat_slot_bytes=$(align_up_int $(( GOLEM_GEMM_BLOCK_M * GOLEM_GEMM_BLOCK_K * ELEM_BYTES )) $LOCAL_ALIGN)
vec_slot_bytes=$(align_up_int $(( GOLEM_GEMM_BLOCK_N * GOLEM_GEMM_BLOCK_K * ELEM_BYTES )) $LOCAL_ALIGN)
out_scratch_bytes=$(align_up_int $(( GOLEM_ARRAY_OUTPUT_SIZE * ELEM_BYTES )) $LOCAL_ALIGN)
out_tile_bytes=$(align_up_int $(( GOLEM_GEMM_BLOCK_M * GOLEM_GEMM_BLOCK_N * ELEM_BYTES )) $LOCAL_ALIGN)
required_global_stride_bytes=$(( LOCAL_DATA_BASE_DEC + GOLEM_DMA_SLOT_COUNT * mat_slot_bytes + GOLEM_DMA_SLOT_COUNT * vec_slot_bytes + out_scratch_bytes + out_tile_bytes + 0x40 + LOCAL_ALIGN ))
if (( GOLEM_GLOBAL_STRIDE_BYTES < required_global_stride_bytes )); then
	echo "[INFO] Expanding GOLEM_GLOBAL_STRIDE_BYTES from ${GOLEM_GLOBAL_STRIDE_BYTES} to ${required_global_stride_bytes} for local GM layout (slots=${GOLEM_DMA_SLOT_COUNT})"
	GOLEM_GLOBAL_STRIDE_BYTES=$required_global_stride_bytes
	GOLEM_GLOBAL_STRIDE_KB=$(( (GOLEM_GLOBAL_STRIDE_BYTES + 1023) / 1024 ))
fi

if [[ "$GOLEM_BIAS_ENABLE" != "0" && "$GOLEM_BIAS_ENABLE" != "1" ]]; then
	echo "[ERROR] --bias-enable 仅支持 0/1，收到: $GOLEM_BIAS_ENABLE" >&2
	exit 1
fi

if [[ "$GOLEM_MATMUL_DTYPE" != "int32" && "$GOLEM_MATMUL_DTYPE" != "fp32" ]]; then
	echo "[ERROR] --dtype 仅支持 int32|fp32，收到: $GOLEM_MATMUL_DTYPE" >&2
	exit 1
fi

if [[ "$GOLEM_MATMUL_DTYPE" == "int32" ]]; then
	if ! [[ "$GOLEM_BIAS_VALUE" =~ ^-?[0-9]+$ ]]; then
		echo "[ERROR] int32 模式下 --bias-value 必须为整数，收到: $GOLEM_BIAS_VALUE" >&2
		exit 1
	fi
elif ! [[ "$GOLEM_BIAS_VALUE" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
	echo "[ERROR] fp32 模式下 --bias-value 必须为数字，收到: $GOLEM_BIAS_VALUE" >&2
	exit 1
fi

if [[ -z "${GOLEM_ROCC_TYPE+x}" || "$GOLEM_ROCC_TYPE" == "golem.RoCCAnalogInt" ]]; then
	if [[ "$GOLEM_MATMUL_DTYPE" == "fp32" ]]; then
		GOLEM_ROCC_TYPE="golem.RoCCAnalogFloat"
	else
		GOLEM_ROCC_TYPE="golem.RoCCAnalogInt"
	fi
fi

if [[ -z "${GOLEM_ARRAY_TYPE+x}" || "$GOLEM_ARRAY_TYPE" == "golem.MVMIntArray" ]]; then
	if [[ "$GOLEM_MATMUL_DTYPE" == "fp32" ]]; then
		GOLEM_ARRAY_TYPE="golem.MVMFloatArray"
	else
		GOLEM_ARRAY_TYPE="golem.MVMIntArray"
	fi
fi

DERIVED_GEMM_K_TILES=$(( GOLEM_GEMM_K / GOLEM_GEMM_BLOCK_K ))
if [[ "$DERIVED_GEMM_K_TILES" -le 0 ]]; then
	echo "[ERROR] 派生 GEMM K tiles 必须为正整数，收到: $DERIVED_GEMM_K_TILES" >&2
	exit 1
fi

if ! [[ "$GOLEM_A_REUSE_N_TILES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_A_REUSE_N_TILES" -le 0 ]]; then
	echo "[ERROR] GOLEM_A_REUSE_N_TILES 必须为正整数，收到: $GOLEM_A_REUSE_N_TILES" >&2
	exit 1
fi

if ! [[ "$GOLEM_B_REUSE_M_TILES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_B_REUSE_M_TILES" -le 0 ]]; then
	echo "[ERROR] GOLEM_B_REUSE_M_TILES 必须为正整数，收到: $GOLEM_B_REUSE_M_TILES" >&2
	exit 1
fi

GOLEM_MEM_NODE_SIZE_BYTES_AUTO=0
if [[ "$GOLEM_MEM_NODE_SIZE_BYTES" == "auto" ]]; then
	GOLEM_MEM_NODE_SIZE_BYTES_AUTO=1
	read -r GOLEM_MEM_NODE_SIZE_BYTES GOLEM_AUTO_MEM_REQUIRED_BYTES GOLEM_AUTO_MEM_DATA_END_BYTES GOLEM_AUTO_MEM_BIAS_STRIDE_BYTES < <(
		derive_auto_mem_node_size \
			"$GOLEM_GEMM_M" "$GOLEM_GEMM_N" "$GOLEM_GEMM_K" \
			"$GOLEM_GEMM_BLOCK_M" "$GOLEM_GEMM_BLOCK_N" "$GOLEM_GEMM_BLOCK_K" \
			"$GOLEM_ELEM_BYTES" "$GOLEM_NUM_MEMORY_NODES" "$GOLEM_TOTAL_GROUPS" \
			"$GOLEM_TOTAL_GEMM_CORES" "$GOLEM_GROUP_MANAGER_ENABLE" \
			"$GOLEM_A_REUSE_N_TILES" "$GOLEM_B_REUSE_M_TILES"
	)
	echo "[INFO] Auto-selected GOLEM_MEM_NODE_SIZE_BYTES=$GOLEM_MEM_NODE_SIZE_BYTES (required=$GOLEM_AUTO_MEM_REQUIRED_BYTES, data_end=$GOLEM_AUTO_MEM_DATA_END_BYTES, bias_stride=$GOLEM_AUTO_MEM_BIAS_STRIDE_BYTES)"
fi

if ! [[ "$GOLEM_MEM_NODE_SIZE_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_MEM_NODE_SIZE_BYTES" -le 0 ]]; then
	echo "[ERROR] GOLEM_MEM_NODE_SIZE_BYTES 必须为正整数或 auto，收到: $GOLEM_MEM_NODE_SIZE_BYTES" >&2
	exit 1
fi

GOLEM_IDENTITY_BASE="$GOLEM_MEM_NODE_SIZE_BYTES"

if ! [[ "$GOLEM_DMA_WINDOW_K_TILES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_WINDOW_K_TILES" -le 0 ]]; then
	echo "[ERROR] GOLEM_DMA_WINDOW_K_TILES 必须为正整数，收到: $GOLEM_DMA_WINDOW_K_TILES" >&2
	exit 1
fi

DERIVED_WCP_RESIDENT_K_TILES=$GOLEM_DMA_WINDOW_K_TILES
if [[ "$DERIVED_WCP_RESIDENT_K_TILES" -gt "$DERIVED_GEMM_K_TILES" ]]; then
	DERIVED_WCP_RESIDENT_K_TILES=$DERIVED_GEMM_K_TILES
fi

if [[ "$GOLEM_A_REUSE_N_TILES" -gt 1 && "$GOLEM_B_REUSE_M_TILES" -gt 1 ]]; then
	if [[ "$GOLEM_A_REUSE_N_TILES" -ne "$GOLEM_B_REUSE_M_TILES" ]]; then
		echo "[ERROR] 当前 2D full-K 第一版要求 square reuse，收到 A=$GOLEM_A_REUSE_N_TILES B=$GOLEM_B_REUSE_M_TILES" >&2
		exit 1
	fi
	# The worker command processor uses slot-driven K windows for 2D reuse, so local slots
	# only need to hold active+prefetch resident windows rather than all K tiles at once.
	if [[ "${GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE:-1}" != "0" ]]; then
		wcp_prefetch_windows=$GOLEM_WCP_PREFETCH_WINDOWS
		if ! [[ "$wcp_prefetch_windows" =~ ^[0-9]+$ ]] || [[ "$wcp_prefetch_windows" -le 0 ]]; then
			echo "[ERROR] GOLEM_WCP_PREFETCH_WINDOWS 必须为正整数，收到: $GOLEM_WCP_PREFETCH_WINDOWS" >&2
			exit 1
		fi
		wcp_window_buffers=$(( wcp_prefetch_windows + 1 ))
		if [[ "$wcp_window_buffers" -lt 2 ]]; then
			wcp_window_buffers=2
		fi
		mat_resident_k=$(( GOLEM_DMA_SLOT_COUNT / (wcp_window_buffers * GOLEM_B_REUSE_M_TILES) ))
		vec_resident_k=$(( GOLEM_DMA_SLOT_COUNT / (wcp_window_buffers * GOLEM_A_REUSE_N_TILES) ))
		if [[ "$mat_resident_k" -le 0 || "$vec_resident_k" -le 0 ]]; then
			echo "[ERROR] 当前 2D K-window 要求 local_slot_count 至少容纳 active+prefetch residentK=1，收到 slots=$GOLEM_DMA_SLOT_COUNT A=$GOLEM_A_REUSE_N_TILES B=$GOLEM_B_REUSE_M_TILES prefetch_windows=$GOLEM_WCP_PREFETCH_WINDOWS buffers=$wcp_window_buffers" >&2
			exit 1
		fi
		slot_resident_k=$mat_resident_k
		if [[ "$vec_resident_k" -lt "$slot_resident_k" ]]; then
			slot_resident_k=$vec_resident_k
		fi
		if [[ "$DERIVED_WCP_RESIDENT_K_TILES" -gt "$slot_resident_k" ]]; then
			DERIVED_WCP_RESIDENT_K_TILES=$slot_resident_k
		fi
	else
		mat_slots_needed=$(( GOLEM_B_REUSE_M_TILES * DERIVED_GEMM_K_TILES ))
		vec_slots_needed=$(( GOLEM_A_REUSE_N_TILES * DERIVED_GEMM_K_TILES ))
		if [[ "$mat_slots_needed" -gt "$GOLEM_DMA_SLOT_COUNT" || "$vec_slots_needed" -gt "$GOLEM_DMA_SLOT_COUNT" ]]; then
			echo "[ERROR] 当前 2D full-K 要求 reuse_m*k_tiles 和 reuse_n*k_tiles <= local_slot_count($GOLEM_DMA_SLOT_COUNT)，收到 mat=$mat_slots_needed vec=$vec_slots_needed" >&2
			exit 1
		fi
	fi
elif [[ "$GOLEM_A_REUSE_N_TILES" -gt 1 && "$DERIVED_GEMM_K_TILES" -gt "$GOLEM_DMA_SLOT_COUNT" ]]; then
	echo "[ERROR] 当前 A-reuse 第一版要求 GEMM_K_TILES <= local_slot_count($GOLEM_DMA_SLOT_COUNT)，收到 K tiles=$DERIVED_GEMM_K_TILES" >&2
	exit 1
elif [[ "$GOLEM_B_REUSE_M_TILES" -gt 1 && "$DERIVED_GEMM_K_TILES" -gt "$GOLEM_DMA_SLOT_COUNT" ]]; then
	echo "[ERROR] 当前 B-reuse 第一版要求 GEMM_K_TILES <= local_slot_count($GOLEM_DMA_SLOT_COUNT)，收到 K tiles=$DERIVED_GEMM_K_TILES" >&2
	exit 1
fi

if [[ "$GOLEM_NUM_MEMORY_NODES" -lt 2 ]]; then
	echo "[ERROR] --num-mem-nodes 至少为 2（1 个 OS 节点 + 至少 1 个数据节点）" >&2
	exit 1
fi

DATA_MEM_NODES=$(( GOLEM_NUM_MEMORY_NODES - 1 ))
if [[ "$DATA_MEM_NODES" -gt "$GOLEM_MESH_DIM_X" ]]; then
	echo "[ERROR] 数据节点数($DATA_MEM_NODES) 不能大于 --mesh-dim-x($GOLEM_MESH_DIM_X)，否则无法在单独数据行按列对齐放置 4 个 HBM 节点" >&2
	exit 1
fi

if [[ "$GOLEM_GROUP_MANAGER_ENABLE" == "1" && "$GOLEM_CTRL_LINK_ENABLE" == "1" && "$GOLEM_TOTAL_CORES" == "16" && "$GOLEM_TOTAL_GEMM_CORES" == "16" ]]; then
	GOLEM_TOTAL_CORES=$((GOLEM_TOTAL_CORES + GOLEM_MESH_DIM_X))
	GOLEM_TOTAL_GEMM_CORES="$GOLEM_TOTAL_CORES"
	echo "[INFO] Group-manager control mode: auto-added one CPU row (num-cores=${GOLEM_TOTAL_CORES}, gemm-cores=${GOLEM_TOTAL_GEMM_CORES})"
fi

if [[ "$GOLEM_GROUP_MANAGER_ENABLE" == "1" && "$GOLEM_CTRL_LINK_ENABLE" == "1" ]]; then
	if [[ "$GOLEM_NUM_ARRAYS" == "1" ]]; then
		GOLEM_NUM_ARRAYS="$GOLEM_GEMM_BLOCK_N"
		echo "[INFO] Ctrl multi-array mode: align numArrays to block_n (numArrays=${GOLEM_NUM_ARRAYS})"
	fi

	if [[ "$GOLEM_NUM_ARRAYS" -lt "$GOLEM_GEMM_BLOCK_N" ]]; then
		echo "[ERROR] numArrays($GOLEM_NUM_ARRAYS) 必须 >= block_n($GOLEM_GEMM_BLOCK_N) 才能按 n_col 对齐 array_id" >&2
		exit 1
	fi

	required_active_workers=$((GOLEM_TOTAL_GROUPS * 4))
	actual_active_workers=$((GOLEM_TOTAL_GEMM_CORES - GOLEM_TOTAL_GROUPS))
	if [[ "$actual_active_workers" -ne "$required_active_workers" ]]; then
		echo "[ERROR] 当前 GroupCtrlEndpoint 管理器固定 4 个 worker slot (req_in_0..3)，需要 active_workers=${required_active_workers}。" >&2
		echo "        但当前 active_workers=${actual_active_workers} (gemm_cores=${GOLEM_TOTAL_GEMM_CORES}, groups=${GOLEM_TOTAL_GROUPS})。" >&2
		echo "        建议设置 GOLEM_TOTAL_GEMM_CORES=$((GOLEM_TOTAL_GROUPS * 5))（例如 groups=4 时为 20）。" >&2
		exit 1
	fi
fi

if [[ "$GOLEM_TOTAL_GEMM_CORES" -gt "$GOLEM_TOTAL_CORES" ]]; then
	echo "[ERROR] --gemm-cores($GOLEM_TOTAL_GEMM_CORES) 不能大于 --num-cores($GOLEM_TOTAL_CORES)" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_NODE_CREDITS" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_NODE_CREDITS" -le 0 ]]; then
	echo "[ERROR] GOLEM_DMA_NODE_CREDITS 必须为正整数，收到: $GOLEM_DMA_NODE_CREDITS" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_READ_RETRY_TICKS" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_READ_RETRY_TICKS" -le 0 ]]; then
	echo "[ERROR] --dma-read-retry-ticks 必须为正整数，收到: $GOLEM_DMA_READ_RETRY_TICKS" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_READ_MAX_RETRIES" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] --dma-read-max-retries 必须为非负整数，收到: $GOLEM_DMA_READ_MAX_RETRIES" >&2
	exit 1
fi

if [[ "$GOLEM_PROGRESS_HEARTBEAT" != "0" && "$GOLEM_PROGRESS_HEARTBEAT" != "1" ]]; then
	echo "[ERROR] --progress-heartbeat 仅支持 0/1，收到: $GOLEM_PROGRESS_HEARTBEAT" >&2
	exit 1
fi

if ! [[ "$GOLEM_PROGRESS_INTERVAL_CYCLES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_PROGRESS_INTERVAL_CYCLES" -le 0 ]]; then
	echo "[ERROR] --progress-interval-cycles 必须为正整数，收到: $GOLEM_PROGRESS_INTERVAL_CYCLES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_BURST_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_BURST_BYTES" -le 0 ]]; then
	echo "[ERROR] --dma-burst-bytes 必须为正整数，收到: $GOLEM_DMA_BURST_BYTES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_PANEL_CHUNK_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_PANEL_CHUNK_BYTES" -le 0 ]]; then
	echo "[ERROR] GOLEM_DMA_PANEL_CHUNK_BYTES 必须为正整数，收到: $GOLEM_DMA_PANEL_CHUNK_BYTES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_CREDIT_CHUNK_BYTES" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_CREDIT_CHUNK_BYTES" -le 0 ]]; then
	echo "[ERROR] GOLEM_DMA_CREDIT_CHUNK_BYTES 必须为正整数，收到: $GOLEM_DMA_CREDIT_CHUNK_BYTES" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_RESPONSE_DRAIN_LIMIT" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] GOLEM_DMA_RESPONSE_DRAIN_LIMIT 必须为非负整数，收到: $GOLEM_DMA_RESPONSE_DRAIN_LIMIT" >&2
	exit 1
fi

if ! [[ "$GOLEM_SCHED_ISSUE_BUDGET_PER_TICK" =~ ^[0-9]+$ ]] || [[ "$GOLEM_SCHED_ISSUE_BUDGET_PER_TICK" -le 0 ]]; then
	echo "[ERROR] GOLEM_SCHED_ISSUE_BUDGET_PER_TICK 必须为正整数，收到: $GOLEM_SCHED_ISSUE_BUDGET_PER_TICK" >&2
	exit 1
fi

if [[ -z "$GOLEM_DMA_NODE_CHUNK_CREDITS" ]]; then
	mat_sched_transfer_bytes=$(( GOLEM_GEMM_BLOCK_M * GOLEM_GEMM_BLOCK_K * ELEM_BYTES ))
	vec_sched_transfer_bytes=$(( GOLEM_GEMM_BLOCK_N * GOLEM_GEMM_BLOCK_K * ELEM_BYTES ))
	mat_transfer_credit_chunks=$(( (mat_sched_transfer_bytes + GOLEM_DMA_CREDIT_CHUNK_BYTES - 1) / GOLEM_DMA_CREDIT_CHUNK_BYTES ))
	vec_transfer_credit_chunks=$(( (vec_sched_transfer_bytes + GOLEM_DMA_CREDIT_CHUNK_BYTES - 1) / GOLEM_DMA_CREDIT_CHUNK_BYTES ))
	wcp_credit_window_buffers=$(( GOLEM_WCP_PREFETCH_WINDOWS + 1 ))
	if [[ "$wcp_credit_window_buffers" -lt 1 ]]; then
		wcp_credit_window_buffers=1
	fi
	if [[ "$GOLEM_GROUP_MANAGER_ENABLE" == "1" ]]; then
		derived_active_workers=$(( GOLEM_TOTAL_GEMM_CORES - GOLEM_TOTAL_GROUPS ))
	else
		derived_active_workers=$GOLEM_TOTAL_GEMM_CORES
	fi
	if [[ "$derived_active_workers" -le 0 ]]; then
		echo "[ERROR] 推导 DMA node chunk credit 需要 active worker > 0，收到 gemm_cores=$GOLEM_TOTAL_GEMM_CORES groups=$GOLEM_TOTAL_GROUPS group_manager=$GOLEM_GROUP_MANAGER_ENABLE" >&2
		exit 1
	fi
	per_worker_ab_credit_chunks=$(( DERIVED_WCP_RESIDENT_K_TILES * (GOLEM_B_REUSE_M_TILES * mat_transfer_credit_chunks + GOLEM_A_REUSE_N_TILES * vec_transfer_credit_chunks) ))
	GOLEM_DMA_NODE_CHUNK_CREDITS=$(( derived_active_workers * wcp_credit_window_buffers * per_worker_ab_credit_chunks ))
fi

if ! [[ "$GOLEM_DMA_NODE_CHUNK_CREDITS" =~ ^[0-9]+$ ]] || [[ "$GOLEM_DMA_NODE_CHUNK_CREDITS" -le 0 ]]; then
	echo "[ERROR] GOLEM_DMA_NODE_CHUNK_CREDITS 必须为正整数，收到: $GOLEM_DMA_NODE_CHUNK_CREDITS" >&2
	exit 1
fi

if ! [[ "$GOLEM_SCHED_SUBMIT_BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$GOLEM_SCHED_SUBMIT_BATCH_SIZE" -le 0 ]]; then
	echo "[ERROR] GOLEM_SCHED_SUBMIT_BATCH_SIZE 必须为正整数，收到: $GOLEM_SCHED_SUBMIT_BATCH_SIZE" >&2
	exit 1
fi

if ! [[ "$GOLEM_SCHED_DONE_BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$GOLEM_SCHED_DONE_BATCH_SIZE" -le 0 ]]; then
	echo "[ERROR] GOLEM_SCHED_DONE_BATCH_SIZE 必须为正整数，收到: $GOLEM_SCHED_DONE_BATCH_SIZE" >&2
	exit 1
fi

if ! [[ "$GOLEM_DMA_STAGGER_CYCLES" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] --dma-stagger-cycles 必须为非负整数，收到: $GOLEM_DMA_STAGGER_CYCLES" >&2
	exit 1
fi

if [[ "$GOLEM_DMA_OVERLAP" != "0" && "$GOLEM_DMA_OVERLAP" != "1" ]]; then
	echo "[ERROR] --dma-overlap 必须为 0 或 1，收到: $GOLEM_DMA_OVERLAP" >&2
	exit 1
fi

for flag_name in GOLEM_SKIP_TENSOR_GEN GOLEM_SKIP_HBM_GEN GOLEM_SKIP_BUILD; do
	flag_value="${!flag_name}"
	if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
		echo "[ERROR] $flag_name 必须为 0 或 1，收到: $flag_value" >&2
		exit 1
	fi
done

if [[ "$GOLEM_CTRL_OVERLAP_AB" != "0" && "$GOLEM_CTRL_OVERLAP_AB" != "1" ]]; then
	echo "[ERROR] GOLEM_CTRL_OVERLAP_AB 必须为 0 或 1，收到: $GOLEM_CTRL_OVERLAP_AB" >&2
	exit 1
fi

if [[ "$GOLEM_ARCH_SCRIPT" == "architecture/ncores_selfcom_dma.py" ]]; then
	if [[ "$GOLEM_CTRL_LINK_ENABLE" == "1" ]]; then
		GOLEM_ARCH_SCRIPT="architecture/ncores_selfcom_dma_ctrl.py"
	else
		GOLEM_ARCH_SCRIPT="architecture/archive/ncores_selfcom_dma.py"
	fi
fi

AUTO_STATS_SUBDIR="overlap${GOLEM_DMA_OVERLAP}/$RUN_ID"
DEFAULT_STATS_ROOT="$ARTIFACT_ROOT/stats"
DEFAULT_MVM_DUMP_ROOT="$ARTIFACT_ROOT/mvm_dumps"
DEFAULT_STDOUT_ROOT="$ARTIFACT_ROOT/stdout"
if [[ "$STATS_DIR_FROM_ENV" -eq 0 ]]; then
	STATS_DIR="$DEFAULT_STATS_ROOT/$AUTO_STATS_SUBDIR"
fi
if [[ "$STATS_FILE_FROM_ENV" -eq 0 ]]; then
	STATS_FILE="$STATS_DIR/stats_selfcom.txt"
fi
if [[ "$CORE_MAP_FILE_FROM_ENV" -eq 0 ]]; then
	CORE_MAP_FILE="$STATS_DIR/core_memory_map.csv"
fi
if [[ "$MVM_VERIFY_SUMMARY_FROM_ENV" -eq 0 ]]; then
	MVM_VERIFY_SUMMARY_FILE="$STATS_DIR/mvm_verify_summary.csv"
fi
HEATMAP_PREFIX="noc_m${GOLEM_GEMM_M}_n${GOLEM_GEMM_N}_k${GOLEM_GEMM_K}_${GOLEM_MATMUL_DTYPE}_ov${GOLEM_DMA_OVERLAP}_${RUN_ID}"
CONTRACT_RESOLVED_FILE="$ARTIFACT_ROOT/contracts/matmul_op_desc_resolved.json"
HBM_METADATA_FILE="$HBM_DIR/hbm_config.env"
BUILD_METADATA_FILE="$SCRIPT_DIR/small/mvm_noc_int_array/riscv64/test_noc_dma.build.env"
EXEC_SUMMARY_FILE="$STATS_DIR/execution_summary.csv"
EXEC_DEBUG_SUMMARY_FILE="$STATS_DIR/execution_debug_summary.csv"
DMA_SUMMARY_FILE="$STATS_DIR/dma_summary.csv"
NOC_SUMMARY_FILE="$STATS_DIR/noc_summary.csv"
MEMORY_SUMMARY_FILE="$STATS_DIR/memory_summary.csv"
NOC_LATENCY_SUMMARY_FILE="$STATS_DIR/noc_latency_summary.csv"
MEMORY_QUEUE_SUMMARY_FILE="$STATS_DIR/memory_queue_summary.csv"
CAUSAL_SUMMARY_FILE="$STATS_DIR/submit_ready_causal_summary.csv"
CAUSAL_TABLE_FILE="$STATS_DIR/submit_ready_causal_table.csv"
SCHED_PRESSURE_SUMMARY_FILE="$STATS_DIR/sched_pressure_summary.csv"
SCHED_PRESSURE_TABLE_FILE="$STATS_DIR/sched_pressure_table.csv"
NOC_HOTSPOT_SUMMARY_FILE="$STATS_DIR/noc_hotspot_summary.csv"
NOC_HOTSPOT_ROUTER_FILE="$STATS_DIR/noc_hotspot_router_table.csv"
NOC_HOTSPOT_PORT_FILE="$STATS_DIR/noc_hotspot_port_table.csv"
if [[ "$MVM_DUMP_DIR_FROM_ENV" -eq 0 ]]; then
	GOLEM_MVM_DUMP_DIR="$DEFAULT_MVM_DUMP_ROOT/$AUTO_STATS_SUBDIR"
fi
if [[ "$STDOUT_DIR_FROM_ENV" -eq 0 ]]; then
	STDOUT_DIR="$DEFAULT_STDOUT_ROOT/$AUTO_STATS_SUBDIR"
fi

if ! [[ "$GOLEM_MESH_DIM_X" =~ ^[0-9]+$ ]] || [[ "$GOLEM_MESH_DIM_X" -le 0 ]]; then
	echo "[ERROR] --mesh-dim-x 必须为正整数，收到: $GOLEM_MESH_DIM_X" >&2
	exit 1
fi

if [[ "$VERIFY_MVM" -eq 1 ]]; then
	GOLEM_MVM_DUMP_ENABLE=1
fi

	mkdir -p "$ARTIFACT_ROOT" "$LOG_DIR" "$HBM_DIR" "$STDOUT_DIR" "$STATS_DIR" "$GOLEM_MVM_DUMP_DIR" "$DRAMSIM_STATS_DIR"

if [[ "$GOLEM_MVM_DUMP_ENABLE" -eq 1 ]]; then
	rm -rf "$GOLEM_MVM_DUMP_DIR"
	mkdir -p "$GOLEM_MVM_DUMP_DIR"
fi

if [[ "$VERIFY_MVM" -eq 1 ]]; then
	rm -f "$MVM_VERIFY_SUMMARY_FILE"
fi

HBM_METADATA_KEYS=(
	GOLEM_GEMM_M
	GOLEM_GEMM_N
	GOLEM_GEMM_K
	GOLEM_GEMM_BLOCK_M
	GOLEM_GEMM_BLOCK_N
	GOLEM_GEMM_BLOCK_K
	GOLEM_MATMUL_DTYPE
	GOLEM_ARRAY_INPUT_SIZE
	GOLEM_ARRAY_OUTPUT_SIZE
	GOLEM_NUM_ARRAYS
	GOLEM_TOTAL_GROUPS
	GOLEM_TOTAL_CORES
	GOLEM_TOTAL_GEMM_CORES
	GOLEM_NUM_MEMORY_NODES
	GOLEM_MEM_NODE_SIZE_BYTES
	GOLEM_HBM_DUMP_OUTPUT
	GOLEM_GLOBAL_STRIDE_BYTES
	GOLEM_A_REUSE_N_TILES
	GOLEM_B_REUSE_M_TILES
	GOLEM_DMA_SLOT_COUNT
)

BUILD_METADATA_KEYS=(
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

if [[ "$LOG_FILE" = /* ]]; then
	LOG_PATH="$LOG_FILE"
else
	log_base="${LOG_FILE##*/}"
	log_ext=""
	log_stem="$log_base"
	if [[ "$log_base" == *.* ]]; then
		log_ext=".${log_base##*.}"
		log_stem="${log_base%.*}"
	fi
	LOG_PATH="$LOG_DIR/${log_stem}_${RUN_ID}${log_ext}"
fi

export GOLEM_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export GOLEM_HBM_DIR="$HBM_DIR"
export GOLEM_STATS_DIR="$STATS_DIR"
export GOLEM_STATS_FILE="$STATS_FILE"
export GOLEM_STDOUT_DIR="$STDOUT_DIR"
export GOLEM_RUN_ID="$RUN_ID"
export GOLEM_CORE_MAP_FILE="$CORE_MAP_FILE"
export GOLEM_PRINT_CORE_MAP="$PRINT_CORE_MAP"
export GOLEM_MVM_VERIFY_SUMMARY_FILE="$MVM_VERIFY_SUMMARY_FILE"
export GOLEM_VERIFY_MVM="$VERIFY_MVM"
export GOLEM_VERIFY_C="$VERIFY_C"
export GOLEM_DRAMSIM3_CONFIG
export VANADIS_PIPE_TRACE="${VANADIS_PIPE_TRACE:-$LOG_DIR/vanadis_trace.txt}"

export GOLEM_TOTAL_GROUPS
export GOLEM_ARRAY_INPUT_SIZE
export GOLEM_ARRAY_OUTPUT_SIZE
export GOLEM_NUM_ARRAYS
export GOLEM_TOTAL_CORES
export GOLEM_TOTAL_GEMM_CORES
export VANADIS_NUM_CORES="$GOLEM_TOTAL_CORES"
export GOLEM_NUM_MEMORY_NODES
export GOLEM_MEMORY_LAYOUT
export GOLEM_MEM_NODE_SIZE_BYTES
export GOLEM_IDENTITY_BASE
export GOLEM_HBM_DUMP_OUTPUT
export GOLEM_GLOBAL_STRIDE_KB
export GOLEM_GLOBAL_STRIDE_BYTES
export GOLEM_DMA_STAGGER_CYCLES
export GOLEM_DMA_OVERLAP
export GOLEM_CTRL_OVERLAP_AB
export GOLEM_GROUP_MANAGER_ENABLE
export GOLEM_CTRL_LINK_ENABLE
export GOLEM_A_REUSE_N_TILES
export GOLEM_B_REUSE_M_TILES
export GOLEM_DMA_NODE_CREDITS
export GOLEM_DMA_NODE_CHUNK_CREDITS
export GOLEM_DMA_PANEL_CHUNK_BYTES
export GOLEM_DMA_CREDIT_CHUNK_BYTES
export GOLEM_DMA_RESPONSE_DRAIN_LIMIT
export GOLEM_SCHED_ISSUE_BUDGET_PER_TICK
export GOLEM_DMA_SLOT_COUNT
export GOLEM_DMA_WINDOW_K_TILES
export GOLEM_GEMM_M
export GOLEM_GEMM_N
export GOLEM_GEMM_K
export GOLEM_ORIG_M
export GOLEM_ORIG_N
export GOLEM_ORIG_K
export GOLEM_GEMM_BLOCK_M
export GOLEM_GEMM_BLOCK_N
export GOLEM_GEMM_BLOCK_K
export GOLEM_BIAS_ENABLE
export GOLEM_BIAS_VALUE
export GOLEM_MATMUL_M="$GOLEM_GEMM_M"
export GOLEM_MATMUL_N="$GOLEM_GEMM_N"
export GOLEM_MATMUL_K="$GOLEM_GEMM_K"
export GOLEM_MATMUL_BLOCK_M="$GOLEM_GEMM_BLOCK_M"
export GOLEM_MATMUL_BLOCK_N="$GOLEM_GEMM_BLOCK_N"
export GOLEM_MATMUL_BLOCK_K="$GOLEM_GEMM_BLOCK_K"
export GOLEM_MATMUL_DTYPE
export GOLEM_GEMM_OUT_LAYOUT="colmajor_tile"
export GOLEM_MATMUL_LAYOUT="row_major"
export GOLEM_MATMUL_TRANSPOSE_A="0"
export GOLEM_MATMUL_TRANSPOSE_B="0"
export GOLEM_DMA_READ_RETRY_TICKS
export GOLEM_DMA_READ_MAX_RETRIES
export GOLEM_DMA_BURST_BYTES
export GOLEM_LATENCY_MVM_GM2IMAT
export GOLEM_LATENCY_MVM_GM2IVEC
export GOLEM_LATENCY_MVM_OVEC2GM
export GOLEM_ARRAY_NUM_CU
export GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE
export GOLEM_ARRAY_PIPELINE_DEPTH
export GOLEM_ARRAY_CLOCK
export GOLEM_MEMCTRL_CLOCK
export GOLEM_PROGRESS_HEARTBEAT
export GOLEM_PROGRESS_INTERVAL_CYCLES
export GOLEM_SST_ARGS
export GOLEM_SST_STAT_LOAD_LEVEL
export GOLEM_SST_ENABLE_ALL_STATS
export GOLEM_EXPORT_NOC_HEATMAPS
export GOLEM_SKIP_TENSOR_GEN
export GOLEM_SKIP_HBM_GEN
export GOLEM_SKIP_BUILD
export GOLEM_BENCH_QUIET_LOGS
export GOLEM_BENCH_DISABLE_SST_STATS
export GOLEM_ROCC_TYPE
export GOLEM_ARRAY_TYPE
export GOLEM_NOC_INPUT_BUF_SIZE
export GOLEM_NOC_OUTPUT_BUF_SIZE
export GOLEM_NOC_LINK_BW
export GOLEM_NOC_XBAR_BW
export GOLEM_NOC_FLIT_SIZE
export GOLEM_NOC_INTER_ROUTER_NO_CUT
export GOLEM_NOC_LOCAL_NO_CUT
export GOLEM_MESH_DIM_X
export GOLEM_GM_BUFFER_LENGTH
export GOLEM_ROCC_VERBOSE
export GOLEM_GM_VERBOSE
export GOLEM_GM_DUMP_DATA
export GOLEM_MVM_DUMP_ENABLE
export GOLEM_MVM_DUMP_DIR
export GOLEM_MVM_DUMP_MODE
export GOLEM_WCP_PREFETCH_WINDOWS
export GOLEM_SCHED_SUBMIT_BATCH_SIZE
export GOLEM_SCHED_DONE_BATCH_SIZE
export GOLEM_TENSOR_A_FILE="$TENSOR_A_FILE"
export GOLEM_TENSOR_B_FILE="$TENSOR_B_FILE"
export GOLEM_DUMP_C_FILE="$DUMP_C_FILE"
export GOLEM_TENSOR_SOURCE="$TENSOR_SOURCE"
export GOLEM_TENSOR_DIR="$TENSOR_DIR"

echo "[RUN] Configuration:"
if [[ -n "$AUTO_PRESET_FILE" ]]; then
	echo "  AUTO_PRESET_FILE=$AUTO_PRESET_FILE"
fi
echo "  GOLEM_TOTAL_GROUPS=$GOLEM_TOTAL_GROUPS"
echo "  GOLEM_ARRAY_INPUT_SIZE=$GOLEM_ARRAY_INPUT_SIZE"
echo "  GOLEM_ARRAY_OUTPUT_SIZE=$GOLEM_ARRAY_OUTPUT_SIZE"
echo "  GOLEM_NUM_ARRAYS=$GOLEM_NUM_ARRAYS"
echo "  GOLEM_TOTAL_CORES=$GOLEM_TOTAL_CORES"
echo "  GOLEM_TOTAL_GEMM_CORES=$GOLEM_TOTAL_GEMM_CORES"
echo "  GOLEM_NUM_MEMORY_NODES=$GOLEM_NUM_MEMORY_NODES"
echo "  GOLEM_MEMORY_LAYOUT=$GOLEM_MEMORY_LAYOUT"
echo "  GOLEM_MEM_NODE_SIZE_BYTES=$GOLEM_MEM_NODE_SIZE_BYTES"
echo "  GOLEM_IDENTITY_BASE=${GOLEM_IDENTITY_BASE:-<unset>}"
echo "  GOLEM_HBM_DUMP_OUTPUT=$GOLEM_HBM_DUMP_OUTPUT"
echo "  GOLEM_GLOBAL_STRIDE_KB=$GOLEM_GLOBAL_STRIDE_KB"
echo "  GOLEM_GLOBAL_STRIDE_BYTES=$GOLEM_GLOBAL_STRIDE_BYTES"
echo "  GOLEM_DMA_STAGGER_CYCLES=$GOLEM_DMA_STAGGER_CYCLES"
echo "  GOLEM_DMA_OVERLAP=$GOLEM_DMA_OVERLAP"
echo "  GOLEM_CTRL_OVERLAP_AB=$GOLEM_CTRL_OVERLAP_AB"
echo "  GOLEM_GROUP_MANAGER_ENABLE=$GOLEM_GROUP_MANAGER_ENABLE"
echo "  GOLEM_CTRL_LINK_ENABLE=$GOLEM_CTRL_LINK_ENABLE"
echo "  GOLEM_A_REUSE_N_TILES=$GOLEM_A_REUSE_N_TILES"
echo "  GOLEM_B_REUSE_M_TILES=$GOLEM_B_REUSE_M_TILES"
echo "  GOLEM_DMA_NODE_CREDITS(legacy transfer credits)=$GOLEM_DMA_NODE_CREDITS"
echo "  GOLEM_DMA_NODE_CHUNK_CREDITS=$GOLEM_DMA_NODE_CHUNK_CREDITS"
echo "  GOLEM_DMA_PANEL_CHUNK_BYTES=$GOLEM_DMA_PANEL_CHUNK_BYTES"
echo "  GOLEM_DMA_CREDIT_CHUNK_BYTES=$GOLEM_DMA_CREDIT_CHUNK_BYTES"
echo "  GOLEM_DMA_RESPONSE_DRAIN_LIMIT=$GOLEM_DMA_RESPONSE_DRAIN_LIMIT"
echo "  GOLEM_SCHED_ISSUE_BUDGET_PER_TICK=$GOLEM_SCHED_ISSUE_BUDGET_PER_TICK"
echo "  GOLEM_DMA_SLOT_COUNT=$GOLEM_DMA_SLOT_COUNT"
echo "  GOLEM_DMA_WINDOW_K_TILES=$GOLEM_DMA_WINDOW_K_TILES"
echo "  GOLEM_WCP_PREFETCH_WINDOWS=$GOLEM_WCP_PREFETCH_WINDOWS"
echo "  GOLEM_WCP_RESIDENT_K_TILES(derived)=$DERIVED_WCP_RESIDENT_K_TILES"
echo "  GOLEM_SCHED_SUBMIT_BATCH_SIZE=$GOLEM_SCHED_SUBMIT_BATCH_SIZE"
echo "  GOLEM_SCHED_DONE_BATCH_SIZE=$GOLEM_SCHED_DONE_BATCH_SIZE"
echo "  GOLEM_GEMM_K_TILES(derived)=$DERIVED_GEMM_K_TILES"
echo "  GOLEM_GEMM_M=$GOLEM_GEMM_M"
echo "  GOLEM_GEMM_N=$GOLEM_GEMM_N"
echo "  GOLEM_GEMM_K=$GOLEM_GEMM_K"
echo "  ORIG_M/N/K=$GOLEM_ORIG_M/$GOLEM_ORIG_N/$GOLEM_ORIG_K"
echo "  PADDED_M/N/K=$GOLEM_GEMM_M/$GOLEM_GEMM_N/$GOLEM_GEMM_K"
echo "  GOLEM_MATMUL_BLOCK_M=$GOLEM_MATMUL_BLOCK_M"
echo "  GOLEM_MATMUL_BLOCK_N=$GOLEM_MATMUL_BLOCK_N"
echo "  GOLEM_MATMUL_BLOCK_K=$GOLEM_MATMUL_BLOCK_K"
echo "  GOLEM_BIAS_ENABLE=$GOLEM_BIAS_ENABLE"
echo "  GOLEM_BIAS_VALUE=$GOLEM_BIAS_VALUE"
echo "  GOLEM_MATMUL_DTYPE=$GOLEM_MATMUL_DTYPE"
echo "  VANADIS_CPU_CLOCK=${VANADIS_CPU_CLOCK:-2.3GHz}"
echo "  GOLEM_ARRAY_CLOCK=$GOLEM_ARRAY_CLOCK"
echo "  GOLEM_MEMCTRL_CLOCK=$GOLEM_MEMCTRL_CLOCK"
echo "  GOLEM_MATMUL_LAYOUT=$GOLEM_MATMUL_LAYOUT"
echo "  GOLEM_DMA_READ_RETRY_TICKS=$GOLEM_DMA_READ_RETRY_TICKS"
echo "  GOLEM_DMA_READ_MAX_RETRIES=$GOLEM_DMA_READ_MAX_RETRIES"
echo "  GOLEM_DMA_BURST_BYTES=$GOLEM_DMA_BURST_BYTES"
echo "  GOLEM_LATENCY_MVM_GM2IMAT=$GOLEM_LATENCY_MVM_GM2IMAT"
echo "  GOLEM_LATENCY_MVM_GM2IVEC=$GOLEM_LATENCY_MVM_GM2IVEC"
echo "  GOLEM_LATENCY_MVM_OVEC2GM=$GOLEM_LATENCY_MVM_OVEC2GM"
echo "  GOLEM_ARRAY_NUM_CU=$GOLEM_ARRAY_NUM_CU"
echo "  GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE=$GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE"
echo "  GOLEM_ARRAY_PIPELINE_DEPTH=$GOLEM_ARRAY_PIPELINE_DEPTH"
echo "  GOLEM_PROGRESS_HEARTBEAT=$GOLEM_PROGRESS_HEARTBEAT"
echo "  GOLEM_PROGRESS_INTERVAL_CYCLES=$GOLEM_PROGRESS_INTERVAL_CYCLES"
echo "  GOLEM_SST_ARGS=${GOLEM_SST_ARGS:-<none>}"
echo "  GOLEM_SST_STAT_LOAD_LEVEL=$GOLEM_SST_STAT_LOAD_LEVEL"
echo "  GOLEM_SST_ENABLE_ALL_STATS=$GOLEM_SST_ENABLE_ALL_STATS"
echo "  GOLEM_EXPORT_NOC_HEATMAPS=$GOLEM_EXPORT_NOC_HEATMAPS"
echo "  GOLEM_SKIP_TENSOR_GEN=$GOLEM_SKIP_TENSOR_GEN"
echo "  GOLEM_SKIP_HBM_GEN=$GOLEM_SKIP_HBM_GEN"
echo "  GOLEM_SKIP_BUILD=$GOLEM_SKIP_BUILD"
echo "  GOLEM_BENCH_QUIET_LOGS=$GOLEM_BENCH_QUIET_LOGS"
echo "  GOLEM_BENCH_DISABLE_SST_STATS=$GOLEM_BENCH_DISABLE_SST_STATS"
echo "  GOLEM_ROCC_TYPE=$GOLEM_ROCC_TYPE"
echo "  GOLEM_ARRAY_TYPE=$GOLEM_ARRAY_TYPE"
echo "  GOLEM_NOC_INPUT_BUF_SIZE=$GOLEM_NOC_INPUT_BUF_SIZE"
echo "  GOLEM_NOC_OUTPUT_BUF_SIZE=$GOLEM_NOC_OUTPUT_BUF_SIZE"
echo "  GOLEM_NOC_LINK_BW=$GOLEM_NOC_LINK_BW"
echo "  GOLEM_NOC_XBAR_BW=$GOLEM_NOC_XBAR_BW"
echo "  GOLEM_NOC_FLIT_SIZE=$GOLEM_NOC_FLIT_SIZE"
echo "  GOLEM_NOC_INTER_ROUTER_NO_CUT=$GOLEM_NOC_INTER_ROUTER_NO_CUT"
echo "  GOLEM_NOC_LOCAL_NO_CUT=$GOLEM_NOC_LOCAL_NO_CUT"
echo "  GOLEM_MESH_DIM_X=$GOLEM_MESH_DIM_X"
echo "  GOLEM_GM_BUFFER_LENGTH=$GOLEM_GM_BUFFER_LENGTH"
echo "  GOLEM_ROCC_VERBOSE=$GOLEM_ROCC_VERBOSE"
echo "  GOLEM_GM_VERBOSE=$GOLEM_GM_VERBOSE"
echo "  GOLEM_GM_DUMP_DATA=$GOLEM_GM_DUMP_DATA"
echo "  GOLEM_DMA_TRACE=$GOLEM_DMA_TRACE"
echo "  GOLEM_REQUEST_SCHEDULER_TRACE=$GOLEM_REQUEST_SCHEDULER_TRACE"
echo "  GOLEM_LLSC_TRACE=$GOLEM_LLSC_TRACE"
echo "  GOLEM_MVM_DUMP_ENABLE=$GOLEM_MVM_DUMP_ENABLE"
echo "  GOLEM_MVM_DUMP_DIR=$GOLEM_MVM_DUMP_DIR"
echo "  GOLEM_MVM_DUMP_MODE=$GOLEM_MVM_DUMP_MODE"
echo "  GOLEM_HBM_DIR=$GOLEM_HBM_DIR"
echo "  GOLEM_STATS_DIR=$GOLEM_STATS_DIR"
echo "  GOLEM_STATS_FILE=$GOLEM_STATS_FILE"
echo "  GOLEM_PRINT_CORE_MAP=$GOLEM_PRINT_CORE_MAP"
echo "  GOLEM_CORE_MAP_FILE=$GOLEM_CORE_MAP_FILE"
echo "  GOLEM_VERIFY_MVM=$GOLEM_VERIFY_MVM"
echo "  GOLEM_VERIFY_C=$GOLEM_VERIFY_C"
echo "  GOLEM_TENSOR_SOURCE=$GOLEM_TENSOR_SOURCE"
echo "  GOLEM_TENSOR_DIR=$GOLEM_TENSOR_DIR"
echo "  GOLEM_MVM_VERIFY_SUMMARY_FILE=$GOLEM_MVM_VERIFY_SUMMARY_FILE"
echo "  GOLEM_TENSOR_A_FILE=${GOLEM_TENSOR_A_FILE:-<none>}"
echo "  GOLEM_TENSOR_B_FILE=${GOLEM_TENSOR_B_FILE:-<none>}"
echo "  GOLEM_BIAS_FILE=${GOLEM_BIAS_FILE:-<none>}"
echo "  GOLEM_POOL1_FILE=${GOLEM_POOL1_FILE:-<none>}"
echo "  GOLEM_CONV2_BPACK_FILE=${GOLEM_CONV2_BPACK_FILE:-<none>}"
echo "  GOLEM_CONV2_BIAS_FILE=${GOLEM_CONV2_BIAS_FILE:-<none>}"
echo "  GOLEM_FC1_WEIGHT_FILE=${GOLEM_FC1_WEIGHT_FILE:-<none>}"
echo "  GOLEM_FC1_BIAS_FILE=${GOLEM_FC1_BIAS_FILE:-<none>}"
echo "  GOLEM_FC2_WEIGHT_FILE=${GOLEM_FC2_WEIGHT_FILE:-<none>}"
echo "  GOLEM_FC2_BIAS_FILE=${GOLEM_FC2_BIAS_FILE:-<none>}"
echo "  GOLEM_FC3_WEIGHT_FILE=${GOLEM_FC3_WEIGHT_FILE:-<none>}"
echo "  GOLEM_FC3_BIAS_FILE=${GOLEM_FC3_BIAS_FILE:-<none>}"
echo "  GOLEM_DUMP_C_FILE=${GOLEM_DUMP_C_FILE:-<none>}"
	echo "  GOLEM_STDOUT_DIR=$GOLEM_STDOUT_DIR"
	echo "  GOLEM_RUN_ID=$GOLEM_RUN_ID"
	echo "  GOLEM_DRAMSIM3_OUT_DIR=$DRAMSIM_STATS_DIR"
	echo "  LOG_PATH=$LOG_PATH"

GEN_HBM_CMD=(python3 tools/gen_hbm_init.py)
SST_CMD=(sst)
if [[ -n "$GOLEM_SST_ARGS" ]]; then
	read -r -a SST_EXTRA_ARGS <<< "$GOLEM_SST_ARGS"
	SST_CMD+=("${SST_EXTRA_ARGS[@]}")
fi
SST_CMD+=("$GOLEM_ARCH_SCRIPT")
SST_CMD_DISPLAY="$(printf '%q ' "${SST_CMD[@]}")"
SST_CMD_DISPLAY="${SST_CMD_DISPLAY% }"
SAMPLE_TENSOR_CMD=()
if [[ "$TENSOR_SOURCE" == "sample" ]]; then
	SAMPLE_TENSOR_CMD=(python3 tools/gen_sample_tensors.py --m "$GOLEM_GEMM_M" --n "$GOLEM_GEMM_N" --k "$GOLEM_GEMM_K" --dtype "$GOLEM_MATMUL_DTYPE" --a-out "$TENSOR_A_FILE" --b-out "$TENSOR_B_FILE")
fi
if [[ "$TENSOR_SOURCE" == "file" || "$TENSOR_SOURCE" == "sample" ]]; then
	GEN_HBM_CMD+=(--a-file "$TENSOR_A_FILE")
	GEN_HBM_CMD+=(--b-file "$TENSOR_B_FILE")
fi
if [[ -n "${GOLEM_BIAS_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--bias-file "$GOLEM_BIAS_FILE")
fi
if [[ -n "${GOLEM_POOL1_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--pool1-file "$GOLEM_POOL1_FILE")
fi
if [[ -n "${GOLEM_CONV2_BPACK_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--conv2-bpack-file "$GOLEM_CONV2_BPACK_FILE")
fi
if [[ -n "${GOLEM_CONV2_BIAS_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--conv2-bias-file "$GOLEM_CONV2_BIAS_FILE")
fi
if [[ -n "${GOLEM_FC1_WEIGHT_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc1-weight-file "$GOLEM_FC1_WEIGHT_FILE")
fi
if [[ -n "${GOLEM_FC1_BIAS_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc1-bias-file "$GOLEM_FC1_BIAS_FILE")
fi
if [[ -n "${GOLEM_FC2_WEIGHT_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc2-weight-file "$GOLEM_FC2_WEIGHT_FILE")
fi
if [[ -n "${GOLEM_FC2_BIAS_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc2-bias-file "$GOLEM_FC2_BIAS_FILE")
fi
if [[ -n "${GOLEM_FC3_WEIGHT_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc3-weight-file "$GOLEM_FC3_WEIGHT_FILE")
fi
if [[ -n "${GOLEM_FC3_BIAS_FILE:-}" ]]; then
	GEN_HBM_CMD+=(--fc3-bias-file "$GOLEM_FC3_BIAS_FILE")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
	echo "[DRY-RUN] Commands to execute:"
	print_progress_bar 0 "准备中"
	if [[ "$TENSOR_SOURCE" == "sample" ]]; then
		if [[ "$GOLEM_SKIP_TENSOR_GEN" -eq 1 || "$GOLEM_SKIP_HBM_GEN" -eq 1 ]]; then
			echo "  skip sample tensor generation"
		else
			echo "  ${SAMPLE_TENSOR_CMD[*]}"
		fi
	fi
	if [[ "$GOLEM_SKIP_HBM_GEN" -eq 1 ]]; then
		echo "  skip HBM generation and reuse $HBM_DIR/hbm_init_node*.bin"
	else
		echo "  ${GEN_HBM_CMD[*]}"
	fi
	if [[ "$GOLEM_SKIP_BUILD" -eq 1 ]]; then
		echo "  skip build and reuse small/mvm_noc_int_array/riscv64/test_noc_dma"
	else
		echo "  (cd small/mvm_noc_int_array && make clean ARCH=riscv64 && make ARCH=riscv64 CFLAGS=\"-DGOLEM_ARRAY_INPUT_SIZE=${GOLEM_ARRAY_INPUT_SIZE} -DGOLEM_ARRAY_OUTPUT_SIZE=${GOLEM_ARRAY_OUTPUT_SIZE} -DGOLEM_TOTAL_GROUPS=${GOLEM_TOTAL_GROUPS} -DGOLEM_TOTAL_CORES=${GOLEM_TOTAL_CORES} -DGOLEM_TOTAL_GEMM_CORES=${GOLEM_TOTAL_GEMM_CORES} -DGOLEM_NUM_ARRAYS=${GOLEM_NUM_ARRAYS} -DGOLEM_NUM_MEMORY_NODES=${GOLEM_NUM_MEMORY_NODES} -DGOLEM_MEM_NODE_SIZE_BYTES=${GOLEM_MEM_NODE_SIZE_BYTES} -DGOLEM_GLOBAL_STRIDE_BYTES=${GOLEM_GLOBAL_STRIDE_BYTES} -DGOLEM_DMA_STAGGER_CYCLES=${GOLEM_DMA_STAGGER_CYCLES} -DGOLEM_DMA_OVERLAP=${GOLEM_DMA_OVERLAP} -DGOLEM_CTRL_OVERLAP_AB=${GOLEM_CTRL_OVERLAP_AB} -DGOLEM_GROUP_MANAGER_ENABLE=${GOLEM_GROUP_MANAGER_ENABLE} -DGOLEM_CTRL_LINK_ENABLE=${GOLEM_CTRL_LINK_ENABLE} -DGOLEM_A_REUSE_N_TILES=${GOLEM_A_REUSE_N_TILES} -DGOLEM_B_REUSE_M_TILES=${GOLEM_B_REUSE_M_TILES} -DGOLEM_DMA_SLOT_COUNT=${GOLEM_DMA_SLOT_COUNT} -DGOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=${GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE:-0} -DGOLEM_BIAS_ENABLE=${GOLEM_BIAS_ENABLE} -DGOLEM_BIAS_VALUE=${GOLEM_BIAS_VALUE}\")"
	fi
	echo "  GOLEM_MESH_DIM_X=$GOLEM_MESH_DIM_X GOLEM_MVM_DUMP_ENABLE=$GOLEM_MVM_DUMP_ENABLE GOLEM_MVM_DUMP_DIR=$GOLEM_MVM_DUMP_DIR GOLEM_MVM_DUMP_MODE=$GOLEM_MVM_DUMP_MODE GOLEM_CTRL_LINK_ENABLE=$GOLEM_CTRL_LINK_ENABLE $SST_CMD_DISPLAY > $LOG_PATH 2>&1"
	echo "  mv dramsim3*.{txt,json} $DRAMSIM_STATS_DIR/ (if generated)"
	echo "  mv stdout-*/stderr-* $STDOUT_DIR/"
	if [[ "$VERIFY_MVM" -eq 1 ]]; then
		echo "  python3 stats/verify_mvm_dumps.py --dump-dir $GOLEM_MVM_DUMP_DIR --summary $MVM_VERIFY_SUMMARY_FILE"
	fi
	if [[ -n "$DUMP_C_FILE" ]]; then
		echo "  python3 tools/unpack_c_from_hbm.py --out-file $DUMP_C_FILE"
	fi
	if [[ "$VERIFY_C" -eq 1 ]]; then
		echo "  python3 verify/verify_c_against_golden.py --dtype $GOLEM_MATMUL_DTYPE --a-file $TENSOR_A_FILE --b-file $TENSOR_B_FILE --c-file $DUMP_C_FILE --m $GOLEM_GEMM_M --n $GOLEM_GEMM_N --k $GOLEM_GEMM_K --bias-enable $GOLEM_BIAS_ENABLE --bias-value $GOLEM_BIAS_VALUE"
	fi
	echo "  python3 stats/extract_latency_csv.py --log $LOG_PATH --log-dir $STDOUT_DIR --summary $EXEC_SUMMARY_FILE"
	echo "  python3 stats/extract_dma_read_stats_csv.py --log $LOG_PATH --log-dir $STDOUT_DIR --summary $DMA_SUMMARY_FILE"
	echo "  python3 stats/extract_noc_summary_csv.py --input-file $GOLEM_STATS_FILE --link-bw $GOLEM_NOC_LINK_BW --output $NOC_SUMMARY_FILE"
	echo "  python3 stats/extract_noc_hotspot_csv.py --input-file $GOLEM_STATS_FILE --link-bw $GOLEM_NOC_LINK_BW --summary $NOC_HOTSPOT_SUMMARY_FILE --router-table $NOC_HOTSPOT_ROUTER_FILE --port-table $NOC_HOTSPOT_PORT_FILE"
	echo "  python3 stats/extract_noc_latency_summary_csv.py --log $LOG_PATH --log-dir $STDOUT_DIR --output $NOC_LATENCY_SUMMARY_FILE"
	echo "  python3 stats/extract_memory_summary_csv.py --json $DRAMSIM_STATS_DIR/dramsim3.json --txt $DRAMSIM_STATS_DIR/dramsim3.txt --output $MEMORY_SUMMARY_FILE"
	echo "  python3 stats/extract_memory_queue_summary_csv.py --log $LOG_PATH --log-dir $STDOUT_DIR --output $MEMORY_QUEUE_SUMMARY_FILE"
	echo "  python3 stats/extract_submit_ready_causal_csv.py --log $LOG_PATH --log-dir $STDOUT_DIR --noc-latency-summary $NOC_LATENCY_SUMMARY_FILE --memory-queue-summary $MEMORY_QUEUE_SUMMARY_FILE --sched-clock 1GHz --memory-clock $GOLEM_MEMCTRL_CLOCK --summary $CAUSAL_SUMMARY_FILE --table $CAUSAL_TABLE_FILE"
	echo "  append run summary -> $RUN_SUMMARY_CSV"
	if [[ "$PRINT_CORE_MAP" -eq 1 ]]; then
		echo "  core map file: $CORE_MAP_FILE"
	fi
	exit 0
fi

print_progress_bar 0 "开始"

echo "[1/4] Preparing HBM init files..."
if [[ "$GOLEM_SKIP_HBM_GEN" -eq 1 ]]; then
	missing_hbm=0
	for ((node_idx = 1; node_idx < GOLEM_NUM_MEMORY_NODES; node_idx++)); do
		if [[ ! -f "$HBM_DIR/hbm_init_node${node_idx}.bin" ]]; then
			echo "[ERROR] GOLEM_SKIP_HBM_GEN=1 但缺少 $HBM_DIR/hbm_init_node${node_idx}.bin" >&2
			missing_hbm=1
		fi
	done
	if [[ "$missing_hbm" -ne 0 ]]; then
		exit 1
	fi
	if [[ -f "$HBM_METADATA_FILE" ]]; then
		validate_metadata_file "$HBM_METADATA_FILE" "HBM" "${HBM_METADATA_KEYS[@]}"
	else
		validate_hbm_contract_fallback "$CONTRACT_RESOLVED_FILE"
	fi
	echo "[INFO] Reusing existing HBM backing files in $HBM_DIR"
elif [[ "$TENSOR_SOURCE" == "sample" && "$GOLEM_SKIP_TENSOR_GEN" -eq 1 ]]; then
	expected_a_bytes=$(( GOLEM_GEMM_M * GOLEM_GEMM_K * GOLEM_ELEM_BYTES ))
	expected_b_bytes=$(( GOLEM_GEMM_K * GOLEM_GEMM_N * GOLEM_ELEM_BYTES ))
	validate_tensor_file_size "$TENSOR_A_FILE" "A" "$expected_a_bytes"
	validate_tensor_file_size "$TENSOR_B_FILE" "B" "$expected_b_bytes"
	echo "[INFO] Reusing sample tensors: A=$TENSOR_A_FILE B=$TENSOR_B_FILE"
	"${GEN_HBM_CMD[@]}"
	write_metadata_file "$HBM_METADATA_FILE" "${HBM_METADATA_KEYS[@]}"
elif [[ "$TENSOR_SOURCE" == "sample" ]]; then
	"${SAMPLE_TENSOR_CMD[@]}"
	"${GEN_HBM_CMD[@]}"
	write_metadata_file "$HBM_METADATA_FILE" "${HBM_METADATA_KEYS[@]}"
else
	"${GEN_HBM_CMD[@]}"
	write_metadata_file "$HBM_METADATA_FILE" "${HBM_METADATA_KEYS[@]}"
fi
print_progress_bar 1 "HBM 初始化完成"

echo "[2/4] Building test binary..."
if [[ "$GOLEM_SKIP_BUILD" -eq 1 ]]; then
	if [[ ! -x "$SCRIPT_DIR/small/mvm_noc_int_array/riscv64/test_noc_dma" ]]; then
		echo "[ERROR] GOLEM_SKIP_BUILD=1 但缺少可执行文件 small/mvm_noc_int_array/riscv64/test_noc_dma" >&2
		exit 1
	fi
	validate_metadata_file "$BUILD_METADATA_FILE" "test_noc_dma build" "${BUILD_METADATA_KEYS[@]}"
	echo "[INFO] Reusing existing test binary: small/mvm_noc_int_array/riscv64/test_noc_dma"
else
	pushd small/mvm_noc_int_array >/dev/null
	make clean ARCH=riscv64
	make ARCH=riscv64 CFLAGS="-DGOLEM_ARRAY_INPUT_SIZE=${GOLEM_ARRAY_INPUT_SIZE} -DGOLEM_ARRAY_OUTPUT_SIZE=${GOLEM_ARRAY_OUTPUT_SIZE} -DGOLEM_TOTAL_GROUPS=${GOLEM_TOTAL_GROUPS} -DGOLEM_TOTAL_CORES=${GOLEM_TOTAL_CORES} -DGOLEM_TOTAL_GEMM_CORES=${GOLEM_TOTAL_GEMM_CORES} -DGOLEM_NUM_ARRAYS=${GOLEM_NUM_ARRAYS} -DGOLEM_NUM_MEMORY_NODES=${GOLEM_NUM_MEMORY_NODES} -DGOLEM_MEM_NODE_SIZE_BYTES=${GOLEM_MEM_NODE_SIZE_BYTES} -DGOLEM_GLOBAL_STRIDE_BYTES=${GOLEM_GLOBAL_STRIDE_BYTES} -DGOLEM_GEMM_M=${GOLEM_GEMM_M} -DGOLEM_GEMM_N=${GOLEM_GEMM_N} -DGOLEM_GEMM_K=${GOLEM_GEMM_K} -DGOLEM_GEMM_BLOCK_M=${GOLEM_GEMM_BLOCK_M} -DGOLEM_GEMM_BLOCK_N=${GOLEM_GEMM_BLOCK_N} -DGOLEM_GEMM_BLOCK_K=${GOLEM_GEMM_BLOCK_K} -DGOLEM_DMA_STAGGER_CYCLES=${GOLEM_DMA_STAGGER_CYCLES} -DGOLEM_DMA_OVERLAP=${GOLEM_DMA_OVERLAP} -DGOLEM_CTRL_OVERLAP_AB=${GOLEM_CTRL_OVERLAP_AB} -DGOLEM_GROUP_MANAGER_ENABLE=${GOLEM_GROUP_MANAGER_ENABLE} -DGOLEM_CTRL_LINK_ENABLE=${GOLEM_CTRL_LINK_ENABLE} -DGOLEM_A_REUSE_N_TILES=${GOLEM_A_REUSE_N_TILES} -DGOLEM_B_REUSE_M_TILES=${GOLEM_B_REUSE_M_TILES} -DGOLEM_DMA_SLOT_COUNT=${GOLEM_DMA_SLOT_COUNT} -DGOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=${GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE:-0} -DGOLEM_BIAS_ENABLE=${GOLEM_BIAS_ENABLE} -DGOLEM_BIAS_VALUE=${GOLEM_BIAS_VALUE}"
	popd >/dev/null
	write_metadata_file "$BUILD_METADATA_FILE" "${BUILD_METADATA_KEYS[@]}"
fi
print_progress_bar 2 "编译完成"

echo "[3/4] Running SST..."
rm -f "$SCRIPT_DIR"/dramsim3.txt "$SCRIPT_DIR"/dramsim3.json "$SCRIPT_DIR"/dramsim3epoch.json
setsid "${SST_CMD[@]}" > "$LOG_PATH" 2>&1 &
SST_PID=$!
SST_FATAL=0
SST_COMPLETE=0
progress=50
bar_width=40
phase="架构初始化"
while sst_pid_is_running "$SST_PID"; do
	if sst_log_has_fatal "$LOG_PATH"; then
		SST_FATAL=1
		phase="检测到致命错误"
		break
	fi
	if sst_log_has_complete "$LOG_PATH"; then
		SST_COMPLETE=1
		phase="仿真完成"
		break
	fi
	if supports_fancy_output; then
		info="$(estimate_sst_progress_info "$LOG_PATH")"
		new_progress="${info%%|*}"
		new_phase="${info#*|}"
		if [[ "$new_progress" -gt "$progress" ]]; then
			progress="$new_progress"
		fi
		phase="$new_phase"
		bar="$(render_inline_progress_bar "$progress" "$bar_width")"
		printf "\r[3/4] Running SST... %s %3d%% [%s]\033[K" "$bar" "$progress" "$phase"
	fi
	sleep 0.2
done

if [[ "$SST_FATAL" -eq 1 ]]; then
	terminate_sst_process_tree "$SST_PID"
	wait_for_sst_exit "$SST_PID" 2 || true
	if supports_fancy_output; then
		printf "\n"
	fi
	print_sst_failure_context "$LOG_PATH"
	exit 1
fi

if [[ "$SST_COMPLETE" -eq 1 ]]; then
	if ! wait_for_sst_exit "$SST_PID" 2; then
		echo "[WARN] SST completed in log but process is still exiting; continue to post-processing."
	fi
	if supports_fancy_output; then
		bar="$(render_inline_progress_bar 100 "$bar_width")"
		printf "\r[3/4] Running SST... %s 100%% [阶段:完成]\033[K\n" "$bar"
	fi
else
	if ! wait "$SST_PID"; then
		if supports_fancy_output; then
			printf "\n"
		fi
		print_sst_failure_context "$LOG_PATH"
		exit 1
	fi

	if supports_fancy_output; then
		bar="$(render_inline_progress_bar 100 "$bar_width")"
		printf "\r[3/4] Running SST... %s 100%% [阶段:完成]\033[K\n" "$bar"
	fi
fi

shopt -s nullglob
for f in "$SCRIPT_DIR"/dramsim3.txt "$SCRIPT_DIR"/dramsim3.json "$SCRIPT_DIR"/dramsim3epoch.json; do
	if [[ -f "$f" ]]; then
		mv "$f" "$DRAMSIM_STATS_DIR/"
	fi
done
shopt -u nullglob
print_progress_bar 3 "SST 运行完成"

shopt -s nullglob
for f in "$SCRIPT_DIR"/stdout-* "$SCRIPT_DIR"/stderr-*; do
	mv "$f" "$STDOUT_DIR/"
done
shopt -u nullglob

if [[ "$VERIFY_MVM" -eq 1 ]]; then
	echo "[4/4] Verifying MVM dumps..."
	python3 "$SCRIPT_DIR/stats/verify_mvm_dumps.py" --dump-dir "$GOLEM_MVM_DUMP_DIR" --summary "$MVM_VERIFY_SUMMARY_FILE"
fi

if [[ -n "$DUMP_C_FILE" ]]; then
	echo "[4/4] Unpacking C tensor from HBM output..."
	python3 "$SCRIPT_DIR/tools/unpack_c_from_hbm.py" --out-file "$DUMP_C_FILE"
fi

if [[ "$VERIFY_C" -eq 1 ]]; then
	echo "[4/4] Verifying C against A@B golden..."
	python3 "$SCRIPT_DIR/verify/verify_c_against_golden.py" \
		--dtype "$GOLEM_MATMUL_DTYPE" \
		--a-file "$TENSOR_A_FILE" \
		--b-file "$TENSOR_B_FILE" \
		--c-file "$DUMP_C_FILE" \
		--m "$GOLEM_GEMM_M" \
		--n "$GOLEM_GEMM_N" \
		--k "$GOLEM_GEMM_K" \
		--bias-enable "$GOLEM_BIAS_ENABLE" \
		--bias-value "$GOLEM_BIAS_VALUE"
fi

echo "[4/4] Exporting execution summary..."
if [[ -f "$SCRIPT_DIR/stats/extract_latency_csv.py" ]]; then
	if ! python3 "$SCRIPT_DIR/stats/extract_latency_csv.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--debug-summary "$EXEC_DEBUG_SUMMARY_FILE" \
		--stats-file "$GOLEM_STATS_FILE" \
		--gemm-m "$GOLEM_GEMM_M" \
		--gemm-n "$GOLEM_GEMM_N" \
		--gemm-k "$GOLEM_GEMM_K" \
		--active-worker-cores "$((GOLEM_TOTAL_GEMM_CORES - GOLEM_TOTAL_GROUPS))" \
		--num-arrays "$GOLEM_NUM_ARRAYS" \
		--array-num-cu "$GOLEM_ARRAY_NUM_CU" \
		--mac-per-cu-per-cycle "$GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE" \
		--summary "$EXEC_SUMMARY_FILE"; then
		echo "[WARN] Execution summary extraction failed. Ensure test.log contains LATENCY(cycles) lines."
	fi
else
	echo "[WARN] stats/extract_latency_csv.py not found, skip execution summary extraction."
fi

echo "[4/4] Exporting DMA summary..."
if [[ -f "$SCRIPT_DIR/stats/extract_dma_read_stats_csv.py" ]]; then
	if ! python3 "$SCRIPT_DIR/stats/extract_dma_read_stats_csv.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--summary "$DMA_SUMMARY_FILE"; then
		echo "[WARN] DMA summary extraction failed. Ensure log contains 'DMA READ stats' lines."
	fi
else
	echo "[WARN] stats/extract_dma_read_stats_csv.py not found, skip DMA summary extraction."
fi

echo "[4/4] Exporting NoC summary..."
if [[ -f "$SCRIPT_DIR/stats/extract_noc_summary_csv.py" ]]; then
	if [[ -f "$GOLEM_STATS_FILE" ]]; then
		if ! python3 "$SCRIPT_DIR/stats/extract_noc_summary_csv.py" \
			--input-file "$GOLEM_STATS_FILE" \
			--link-bw "$GOLEM_NOC_LINK_BW" \
			--output "$NOC_SUMMARY_FILE"; then
			echo "[WARN] NoC summary extraction failed."
		fi
	else
		echo "[WARN] $GOLEM_STATS_FILE not found, skip NoC summary extraction."
	fi
else
	echo "[WARN] stats/extract_noc_summary_csv.py not found, skip NoC summary extraction."
fi

echo "[4/4] Exporting NoC hotspot tables..."
if [[ -f "$SCRIPT_DIR/stats/extract_noc_hotspot_csv.py" ]]; then
	if [[ -f "$GOLEM_STATS_FILE" ]]; then
		if ! python3 "$SCRIPT_DIR/stats/extract_noc_hotspot_csv.py" \
			--input-file "$GOLEM_STATS_FILE" \
			--link-bw "$GOLEM_NOC_LINK_BW" \
			--summary "$NOC_HOTSPOT_SUMMARY_FILE" \
			--router-table "$NOC_HOTSPOT_ROUTER_FILE" \
			--port-table "$NOC_HOTSPOT_PORT_FILE"; then
			echo "[WARN] NoC hotspot extraction failed."
		fi
	else
		echo "[WARN] $GOLEM_STATS_FILE not found, skip NoC hotspot extraction."
	fi
else
	echo "[WARN] stats/extract_noc_hotspot_csv.py not found, skip NoC hotspot extraction."
fi

echo "[4/4] Exporting NoC latency summary..."
if [[ -f "$SCRIPT_DIR/stats/extract_noc_latency_summary_csv.py" ]]; then
	if ! python3 "$SCRIPT_DIR/stats/extract_noc_latency_summary_csv.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--output "$NOC_LATENCY_SUMMARY_FILE"; then
		echo "[WARN] NoC latency summary extraction failed."
	fi
else
	echo "[WARN] stats/extract_noc_latency_summary_csv.py not found, skip NoC latency summary extraction."
fi

if [[ "${GOLEM_EXPORT_NOC_HEATMAPS:-0}" -eq 1 ]]; then
	echo "[4/4] Exporting NoC heatmaps..."
	if [[ -f "$SCRIPT_DIR/stats/visualize_noc_routers.py" ]]; then
		if [[ -f "$GOLEM_STATS_FILE" ]]; then
			if ! python3 "$SCRIPT_DIR/stats/visualize_noc_routers.py" \
				--input-file "$GOLEM_STATS_FILE" \
				--output-dir "$STATS_DIR" \
				--output-prefix "$HEATMAP_PREFIX"; then
				echo "[WARN] NoC heatmap generation failed."
			fi
		else
			echo "[WARN] $GOLEM_STATS_FILE not found, skip NoC heatmap generation."
		fi
	else
		echo "[WARN] stats/visualize_noc_routers.py not found, skip NoC heatmap generation."
	fi
fi

echo "[4/4] Exporting memory summary..."
if [[ -f "$SCRIPT_DIR/stats/extract_memory_summary_csv.py" ]]; then
	if [[ -f "$DRAMSIM_STATS_DIR/dramsim3.json" && -f "$DRAMSIM_STATS_DIR/dramsim3.txt" ]]; then
		if ! python3 "$SCRIPT_DIR/stats/extract_memory_summary_csv.py" \
			--json "$DRAMSIM_STATS_DIR/dramsim3.json" \
			--txt "$DRAMSIM_STATS_DIR/dramsim3.txt" \
			--output "$MEMORY_SUMMARY_FILE"; then
			echo "[WARN] Memory summary extraction failed."
		fi
	else
		echo "[WARN] DRAMSim3 stats files not found, skip memory summary extraction."
	fi
else
	echo "[WARN] stats/extract_memory_summary_csv.py not found, skip memory summary extraction."
fi

echo "[4/4] Exporting memory queue summary..."
if [[ -f "$SCRIPT_DIR/stats/extract_memory_queue_summary_csv.py" ]]; then
	if ! python3 "$SCRIPT_DIR/stats/extract_memory_queue_summary_csv.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--output "$MEMORY_QUEUE_SUMMARY_FILE"; then
		echo "[WARN] Memory queue summary extraction failed."
	fi
else
	echo "[WARN] stats/extract_memory_queue_summary_csv.py not found, skip memory queue summary extraction."
fi

echo "[4/4] Exporting submit-ready causal breakdown..."
if [[ -f "$SCRIPT_DIR/stats/extract_submit_ready_causal_csv.py" ]]; then
	if ! python3 "$SCRIPT_DIR/stats/extract_submit_ready_causal_csv.py" \
		--log "$LOG_PATH" \
		--log-dir "$STDOUT_DIR" \
		--noc-latency-summary "$NOC_LATENCY_SUMMARY_FILE" \
		--memory-queue-summary "$MEMORY_QUEUE_SUMMARY_FILE" \
		--sched-clock "1GHz" \
		--memory-clock "$GOLEM_MEMCTRL_CLOCK" \
		--summary "$CAUSAL_SUMMARY_FILE" \
		--table "$CAUSAL_TABLE_FILE"; then
		echo "[WARN] submit-ready causal breakdown extraction failed."
	fi
else
	echo "[WARN] stats/extract_submit_ready_causal_csv.py not found, skip causal breakdown extraction."
fi

echo "[4/4] Exporting scheduler pressure summary..."
if [[ -f "$SCRIPT_DIR/stats/extract_sched_pressure_csv.py" ]]; then
    if ! python3 "$SCRIPT_DIR/stats/extract_sched_pressure_csv.py" \
        --log "$LOG_PATH" \
        --log-dir "$STDOUT_DIR" \
        --summary "$SCHED_PRESSURE_SUMMARY_FILE" \
        --table "$SCHED_PRESSURE_TABLE_FILE"; then
        echo "[WARN] scheduler pressure extraction failed."
    fi
else
    echo "[WARN] stats/extract_sched_pressure_csv.py not found, skip scheduler pressure extraction."
fi

echo "[4/4] Appending run summary CSV..."
RUN_START_EPOCH="$RUN_START_EPOCH" RUN_SUMMARY_CSV="$RUN_SUMMARY_CSV" LOG_PATH="$LOG_PATH" STATS_DIR="$STATS_DIR" python3 - <<'PY'
import csv
import datetime as dt
import math
import os
import re
from pathlib import Path

run_start = int(os.environ.get("RUN_START_EPOCH", "0") or "0")
run_end = int(dt.datetime.now().timestamp())
wall_time_sec = max(0, run_end - run_start)

log_path = Path(os.environ["LOG_PATH"])
stats_dir = Path(os.environ["STATS_DIR"])
execution_summary = stats_dir / "execution_summary.csv"
dma_summary = stats_dir / "dma_summary.csv"
noc_summary = stats_dir / "noc_summary.csv"
memory_summary = stats_dir / "memory_summary.csv"
noc_latency_summary = stats_dir / "noc_latency_summary.csv"
memory_queue_summary = stats_dir / "memory_queue_summary.csv"
causal_summary = stats_dir / "submit_ready_causal_summary.csv"
noc_hotspot_summary = stats_dir / "noc_hotspot_summary.csv"
out_csv = Path(os.environ["RUN_SUMMARY_CSV"])


def read_metric_value_csv(path: Path):
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    header = rows[0]
    out = {}
    if header == ["metric", "value"]:
        for row in rows[1:]:
            if len(row) >= 2:
                out[row[0]] = row[1]
        return out
    if header[:2] == ["metric", "mean"]:
        for row in rows[1:]:
            if len(row) >= 2 and row[0].endswith("_share_pct"):
                out[row[0]] = row[1]
            if len(row) >= 2:
                out[f"{row[0]}_mean"] = row[1]
            if len(row) >= 4:
                out[f"{row[0]}_p95"] = row[3]
            if len(row) >= 5:
                out[f"{row[0]}_min"] = row[4]
            if len(row) >= 6:
                out[f"{row[0]}_max"] = row[5]
            if len(row) >= 7:
                out[f"{row[0]}_sum"] = row[6]
        return out
    return {}


def _to_int(value, default=0):
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _read_ini_number(path: Path, key: str, default=0.0):
    if not path.exists():
        return default
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^;#\s]+)")
    for line in path.read_text(errors="ignore").splitlines():
        m = pat.match(line)
        if not m:
            continue
        return _to_float(m.group(1), default)
    return default


def _hbm_read_cmd_bytes(default=64.0):
    if "GOLEM_HBM_READ_BYTES_PER_CMD" in os.environ:
        return _to_float(os.environ.get("GOLEM_HBM_READ_BYTES_PER_CMD"), default)
    config = Path(os.environ.get("GOLEM_DRAMSIM3_CONFIG", ""))
    bl = _read_ini_number(config, "BL", 4.0)
    bus_width = _read_ini_number(config, "bus_width", 128.0)
    if bl > 0 and bus_width > 0:
        return bl * bus_width / 8.0
    return default


def _hbm_tccd_l_cycles(default=3.0):
    if "GOLEM_HBM_TCCD_L_CYCLES" in os.environ:
        return _to_float(os.environ.get("GOLEM_HBM_TCCD_L_CYCLES"), default)
    config = Path(os.environ.get("GOLEM_DRAMSIM3_CONFIG", ""))
    return _read_ini_number(config, "tCCD_L", default)


def compute_hbm_readonly_tccdl_util_pct(execution, memory, backend_window_cycles, backend_active_cycles):
    elem_bytes = 4
    m = _to_int(os.environ.get("GOLEM_GEMM_M", ""))
    n = _to_int(os.environ.get("GOLEM_GEMM_N", ""))
    k = _to_int(os.environ.get("GOLEM_GEMM_K", ""))
    block_m = _to_int(os.environ.get("GOLEM_GEMM_BLOCK_M", ""))
    block_n = _to_int(os.environ.get("GOLEM_GEMM_BLOCK_N", ""))
    block_k = _to_int(os.environ.get("GOLEM_GEMM_BLOCK_K", ""))
    a_reuse_n = max(1, _to_int(os.environ.get("GOLEM_A_REUSE_N_TILES", "1"), 1))
    b_reuse_m = max(1, _to_int(os.environ.get("GOLEM_B_REUSE_M_TILES", "1"), 1))
    if min(m, n, k, block_m, block_n, block_k) <= 0:
        return "", "", "", "", "", "", ""

    m_tiles = math.ceil(m / block_m)
    n_tiles = math.ceil(n / block_n)
    k_tiles = math.ceil(k / block_k)
    m_groups = math.ceil(m_tiles / b_reuse_m)
    n_groups = math.ceil(n_tiles / a_reuse_n)
    useful_read_bytes = (
        m_groups
        * n_groups
        * k_tiles
        * elem_bytes
        * (
            b_reuse_m * block_m * block_k
            + a_reuse_n * block_n * block_k
        )
    )

    data_nodes = max(1, _to_int(os.environ.get("GOLEM_NUM_MEMORY_NODES", "1"), 1) - 1)
    channel_count = max(1, _to_int(memory.get("channel_count", "16"), 16))
    bytes_per_read_cmd = _hbm_read_cmd_bytes(64.0)
    tccd_l_cycles = _hbm_tccd_l_cycles(3.0)
    if bytes_per_read_cmd <= 0 or tccd_l_cycles <= 0:
        return "", "", "", "", str(useful_read_bytes), "", ""

    roofline_bytes_per_cycle = data_nodes * channel_count * bytes_per_read_cmd / tccd_l_cycles
    gemm_system_cycles = _to_float(execution.get("gemm_system_latency_cycles", ""), 0.0)
    system_pressure_pct = ""
    if gemm_system_cycles > 0:
        system_pressure_pct = f"{100.0 * useful_read_bytes / (gemm_system_cycles * roofline_bytes_per_cycle):.6f}"

    worker_total_cycles = _to_float(execution.get("total_cycles", ""), 0.0)
    worker_pressure_pct = ""
    if worker_total_cycles > 0:
        worker_pressure_pct = f"{100.0 * useful_read_bytes / (worker_total_cycles * roofline_bytes_per_cycle):.6f}"

    backend_window_util_pct = ""
    if backend_window_cycles > 0:
        backend_window_util_pct = f"{100.0 * useful_read_bytes / (backend_window_cycles * roofline_bytes_per_cycle):.6f}"

    backend_util_pct = ""
    if backend_active_cycles > 0:
        backend_util_pct = f"{100.0 * useful_read_bytes / (backend_active_cycles * roofline_bytes_per_cycle):.6f}"

    return (
        system_pressure_pct,
        backend_window_util_pct,
        backend_util_pct,
        worker_pressure_pct,
        str(useful_read_bytes),
        f"{roofline_bytes_per_cycle:.6f}",
        f"{tccd_l_cycles:.6f}",
    )

simulated_time = ""
backend_read_window_cycles = 0
backend_read_active_cycles = 0
if log_path.exists():
    sim_pat = re.compile(r"Simulation is complete, simulated time:\s*(.+)$")
    backend_window_pat = re.compile(r"DRAMSIM3_BACKEND_READ_SERVICE_WINDOW_GLOBAL\b.*\bwindow_cycles=(\d+)")
    backend_active_pat = re.compile(r"DRAMSIM3_BACKEND_READ_ACTIVE_WINDOW_GLOBAL\b.*\bactive_cycles=(\d+)")
    for line in log_path.read_text(errors="ignore").splitlines():
        m = sim_pat.search(line)
        if m:
            simulated_time = m.group(1).strip()
        m = backend_window_pat.search(line)
        if m:
            backend_read_window_cycles = _to_int(m.group(1), 0)
        m = backend_active_pat.search(line)
        if m:
            backend_read_active_cycles = _to_int(m.group(1), 0)

execution = read_metric_value_csv(execution_summary)
dma = read_metric_value_csv(dma_summary)
noc = read_metric_value_csv(noc_summary)
memory = read_metric_value_csv(memory_summary)
noc_latency = read_metric_value_csv(noc_latency_summary)
memory_queue = read_metric_value_csv(memory_queue_summary)
causal = read_metric_value_csv(causal_summary)
hotspot = read_metric_value_csv(noc_hotspot_summary)
hbm_util_pct, hbm_backend_window_util_pct, hbm_backend_active_util_pct, hbm_worker_pressure_pct, hbm_useful_read_bytes, hbm_roofline_bpc, hbm_tccd_l_cycles = compute_hbm_readonly_tccdl_util_pct(
    execution, memory, backend_read_window_cycles, backend_read_active_cycles
)

record = {
    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    "run_id": os.environ.get("GOLEM_RUN_ID", ""),
    "log_file": str(log_path),
    "overlap": f"overlap{os.environ.get('GOLEM_DMA_OVERLAP', '0')}",
    "array_input_size": os.environ.get("GOLEM_ARRAY_INPUT_SIZE", ""),
    "array_output_size": os.environ.get("GOLEM_ARRAY_OUTPUT_SIZE", ""),
    "gemm_m": os.environ.get("GOLEM_GEMM_M", ""),
    "gemm_n": os.environ.get("GOLEM_GEMM_N", ""),
    "gemm_k": os.environ.get("GOLEM_GEMM_K", ""),
    "block_m": os.environ.get("GOLEM_GEMM_BLOCK_M", ""),
    "block_n": os.environ.get("GOLEM_GEMM_BLOCK_N", ""),
    "block_k": os.environ.get("GOLEM_GEMM_BLOCK_K", ""),
    "bias_enable": os.environ.get("GOLEM_BIAS_ENABLE", ""),
    "bias_value": os.environ.get("GOLEM_BIAS_VALUE", ""),
    "num_cores": os.environ.get("GOLEM_TOTAL_CORES", ""),
    "gemm_cores": os.environ.get("GOLEM_TOTAL_GEMM_CORES", ""),
    "num_mem_nodes": os.environ.get("GOLEM_NUM_MEMORY_NODES", ""),
    "mem_node_size_bytes": os.environ.get("GOLEM_MEM_NODE_SIZE_BYTES", ""),
    "hbm_dump_output": os.environ.get("GOLEM_HBM_DUMP_OUTPUT", ""),
    "dma_node_credits": os.environ.get("GOLEM_DMA_NODE_CREDITS", ""),
    "dma_node_chunk_credits": os.environ.get("GOLEM_DMA_NODE_CHUNK_CREDITS", ""),
    "dma_panel_chunk_bytes": os.environ.get("GOLEM_DMA_PANEL_CHUNK_BYTES", ""),
    "dma_credit_chunk_bytes": os.environ.get("GOLEM_DMA_CREDIT_CHUNK_BYTES", ""),
    "wcp_prefetch_windows": os.environ.get("GOLEM_WCP_PREFETCH_WINDOWS", ""),
    "submit_batch_size": os.environ.get("GOLEM_SCHED_SUBMIT_BATCH_SIZE", ""),
    "done_batch_size": os.environ.get("GOLEM_SCHED_DONE_BATCH_SIZE", ""),
    "dma_retry_ticks": os.environ.get("GOLEM_DMA_READ_RETRY_TICKS", ""),
    "dma_burst_bytes": os.environ.get("GOLEM_DMA_BURST_BYTES", ""),
    "dma_response_drain_limit": os.environ.get("GOLEM_DMA_RESPONSE_DRAIN_LIMIT", ""),
    "dma_stagger_cycles": os.environ.get("GOLEM_DMA_STAGGER_CYCLES", ""),
    "ctrl_overlap_ab": os.environ.get("GOLEM_CTRL_OVERLAP_AB", ""),
    "noc_link_bw": os.environ.get("GOLEM_NOC_LINK_BW", ""),
    "noc_xbar_bw": os.environ.get("GOLEM_NOC_XBAR_BW", ""),
    "noc_flit_size": os.environ.get("GOLEM_NOC_FLIT_SIZE", ""),
    "dirctrl_highlink_bw": os.environ.get("GOLEM_DIRCTRL_HIGHLINK_BW", ""),
    "wall_time_sec": str(wall_time_sec),
     "simulated_time": simulated_time,
     "exec_total_cycles": execution.get("total_cycles", ""),
     "gemm_system_latency_cycles": execution.get("gemm_system_latency_cycles", ""),
     "gemm_system_start_cycle": execution.get("gemm_system_start_cycle", ""),
     "gemm_system_end_cycle": execution.get("gemm_system_end_cycle", ""),
     "exec_avg_throughput_ops_per_cycle": execution.get("avg_throughput_ops_per_cycle", ""),
     "exec_system_avg_throughput_ops_per_cycle": execution.get("system_avg_throughput_ops_per_cycle", ""),
     "exec_peak_throughput_ops_per_cycle": execution.get("peak_throughput_ops_per_cycle", ""),
     "exec_array_utilization_pct": execution.get("array_utilization_pct", ""),
     "exec_system_array_utilization_pct": execution.get("system_array_utilization_pct", ""),
     "exec_worker_avg_array_efficiency_pct": execution.get("worker_avg_array_efficiency_pct", ""),
     "exec_worker_p95_total_cycles": execution.get("worker_p95_total_cycles", ""),
     "exec_worker_max_total_cycles": execution.get("worker_max_total_cycles", ""),
     "exec_breakdown_compute_active_time": execution.get("compute_active_time", ""),
     "exec_breakdown_prefetch_wait_time": execution.get("prefetch_wait_time", ""),
     "exec_breakdown_writeback_wait_time": execution.get("writeback_wait_time", ""),
     "exec_breakdown_control_other_time": execution.get("control_other_time", ""),
     "debug_sched_protocol_mean": execution.get("debug_sched_protocol_mean", ""),
     "debug_group_wait_mean": execution.get("debug_group_wait_mean", ""),
     "dma_timeout_retry_sum": dma.get("timeout_retry_sum", ""),
    "dma_read_issue_count_sum": dma.get("read_issue_count_sum", ""),
    "dma_write_issue_count_sum": dma.get("write_issue_count_sum", ""),
    "dma_read_bytes_total_sum": dma.get("read_bytes_total_sum", ""),
    "dma_write_bytes_total_sum": dma.get("write_bytes_total_sum", ""),
    "dma_write_timeout_retry_sum": dma.get("write_timeout_retry_sum", ""),
    "dma_completion_sum": dma.get("completion_sum", ""),
    "dma_write_completion_sum": dma.get("write_completion_sum", ""),
    "dma_wait_count_sum": dma.get("wait_count_sum", ""),
    "dma_avg_rtt_cycles_mean": dma.get("avg_rtt_cycles_mean", ""),
    "dma_max_rtt_cycles_p95": dma.get("max_rtt_cycles_p95", ""),
    "dma_strict_rtt_samples_sum": dma.get("strict_rtt_samples_sum", ""),
    "dma_strict_rtt_cycles_sum": dma.get("strict_rtt_cycles_sum_sum", ""),
    "dma_strict_avg_rtt_cycles_mean": dma.get("strict_avg_rtt_cycles_mean", ""),
    "dma_strict_max_rtt_cycles_max": dma.get("strict_max_rtt_cycles_max", ""),
    "dma_strict_e2e_rtt_samples_sum": dma.get("strict_e2e_rtt_samples_sum", ""),
    "dma_strict_e2e_rtt_cycles_sum": dma.get("strict_e2e_rtt_cycles_sum_sum", ""),
    "dma_strict_avg_e2e_rtt_cycles_mean": dma.get("strict_avg_e2e_rtt_cycles_mean", ""),
    "dma_strict_max_e2e_rtt_cycles_max": dma.get("strict_max_e2e_rtt_cycles_max", ""),
    "noc_total_xbar_stalls": noc.get("total_xbar_stalls", ""),
    "noc_hotspot_top5pct_port_util_pct": noc.get("hotspot_top5pct_port_util_pct", ""),
    "noc_max_port_util_pct": noc.get("max_port_util_pct", ""),
    "noc_total_output_port_stalls": hotspot.get("total_output_port_stalls", ""),
    "noc_hotspot_top1_router": hotspot.get("top1_router", ""),
    "noc_hotspot_top1_router_xbar_share_pct": hotspot.get("top1_router_xbar_share_pct", ""),
    "noc_hotspot_top2_router": hotspot.get("top2_router", ""),
    "noc_hotspot_top2_router_xbar_share_pct": hotspot.get("top2_router_xbar_share_pct", ""),
    "noc_hotspot_top3_router": hotspot.get("top3_router", ""),
    "noc_hotspot_top3_router_xbar_coverage_pct": hotspot.get("top3_router_xbar_coverage_pct", ""),
    "noc_hotspot_top1_port_router": hotspot.get("top1_port_router", ""),
    "noc_hotspot_top1_port": hotspot.get("top1_port", ""),
    "noc_hotspot_top1_port_xbar_share_pct": hotspot.get("top1_port_xbar_share_pct", ""),
    "noc_avg_packet_latency_ns": noc_latency.get("noc_avg_packet_latency_ns", ""),
    "noc_p99_packet_latency_ns": noc_latency.get("noc_p99_packet_latency_ns", ""),
    "memory_avg_read_latency_cycles": memory.get("mem_avg_read_latency_cycles", ""),
    "memory_p95_read_latency_bucket_cycles": memory.get("mem_p95_read_latency_bucket_cycles", ""),
    "memory_read_tail_ge_100_pct": memory.get("mem_read_tail_ge_100_pct", ""),
    "hbm_utilization_pct": hbm_util_pct,
    "hbm_pressure_vs_gemm_system_pct": hbm_util_pct,
    "hbm_backend_service_window_utilization_pct": hbm_backend_window_util_pct,
    "hbm_backend_active_utilization_pct": hbm_backend_active_util_pct,
    "hbm_backend_read_window_cycles": str(backend_read_window_cycles) if backend_read_window_cycles > 0 else "",
    "hbm_backend_read_active_cycles": str(backend_read_active_cycles) if backend_read_active_cycles > 0 else "",
    "hbm_pressure_vs_worker_avg_pct": hbm_worker_pressure_pct,
    "hbm_useful_read_bytes": hbm_useful_read_bytes,
    "hbm_tccdl_roofline_bytes_per_cycle": hbm_roofline_bpc,
    "hbm_tccd_l_cycles": hbm_tccd_l_cycles,
    "hbm_channel_bandwidth_imbalance": memory.get("hbm_channel_bandwidth_imbalance", ""),
    "memory_queue_delay_avg_cycles": memory_queue.get("memory_queue_delay_avg_cycles", ""),
    "memory_queue_delay_p99_cycles": memory_queue.get("memory_queue_delay_p99_cycles", ""),
    "memory_backend_read_latency_avg_cycles": memory_queue.get("memory_backend_read_latency_avg_cycles", ""),
    "memory_backend_read_latency_p99_cycles": memory_queue.get("memory_backend_read_latency_p99_cycles", ""),
    "causal_model_source": causal.get("causal_model_source", ""),
    "causal_memnic_cycle_scale": causal.get("memnic_cycle_scale", ""),
    "causal_event_full_match_count": causal.get("event_full_match_count", ""),
    "causal_event_invalid_order_count": causal.get("event_invalid_order_count", ""),
    "causal_issue_to_pending_mat_mean_cycles": causal.get("causal_issue_to_pending_mat_mean_cycles", ""),
    "causal_issue_to_pending_vec_mean_cycles": causal.get("causal_issue_to_pending_vec_mean_cycles", ""),
    "causal_forward_to_memnic_mean_cycles": causal.get("causal_forward_to_memnic_mean_cycles", ""),
    "causal_memory_service_mean_cycles": causal.get("causal_memory_service_mean_cycles", ""),
    "causal_return_path_mat_mean_cycles": causal.get("causal_return_path_mat_mean_cycles", ""),
    "causal_return_path_vec_mean_cycles": causal.get("causal_return_path_vec_mean_cycles", ""),
    "causal_return_path_mat_mean_share_pct": causal.get("causal_return_path_mat_mean_share_pct", ""),
    "causal_return_path_vec_mean_share_pct": causal.get("causal_return_path_vec_mean_share_pct", ""),
    "causal_issue_to_pending_mat_p95_cycles": causal.get("causal_issue_to_pending_mat_p95_cycles", ""),
    "causal_issue_to_pending_vec_p95_cycles": causal.get("causal_issue_to_pending_vec_p95_cycles", ""),
    "causal_forward_to_memnic_p95_cycles": causal.get("causal_forward_to_memnic_p95_cycles", ""),
    "causal_memory_service_p95_cycles": causal.get("causal_memory_service_p95_cycles", ""),
    "causal_return_path_mat_p95_cycles": causal.get("causal_return_path_mat_p95_cycles", ""),
    "causal_return_path_vec_p95_cycles": causal.get("causal_return_path_vec_p95_cycles", ""),
}

out_csv.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(record.keys())
write_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
if out_csv.exists() and out_csv.stat().st_size > 0:
    with out_csv.open(newline="") as f:
        existing_rows = list(csv.reader(f))
    existing_header = existing_rows[0] if existing_rows else []
    if existing_header != fieldnames:
        backup_path = out_csv.with_name(out_csv.stem + ".legacy_backup.csv")
        out_csv.replace(backup_path)
        print(f"[INFO] run summary header changed; archived legacy file to {backup_path}")
        write_header = True
with out_csv.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        w.writeheader()
    w.writerow(record)

print(f"[OK] run summary appended: {out_csv}")
PY

echo "[4/4] Done. Log: $LOG_PATH"
echo "[4/4] Stdout/Stderr dir: $STDOUT_DIR"
echo "[4/4] HBM dir: $HBM_DIR"
echo "[4/4] DRAMSim3 stats dir: $DRAMSIM_STATS_DIR"
echo "[4/4] Key stats files:"
echo "  $EXEC_SUMMARY_FILE"
echo "  $DMA_SUMMARY_FILE"
echo "  $NOC_SUMMARY_FILE"
echo "  $NOC_HOTSPOT_SUMMARY_FILE"
echo "  $NOC_HOTSPOT_ROUTER_FILE"
echo "  $NOC_HOTSPOT_PORT_FILE"
echo "  $NOC_LATENCY_SUMMARY_FILE"
if [[ "${GOLEM_EXPORT_NOC_HEATMAPS:-0}" -eq 1 ]]; then
	echo "  $STATS_DIR/${HEATMAP_PREFIX}_heatmap.png"
	echo "  $STATS_DIR/${HEATMAP_PREFIX}_send_packets_heatmap.png"
	echo "  $STATS_DIR/${HEATMAP_PREFIX}_stalls_line.png"
	echo "  $STATS_DIR/${HEATMAP_PREFIX}_combined.png"
fi
echo "  $MEMORY_SUMMARY_FILE"
echo "  $MEMORY_QUEUE_SUMMARY_FILE"
echo "  $CAUSAL_SUMMARY_FILE"
echo "  $CAUSAL_TABLE_FILE"
echo "  $RUN_SUMMARY_CSV"
echo "  $GOLEM_STATS_FILE"
echo "  $DRAMSIM_STATS_DIR/dramsim3.json"
echo "  $DRAMSIM_STATS_DIR/dramsim3.txt"
if [[ "$PRINT_CORE_MAP" -eq 1 ]]; then
	echo "  $CORE_MAP_FILE"
fi
if [[ "$VERIFY_MVM" -eq 1 ]]; then
	echo "  $MVM_VERIFY_SUMMARY_FILE"
fi
if [[ -n "$DUMP_C_FILE" ]]; then
	echo "  $DUMP_C_FILE"
fi
print_progress_bar 4 "全部完成"
echo "Tip: grep -E \"\[VERIFY\]|Stage完成|ERROR|FAILED\" $LOG_PATH"
