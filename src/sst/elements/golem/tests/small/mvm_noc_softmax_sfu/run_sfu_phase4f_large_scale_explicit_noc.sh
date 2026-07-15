#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHILD_RUNNER="$SCRIPT_DIR/run_sfu_unified_job_distributed_scaling.sh"
COLLECTOR="$SCRIPT_DIR/plot_sfu_phase4f_large_scale.py"
VERIFIER="$SCRIPT_DIR/verify_softmax_sfu_against_golden.py"
SCHEMA="phase4f-parent-v1"

error() {
	echo "[PHASE4F][ERROR] $*" >&2
}

require_unset_or_equal() {
	local name="$1"
	local expected="$2"
	if [[ -n "${!name+x}" && "${!name}" != "$expected" ]]; then
		error "$name conflicts with canonical value: expected='$expected' actual='${!name}'"
		exit 2
	fi
	printf -v "$name" '%s' "$expected"
	export "$name"
}

require_unset() {
	local name="$1"
	if [[ -n "${!name+x}" ]]; then
		error "$name is controlled by the Phase 4F parent runner and must be unset"
		exit 2
	fi
}

require_unset_or_equal GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT explicit_noc
require_unset_or_equal GOLEM_SFU_VN_SWEEP 1
require_unset_or_equal GOLEM_SFU_REDUCTION_VN 0
require_unset_or_equal GOLEM_DMA_RESPONSE_VN 0
require_unset_or_equal GOLEM_NOC_LINK_BW 1200GB/s
require_unset_or_equal GOLEM_NOC_XBAR_BW 1200GB/s
require_unset_or_equal GOLEM_DIRCTRL_HIGHLINK_BW 1200GB/s
require_unset_or_equal GOLEM_NOC_INPUT_BUF_SIZE 512KB
require_unset_or_equal GOLEM_NOC_OUTPUT_BUF_SIZE 512KB
require_unset_or_equal GOLEM_NOC_FLIT_SIZE 128B
require_unset_or_equal GOLEM_GM_BUFFER_LENGTH 1024KB
require_unset_or_equal GOLEM_NOC_INTER_ROUTER_NO_CUT 0
require_unset_or_equal GOLEM_NOC_LOCAL_NO_CUT 0
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_CHUNK_ELEMS 256
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_STAGING_ROWS 4
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_JOB_ROWS 4
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_RETRY_TICKS 1024
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_MAX_RETRIES 8

require_unset GOLEM_SFU_DISTRIBUTED_PIPELINE_ARGS
require_unset GOLEM_SWEEP_ROOT
require_unset GOLEM_DRY_RUN_SWEEP
require_unset GOLEM_STOP_ON_FAIL
require_unset GOLEM_SFU_DISTRIBUTED_POINT_LIST
require_unset GOLEM_TIMEOUT_512
require_unset GOLEM_TIMEOUT_1024

DRY_RUN="${GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN:-0}"
STOP_ON_FAIL="${GOLEM_PHASE4F_LARGE_SCALE_STOP_ON_FAIL:-1}"
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
	error "GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN must be 0 or 1; got '$DRY_RUN'"
	exit 2
fi
if [[ "$STOP_ON_FAIL" != 0 && "$STOP_ON_FAIL" != 1 ]]; then
	error "GOLEM_PHASE4F_LARGE_SCALE_STOP_ON_FAIL must be 0 or 1; got '$STOP_ON_FAIL'"
	exit 2
fi

DEFAULT_POINTS="16:512:16:16 16:1024:16:16 16:2048:16:16 16:4096:16:16 16:4096:4:4 16:4096:8:8 64:4096:16:16 256:4096:16:16"
if [[ -n "${GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST+x}" ]]; then
	POINT_LIST="$GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST"
else
	POINT_LIST="$DEFAULT_POINTS"
fi
read -r -a point_tokens <<< "$POINT_LIST"
if [[ ${#point_tokens[@]} -eq 0 ]]; then
	error "GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST must not be empty"
	exit 2
fi

declare -A seen_points=()
declare -a points=()
for point in "${point_tokens[@]}"; do
	case "$point" in
		16:512:16:16) point_data="A:16:512:16:16:134217728:900" ;;
		16:1024:16:16) point_data="A:16:1024:16:16:134217728:1800" ;;
		16:2048:16:16) point_data="A:16:2048:16:16:268435456:2400" ;;
		16:4096:16:16) point_data="A:16:4096:16:16:268435456:3600" ;;
		16:4096:4:4) point_data="B:16:4096:4:4:268435456:3600" ;;
		16:4096:8:8) point_data="B:16:4096:8:8:268435456:3600" ;;
		64:4096:16:16) point_data="C:64:4096:16:16:268435456:7200" ;;
		256:4096:16:16) point_data="C:256:4096:16:16:268435456:14400" ;;
		*)
			error "invalid Phase 4F point '$point'"
			exit 2
			;;
	esac
	if [[ -n "${seen_points[$point]+x}" ]]; then
		error "duplicate Phase 4F point '$point'"
		exit 2
	fi
	seen_points[$point]=1
	points+=("$point_data")
done

ROOT="${GOLEM_PHASE4F_LARGE_SCALE_ROOT:-}"
if [[ -z "$ROOT" ]]; then
	error "GOLEM_PHASE4F_LARGE_SCALE_ROOT must name a fresh absolute root"
	exit 2
fi
if [[ "$ROOT" != /* ]]; then
	error "GOLEM_PHASE4F_LARGE_SCALE_ROOT must be absolute: '$ROOT'"
	exit 2
fi
mkdir -p "$ROOT"
exec 9>"$ROOT/.phase4f.lock"
if ! flock -n 9; then
	error "Phase 4F parent root is locked: $ROOT"
	exit 2
fi

SCHEMA_FILE="$ROOT/parent_schema"
if [[ -f "$SCHEMA_FILE" ]]; then
	read -r existing_schema < "$SCHEMA_FILE" || existing_schema=""
	if [[ "$existing_schema" != "$SCHEMA" ]]; then
		error "parent schema mismatch in $SCHEMA_FILE: expected='$SCHEMA' actual='$existing_schema'"
		exit 2
	fi
else
	unexpected="$(find "$ROOT" -mindepth 1 -maxdepth 1 ! -name .phase4f.lock -print -quit)"
	if [[ -n "$unexpected" ]]; then
		error "parent root is not fresh and has no schema: $ROOT (found $unexpected)"
		exit 2
	fi
	printf '%s\n' "$SCHEMA" > "$SCHEMA_FILE"
fi

mkdir -p "$ROOT/children" "$ROOT/completed"
PARENT_MANIFEST="$ROOT/large_scale_manifest.csv"
PARENT_HEADER="run_id,stage,rows,dim,chunk_elems,worker_cores,band_cores,transport,reduction_vn,num_vns,dma_response_vn,noc_link_bw,noc_xbar_bw,dirctrl_highlink_bw,noc_input_buffer,noc_output_buffer,gm_buffer,flit_size,mem_node_size,retry_ticks,max_retries,timeout_sec,status,exit_code,artifact_validation,golden_checked,golden_mismatches,transport_events,transport_immediate,transport_queued,transport_rejected,transport_stale,inbox_high_water,latency_avg_cycles,latency_max_cycles,total_send_packets,total_send_bits,total_xbar_stalls,simulated_time_us,wall_time_sec,dma_timeout_retry,dma_timeout_exhausted,dma_write_timeout_retry,output_sha256,child_root"
if [[ -f "$PARENT_MANIFEST" ]]; then
	read -r existing_parent_header < "$PARENT_MANIFEST" || existing_parent_header=""
	existing_parent_header="${existing_parent_header%$'\r'}"
	if [[ "$existing_parent_header" != "$PARENT_HEADER" ]]; then
		error "parent manifest schema mismatch in $PARENT_MANIFEST"
		exit 2
	fi
	if ! awk -F, 'NR > 1 { key=$2 FS $3 FS $4 FS $6 FS $7; if (seen[key]++) exit 1 }' "$PARENT_MANIFEST"; then
		error "duplicate parent identity in $PARENT_MANIFEST"
		exit 2
	fi
fi
STATUS_FILE="$ROOT/point_status.csv"
STATUS_HEADER="stage,rows,dim,worker_cores,band_cores,status,exit_code,transport,reduction_vn,num_vns,dma_response_vn,noc_link_bw,noc_xbar_bw,dirctrl_highlink_bw,noc_input_buffer,noc_output_buffer,gm_buffer,flit_size,inter_router_no_cut,local_no_cut,mem_node_size,retry_ticks,max_retries,timeout_sec,child_root"
if [[ -f "$STATUS_FILE" ]]; then
	read -r existing_header < "$STATUS_FILE" || existing_header=""
	if [[ "$existing_header" != "$STATUS_HEADER" ]]; then
		error "point status schema mismatch in $STATUS_FILE"
		exit 2
	fi
else
	printf '%s\n' "$STATUS_HEADER" > "$STATUS_FILE"
fi

CHILD_RUNNER_SHA="$(sha256sum "$CHILD_RUNNER" | awk '{print $1}')"

marker_value() {
	local marker="$1"
	local key="$2"
	local matches
	matches="$(awk -F= -v key="$key" '$1 == key { count++; sub(/^[^=]*=/, ""); print } END { if (count != 1) exit 1 }' "$marker")" || return 1
	printf '%s' "$matches"
}

write_marker() {
	local marker="$1"
	local state="$2"
	local signature="$3"
	local signature_sha="$4"
	local pipeline_sha="$5"
	local output_sha="${6:-}"
	local temp="${marker}.tmp.$$"
	{
		printf 'schema=%s\n' "$SCHEMA"
		printf 'state=%s\n' "$state"
		printf 'signature_sha256=%s\n' "$signature_sha"
		printf 'signature=%s\n' "$signature"
		printf 'child_runner_sha256=%s\n' "$CHILD_RUNNER_SHA"
		printf 'pipeline_args_sha256=%s\n' "$pipeline_sha"
		printf 'output_sha256=%s\n' "$output_sha"
	} > "$temp"
	mv "$temp" "$marker"
}

validate_marker() {
	local marker="$1"
	local signature="$2"
	local signature_sha="$3"
	local pipeline_sha="$4"
	local key value
	[[ -f "$marker" ]] || return 1
	for key in schema state signature_sha256 signature child_runner_sha256 pipeline_args_sha256 output_sha256; do
		value="$(marker_value "$marker" "$key")" || return 1
	done
	[[ "$(wc -l < "$marker")" -eq 7 ]] || return 1
	[[ "$(marker_value "$marker" schema)" == "$SCHEMA" ]] || return 1
	[[ "$(marker_value "$marker" signature)" == "$signature" ]] || return 1
	[[ "$(marker_value "$marker" signature_sha256)" == "$signature_sha" ]] || return 1
	[[ "$(marker_value "$marker" child_runner_sha256)" == "$CHILD_RUNNER_SHA" ]] || return 1
	[[ "$(marker_value "$marker" pipeline_args_sha256)" == "$pipeline_sha" ]] || return 1
	case "$(marker_value "$marker" state)" in
		DRYRUN|FAIL|TIMEOUT|ARTIFACT_FAIL) ;;
		PASS)
			[[ "$(marker_value "$marker" output_sha256)" =~ ^[0-9a-f]{64}$ ]] || return 1
			;;
		*) return 1 ;;
	esac
}

record_status() {
	local stage="$1" rows="$2" dim="$3" workers="$4" bands="$5"
	local status="$6" exit_code="$7" mem="$8" timeout_sec="$9" child_root="${10}"
	local temp="${STATUS_FILE}.tmp.$$"
	awk -F, -v stage="$stage" -v rows="$rows" -v dim="$dim" -v workers="$workers" -v bands="$bands" \
		'NR == 1 || !($1 == stage && $2 == rows && $3 == dim && $4 == workers && $5 == bands)' \
		"$STATUS_FILE" > "$temp"
	printf '%s,%s,%s,%s,%s,%s,%s,explicit_noc,0,3,0,1200GB/s,1200GB/s,1200GB/s,512KB,512KB,1024KB,128B,0,0,%s,1024,8,%s,%s\n' \
		"$stage" "$rows" "$dim" "$workers" "$bands" "$status" "$exit_code" \
		"$mem" "$timeout_sec" "$child_root" >> "$temp"
	mv "$temp" "$STATUS_FILE"
}

collect_point() {
	local stage="$1" rows="$2" dim="$3" workers="$4" bands="$5" child_root="$6"
	python3 "$COLLECTOR" collect \
		--child-root "$child_root" --stage "$stage" --rows "$rows" --dim "$dim" \
		--workers "$workers" --bands "$bands" --parent-manifest "$PARENT_MANIFEST" \
		--verifier "$VERIFIER"
}

overall_rc=0
for point_data in "${points[@]}"; do
	IFS=: read -r stage rows dim workers bands mem_node_size timeout_sec <<< "$point_data"
	identity="stage_${stage}_r${rows}_d${dim}_w${workers}_b${bands}"
	child_root="$ROOT/children/$identity"
	marker="$ROOT/completed/${identity}.marker"
	pipeline_args="--noc-in-buf 512KB --noc-out-buf 512KB --noc-link-bw 1200GB/s --noc-xbar-bw 1200GB/s --noc-flit-size 128B --gm-buf 1024KB --mem-node-size $mem_node_size"
	pipeline_sha="$({ printf '%s\0' "$pipeline_args"; } | sha256sum | awk '{print $1}')"
	signature="schema=$SCHEMA;stage=$stage;rows=$rows;dim=$dim;workers=$workers;bands=$bands;cooperative_groups=1;transport=explicit_noc;request_vn=0;ordinary_response_vn=1;reduction_vn=0;num_vns=3;dma_response_vn=0;noc_link_bw=1200GB/s;noc_xbar_bw=1200GB/s;dirctrl_highlink_bw=1200GB/s;noc_input_buffer=512KB;noc_output_buffer=512KB;gm_buffer=1024KB;flit_size=128B;inter_router_no_cut=0;local_no_cut=0;mem_node_size=$mem_node_size;timeout_sec=$timeout_sec;chunk=256;staging_rows=4;job_rows=4;retry_ticks=1024;max_retries=8;child_runner_sha256=$CHILD_RUNNER_SHA;pipeline_args_sha256=$pipeline_sha"
	signature_sha="$({ printf '%s' "$signature"; } | sha256sum | awk '{print $1}')"

	if [[ -f "$marker" ]]; then
		if ! validate_marker "$marker" "$signature" "$signature_sha" "$pipeline_sha"; then
			error "invalid or drifted marker for point=$rows:$dim:$workers:$bands marker=$marker child_root=$child_root"
			exit 2
		fi
		marker_state="$(marker_value "$marker" state)"
		if [[ "$marker_state" == PASS ]]; then
			if collect_output="$(collect_point "$stage" "$rows" "$dim" "$workers" "$bands" "$child_root" 2>&1)"; then
				collect_sha="${collect_output##*output_sha256=}"
				if [[ "$collect_sha" != "$(marker_value "$marker" output_sha256)" ]]; then
					error "cached output hash drift point=$rows:$dim:$workers:$bands child_root=$child_root marker=$marker"
					exit 3
				fi
				echo "[PHASE4F] validated cached PASS point=$rows:$dim:$workers:$bands child_root=$child_root $collect_output"
				record_status "$stage" "$rows" "$dim" "$workers" "$bands" PASS 0 "$mem_node_size" "$timeout_sec" "$child_root"
				continue
			fi
			error "cached artifact validation failed point=$rows:$dim:$workers:$bands child_root=$child_root: $collect_output"
			exit 3
		fi
	fi

	if [[ "$DRY_RUN" == 1 ]]; then
		echo "[PHASE4F][DRY-RUN] point=$rows:$dim:$workers:$bands stage=$stage mem_node_size=$mem_node_size timeout_sec=$timeout_sec child_root=$child_root"
	else
		echo "[PHASE4F] run point=$rows:$dim:$workers:$bands stage=$stage mem_node_size=$mem_node_size timeout_sec=$timeout_sec child_root=$child_root"
	fi
	set +e
	GOLEM_SWEEP_ROOT="$child_root" \
	GOLEM_DRY_RUN_SWEEP="$DRY_RUN" \
	GOLEM_STOP_ON_FAIL=1 \
	GOLEM_SFU_DISTRIBUTED_POINT_LIST="$rows:$dim:$workers:$bands" \
	GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc \
	GOLEM_SFU_VN_SWEEP=1 GOLEM_SFU_REDUCTION_VN=0 GOLEM_DMA_RESPONSE_VN=0 \
	GOLEM_SFU_DISTRIBUTED_CHUNK_ELEMS=256 \
	GOLEM_SFU_DISTRIBUTED_STAGING_ROWS=4 GOLEM_SFU_DISTRIBUTED_JOB_ROWS=4 \
	GOLEM_SFU_DISTRIBUTED_RETRY_TICKS=1024 GOLEM_SFU_DISTRIBUTED_MAX_RETRIES=8 \
	GOLEM_TIMEOUT_512="$timeout_sec" GOLEM_TIMEOUT_1024="$timeout_sec" \
	GOLEM_SFU_DISTRIBUTED_PIPELINE_ARGS="$pipeline_args" \
		bash "$CHILD_RUNNER"
	child_rc=$?
	set -e

	status=FAIL
	exit_code="$child_rc"
	output_sha=""
	if [[ "$child_rc" -eq 124 ]]; then
		status=TIMEOUT
	elif [[ "$child_rc" -eq 0 && "$DRY_RUN" == 1 ]]; then
		status=DRYRUN
	elif [[ "$child_rc" -eq 0 ]]; then
		if collect_output="$(collect_point "$stage" "$rows" "$dim" "$workers" "$bands" "$child_root" 2>&1)"; then
			status=PASS
			exit_code=0
			output_sha="${collect_output##*output_sha256=}"
			echo "[PHASE4F] PASS point=$rows:$dim:$workers:$bands child_root=$child_root $collect_output"
		else
			status=ARTIFACT_FAIL
			exit_code=3
			error "artifact validation failed point=$rows:$dim:$workers:$bands child_root=$child_root: $collect_output"
		fi
	fi

	write_marker "$marker" "$status" "$signature" "$signature_sha" "$pipeline_sha" "$output_sha"
	record_status "$stage" "$rows" "$dim" "$workers" "$bands" "$status" "$exit_code" "$mem_node_size" "$timeout_sec" "$child_root"
	if [[ "$status" != PASS && "$status" != DRYRUN ]]; then
		error "point=$rows:$dim:$workers:$bands stage=$stage status=$status exit_code=$exit_code child_root=$child_root"
		overall_rc="$exit_code"
		if [[ "$stage" == C || "$STOP_ON_FAIL" == 1 ]]; then
			exit "$exit_code"
		fi
	fi
done

echo "[PHASE4F] root: $ROOT"
echo "[PHASE4F] manifest: $PARENT_MANIFEST"
exit "$overall_rc"
