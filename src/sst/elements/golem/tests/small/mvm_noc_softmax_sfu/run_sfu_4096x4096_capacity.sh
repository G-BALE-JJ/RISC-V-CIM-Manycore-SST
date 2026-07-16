#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHILD_RUNNER="$SCRIPT_DIR/run_sfu_unified_job_distributed_scaling.sh"
CAPACITY_TOOL="$SCRIPT_DIR/sfu_4096x4096_capacity.py"
PARSER="$SCRIPT_DIR/plot_sfu_phase4f_large_scale.py"
VERIFIER="$SCRIPT_DIR/verify_softmax_sfu_against_golden.py"
HBM_GENERATOR="$TESTS_DIR/tools/gen_hbm_init.py"
SCHEMA="sfu-4096-capacity-parent-v1"

error() {
	echo "[SFU-CAPACITY][ERROR] $*" >&2
}

require_unset_or_equal() {
	local name="$1" expected="$2"
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
		error "$name is controlled by the capacity parent runner and must be unset"
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
require_unset_or_equal GOLEM_SFU_CAPACITY_STOP_ON_FAIL 1
require_unset_or_equal TMPDIR /data4/jjgong/tmp

require_unset GOLEM_SWEEP_ROOT
require_unset GOLEM_DRY_RUN_SWEEP
require_unset GOLEM_STOP_ON_FAIL
require_unset GOLEM_SFU_DISTRIBUTED_POINT_LIST
require_unset GOLEM_SFU_DISTRIBUTED_PIPELINE_ARGS
require_unset GOLEM_TIMEOUT_512
require_unset GOLEM_TIMEOUT_1024

DRY_RUN="${GOLEM_SFU_CAPACITY_DRY_RUN:-0}"
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
	error "GOLEM_SFU_CAPACITY_DRY_RUN must be 0 or 1; got '$DRY_RUN'"
	exit 2
fi

DEFAULT_POINTS="512:4096:16:16 1024:4096:16:16 2048:4096:16:16 4096:4096:16:16"
POINT_LIST="${GOLEM_SFU_CAPACITY_POINT_LIST-$DEFAULT_POINTS}"
case "$POINT_LIST" in
	"512:4096:16:16"|\
	"512:4096:16:16 1024:4096:16:16"|\
	"512:4096:16:16 1024:4096:16:16 2048:4096:16:16"|\
	"$DEFAULT_POINTS") ;;
	*)
		error "GOLEM_SFU_CAPACITY_POINT_LIST must be a nonempty ordered canonical prefix; got '$POINT_LIST'"
		exit 2
		;;
esac
read -r -a points <<< "$POINT_LIST"

ROOT="${GOLEM_SFU_CAPACITY_ROOT:-}"
if [[ -z "$ROOT" || "$ROOT" != /* ]]; then
	error "GOLEM_SFU_CAPACITY_ROOT must name a fresh absolute root; got '$ROOT'"
	exit 2
fi
mkdir -p "$ROOT"
exec 9>"$ROOT/.capacity.lock"
if ! flock -n 9; then
	error "capacity root is locked: $ROOT"
	exit 2
fi

SCHEMA_FILE="$ROOT/parent_schema"
if [[ -f "$SCHEMA_FILE" ]]; then
	read -r existing_schema < "$SCHEMA_FILE" || existing_schema=""
	if [[ "$existing_schema" != "$SCHEMA" ]]; then
		error "parent schema mismatch: expected='$SCHEMA' actual='$existing_schema'"
		exit 2
	fi
else
	unexpected="$(find "$ROOT" -mindepth 1 -maxdepth 1 ! -name .capacity.lock -print -quit)"
	if [[ -n "$unexpected" ]]; then
		error "capacity root is not fresh and has no schema: found $unexpected"
		exit 2
	fi
	printf '%s\n' "$SCHEMA" > "$SCHEMA_FILE"
fi

if [[ -L "$ROOT/children" || -L "$ROOT/completed" ]]; then
	error "controlled capacity directory must not be a symlink"
	exit 2
fi
mkdir -p "$ROOT/children" "$ROOT/completed"

STATUS_FILE="$ROOT/capacity_status.csv"
STATUS_HEADER="rows,dim,worker_cores,band_cores,status,exit_code,timeout_sec,wall_time_sec,log_path,child_root"
if [[ -f "$STATUS_FILE" ]]; then
	read -r existing_header < "$STATUS_FILE" || existing_header=""
	if [[ "$existing_header" != "$STATUS_HEADER" ]]; then
		error "capacity status schema mismatch"
		exit 2
	fi
else
	printf '%s\n' "$STATUS_HEADER" > "$STATUS_FILE"
fi

PARENT_MANIFEST="$ROOT/capacity_manifest.csv"
PREFLIGHT="$ROOT/capacity_preflight.csv"
CHILD_SHA="$(sha256sum "$CHILD_RUNNER" | awk '{print $1}')"
CAPACITY_SHA="$(sha256sum "$CAPACITY_TOOL" | awk '{print $1}')"
PARSER_SHA="$(sha256sum "$PARSER" | awk '{print $1}')"
HBM_SHA="$(sha256sum "$HBM_GENERATOR" | awk '{print $1}')"
VERIFIER_SHA="$(sha256sum "$VERIFIER" | awk '{print $1}')"

timeout_for_rows() {
	case "$1" in
		512) echo 3600 ;;
		1024) echo 7200 ;;
		2048) echo 10800 ;;
		4096) echo 14400 ;;
		*) return 1 ;;
	esac
}

record_status() {
	local rows="$1" status="$2" exit_code="$3" timeout_sec="$4"
	local wall_time="$5" log_path="$6" child_root="$7"
	local temporary="${STATUS_FILE}.tmp.$$"
	awk -F, -v rows="$rows" 'NR == 1 || $1 != rows' "$STATUS_FILE" > "$temporary"
	printf '%s,4096,16,16,%s,%s,%s,%s,%s,%s\n' \
		"$rows" "$status" "$exit_code" "$timeout_sec" "$wall_time" \
		"$log_path" "$child_root" >> "$temporary"
	mv "$temporary" "$STATUS_FILE"
}

marker_value() {
	local marker="$1" key="$2"
	awk -F= -v key="$key" '$1 == key { count++; sub(/^[^=]*=/, ""); print } END { if (count != 1) exit 1 }' "$marker"
}

write_marker() {
	local marker="$1" state="$2" exit_code="$3" signature="$4"
	local signature_sha="$5" pipeline_sha="$6" child_root="$7"
	local output_sha="$8" wall_time="$9" log_path="${10}"
	local temporary="${marker}.tmp.$$"
	{
		printf 'schema=%s\n' "$SCHEMA"
		printf 'state=%s\n' "$state"
		printf 'exit_code=%s\n' "$exit_code"
		printf 'signature_sha256=%s\n' "$signature_sha"
		printf 'signature=%s\n' "$signature"
		printf 'child_runner_sha256=%s\n' "$CHILD_SHA"
		printf 'capacity_tool_sha256=%s\n' "$CAPACITY_SHA"
		printf 'parser_sha256=%s\n' "$PARSER_SHA"
		printf 'hbm_generator_sha256=%s\n' "$HBM_SHA"
		printf 'verifier_sha256=%s\n' "$VERIFIER_SHA"
		printf 'pipeline_args_sha256=%s\n' "$pipeline_sha"
		printf 'child_root=%s\n' "$child_root"
		printf 'output_sha256=%s\n' "$output_sha"
		printf 'wall_time_sec=%s\n' "$wall_time"
		printf 'log_path=%s\n' "$log_path"
	} > "$temporary"
	mv "$temporary" "$marker"
}

validate_marker() {
	local marker="$1" signature="$2" signature_sha="$3" pipeline_sha="$4"
	local key
	[[ -f "$marker" && ! -L "$marker" ]] || return 1
	for key in schema state exit_code signature_sha256 signature child_runner_sha256 \
		capacity_tool_sha256 parser_sha256 hbm_generator_sha256 verifier_sha256 \
		pipeline_args_sha256 child_root output_sha256 wall_time_sec log_path; do
		marker_value "$marker" "$key" >/dev/null || return 1
	done
	[[ "$(wc -l < "$marker")" -eq 15 ]] || return 1
	[[ "$(marker_value "$marker" schema)" == "$SCHEMA" ]] || return 1
	[[ "$(marker_value "$marker" signature)" == "$signature" ]] || return 1
	[[ "$(marker_value "$marker" signature_sha256)" == "$signature_sha" ]] || return 1
	[[ "$(marker_value "$marker" child_runner_sha256)" == "$CHILD_SHA" ]] || return 1
	[[ "$(marker_value "$marker" capacity_tool_sha256)" == "$CAPACITY_SHA" ]] || return 1
	[[ "$(marker_value "$marker" parser_sha256)" == "$PARSER_SHA" ]] || return 1
	[[ "$(marker_value "$marker" hbm_generator_sha256)" == "$HBM_SHA" ]] || return 1
	[[ "$(marker_value "$marker" verifier_sha256)" == "$VERIFIER_SHA" ]] || return 1
	[[ "$(marker_value "$marker" pipeline_args_sha256)" == "$pipeline_sha" ]] || return 1
	local child_root
	child_root="$(marker_value "$marker" child_root)"
	[[ "$child_root" == "$ROOT/children/"* && -d "$child_root" && ! -L "$child_root" ]] || return 1
}

next_attempt_root() {
	local base="$1" index=1 candidate
	while :; do
		printf -v candidate '%s/attempt-%04d' "$base" "$index"
		if [[ ! -e "$candidate" ]]; then
			printf '%s' "$candidate"
			return
		fi
		index=$((index + 1))
	done
}

is_mpi_environment_failure() {
	local log_path="$1"
	[[ -f "$log_path" ]] || return 1
	rg -q "No network interfaces were found for out-of-band communications" "$log_path" && \
		rg -q "MPI_Init|MPI_INIT" "$log_path"
}

archive_environment_failure() {
	local marker="$1" identity="$2" child_root="$3"
	local attempt archive
	attempt="$(basename "$child_root")"
	archive="$ROOT/completed/${identity}.${attempt}.environment-fail.marker"
	if [[ ! -e "$archive" ]]; then
		cp "$marker" "$archive"
	fi
	echo "$archive"
}

collect_point() {
	local rows="$1" child_root="$2"
	python3 "$CAPACITY_TOOL" collect \
		--child-root "$child_root" --rows "$rows" \
		--parent-manifest "$PARENT_MANIFEST" --verifier "$VERIFIER"
}

for point in "${points[@]}"; do
	IFS=: read -r rows dim workers bands <<< "$point"
	timeout_sec="$(timeout_for_rows "$rows")"
	identity="r${rows}_d4096_w16_b16"
	attempt_base="$ROOT/children/$identity"
	marker="$ROOT/completed/${identity}.marker"
	pipeline_args="--noc-in-buf 512KB --noc-out-buf 512KB --noc-link-bw 1200GB/s --noc-xbar-bw 1200GB/s --noc-flit-size 128B --gm-buf 1024KB --mem-node-size 268435456"
	pipeline_sha="$({ printf '%s\0' "$pipeline_args"; } | sha256sum | awk '{print $1}')"
	signature="schema=$SCHEMA;rows=$rows;dim=4096;workers=16;bands=16;cooperative_groups=1;transport=explicit_noc;request_vn=0;ordinary_response_vn=1;reduction_vn=0;num_vns=3;dma_response_vn=0;noc_link_bw=1200GB/s;noc_xbar_bw=1200GB/s;dirctrl_highlink_bw=1200GB/s;noc_input_buffer=512KB;noc_output_buffer=512KB;gm_buffer=1024KB;flit_size=128B;inter_router_no_cut=0;local_no_cut=0;mem_node_size=268435456;timeout_sec=$timeout_sec;chunk=256;staging_rows=4;job_rows=4;retry_ticks=1024;max_retries=8;child_runner_sha256=$CHILD_SHA;capacity_tool_sha256=$CAPACITY_SHA;parser_sha256=$PARSER_SHA;hbm_generator_sha256=$HBM_SHA;verifier_sha256=$VERIFIER_SHA;pipeline_args_sha256=$pipeline_sha"
	signature_sha="$({ printf '%s' "$signature"; } | sha256sum | awk '{print $1}')"

	if [[ -L "$attempt_base" ]]; then
		error "attempt base must not be a symlink: $attempt_base"
		exit 2
	fi
	mkdir -p "$attempt_base"

	if [[ -f "$marker" ]]; then
		if ! validate_marker "$marker" "$signature" "$signature_sha" "$pipeline_sha"; then
			error "invalid or drifted marker: $marker"
			exit 2
		fi
		marker_state="$(marker_value "$marker" state)"
		marker_exit="$(marker_value "$marker" exit_code)"
		marker_child="$(marker_value "$marker" child_root)"
		case "$marker_state" in
			PASS)
				run_id="sfu_job_dist_r${rows}_d4096_w16_bc16_g1_vn0"
				output="$marker_child/outputs/${run_id}.bin"
				[[ -f "$output" ]] || { error "cached output is missing: $output"; exit 3; }
				actual_sha="$(sha256sum "$output" | awk '{print $1}')"
				[[ "$actual_sha" == "$(marker_value "$marker" output_sha256)" ]] || {
					error "cached output hash drift: $output"
					exit 3
				}
				collect_output="$(collect_point "$rows" "$marker_child")" || {
					error "cached artifact validation failed: $marker_child"
					exit 3
				}
				record_status "$rows" PASS 0 "$timeout_sec" \
					"$(marker_value "$marker" wall_time_sec)" \
					"$(marker_value "$marker" log_path)" "$marker_child"
				echo "[SFU-CAPACITY] validated cached PASS point=$point child_root=$marker_child $collect_output"
				continue
				;;
			TIMEOUT)
				error "recorded TIMEOUT requires later analysis; point=$point child_root=$marker_child"
				exit 124
				;;
			ENVIRONMENT_FAIL)
				archive="$(archive_environment_failure "$marker" "$identity" "$marker_child")"
				echo "[SFU-CAPACITY] archived environment failure point=$point marker=$archive"
				;;
			FAIL)
				marker_log="$(marker_value "$marker" log_path)"
				if is_mpi_environment_failure "$marker_log"; then
					archive="$(archive_environment_failure "$marker" "$identity" "$marker_child")"
					echo "[SFU-CAPACITY] reclassified and archived MPI environment failure point=$point marker=$archive"
				else
					error "recorded FAIL blocks automatic retry; point=$point child_root=$marker_child"
					exit "$marker_exit"
				fi
				;;
			ARTIFACT_FAIL)
				error "recorded ARTIFACT_FAIL blocks automatic retry; point=$point child_root=$marker_child"
				exit "$marker_exit"
				;;
			DRYRUN) ;;
			*) error "unsupported marker state: $marker_state"; exit 2 ;;
		esac
	fi

	if [[ "$DRY_RUN" == 0 ]]; then
		python3 "$CAPACITY_TOOL" preflight --root "$ROOT" --tmpdir "$TMPDIR" \
			--output "$PREFLIGHT" --point-list "$POINT_LIST"
	fi

	child_root="$(next_attempt_root "$attempt_base")"
	mkdir "$child_root"
	if [[ "$DRY_RUN" == 1 ]]; then
		echo "[SFU-CAPACITY][DRY-RUN] point=$point mem_node_size=268435456 timeout_sec=$timeout_sec child_root=$child_root"
	else
		echo "[SFU-CAPACITY] run point=$point mem_node_size=268435456 timeout_sec=$timeout_sec child_root=$child_root"
	fi

	start_sec="$(date +%s)"
	set +e
	GOLEM_SWEEP_ROOT="$child_root" \
	GOLEM_DRY_RUN_SWEEP="$DRY_RUN" \
	GOLEM_STOP_ON_FAIL=1 \
	GOLEM_SFU_DISTRIBUTED_POINT_LIST="$point" \
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
	wall_time=$(( $(date +%s) - start_sec ))
	log_path="$(find "$child_root/logs" -maxdepth 1 -type f -print -quit 2>/dev/null || true)"
	log_path="${log_path:-$child_root/logs}"

	state=FAIL
	exit_code="$child_rc"
	output_sha=""
	if [[ "$child_rc" -eq 124 ]]; then
		state=TIMEOUT
	elif [[ "$child_rc" -eq 0 && "$DRY_RUN" == 1 ]]; then
		state=DRYRUN
	elif [[ "$child_rc" -eq 0 ]]; then
		if collect_output="$(collect_point "$rows" "$child_root")"; then
			state=PASS
			exit_code=0
			output_sha="${collect_output##*output_sha256=}"
			echo "[SFU-CAPACITY] PASS point=$point child_root=$child_root $collect_output"
		else
			state=ARTIFACT_FAIL
			exit_code=3
			error "artifact validation failed: point=$point child_root=$child_root"
		fi
	elif is_mpi_environment_failure "$log_path"; then
		state=ENVIRONMENT_FAIL
	fi

	write_marker "$marker" "$state" "$exit_code" "$signature" "$signature_sha" \
		"$pipeline_sha" "$child_root" "$output_sha" "$wall_time" "$log_path"
	record_status "$rows" "$state" "$exit_code" "$timeout_sec" "$wall_time" \
		"$log_path" "$child_root"
	if [[ "$state" != PASS && "$state" != DRYRUN ]]; then
		error "point=$point status=$state exit_code=$exit_code wall_time_sec=$wall_time child_root=$child_root log_path=$log_path"
		exit "$exit_code"
	fi
done

echo "[SFU-CAPACITY] root: $ROOT"
echo "[SFU-CAPACITY] status: $STATUS_FILE"
echo "[SFU-CAPACITY] manifest: $PARENT_MANIFEST"
