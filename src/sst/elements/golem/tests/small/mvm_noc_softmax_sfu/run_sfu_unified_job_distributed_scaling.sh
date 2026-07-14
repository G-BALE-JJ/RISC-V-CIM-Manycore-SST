#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

STAMP="${GOLEM_SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S_%N)_$$}"
SWEEP_ROOT="${GOLEM_SWEEP_ROOT:-$TESTS_DIR/artifacts/sweeps/sfu_unified_job_distributed_scaling_$STAMP}"
mkdir -p "$SWEEP_ROOT" "$SWEEP_ROOT/completed" "$SWEEP_ROOT/inputs" "$SWEEP_ROOT/outputs"
SWEEP_ROOT="$(cd "$SWEEP_ROOT" && pwd)"
MANIFEST="$SWEEP_ROOT/sweep_manifest.csv"

ROWS="${GOLEM_SFU_DISTRIBUTED_ROWS:-16}"
STAGING_ROWS="${GOLEM_SFU_DISTRIBUTED_STAGING_ROWS:-4}"
JOB_ROWS="${GOLEM_SFU_DISTRIBUTED_JOB_ROWS:-4}"
CHUNK_ELEMS="${GOLEM_SFU_DISTRIBUTED_CHUNK_ELEMS:-256}"
RETRY_TICKS="${GOLEM_SFU_DISTRIBUTED_RETRY_TICKS:-1024}"
MAX_RETRIES="${GOLEM_SFU_DISTRIBUTED_MAX_RETRIES:-8}"
REDUCTION_TRANSPORT="${GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT:-modeled_noc}"

case "$REDUCTION_TRANSPORT" in
	modeled_noc|noc_model|noc|explicit_noc)
		;;
	*)
		echo "[SFU-DISTRIBUTED-SCALING][ERROR] distributed scaling requires modeled_noc or explicit_noc reduction transport for reduction message counter validation; got '$REDUCTION_TRANSPORT'" >&2
		exit 2
		;;
esac

if [[ ! -f "$MANIFEST" ]]; then
	printf "run_id,rows,dim,chunk_elems,worker_cores,band_cores,cooperative_groups,staging_rows,job_rows,retry_ticks,max_retries,status,exit_code,timeout_sec,artifact_validation\n" > "$MANIFEST"
fi

pipeline_args=()
if [[ -n "${GOLEM_SFU_DISTRIBUTED_PIPELINE_ARGS:-}" ]]; then
	read -r -a pipeline_args <<< "$GOLEM_SFU_DISTRIBUTED_PIPELINE_ARGS"
fi

timeout_for_dim() {
	local dim="$1"
	if (( dim <= 512 )); then
		echo "${GOLEM_TIMEOUT_512:-900}"
	else
		echo "${GOLEM_TIMEOUT_1024:-1800}"
	fi
}

record_manifest() {
	local run_id="$1"
	local rows="$2"
	local dim="$3"
	local worker_cores="$4"
	local band_cores="$5"
	local cooperative_groups="$6"
	local status="$7"
	local exit_code="$8"
	local timeout_sec="$9"
	local artifact_validation="${10}"
	printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
		"$run_id" "$rows" "$dim" "$CHUNK_ELEMS" "$worker_cores" "$band_cores" \
		"$cooperative_groups" "$STAGING_ROWS" "$JOB_ROWS" "$RETRY_TICKS" \
		"$MAX_RETRIES" "$status" "$exit_code" "$timeout_sec" "$artifact_validation" \
		>> "$MANIFEST"
}

dma_metric_sum() {
	local file="$1"
	local metric="$2"
	awk -F, -v metric="$metric" '$1 == metric { print int($7 + 0); found = 1 } END { if (!found) exit 1 }' "$file"
}

sfu_stat_sum() {
	local file="$1"
	local stat="$2"
	awk -F, -v stat="$stat" '$2 == stat { sum += $7; found = 1 } END { if (!found) exit 1; print int(sum) }' "$file"
}

sfu_stat_active_count() {
	local file="$1"
	local stat="$2"
	awk -F, -v stat="$stat" '$2 == stat && ($7 + 0) > 0 { count++ } END { print int(count) }' "$file"
}

require_equal() {
	local name="$1"
	local actual="$2"
	local expected="$3"
	if [[ "$actual" != "$expected" ]]; then
		echo "[SFU-DISTRIBUTED-SCALING][ERROR] $name expected=$expected actual=$actual" >&2
		return 1
	fi
}

point_signature() {
	local rows="$1"
	local dim="$2"
	local worker_cores="$3"
	local band_cores="$4"
	local pipeline_args_sha256
	pipeline_args_sha256="$({
		printf "%s\0" "${#pipeline_args[@]}"
		printf "%s\0" "${pipeline_args[@]}"
	} | sha256sum | awk '{ print $1 }')" || return 1
	printf "rows=%s;dim=%s;chunk=%s;workers=%s;band=%s;staging=%s;job=%s;retry=%s;max_retries=%s;reduction_transport=%s;pipeline_args_sha256=%s" \
		"$rows" "$dim" "$CHUNK_ELEMS" "$worker_cores" "$band_cores" \
		"$STAGING_ROWS" "$JOB_ROWS" "$RETRY_TICKS" "$MAX_RETRIES" \
		"$REDUCTION_TRANSPORT" "$pipeline_args_sha256"
}

marker_value() {
	local marker="$1"
	local key="$2"
	local line
	while IFS= read -r line; do
		if [[ "$line" == "$key="* ]]; then
			printf "%s" "${line#*=}"
			return 0
		fi
	done < "$marker"
	return 1
}

file_sha256() {
	local file="$1"
	sha256sum "$file" | awk '{ print $1 }'
}

validate_point_artifacts() {
	local run_id="$1"
	local rows="$2"
	local dim="$3"
	local worker_cores="$4"
	local band_cores="$5"
	local stdout_dir="$SWEEP_ROOT/stdout/overlap0/$run_id"
	local stats_dir="$SWEEP_ROOT/stats/overlap0/$run_id"
	local stats_file="$stats_dir/stats_selfcom.txt"
	local dma_file="$stats_dir/dma_summary.csv"
	local output_file="$SWEEP_ROOT/outputs/${run_id}.bin"

	if [[ ! -d "$stdout_dir" || ! -f "$stats_file" || ! -f "$dma_file" ||
	      ! -f "$output_file" ]]; then
		echo "[SFU-DISTRIBUTED-SCALING][ERROR] missing artifacts for $run_id" >&2
		return 1
	fi
	local output_bytes
	output_bytes="$(stat -c %s "$output_file")" || return 1
	require_equal "output tensor bytes" "$output_bytes" "$((rows * dim * 4))" || return 1

	local pass_count=0
	local file
	for file in "$stdout_dir"/stdout-*; do
		[[ -f "$file" ]] || continue
		if rg -q "mode=sfu-standalone-job-softmax.*rows=$rows dim=$dim.*worker_cores=$worker_cores.*staging_rows=$STAGING_ROWS job_rows=$JOB_ROWS band_cores=$band_cores.*distributed_columns=1 PASS" "$file"; then
			pass_count=$((pass_count + 1))
		fi
	done
	require_equal "physical PASS cores" "$pass_count" "$band_cores" || return 1

	local active_ops active_max active_sum active_norm
	active_ops="$(sfu_stat_active_count "$stats_file" sfu_ops_issued)" || return 1
	active_max="$(sfu_stat_active_count "$stats_file" sfu_job_softmax_max_chunks)" || return 1
	active_sum="$(sfu_stat_active_count "$stats_file" sfu_job_softmax_sum_chunks)" || return 1
	active_norm="$(sfu_stat_active_count "$stats_file" sfu_job_softmax_norm_chunks)" || return 1
	require_equal "active SFU ops" "$active_ops" "$band_cores" || return 1
	require_equal "active SFU max" "$active_max" "$band_cores" || return 1
	require_equal "active SFU sum" "$active_sum" "$band_cores" || return 1
	require_equal "active SFU normalize" "$active_norm" "$band_cores" || return 1

	local expected_ops=$(( (rows / JOB_ROWS) * worker_cores ))
	local expected_worker_rows=$(( rows * worker_cores ))
	local expected_chunks=0
	local worker slice_begin slice_end slice_elems chunks
	for ((worker = 0; worker < worker_cores; ++worker)); do
		slice_begin=$(( dim * worker / worker_cores ))
		slice_end=$(( dim * (worker + 1) / worker_cores ))
		slice_elems=$(( slice_end - slice_begin ))
		chunks=$(( (slice_elems + CHUNK_ELEMS - 1) / CHUNK_ELEMS ))
		expected_chunks=$(( expected_chunks + rows * chunks ))
	done
	local actual
	actual="$(sfu_stat_sum "$stats_file" sfu_ops_issued)" || return 1
	require_equal "SFU ops total" "$actual" "$expected_ops" || return 1
	actual="$(sfu_stat_sum "$stats_file" sfu_job_softmax_max_chunks)" || return 1
	require_equal "SFU max chunks" "$actual" "$expected_chunks" || return 1
	actual="$(sfu_stat_sum "$stats_file" sfu_job_softmax_sum_chunks)" || return 1
	require_equal "SFU sum chunks" "$actual" "$expected_chunks" || return 1
	actual="$(sfu_stat_sum "$stats_file" sfu_job_softmax_norm_chunks)" || return 1
	require_equal "SFU normalize chunks" "$actual" "$expected_chunks" || return 1
	actual="$(sfu_stat_sum "$stats_file" sfu_partial_submits)" || return 1
	require_equal "SFU partial submits" "$actual" "$((expected_worker_rows * 2))" || return 1
	actual="$(sfu_stat_sum "$stats_file" sfu_partial_done)" || return 1
	require_equal "SFU partial done" "$actual" "$expected_worker_rows" || return 1
	actual="$(sfu_stat_sum "$stats_file" sfu_reduction_max_requests)" || return 1
	require_equal "SFU max reduction requests" "$actual" "$expected_worker_rows" || return 1
	actual="$(sfu_stat_sum "$stats_file" sfu_reduction_max_responses)" || return 1
	require_equal "SFU max reduction responses" "$actual" "$expected_worker_rows" || return 1
	actual="$(sfu_stat_sum "$stats_file" sfu_reduction_sum_requests)" || return 1
	require_equal "SFU sum reduction requests" "$actual" "$expected_worker_rows" || return 1
	actual="$(sfu_stat_sum "$stats_file" sfu_reduction_sum_responses)" || return 1
	require_equal "SFU sum reduction responses" "$actual" "$expected_worker_rows" || return 1
	actual="$(sfu_stat_sum "$stats_file" sfu_retry_events)" || return 1
	require_equal "SFU retry events" "$actual" 0 || return 1

	if [[ "$REDUCTION_TRANSPORT" == "explicit_noc" ]]; then
		local expected_transport_events=$(( expected_worker_rows * 4 ))
		local sfu_transport_received
		sfu_transport_received="$(sfu_stat_sum "$stats_file" sfu_reduction_transport_received)" || return 1
		require_equal "SFU reduction transport receives" "$sfu_transport_received" "$expected_transport_events" || return 1

		if rg -q ",gmem_reduction_send_immediate," "$stats_file" && \
		   rg -q ",gmem_reduction_send_queued," "$stats_file" && \
		   rg -q ",gmem_reduction_send_rejected," "$stats_file" && \
		   rg -q ",gmem_reduction_received," "$stats_file"; then
			local transport_immediate transport_queued transport_rejected gmem_delivery_count
			transport_immediate="$(sfu_stat_sum "$stats_file" gmem_reduction_send_immediate)" || return 1
			transport_queued="$(sfu_stat_sum "$stats_file" gmem_reduction_send_queued)" || return 1
			transport_rejected="$(sfu_stat_sum "$stats_file" gmem_reduction_send_rejected)" || return 1
			gmem_delivery_count="$(sfu_stat_sum "$stats_file" gmem_reduction_received)" || return 1
			actual=$(( transport_immediate + transport_queued ))
			require_equal "GlobalMemory reduction transport sends" "$actual" "$expected_transport_events" || return 1
			require_equal "GlobalMemory reduction transport rejected sends" "$transport_rejected" 0 || return 1
			require_equal "GlobalMemory reduction transport receives" "$gmem_delivery_count" "$expected_transport_events" || return 1
		else
			echo "[SFU-DISTRIBUTED-SCALING][INFO] GlobalMemory reduction transport stats unavailable; SFU receive gate remains authoritative" >&2
		fi
	fi

	local expected_dma_ops=$(( rows * worker_cores ))
	local expected_bytes=$(( rows * dim * 4 ))
	actual="$(dma_metric_sum "$dma_file" read_issue_count)" || return 1
	require_equal "DMA read issue" "$actual" "$expected_dma_ops" || return 1
	actual="$(dma_metric_sum "$dma_file" write_issue_count)" || return 1
	require_equal "DMA write issue" "$actual" "$expected_dma_ops" || return 1
	actual="$(dma_metric_sum "$dma_file" completion)" || return 1
	require_equal "DMA completion" "$actual" "$expected_dma_ops" || return 1
	actual="$(dma_metric_sum "$dma_file" write_completion)" || return 1
	require_equal "DMA write completion" "$actual" "$expected_dma_ops" || return 1
	actual="$(dma_metric_sum "$dma_file" read_bytes_total)" || return 1
	require_equal "DMA read bytes" "$actual" "$expected_bytes" || return 1
	actual="$(dma_metric_sum "$dma_file" write_bytes_total)" || return 1
	require_equal "DMA write bytes" "$actual" "$expected_bytes" || return 1
	actual="$(dma_metric_sum "$dma_file" timeout_retry)" || return 1
	require_equal "DMA timeout retry" "$actual" 0 || return 1
	actual="$(dma_metric_sum "$dma_file" timeout_exhausted)" || return 1
	require_equal "DMA timeout exhausted" "$actual" 0 || return 1
	actual="$(dma_metric_sum "$dma_file" write_timeout_retry)" || return 1
	require_equal "DMA write timeout retry" "$actual" 0 || return 1
	return 0
}

validate_cached_point() {
	local marker="$1"
	local signature="$2"
	local run_id="$3"
	local rows="$4"
	local dim="$5"
	local worker_cores="$6"
	local band_cores="$7"
	local output_file="$SWEEP_ROOT/outputs/${run_id}.bin"
	local cached_signature cached_output_sha256 actual_output_sha256

	cached_signature="$(marker_value "$marker" signature)" || return 1
	[[ "$cached_signature" == "$signature" ]] || return 1
	validate_point_artifacts "$run_id" "$rows" "$dim" "$worker_cores" "$band_cores" || return 1
	cached_output_sha256="$(marker_value "$marker" output_sha256)" || return 1
	actual_output_sha256="$(file_sha256 "$output_file")" || return 1
	require_equal "output tensor SHA-256" "$actual_output_sha256" "$cached_output_sha256" || return 1
	return 0
}

validate_point_shape() {
	local rows="$1"
	local dim="$2"
	local worker_cores="$3"
	local band_cores="$4"
	if (( rows <= 0 || dim <= 0 || worker_cores <= 0 || band_cores <= 0 ||
	      band_cores > 16 || worker_cores > band_cores || band_cores % worker_cores != 0 ||
	      worker_cores > dim || STAGING_ROWS <= 0 || JOB_ROWS <= 0 ||
	      rows % STAGING_ROWS != 0 || STAGING_ROWS % JOB_ROWS != 0 )); then
		echo "[SFU-DISTRIBUTED-SCALING][ERROR] invalid point rows=$rows dim=$dim worker_cores=$worker_cores band_cores=$band_cores staging_rows=$STAGING_ROWS job_rows=$JOB_ROWS" >&2
		return 1
	fi
}

run_point() {
	local rows="$1"
	local dim="$2"
	local worker_cores="$3"
	local band_cores="$4"
	local cooperative_groups=$(( band_cores / worker_cores ))
	local run_id="sfu_job_dist_r${rows}_d${dim}_w${worker_cores}_bc${band_cores}_g${cooperative_groups}"
	local timeout_sec
	timeout_sec="$(timeout_for_dim "$dim")"
	local marker="$SWEEP_ROOT/completed/${run_id}.pass"
	local signature
	signature="$(point_signature "$rows" "$dim" "$worker_cores" "$band_cores")"

	validate_point_shape "$rows" "$dim" "$worker_cores" "$band_cores"
	if [[ -f "$marker" ]]; then
		if validate_cached_point "$marker" "$signature" "$run_id" "$rows" "$dim" \
		   "$worker_cores" "$band_cores"; then
			echo "[SFU-DISTRIBUTED-SCALING] skip validated PASS $run_id"
			record_manifest "$run_id" "$rows" "$dim" "$worker_cores" "$band_cores" \
				"$cooperative_groups" PASS 0 "$timeout_sec" CACHED
			return 0
		fi
		echo "[SFU-DISTRIBUTED-SCALING] stale or invalid marker; rerun $run_id"
	fi

	if [[ "${GOLEM_DRY_RUN_SWEEP:-0}" != "0" ]]; then
		echo "[SFU-DISTRIBUTED-SCALING][DRY-RUN] $run_id rows=$rows dim=$dim workers=$worker_cores band_cores=$band_cores groups=$cooperative_groups"
		record_manifest "$run_id" "$rows" "$dim" "$worker_cores" "$band_cores" \
			"$cooperative_groups" DRYRUN 0 "$timeout_sec" NOT_RUN
		return 0
	fi

	echo "[SFU-DISTRIBUTED-SCALING] run $run_id timeout=${timeout_sec}s"
	set +e
	timeout "$timeout_sec" env \
		GOLEM_RUN_ID="$run_id" \
		GOLEM_ARTIFACT_ROOT="$SWEEP_ROOT" \
		GOLEM_HBM_DIR="$SWEEP_ROOT/hbm" \
		GOLEM_RUN_SUMMARY_CSV="$SWEEP_ROOT/stats/run_summary.csv" \
		GOLEM_TENSOR_DIR="$SWEEP_ROOT/inputs" \
		GOLEM_SOFTMAX_C_FILE="$SWEEP_ROOT/outputs/${run_id}.bin" \
		GOLEM_SKIP_TENSOR_GEN=0 \
		GOLEM_SKIP_HBM_GEN=0 \
		GOLEM_SKIP_BUILD=0 \
		GOLEM_HBM_DUMP_OUTPUT=1 \
		GOLEM_SFU_ENABLE=1 \
		GOLEM_SFU_PERF_PROFILE=0 \
		GOLEM_SFU_STANDALONE_SOFTMAX=1 \
		GOLEM_SFU_JOB_SOFTMAX=1 \
		GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM=1 \
		GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS=1 \
		GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT="$REDUCTION_TRANSPORT" \
		GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS="$STAGING_ROWS" \
		GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS="$JOB_ROWS" \
		GOLEM_SFU_JOB_SOFTMAX_BAND_CORES="$band_cores" \
		GOLEM_SFU_JOB_SOFTMAX_CHUNK_ELEMS="$CHUNK_ELEMS" \
		GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES="$worker_cores" \
		GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS=1 \
		GOLEM_SFU_PRIMITIVE_SOFTMAX=0 \
		GOLEM_SOFTMAX_VERIFY_REFERENCE=logits \
		GOLEM_A_REUSE_N_TILES=1 \
		GOLEM_B_REUSE_M_TILES=1 \
		GOLEM_DMA_SLOT_COUNT=16 \
		GOLEM_DMA_READ_RETRY_TICKS="$RETRY_TICKS" \
		GOLEM_DMA_READ_MAX_RETRIES="$MAX_RETRIES" \
		GOLEM_SST_ENABLE_ALL_STATS=1 \
		GOLEM_SST_STAT_LOAD_LEVEL=1 \
		GOLEM_BENCH_DISABLE_SST_STATS=0 \
		"$SCRIPT_DIR/run_noc_dma_softmax_sfu_pipeline.sh" \
		--groups 4 --array-in 64 --array-out 4 --num-arrays 64 \
		--num-cores 16 --gemm-cores 16 --num-mem-nodes 5 --mesh-dim-x 4 \
		--mem-node-size 134217728 --global-stride-kb 512 \
		--gemm-m "$rows" --gemm-n "$dim" --gemm-k "$dim" \
		--gemm-block-m 4 --gemm-block-n 64 --gemm-block-k 64 \
		--verify-softmax --group-manager-enable 0 --ctrl-link-enable 0 \
		--softmax-reference logits \
		--softmax-logits-file "$SWEEP_ROOT/inputs/softmax_logits_${rows}x${dim}.bin" \
		"${pipeline_args[@]}" \
		--softmax-c-file "$SWEEP_ROOT/outputs/${run_id}.bin"
	local exit_code=$?
	set -e

	local status=FAIL
	local artifact_validation=NOT_RUN
	local output_sha256=""
	if [[ "$exit_code" -eq 124 ]]; then
		status=TIMEOUT
	elif [[ "$exit_code" -eq 0 ]]; then
		if validate_point_artifacts "$run_id" "$rows" "$dim" "$worker_cores" "$band_cores" && \
		   output_sha256="$(file_sha256 "$SWEEP_ROOT/outputs/${run_id}.bin")"; then
			status=PASS
			artifact_validation=PASS
			printf "signature=%s\noutput_sha256=%s\nrun_id=%s\n" \
				"$signature" "$output_sha256" "$run_id" > "$marker"
		else
			status=ARTIFACT_FAIL
			artifact_validation=FAIL
			exit_code=3
		fi
	fi
	record_manifest "$run_id" "$rows" "$dim" "$worker_cores" "$band_cores" \
		"$cooperative_groups" "$status" "$exit_code" "$timeout_sec" "$artifact_validation"

	if [[ "$status" != PASS ]]; then
		echo "[SFU-DISTRIBUTED-SCALING][ERROR] $run_id status=$status exit_code=$exit_code" >&2
		if [[ "${GOLEM_STOP_ON_FAIL:-1}" != "0" ]]; then
			return "$exit_code"
		fi
	fi
}

run_explicit_point_list() {
	local points=()
	read -r -a points <<< "$GOLEM_SFU_DISTRIBUTED_POINT_LIST"
	local point rows dim worker_cores band_cores extra
	for point in "${points[@]}"; do
		IFS=: read -r rows dim worker_cores band_cores extra <<< "$point"
		if [[ -z "${rows:-}" || -z "${dim:-}" || -z "${worker_cores:-}" ||
		      -z "${band_cores:-}" || -n "${extra:-}" ]]; then
			echo "[SFU-DISTRIBUTED-SCALING][ERROR] invalid point '$point'; expected rows:dim:worker_cores:band_cores" >&2
			return 2
		fi
		run_point "$rows" "$dim" "$worker_cores" "$band_cores"
	done
}

run_representative_matrix() {
	run_point "$ROWS" 512 4 4
	run_point "$ROWS" 512 4 16
	run_point "$ROWS" 512 8 16
	run_point "$ROWS" 512 16 16
	run_point "$ROWS" 1024 4 4
	run_point "$ROWS" 1024 4 16
	run_point "$ROWS" 1024 8 16
	run_point "$ROWS" 1024 16 16
}

if [[ -n "${GOLEM_SFU_DISTRIBUTED_POINT_LIST:-}" ]]; then
	run_explicit_point_list
else
	run_representative_matrix
fi

echo "[SFU-DISTRIBUTED-SCALING] root: $SWEEP_ROOT"
echo "[SFU-DISTRIBUTED-SCALING] manifest: $MANIFEST"
