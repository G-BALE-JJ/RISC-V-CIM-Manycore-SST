#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/test_flash_attention.sh [--timeout SEC] [--mpi-ranks 1|2|4]

Run the active FlashAttention E3 regression from this worktree. The 2- and
4-rank modes verify query-block placement. Build first with
scripts/build_and_install_local.sh.
USAGE
}

TIMEOUT=600
MPI_RANKS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --mpi-ranks) MPI_RANKS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$MPI_RANKS" != "1" && "$MPI_RANKS" != "2" && "$MPI_RANKS" != "4" ]]; then
  echo "--mpi-ranks must be 1, 2, or 4 for the frozen E3 baselines" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
ATTENTION_DIR="$WORKTREE_ROOT/src/sst/elements/golem/tests/small/muticore_attention"
if [[ "$MPI_RANKS" == "1" ]]; then
  BASELINE_JSON="$WORKTREE_ROOT/baseline/e3/result.json"
  ARTIFACT_ROOT="/tmp/fused_attention_e3_s1024_d128"
elif [[ "$MPI_RANKS" == "2" ]]; then
  BASELINE_JSON="$WORKTREE_ROOT/baseline/e3/mpi2/result.json"
  ARTIFACT_ROOT="/tmp/fused_attention_e3_s1024_d128_mpi2"
else
  BASELINE_JSON="$WORKTREE_ROOT/baseline/e3/mpi4/result.json"
  ARTIFACT_ROOT="/tmp/fused_attention_e3_s1024_d128_mpi4"
fi

# Ensure SST resolves this worktree's freshly installed element library.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_local_install.sh"
# run_noc_dma_pipeline.sh can discover a stale build-tree library; the
# regression entry point must use the current worktree install explicitly.
export SST_LIB_PATH="$WORKTREE_ROOT/install/lib/sst-elements-library"
if [[ ! -f "$SST_LIB_PATH/libgolem.so" ]]; then
  echo "[ERROR] Missing local element library: $SST_LIB_PATH/libgolem.so" >&2
  echo "        Build it first with scripts/build_and_install_local.sh" >&2
  exit 1
fi

python3 -m unittest "$ATTENTION_DIR/test_flash_attention_baseline_contract.py"
bash -n "$ATTENTION_DIR/run_flash_attention.sh" "$ATTENTION_DIR/run_fused_attention_scale.sh"

echo "[FLASH] Running E3 baseline with $MPI_RANKS MPI rank(s)"
GOLEM_MPI_RANKS="$MPI_RANKS" "$ATTENTION_DIR/run_flash_attention.sh" \
  --timeout "$TIMEOUT" --artifact-root "$ARTIFACT_ROOT"

python3 - "$BASELINE_JSON" "$ARTIFACT_ROOT/fused_attention_result.json" \
  "$ARTIFACT_ROOT/attention_lifecycle.json" \
  "$ARTIFACT_ROOT/attention_mpi_partition.json" "$MPI_RANKS" <<'PY'
import json
import math
import sys

baseline_path, result_path, lifecycle_path, partition_path, mpi_ranks_raw = sys.argv[1:]
mpi_ranks = int(mpi_ranks_raw)
with open(baseline_path, encoding="ascii") as stream:
    baseline = json.load(stream)
with open(result_path, encoding="ascii") as stream:
    result = json.load(stream)
with open(lifecycle_path, encoding="ascii") as stream:
    lifecycle = json.load(stream)

checks = {
    "verification.status": result.get("status") == baseline["verification"]["status"],
    "verification.checked": result.get("checked") == baseline["verification"]["checked"],
    "verification.mismatches": result.get("mismatches") == baseline["verification"]["mismatches"],
    "verification.shape": result.get("shape") == {
        "queries": baseline["shape"]["queries"],
        "keys": baseline["shape"]["keys"],
        "head_dim": baseline["shape"]["head_dim"],
    },
    "verification.score_probability_hbm_bytes": result.get("score_probability_hbm_bytes") == baseline["verification"]["score_probability_hbm_bytes"],
    "verification.max_abs_error": math.isclose(
        result.get("max_abs_error", math.inf),
        baseline["verification"]["max_abs_error"],
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ),
    "lifecycle.status": lifecycle.get("status") == baseline["lifecycle"]["status"],
    "lifecycle.order_valid": lifecycle.get("lifecycle", {}).get("worker_critical_path", {}).get("order_valid") is True,
    "lifecycle.conservation_valid": lifecycle.get("lifecycle", {}).get("worker_critical_path", {}).get("inter_tile_breakdown", {}).get("conservation_valid") is True,
    "lifecycle.accelerator_completion_cycles": lifecycle.get("lifecycle", {}).get("accelerator_completion_cycles") == baseline["lifecycle"]["accelerator_completion_cycles"],
    "lifecycle.wait_return_cycles": lifecycle.get("lifecycle", {}).get("wait_return_cycles") == baseline["lifecycle"]["wait_return_cycles"],
    "topology.mpi_ranks": baseline.get("topology", {}).get("mpi_ranks") == mpi_ranks,
}
if mpi_ranks > 1:
    with open(partition_path, encoding="ascii") as stream:
        partition = json.load(stream)
    checks.update({
        "partition.status": partition.get("status") == "PASS",
        "partition.mpi_ranks": partition.get("mpi_ranks") == mpi_ranks,
        "partition.core_ranks": partition.get("observed_core_ranks") == partition.get("expected_core_ranks"),
        "partition.stats_files": len(partition.get("ranked_stats_files", [])) == mpi_ranks,
    })
failed = [name for name, passed in checks.items() if not passed]
if failed:
    print("[FLASH] E3 frozen baseline FAIL: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print(f"[FLASH] E3 {mpi_ranks}-rank frozen baseline MATCH")
PY
echo "[FLASH] E3 regression PASS"
