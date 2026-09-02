#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/test_flash_attention.sh [--timeout SEC]

Run the active FlashAttention E3 regression from this worktree. Build first
with scripts/build_and_install_local.sh.
USAGE
}

TIMEOUT=600
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
ATTENTION_DIR="$WORKTREE_ROOT/src/sst/elements/golem/tests/small/muticore_attention"
BASELINE_JSON="$WORKTREE_ROOT/baseline/e3/result.json"
ARTIFACT_ROOT="/tmp/fused_attention_e3_s1024_d128"

# Ensure SST resolves this worktree's freshly installed element library.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_local_install.sh"
# run_noc_dma_pipeline.sh can discover a stale build-tree library; the
# regression entry point must use the current worktree install explicitly.
export SST_LIB_PATH="$WORKTREE_ROOT/install/lib/sst-elements-library"
export SST_SOFTMAX_LD_LIBRARY_PATH="$SST_LIB_PATH:${LD_LIBRARY_PATH:-}"
if [[ ! -f "$SST_LIB_PATH/libgolem.so" ]]; then
  echo "[ERROR] Missing local element library: $SST_LIB_PATH/libgolem.so" >&2
  echo "        Build it first with scripts/build_and_install_local.sh" >&2
  exit 1
fi

python3 -m unittest "$ATTENTION_DIR/test_flash_attention_baseline_contract.py"
bash -n "$ATTENTION_DIR/run_flash_attention.sh" "$ATTENTION_DIR/run_fused_attention_scale.sh"

echo "[FLASH] Running E3 baseline"
"$ATTENTION_DIR/run_flash_attention.sh" --timeout "$TIMEOUT"

python3 - "$BASELINE_JSON" "$ARTIFACT_ROOT/fused_attention_result.json" "$ARTIFACT_ROOT/attention_lifecycle.json" <<'PY'
import json
import math
import sys

baseline_path, result_path, lifecycle_path = sys.argv[1:]
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
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    print("[FLASH] E3 frozen baseline FAIL: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("[FLASH] E3 frozen baseline MATCH")
PY
echo "[FLASH] E3 regression PASS"
