#!/usr/bin/env bash
set -euo pipefail

# Canonical local FlashAttention smoke/regression entry point.
# Keep the implementation in run_fused_attention_scale.sh so the historical
# scale profiles remain available without making them part of the default UI.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCALE_RUNNER="$SCRIPT_DIR/run_fused_attention_scale.sh"

PROFILE=e3
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: run_flash_attention.sh [--profile e3|e4|e5] [scale-runner options]

Runs the locally built FlashAttention scale path. E3 is the default verified
baseline (S1024,D128). E4 is an optional pressure case; E5 requires
--allow-expensive. Archive attention profiles are not part of this entrypoint.
USAGE
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

case "$PROFILE" in
  e3|e4|e5) ;;
  *)
    echo "Unknown FlashAttention profile: $PROFILE (expected e3, e4, or e5)" >&2
    exit 2
    ;;
esac

exec "$SCALE_RUNNER" --scale-point "$PROFILE" "${ARGS[@]}"
