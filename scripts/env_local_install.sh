#!/usr/bin/env bash
# Source this file before running experiments from this worktree:
#   source scripts/env_local_install.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	echo "source this script instead of executing it:" >&2
	echo "  source scripts/env_local_install.sh" >&2
	exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ELEMENTS_WORKTREE="$(cd "$SCRIPT_DIR/.." && pwd -P)"

export SST_CORE_PREFIX="${SST_CORE_PREFIX:-/data4/lishun/pkg/sst_install}"
export SST_ELEMENTS_INSTALL_PREFIX="${SST_ELEMENTS_INSTALL_PREFIX:-$ELEMENTS_WORKTREE/install}"

case ":${PATH:-}:" in
	*":$SST_CORE_PREFIX/bin:"*) ;;
	*) export PATH="$SST_CORE_PREFIX/bin${PATH:+:$PATH}" ;;
esac

for lib_dir in "$SST_ELEMENTS_INSTALL_PREFIX/lib" "$SST_CORE_PREFIX/lib"; do
	case ":${LD_LIBRARY_PATH:-}:" in
		*":$lib_dir:"*) ;;
		*) export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
	esac
done

elements_lib_path="$SST_ELEMENTS_INSTALL_PREFIX/lib/sst-elements-library"
sst_lib_arg="--add-lib-path=$elements_lib_path"
case " ${GOLEM_SST_ARGS:-} " in
	*" $sst_lib_arg "*) ;;
	*) export GOLEM_SST_ARGS="$sst_lib_arg${GOLEM_SST_ARGS:+ $GOLEM_SST_ARGS}" ;;
esac

echo "[OK] SST core: $SST_CORE_PREFIX"
echo "[OK] SST elements install: $SST_ELEMENTS_INSTALL_PREFIX"
echo "[OK] GOLEM_SST_ARGS=$GOLEM_SST_ARGS"
