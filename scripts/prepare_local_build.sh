#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage: scripts/prepare_local_build.sh [BUILD_ROOT]

Prepare a local SST source/build tree under this full-layout worktree.

Defaults:
  BUILD_ROOT          ./build/sst-elements
  SST_CORE_PREFIX     /data4/lishun/pkg/sst_install
  SST_DRAMSIM3_PREFIX /data4/lishun/pkg/DRAMsim3

The prepared tree keeps src/sst/elements as a symlink to this worktree's
src/sst/elements directory.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
ELEMENTS_SOURCE="$WORKTREE_ROOT/src/sst/elements"
BUILD_ROOT="${1:-$WORKTREE_ROOT/build/sst-elements}"
INSTALL_PREFIX="$WORKTREE_ROOT/install"
SST_CORE_PREFIX="${SST_CORE_PREFIX:-/data4/lishun/pkg/sst_install}"
SST_DRAMSIM3_PREFIX="${SST_DRAMSIM3_PREFIX:-/data4/lishun/pkg/DRAMsim3}"

if [[ ! -d "$ELEMENTS_SOURCE/golem" || ! -f "$WORKTREE_ROOT/autogen.sh" || ! -f "$WORKTREE_ROOT/configure.ac" ]]; then
	echo "[ERROR] This must be run from a full SST source layout with src/sst/elements." >&2
	exit 1
fi

if [[ -e "$BUILD_ROOT" ]]; then
	if [[ -L "$BUILD_ROOT/src/sst/elements" ]]; then
		current_target="$(readlink -f "$BUILD_ROOT/src/sst/elements")"
		if [[ "$current_target" == "$ELEMENTS_SOURCE" ]]; then
			echo "[INFO] Local build tree already prepared: $BUILD_ROOT"
			exit 0
		fi
	fi
	echo "[ERROR] BUILD_ROOT already exists and is not prepared for this worktree: $BUILD_ROOT" >&2
	echo "        Choose another path or remove it manually after checking its contents." >&2
	exit 1
fi

mkdir -p "$(dirname "$BUILD_ROOT")"

rsync -a \
	--exclude='.git/' \
	--exclude='build/' \
	--exclude='install/' \
	--exclude='src/sst/elements/' \
	--exclude='autom4te.cache/' \
	--exclude='config.log' \
	--exclude='config.status' \
	--exclude='libtool' \
	--exclude='Makefile' \
	--exclude='*.o' \
	--exclude='*.lo' \
	--exclude='*.la' \
	--exclude='.deps/' \
	--exclude='.libs/' \
	"$WORKTREE_ROOT/" "$BUILD_ROOT/"

mkdir -p "$BUILD_ROOT/src/sst"
ln -s "$ELEMENTS_SOURCE" "$BUILD_ROOT/src/sst/elements"

cat <<EOF
[OK] Prepared local build tree:
  $BUILD_ROOT

Use it when you want to build:
  cd "$BUILD_ROOT"
  ./autogen.sh
  ./configure --prefix="$INSTALL_PREFIX" --with-sst-core="$SST_CORE_PREFIX" --with-dramsim3="$SST_DRAMSIM3_PREFIX"
  make -j
  make install

Before running this worktree's experiments:
  cd "$WORKTREE_ROOT"
  source scripts/env_local_install.sh

Elements source used by that build:
  $BUILD_ROOT/src/sst/elements -> $ELEMENTS_SOURCE
EOF
