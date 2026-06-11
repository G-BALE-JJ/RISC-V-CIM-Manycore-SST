#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage: scripts/prepare_local_build.sh [BUILD_ROOT]

Prepare a local SST source/build tree under this worktree.

Defaults:
  BUILD_ROOT             ./build/sst-elements
  SST_ELEMENTS_TEMPLATE  /data4/lishun/pkg/sst-elements

The prepared tree keeps src/sst/elements as a symlink to this worktree, so
configure/make and golem/tests runs use the branch checked out here instead of
the old temporary experiment tree.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ELEMENTS_WORKTREE="$(cd "$SCRIPT_DIR/.." && pwd -P)"
TEMPLATE="${SST_ELEMENTS_TEMPLATE:-/data4/lishun/pkg/sst-elements}"
BUILD_ROOT="${1:-$ELEMENTS_WORKTREE/build/sst-elements}"

if [[ ! -d "$TEMPLATE/src/sst" || ! -f "$TEMPLATE/autogen.sh" || ! -f "$TEMPLATE/configure.ac" ]]; then
	echo "[ERROR] SST_ELEMENTS_TEMPLATE must point to a full SST source tree: $TEMPLATE" >&2
	exit 1
fi

if [[ -e "$BUILD_ROOT" ]]; then
	if [[ -L "$BUILD_ROOT/src/sst/elements" ]]; then
		current_target="$(readlink -f "$BUILD_ROOT/src/sst/elements")"
		if [[ "$current_target" == "$ELEMENTS_WORKTREE" ]]; then
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
	"$TEMPLATE/" "$BUILD_ROOT/"

mkdir -p "$BUILD_ROOT/src/sst"
ln -s "$ELEMENTS_WORKTREE" "$BUILD_ROOT/src/sst/elements"

cat <<EOF
[OK] Prepared local build tree:
  $BUILD_ROOT

Use it when you want to build:
  cd "$BUILD_ROOT"
  ./autogen.sh
  ./configure <your usual configure flags>
  make -j

Elements source used by that build:
  $BUILD_ROOT/src/sst/elements -> $ELEMENTS_WORKTREE
EOF
