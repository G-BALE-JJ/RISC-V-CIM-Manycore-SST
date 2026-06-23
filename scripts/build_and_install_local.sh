#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/build_and_install_local.sh [--clean] [--reconfigure] [--no-autogen] [--jobs N]

Build and install this full-layout worktree's SST elements into a worktree-local
prefix.

Defaults:
  BUILD_ROOT          ./build/sst-elements
  INSTALL_PREFIX      ./install
  SST_CORE_PREFIX     /data4/lishun/pkg/sst_install
  SST_DRAMSIM3_PREFIX /data4/lishun/pkg/DRAMsim3
  JOBS                nproc

Override defaults with environment variables, for example:
  SST_CORE_PREFIX=/path/to/sst_core_install \
  SST_DRAMSIM3_PREFIX=/path/to/DRAMsim3 \
  scripts/build_and_install_local.sh

Options:
  --clean        Remove BUILD_ROOT and INSTALL_PREFIX before building.
  --reconfigure  Remove configure outputs in BUILD_ROOT before configuring.
  --no-autogen   Skip ./autogen.sh.
  --jobs N       Parallel make jobs.
EOF
}

clean=0
reconfigure=0
run_autogen=1
jobs="${JOBS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --clean)
            clean=1
            shift
            ;;
        --reconfigure)
            reconfigure=1
            shift
            ;;
        --no-autogen)
            run_autogen=0
            shift
            ;;
        --jobs)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ || "$2" -le 0 ]]; then
                echo "[ERROR] --jobs requires a positive integer" >&2
                exit 1
            fi
            jobs="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
BUILD_ROOT="${BUILD_ROOT:-$WORKTREE_ROOT/build/sst-elements}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$WORKTREE_ROOT/install}"
SST_CORE_PREFIX="${SST_CORE_PREFIX:-/data4/jjgong/local/sstcore}"
SST_DRAMSIM3_PREFIX="${SST_DRAMSIM3_PREFIX:-/data4/lishun/pkg/DRAMsim3}"
INSTALL_HOME="$BUILD_ROOT/.sst-home"
INSTALL_HOME_CONFIG="$INSTALL_HOME/.sst/sstsimulator.conf"

if [[ -z "$jobs" ]]; then
    jobs="$(nproc 2>/dev/null || echo 1)"
fi

if [[ ! -x "$SST_CORE_PREFIX/bin/sst-config" ]]; then
    echo "[ERROR] Missing SST core install: $SST_CORE_PREFIX/bin/sst-config" >&2
    exit 1
fi

if [[ ! -d "$SST_DRAMSIM3_PREFIX" ]]; then
    echo "[ERROR] Missing DRAMSim3 prefix: $SST_DRAMSIM3_PREFIX" >&2
    exit 1
fi

if [[ "$clean" -eq 1 ]]; then
    echo "[INFO] Removing local build/install directories"
    rm -rf "$BUILD_ROOT" "$INSTALL_PREFIX"
fi

"$SCRIPT_DIR/prepare_local_build.sh" "$BUILD_ROOT"

if [[ ! -d "$BUILD_ROOT/src/sst/elements/golem" || -L "$BUILD_ROOT/src/sst/elements" ]]; then
	echo "[ERROR] $BUILD_ROOT/src/sst/elements is not a copied elements source tree" >&2
	exit 1
fi

cd "$BUILD_ROOT"

if [[ "$reconfigure" -eq 1 ]]; then
    echo "[INFO] Removing configure outputs"
    rm -f Makefile config.log config.status libtool
    find . -name Makefile -type f -delete
fi

if [[ "$run_autogen" -eq 1 ]]; then
    echo "[1/4] Running autogen.sh"
    ./autogen.sh
else
    echo "[1/4] Skipping autogen.sh"
fi

echo "[2/4] Configuring local install"
./configure \
    --prefix="$INSTALL_PREFIX" \
    --with-sst-core="$SST_CORE_PREFIX" \
    --with-dramsim3="$SST_DRAMSIM3_PREFIX"

#  关键修复：将 libtool 软链接到源码树根，供 out-of-tree build 使用
SRCDIR_ROOT="$(cd "$BUILD_ROOT/src/sst/elements/../../../../.." && pwd -P)"
# 即 WORKTREE_ROOT 本身（src 的上一级）
ln -sf "$BUILD_ROOT/libtool" "$WORKTREE_ROOT/libtool"
# 新增：软链接 configure 生成的头文件到源码树 
echo "[INFO] Symlinking build-generated headers into source tree for out-of-tree build"
find "$BUILD_ROOT/src" -name "*.h" | while read -r build_h; do
    rel="${build_h#$BUILD_ROOT/src/}"
    src_h="$WORKTREE_ROOT/src/$rel"
    if [[ ! -f "$src_h" && ! -L "$src_h" ]]; then
        echo "  Linking: $rel"
        mkdir -p "$(dirname "$src_h")"
        ln -sf "$build_h" "$src_h"
    fi
done

echo "[3/4] Building with $jobs jobs"

export TMPDIR="$BUILD_ROOT/.tmp"
mkdir -p "$TMPDIR"

make -j"$jobs"

echo "[4/4] Installing to $INSTALL_PREFIX"
mkdir -p "$INSTALL_HOME/.sst"
if [[ ! -f "$INSTALL_HOME_CONFIG" ]]; then
	cp "$SST_CORE_PREFIX/etc/sst/sstsimulator.conf" "$INSTALL_HOME_CONFIG"
fi
HOME="$INSTALL_HOME" make install

cat <<EOF
[OK] Build and install complete.

Before running experiments from this worktree:
  cd "$WORKTREE_ROOT"
  source scripts/env_local_install.sh
  cd "$BUILD_ROOT/src/sst/elements/golem/tests"

Installed element libraries:
  $INSTALL_PREFIX/lib/sst-elements-library
EOF