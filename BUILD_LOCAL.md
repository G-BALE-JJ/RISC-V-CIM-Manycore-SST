# Local Build Layout

This branch uses the full SST source layout. The elements source lives under:

```text
src/sst/elements/
```

The branch is meant to be used as one branch, one worktree, one local build
directory, and one local install prefix.

## One-command build and install

From a fresh checkout/worktree:

```bash
cd /data4/lishun/pkg/wt-huti-v0-full
scripts/build_and_install_local.sh
```

This does all of the following:

```text
prepare build/sst-elements
run ./autogen.sh
run ./configure
run make
run make install
```

By default it installs this worktree's elements under:

```text
/data4/lishun/pkg/wt-huti-v0-full/install/
```

and keeps generated build files under:

```text
/data4/lishun/pkg/wt-huti-v0-full/build/sst-elements/
```

The prepared build tree links `build/sst-elements/src/sst/elements` back to
this worktree's `src/sst/elements`, so builds and `golem/tests` runs use this
branch's sources.

## Defaults

The one-command script assumes these local dependencies:

```text
SST core install:      /data4/lishun/pkg/sst_install
DRAMSim3 source/build: /data4/lishun/pkg/DRAMsim3
```

Override them with environment variables when needed:

```bash
SST_CORE_PREFIX=/path/to/sst_core_install SST_DRAMSIM3_PREFIX=/path/to/DRAMsim3 JOBS=16 scripts/build_and_install_local.sh
```

Useful options:

```bash
scripts/build_and_install_local.sh --clean
scripts/build_and_install_local.sh --reconfigure
scripts/build_and_install_local.sh --jobs 32
```

## Run environment

Before running this branch's experiments, source the local environment:

```bash
cd /data4/lishun/pkg/wt-huti-v0-full
source scripts/env_local_install.sh
cd build/sst-elements/src/sst/elements/golem/tests
```

That keeps `sst` itself coming from the shared SST core install, but points SST
at this worktree's local element library using `--add-lib-path` through
`GOLEM_SST_ARGS`. Different worktrees can then build, install, and run in
parallel without overwriting each other's `libgolem.so`.

`build/` and `install/` are ignored by git and should hold only local generated
files.
