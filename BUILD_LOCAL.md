# Local Build Layout

This branch is meant to be used as one branch, one worktree, one local build
directory, and one local install prefix.

Prepare the local build tree without compiling:

```bash
cd /data4/lishun/pkg/wt-huti-v0
scripts/prepare_local_build.sh
```

This creates:

```text
/data4/lishun/pkg/wt-huti-v0/build/sst-elements/
```

The prepared SST tree links `src/sst/elements` back to this worktree, so future
builds and `golem/tests` runs use the `wt-huti-v0` branch sources instead of the
old temporary experiment directory.

Build and install later from:

```bash
cd /data4/lishun/pkg/wt-huti-v0/build/sst-elements
./autogen.sh
./configure \
  --prefix=/data4/lishun/pkg/wt-huti-v0/install \
  --with-sst-core=/data4/lishun/pkg/sst_install \
  --with-dramsim3=/data4/lishun/pkg/DRAMsim3
make -j
make install
```

This keeps the worktree's element libraries under:

```text
/data4/lishun/pkg/wt-huti-v0/install/lib/sst-elements-library/
```

Before running this branch's experiments, source the local environment:

```bash
cd /data4/lishun/pkg/wt-huti-v0
source scripts/env_local_install.sh
cd build/sst-elements/src/sst/elements/golem/tests
```

That keeps `sst` itself coming from the shared SST core install, but points SST
at this worktree's local element library using `--add-lib-path` through
`GOLEM_SST_ARGS`. Different worktrees can then build, install, and run in
parallel without overwriting each other's `libgolem.so`.

`build/` and `install/` are ignored by git and should hold only local generated
files.
