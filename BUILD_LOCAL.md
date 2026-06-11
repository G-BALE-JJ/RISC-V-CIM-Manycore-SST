# Local Build Layout

This branch is meant to be used as one branch, one worktree, and one local
build directory.

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

Build later from:

```bash
cd /data4/lishun/pkg/wt-huti-v0/build/sst-elements
./autogen.sh
./configure <your usual configure flags>
make -j
```

`build/` is ignored by git and should hold only local generated files.
