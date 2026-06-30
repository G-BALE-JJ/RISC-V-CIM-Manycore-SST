# Progress Log

## Session: 2026-06-29

### Phase 1: Requirements & Discovery

- **Status:** complete
- Actions taken:
  - 确认用户希望 SFU 作为独立子组件，类似 `array`、`global_memory` 挂在 RoCC 下。
  - 确认模块名称只叫 `SFU`，不叫 `SoftmaxSFU`。
  - 分析当前 tile 形状：`block_n=64` 时，`128/256/512` 都存在跨 N tile 行。
  - 确认 tile-local softmax 对大维度矩阵不满足 full row-wise softmax 正确性。
  - 确认第一版直接做跨 tile full row-wise softmax。
  - 确认多核竞争和死锁控制应参考 GEMM DMA / request scheduler。
  - 确认 softmax 方案应更新为 online softmax 统计合并。
- Files created/modified:
  - `design.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 2: Design Documentation

- **Status:** complete
- Actions taken:
  - 将设计文档改为中文。
  - 将传统三阶段 softmax 改为 `tile_m/tile_l -> online merge -> normalize`。
  - 增加“借鉴 GEMM DMA 的竞争与死锁控制”章节。
  - 创建 planning-with-files 要求的持久规划文件。
- Files created/modified:
  - `design.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 3: Implementation Plan

- **Status:** complete
- Actions taken:
  - 新增 `implementation_plan.md`，把后续实现拆成 checker、SFU 子组件骨架、online reducer、RoCC 指令、`cpu_builder.py` 挂载、RISC-V workload、运行脚本和验证八个阶段。
  - 明确第一版数据边界：workload 继续用现有 DMA 把 C tile 搬到 local GM，SFU 从 local GM 读写 tile，`sfu.wait` 后 workload 再写回 HBM/remote C tile。
  - 明确第一版 RoCC 指令编码使用 `0x17` 和 `0x18`，descriptor 地址通过 `rs1` 传入。
  - 明确第一版 reducer 采用 SFU-managed shared state，key 为 `(job_id, global_row)`，用 online softmax 公式合并 partial stats。
  - 明确构建边界：新增 SFU 源文件后需要 golem element reconfigure；只改 workload 时只在新目录构建。
  - 增加兼容性硬边界：所有 SFU 相关改动必须默认关闭，不设置 `GOLEM_SFU_ENABLE` 时不能影响原始 GEMM 功能和已有 GEMM small tests。
- Files created/modified:
  - `implementation_plan.md`
  - `task_plan.md`
  - `progress.md`

### Phase 5: Workload and Verification

- **Status:** in_progress
- Actions taken:
  - 先按 TDD 新增 `test_verify_softmax_sfu_against_golden.py`。
  - 运行测试确认红灯：失败原因是 `verify_softmax_sfu_against_golden.py` 尚不存在。
  - 新增 `verify_softmax_sfu_against_golden.py`，实现 full row-wise `softmax(A @ B)` golden checker。
  - checker 支持 `a_b` 和 `probability` 两种 reference，使用 full row 概率和检查，不接受 tile-local row sum。
  - 单元测试覆盖 full row-wise pass、tile-local softmax fail、probability full-row pass、tile-local row sum fail、大 logits 数值稳定性。
- Files created/modified:
  - `verify_softmax_sfu_against_golden.py`
  - `test_verify_softmax_sfu_against_golden.py`
  - `task_plan.md`
  - `progress.md`

### Phase 4: Component Implementation

- **Status:** in_progress
- Actions taken:
  - 按 TDD 新增 `test_sfu_component_scaffold.py`，先确认红灯：缺少 `sfu/sfu.h`、`sfu/sfu.cc`、`golem.cc` include 和 `Makefile.am` 条目。
  - 新增 `src/sst/elements/golem/sfu/sfu.h`，定义 `SFUAPI` 和 `SFU` 子组件骨架。
  - 新增 `src/sst/elements/golem/sfu/sfu.cc`，实现最小 no-op `issueSoftmaxTile`、`wait`、`bindGlobalMemory`、`setCoreInfo`，并注册 SFU 统计项。
  - 修改 `golem.cc` include `sst/elements/golem/sfu/sfu.h`。
  - 修改 `Makefile.am`，将 `sfu/sfu.h` 和 `sfu/sfu.cc` 加入 `libgolem_la_SOURCES`。
  - `scripts/build_and_install_local.sh --reconfigure --jobs 16` 在 rsync/chgrp 阶段失败，未进入编译；随后在已复制的 build tree 中手动运行 `./autogen.sh`、`./configure ...`、`make -C src/sst/elements/golem -j16`。
  - 按 TDD 新增 `test_rocc_sfu_integration.py`，先确认红灯：缺少 SFU func7、`sfuEnable`、SFU slot 加载、生命周期转发和 tick 分支。
  - 修改 `rocc/roccAnalog.h`：
    - include `sst/elements/golem/sfu/sfu.h`。
    - 新增 `GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE = 0x17` 和 `GOLEM_ROCC_FUNC7_SFU_WAIT = 0x18`，不改变已有 `0x11` 到 `0x16`。
    - 新增 `sfuEnable`，默认从 `params.find<int>("sfuEnable", 0)` 读取，默认关闭。
    - 仅在 `sfuEnable=1` 时加载 RoCC slot `"sfu"`，缺失则 fatal。
    - 绑定 `globalMem`、`coreID` 和 `active_worker_cores` 给 SFU。
    - 在 `init/setup/complete/finish` 中转发 SFU 生命周期。
    - 在非 busy tick 分支中新增 SFU softmax tile 和 wait 的非阻塞处理。
- Files created/modified:
  - `src/sst/elements/golem/sfu/sfu.h`
  - `src/sst/elements/golem/sfu/sfu.cc`
  - `src/sst/elements/golem/golem.cc`
  - `src/sst/elements/golem/Makefile.am`
  - `src/sst/elements/golem/rocc/roccAnalog.h`
  - `test_sfu_component_scaffold.py`
  - `test_rocc_sfu_integration.py`
  - `task_plan.md`
  - `progress.md`

## Test Results

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 文档自检 | 阅读 `design.md` | 包含 SFU 命名、online softmax、GEMM DMA 借鉴方案 | 已写入 | pass |
| 实现计划自检 | 阅读 `implementation_plan.md` | 包含路径、descriptor、RoCC 编码、DMA 边界、构建与验证顺序 | 已写入 | pass |
| Checker 红灯 | 只存在 `test_verify_softmax_sfu_against_golden.py` | 测试因 checker 缺失失败 | 5 failures，提示 checker 文件不存在 | pass |
| Checker 绿灯 | `python3 .../test_verify_softmax_sfu_against_golden.py` | 5 个测试通过 | Ran 5 tests in 0.395s，OK | pass |
| SFU scaffold 红灯 | `python3 .../test_sfu_component_scaffold.py` | 测试因 SFU 骨架缺失失败 | 1 failure, 2 errors，缺少 sfu 文件和注册 include | pass |
| SFU scaffold 绿灯 | `python3 .../test_sfu_component_scaffold.py` | 3 个测试通过 | Ran 3 tests in 0.001s，OK | pass |
| golem 局部编译 | `make -C src/sst/elements/golem -j16` in build tree | `sfu/sfu.cc` 编译并链接进 `libgolem.la` | 编译 `sfu/sfu.lo`，`CXXLD libgolem.la`，exit 0 | pass |
| RoCC SFU 红灯 | `python3 .../test_rocc_sfu_integration.py` | 测试因 RoCC 未接 SFU 失败，旧 GEMM func7 保持通过 | 4 failures，旧 func7 检查通过 | pass |
| RoCC SFU 绿灯 | `python3 .../test_rocc_sfu_integration.py` | 5 个测试通过 | Ran 5 tests in 0.003s，OK | pass |
| RoCC 接入后 golem 局部编译 | `make -C src/sst/elements/golem -j16` in build tree | `roccAnalog.h` 接入 SFU 后可编译链接 | `CXX golem.lo`，`CXXLD libgolem.la`，exit 0 | pass |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-06-29 | 无 | 1 | 当前阶段仅更新文档 |
| 2026-06-29 | Checker 红灯阶段 5 个测试失败 | 1 | 预期失败，原因是 checker 文件尚不存在；随后实现 checker 并转绿 |
| 2026-06-29 | `scripts/build_and_install_local.sh --reconfigure --jobs 16` 失败 | 1 | `rsync -a` 同步 build tree 时多个 `chgrp ... Invalid argument`，未进入编译；在已复制 build tree 手动 autogen/configure/golem 局部 make 验证通过 |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 4 的 SFU 子组件骨架和 RoCC 指令接入已编译通过，`cpu_builder.py` 挂载尚未开始 |
| Where am I going? | 下一步修改 `tests/architecture/cpu_builder.py`，仅在 `GOLEM_SFU_ENABLE=1` 时挂载 `golem.SFU` |
| What's the goal? | 用独立 `golem.SFU` 子组件实现跨 tile online full row-wise softmax |
| What have I learned? | 见 `findings.md` |
| What have I done? | 已创建中文设计文档、持久规划文件、详细实现计划、full row-wise softmax checker 和可编译的 SFU 子组件骨架 |
