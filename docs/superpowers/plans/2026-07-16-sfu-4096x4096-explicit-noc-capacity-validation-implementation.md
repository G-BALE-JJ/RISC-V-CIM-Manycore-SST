# SFU 4096x4096 Explicit-NoC Softmax 容量验证实施计划

> **面向 agentic workers：** 必须逐任务使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 执行本计划；步骤使用复选框跟踪，每个任务完成测试和审查后才能继续。

**目标：** 在固定 `explicit_noc`、16 workers 和成熟 GEMM 网络配置下，按顺序完成 `512/1024/2048/4096 x 4096` 四个真实 SST softmax 点，并以完整 golden、reduction、transport、DMA 和 artifact 证据验证最终 `4096x4096`。

**架构：** 新增容量契约/证据 Python 模块和独立 shell parent runner，继续把现有 `run_sfu_unified_job_distributed_scaling.sh` 作为唯一 child runner。容量模块复用 Phase 4F 已审查的通用 child parser，但拥有独立四点 resolver、preflight、完整矩阵 gate 和 CSV/Markdown 报告；Phase 4F 的 8 点矩阵及历史 artifact 保持不变。

**技术栈：** Bash、Python 3 标准库、unittest、现有 SST pipeline、现有 HBM generator、现有 logits golden verifier、CSV/SST log/stat artifact。

## 全局约束

- 设计依据：`docs/superpowers/specs/2026-07-16-sfu-4096x4096-explicit-noc-capacity-validation-design.md`。
- 默认矩阵严格为 `512:4096:16:16 1024:4096:16:16 2048:4096:16:16 4096:4096:16:16`；override 只能是该序列的非空有序前缀。
- 固定 `explicit_noc`、`num_vns=3`、`reduction_vn=0`、`dma_response_vn=0`、16 workers/16 bands、`chunk=256`、`staging_rows=4`、`job_rows=4`、`retry_ticks=1024`、`max_retries=8`。
- 固定 `mem_node_size=268435456` bytes；禁止 memory capacity sweep。
- 固定 GEMM 网络值：link/xbar/directory highlink `1200GB/s`，NoC input/output buffer `512KB`，GM buffer `1024KB`，flit `128B`，两个 no-cut 值均为 0。
- watchdog 固定为 `512=3600s`、`1024=7200s`、`2048=10800s`、`4096=14400s`；
  exit code 124 必须记录为 `TIMEOUT` 并停止后续点，禁止当场扩大阈值重跑。
- 资源 gate：真实运行前 artifact filesystem 至少 16 GiB free、available host memory 至少 8 GiB、`TMPDIR=/data4/jjgong/tmp` 存在且可写。
- 真实运行始终 stop-on-first-failure；禁止自动改参数、跳点或删除失败 attempt。
- 不修改 production component、GEMM runner/architecture/guest、Phase 4F parent runner、distributed child runner、HBM generator 或 golden verifier。
- 不运行 `modeled_noc`、VN/bandwidth/worker/chunk/job-row DSE，不扩展 `dim=8192`，不做 GEMM+softmax fusion。
- 当前 worktree 有大量既有改动。只精确暂存本计划列出的新文件和 Softmax 文档，不回退、不覆盖、不广泛暂存其他改动。

---

### 任务 1：用 TDD 锁定四点容量契约

**文件：**

- 新建：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/sfu_4096x4096_capacity.py`
- 新建：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_4096x4096_capacity.py`

**接口：**

```python
@dataclasses.dataclass(frozen=True)
class CapacityPoint:
    rows: int
    dim: int
    worker_cores: int
    band_cores: int
    mem_node_size: int
    timeout_sec: int
    rowmajor_region_end: int

@dataclasses.dataclass(frozen=True)
class CapacityEvidence:
    point: CapacityPoint
    elements: int
    tensor_bytes: int
    expected_reduction_each: int
    expected_transport_total: int
    expected_dma_ops: int
    expected_dma_bytes: int
    bias_base: int
    layout_margin_bytes: int

DEFAULT_POINTS: tuple[CapacityPoint, CapacityPoint, CapacityPoint, CapacityPoint]
```

- `resolve_point(rows: int, dim: int, workers: int, bands: int) -> CapacityPoint`
- `parse_point_list(value: str | None) -> tuple[CapacityPoint, CapacityPoint, CapacityPoint, CapacityPoint] | tuple[CapacityPoint, ...]`
- `derive_capacity(point: CapacityPoint) -> CapacityEvidence`

- [ ] **步骤 1：写容量矩阵 RED 测试。** 精确断言：

```python
EXPECTED = (
    (512, 4096, 16, 16, 268435456, 3600, 37748736),
    (1024, 4096, 16, 16, 268435456, 7200, 58720256),
    (2048, 4096, 16, 16, 268435456, 10800, 100663296),
    (4096, 4096, 16, 16, 268435456, 14400, 184549376),
)
```

  测试还必须拒绝 `dim=8192`、rows 不在矩阵、worker/band 非 16、重复点、乱序点、跳过前缀点和空 override。

- [ ] **步骤 2：写派生公式 RED 测试。** 对最终点精确断言：

```python
evidence.elements == 16_777_216
evidence.tensor_bytes == 67_108_864
evidence.expected_reduction_each == 65_536
evidence.expected_transport_total == 262_144
evidence.expected_dma_ops == 65_536
evidence.expected_dma_bytes == 67_108_864
evidence.bias_base == 268_419_072
evidence.layout_margin_bytes == 83_869_696
```

- [ ] **步骤 3：运行 RED。**

```bash
TMPDIR=/data4/jjgong/tmp python3 -m unittest src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_4096x4096_capacity.py -v
```

  预期因 `sfu_4096x4096_capacity` 不存在而失败。

- [ ] **步骤 4：实现最小容量契约。** 使用以下 canonical 常量；只接受默认矩阵有序前缀：

```python
MEM_NODE_SIZE = 268_435_456
BIAS_STRIDE = 16_384
BIAS_BASE = MEM_NODE_SIZE - BIAS_STRIDE
DEFAULT_POINTS = (
    CapacityPoint(512, 4096, 16, 16, MEM_NODE_SIZE, 3_600, 37_748_736),
    CapacityPoint(1024, 4096, 16, 16, MEM_NODE_SIZE, 7_200, 58_720_256),
    CapacityPoint(2048, 4096, 16, 16, MEM_NODE_SIZE, 10_800, 100_663_296),
    CapacityPoint(4096, 4096, 16, 16, MEM_NODE_SIZE, 14_400, 184_549_376),
)

def derive_capacity(point):
    elements = point.rows * point.dim
    tensor_bytes = elements * 4
    reduction_each = point.rows * point.worker_cores
    return CapacityEvidence(
        point=point,
        elements=elements,
        tensor_bytes=tensor_bytes,
        expected_reduction_each=reduction_each,
        expected_transport_total=4 * reduction_each,
        expected_dma_ops=reduction_each,
        expected_dma_bytes=tensor_bytes,
        bias_base=BIAS_BASE,
        layout_margin_bytes=BIAS_BASE - point.rowmajor_region_end,
    )
```

- [ ] **步骤 5：运行 GREEN。** 预期本文件全部 PASS。

- [ ] **步骤 6：运行 Phase 4F 合同回归。**

```bash
TMPDIR=/data4/jjgong/tmp python3 -m unittest src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_phase4f_large_scale.py -v
```

  预期 Phase 4F 原有矩阵和测试全部 PASS。

- [ ] **步骤 7：精确提交。**

```bash
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/sfu_4096x4096_capacity.py src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_4096x4096_capacity.py
git commit -m "test: define 4096 softmax capacity contract"
```

---

### 任务 2：实现资源 preflight 和容量证据报告

**文件：**

- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/sfu_4096x4096_capacity.py`
- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_4096x4096_capacity.py`

**接口：**

```python
@dataclasses.dataclass(frozen=True)
class ResourceSnapshot:
    artifact_free_bytes: int
    available_memory_bytes: int
    tmpdir: pathlib.Path
    tmpdir_writable: bool

```

- `read_resource_snapshot(root: pathlib.Path, tmpdir: pathlib.Path) -> ResourceSnapshot`
- `validate_resource_snapshot(snapshot: ResourceSnapshot) -> None`
- `write_preflight_csv(points: tuple[CapacityPoint, ...], snapshot: ResourceSnapshot, path: pathlib.Path) -> None`
- `collect_child(child_root: pathlib.Path, point: CapacityPoint, verifier: pathlib.Path, parent_manifest: pathlib.Path) -> phase4f.PointRecord`
- `write_capacity_report(root: pathlib.Path, output_dir: pathlib.Path, verifier: pathlib.Path) -> None`

- [ ] **步骤 1：写资源 RED 测试。** 使用注入的 `ResourceSnapshot`，验证：

```python
MIN_ARTIFACT_FREE_BYTES = 16 * 1024**3
MIN_AVAILABLE_MEMORY_BYTES = 8 * 1024**3
CANONICAL_TMPDIR = pathlib.Path("/data4/jjgong/tmp")
```

  分别拒绝磁盘不足、内存不足、错误 TMPDIR 和不可写 TMPDIR；边界值必须通过。测试不得
  依赖当前机器实际 free space。

- [ ] **步骤 2：写 preflight CSV RED 测试。** 要求严格 UTF-8/LF、四行固定顺序，并含：

```text
rows,dim,worker_cores,band_cores,mem_node_size,timeout_sec,
elements,tensor_bytes,rowmajor_region_end,bias_base,layout_margin_bytes,
expected_reduction_each,expected_transport_total,expected_dma_ops,
expected_dma_bytes,artifact_free_bytes,available_memory_bytes,tmpdir,status
```

- [ ] **步骤 3：写 generic child parser 复用 RED 测试。** 构造默认矩阵中的
  `512x4096` `CapacityPoint` 的
  synthetic Phase 4F-style child artifact，并通过：

```python
phase4f_spec = phase4f.PointSpec(
    "CAP", point.rows, point.dim, point.worker_cores, point.band_cores,
    point.mem_node_size, point.timeout_sec,
)
record = phase4f.parse_child_point(child_root, phase4f_spec, verifier)
```

  验证新模块不调用 Phase 4F 的 `resolve_point()`，但仍复用其 golden、network、reduction、
  transport、DMA 和 output hash gate。

- [ ] **步骤 4：写报告 RED 测试。** 四个 synthetic PASS child 必须确定性生成：

```text
report/sfu_4096x4096_capacity_source_data.csv
report/sfu_4096x4096_capacity_summary.md
```

  summary 必须列出四点状态和最终点的 `16,777,216/0`、`65,536`、`262,144`、
  `67,108,864`；缺点、重复点、乱序、失败点伪装为 PASS、child evidence/hash 漂移都必须
  失败。

- [ ] **步骤 5：运行 RED。** 使用任务 1 的 focused 命令，预期新增测试失败。

- [ ] **步骤 6：实现资源和证据功能。** `read_resource_snapshot()` 使用
  `shutil.disk_usage(root).free` 和 `/proc/meminfo` 的 `MemAvailable`；只在真实 runner 中
  使用当前机器快照。报告必须重新调用 `phase4f.parse_child_point()`，不能只信任 parent
  manifest。

- [ ] **步骤 7：实现 CLI。** 精确提供：

```text
python3 sfu_4096x4096_capacity.py preflight --root ROOT --tmpdir /data4/jjgong/tmp --output capacity_preflight.csv
python3 sfu_4096x4096_capacity.py collect --child-root CHILD --rows R --parent-manifest capacity_manifest.csv --verifier VERIFY
python3 sfu_4096x4096_capacity.py report --root ROOT --output-dir ROOT/report --verifier VERIFY
```

  `collect` 必须从 canonical resolver 获得 point metadata；`report` 必须要求完整四点 PASS。

- [ ] **步骤 8：运行 GREEN 和确定性检查。** 连续生成两次 synthetic report，CSV 和
  Markdown 的 SHA-256 必须一致。

- [ ] **步骤 9：提交。**

```bash
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/sfu_4096x4096_capacity.py src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_4096x4096_capacity.py
git commit -m "feat: add 4096 softmax capacity preflight"
```

---

### 任务 3：实现独立 parent runner 和恢复边界

**文件：**

- 新建：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh`
- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_4096x4096_capacity.py`

**接口：**

```text
GOLEM_SFU_CAPACITY_ROOT=<fresh absolute root>
GOLEM_SFU_CAPACITY_DRY_RUN=0|1
GOLEM_SFU_CAPACITY_POINT_LIST="ordered canonical prefix"
GOLEM_SFU_CAPACITY_STOP_ON_FAIL=1
```

- [ ] **步骤 1：写 runner RED 测试。** 断言 default dry-run 严格展开四点；每点 child
  manifest 的 timeout 为 `3600/7200/10800/14400`，pipeline args 都包含：

```text
--noc-in-buf 512KB --noc-out-buf 512KB
--noc-link-bw 1200GB/s --noc-xbar-bw 1200GB/s
--noc-flit-size 128B --gm-buf 1024KB
--mem-node-size 268435456
```

- [ ] **步骤 2：写 inherited-conflict RED 测试。** 对 transport、VN、network、buffer、
  chunk、staging、job rows、retry、pipeline args、TMPDIR 和 stop-on-fail 逐项注入冲突；
  runner 必须 exit 2，且不得创建 child attempt。

- [ ] **步骤 3：写 lifecycle RED 测试。** 覆盖 absolute fresh root、schema
  `sfu-4096-capacity-parent-v1`、root lock、ordered-prefix override、immutable attempt、marker
  完整字段、hash drift、cached output drift、失败状态保留和 stop-on-first-failure。

- [ ] **步骤 4：运行 RED。** 预期 runner 缺失。

- [ ] **步骤 5：实现 runner 固定环境。** 开头必须用 `require_unset_or_equal` 冻结：

```bash
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT explicit_noc
require_unset_or_equal GOLEM_SFU_VN_SWEEP 1
require_unset_or_equal GOLEM_SFU_REDUCTION_VN 0
require_unset_or_equal GOLEM_DMA_RESPONSE_VN 0
require_unset_or_equal GOLEM_NOC_LINK_BW 1200GB/s
require_unset_or_equal GOLEM_NOC_XBAR_BW 1200GB/s
require_unset_or_equal GOLEM_DIRCTRL_HIGHLINK_BW 1200GB/s
require_unset_or_equal GOLEM_NOC_INPUT_BUF_SIZE 512KB
require_unset_or_equal GOLEM_NOC_OUTPUT_BUF_SIZE 512KB
require_unset_or_equal GOLEM_NOC_FLIT_SIZE 128B
require_unset_or_equal GOLEM_GM_BUFFER_LENGTH 1024KB
require_unset_or_equal GOLEM_NOC_INTER_ROUTER_NO_CUT 0
require_unset_or_equal GOLEM_NOC_LOCAL_NO_CUT 0
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_CHUNK_ELEMS 256
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_STAGING_ROWS 4
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_JOB_ROWS 4
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_RETRY_TICKS 1024
require_unset_or_equal GOLEM_SFU_DISTRIBUTED_MAX_RETRIES 8
require_unset_or_equal GOLEM_SFU_CAPACITY_STOP_ON_FAIL 1
require_unset_or_equal TMPDIR /data4/jjgong/tmp
```

- [ ] **步骤 6：实现 point 调用。** 对每点设置同一个 child timeout 到
  `GOLEM_TIMEOUT_512` 和 `GOLEM_TIMEOUT_1024`，并调用：

```bash
GOLEM_SWEEP_ROOT="$child_root" \
GOLEM_DRY_RUN_SWEEP="$DRY_RUN" \
GOLEM_STOP_ON_FAIL=1 \
GOLEM_SFU_DISTRIBUTED_POINT_LIST="$rows:4096:16:16" \
GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc \
GOLEM_SFU_VN_SWEEP=1 \
GOLEM_SFU_REDUCTION_VN=0 \
GOLEM_DMA_RESPONSE_VN=0 \
GOLEM_TIMEOUT_512="$timeout_sec" \
GOLEM_TIMEOUT_1024="$timeout_sec" \
GOLEM_SFU_DISTRIBUTED_PIPELINE_ARGS="$pipeline_args" \
bash "$CHILD_RUNNER"
```

  `DRY_RUN=0` 时必须在第一个新 point 前和每个后续 point 前执行 resource preflight。

- [ ] **步骤 7：实现 marker/status/manifest。** signature 必须包含 schema、完整 point、
  全部固定配置、timeout、child runner、容量模块、Phase 4F parser、HBM generator 和
  golden verifier 的 SHA-256，以及 pipeline args hash。PASS resume 必须重新运行
  `collect` 并比较 output SHA-256；exit code 124 必须写入 `TIMEOUT` marker/status，包含
  wall time 和日志位置，并立即退出。失败和 timeout marker 不得被当成可跳过 PASS。

- [ ] **步骤 8：运行 GREEN、shell syntax 和 dry-run。**

```bash
bash -n src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh
```

```bash
TMPDIR=/data4/jjgong/tmp GOLEM_SFU_CAPACITY_ROOT=/data4/jjgong/tmp/sfu-capacity-dryrun GOLEM_SFU_CAPACITY_DRY_RUN=1 bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh
```

  预期只生成小型 dry-run artifact，不生成 HBM backing files，不启动 SST。

- [ ] **步骤 9：提交。**

```bash
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_4096x4096_capacity.py
git commit -m "feat: orchestrate 4096 softmax capacity ladder"
```

---

### 任务 4：完成 focused 验证和执行前审计

**文件：**

- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/README.md`
- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/task_plan.md`
- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/progress.md`
- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/findings.md`

- [ ] **步骤 1：运行新模块 focused suite。**

```bash
TMPDIR=/data4/jjgong/tmp python3 -m unittest src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_4096x4096_capacity.py -v
```

- [ ] **步骤 2：运行完整 Softmax focused suite。**

```bash
TMPDIR=/data4/jjgong/tmp /data4/jjgong/.venvs/golem-plot/bin/python -m unittest discover -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu -p 'test_*.py'
```

- [ ] **步骤 3：运行静态检查。**

```bash
python3 -m py_compile src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/sfu_4096x4096_capacity.py src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_4096x4096_capacity.py
```

```bash
bash -n src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh
```

```bash
git diff --check -- src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu docs/superpowers
```

- [ ] **步骤 4：记录 GEMM 隔离哈希。** 在 canonical root 中记录但不纳入 source commit：

```bash
sha256sum src/sst/elements/golem/tests/run_noc_dma_pipeline.sh src/sst/elements/golem/tests/architecture/ncores_selfcom_dma_ctrl.py src/sst/elements/golem/tests/small/mvm_noc_int_array/riscv64/test_noc_dma
```

  实验完成后必须对同三项重新计算并逐字比较。若任一项发生变化，停止容量实验结论，先
  运行现有 `run_noc_dma_pipeline.sh` 真实 GEMM 回归并定位原因。

- [ ] **步骤 5：更新 README 和持久记录。** 明确“当前最大已验证仍为 `256x4096`，
  `4096x4096` 尚未运行”，记录 canonical root、固定配置、运行命令和停止条件。

- [ ] **步骤 6：提交文档更新。**

```bash
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/README.md src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/task_plan.md src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/progress.md src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/findings.md
git commit -m "docs: prepare 4096 softmax capacity run"
```

---

### 任务 5：按前缀逐级执行四个真实 SST 点

**Artifact root：**

```text
/data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_4096x4096_capacity_explicit_noc_20260716
```

- [ ] **步骤 1：运行 `512x4096`。**

```bash
TMPDIR=/data4/jjgong/tmp GOLEM_SFU_CAPACITY_ROOT=/data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_4096x4096_capacity_explicit_noc_20260716 GOLEM_SFU_CAPACITY_POINT_LIST='512:4096:16:16' bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh
```

  预期 golden `2,097,152/0`，四类 reduction 各 `8,192`，transport `32,768`，
  DMA bytes `8,388,608`，全部 retry/rejected/stale 为 0。

- [ ] **步骤 2：确认 512 点 PASS 后加入 `1024x4096`。**

```bash
TMPDIR=/data4/jjgong/tmp GOLEM_SFU_CAPACITY_ROOT=/data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_4096x4096_capacity_explicit_noc_20260716 GOLEM_SFU_CAPACITY_POINT_LIST='512:4096:16:16 1024:4096:16:16' bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh
```

  预期先重新验证 512 cached PASS，再得到 golden `4,194,304/0`、单类 reduction
  `16,384`、transport `65,536`、DMA bytes `16,777,216`。

- [ ] **步骤 3：确认前两点 PASS 后加入 `2048x4096`。**

```bash
TMPDIR=/data4/jjgong/tmp GOLEM_SFU_CAPACITY_ROOT=/data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_4096x4096_capacity_explicit_noc_20260716 GOLEM_SFU_CAPACITY_POINT_LIST='512:4096:16:16 1024:4096:16:16 2048:4096:16:16' bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh
```

  预期 golden `8,388,608/0`、单类 reduction `32,768`、transport `131,072`、
  DMA bytes `33,554,432`。

- [ ] **步骤 4：确认前三点 PASS 后运行完整默认矩阵。**

```bash
TMPDIR=/data4/jjgong/tmp GOLEM_SFU_CAPACITY_ROOT=/data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_4096x4096_capacity_explicit_noc_20260716 bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_4096x4096_capacity.sh
```

  最终点预期 golden `16,777,216/0`、四类 reduction 各 `65,536`、transport
  `262,144`、DMA read/write operations `65,536`、DMA read/write bytes
  `67,108,864`，retry/rejected/stale 全为 0。

- [ ] **步骤 5：任何非 PASS 立即停止。** watchdog 超限时由现有 GNU `timeout` 和底层
  pipeline cleanup 终止 SST 进程组。记录 status、exit code、实际 wall time、first
  failing gate、child root、log 和资源快照；不得执行后续更大点，也不得扩大 timeout
  或修改固定配置后重试，原因分析转入后续独立任务。

---

### 任务 6：恢复验证、最终报告和回归收尾

**文件：**

- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/README.md`
- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/task_plan.md`
- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/progress.md`
- 修改：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/findings.md`

- [ ] **步骤 1：完整 resume。** 再次运行默认矩阵命令，预期四点全部重新解析为 cached
  PASS，children 下没有新增 attempt。

- [ ] **步骤 2：生成报告。**

```bash
TMPDIR=/data4/jjgong/tmp python3 src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/sfu_4096x4096_capacity.py report --root /data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_4096x4096_capacity_explicit_noc_20260716 --output-dir /data4/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/artifacts/sweeps/sfu_4096x4096_capacity_explicit_noc_20260716/report --verifier src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/verify_softmax_sfu_against_golden.py
```

- [ ] **步骤 3：确定性重建。** 连续执行两次报告命令，比较 source CSV 和 Markdown 的
  SHA-256；预期完全一致。

- [ ] **步骤 4：重新运行任务 4 的完整 focused suite、syntax、py_compile 和
  `git diff --check`。** 所有命令必须 PASS。

- [ ] **步骤 5：验证 GEMM 隔离哈希。** 与任务 4 的三项哈希逐字一致；确认本阶段没有
  修改 GEMM runner、architecture 或 guest binary。只有不一致时才运行原有真实 GEMM
  pipeline，并把结果作为阻塞项处理。

- [ ] **步骤 6：更新持久记录。** 只有四点全部 PASS 后，README/progress/findings 才能
  写“`4096x4096` 已完成真实 SST 验证”；否则写当前最大 PASS 和首个失败点，不夸大结论。

- [ ] **步骤 7：精确提交最终文档。**

```bash
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/README.md src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/task_plan.md src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/progress.md src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/findings.md
git commit -m "docs: record 4096 softmax capacity result"
```

## 最终验收清单

- [ ] 四点真实 SST 按顺序完成，或明确记录首个失败边界。
- [ ] `4096x4096` golden 为 `16,777,216 checked / 0 mismatches`。
- [ ] 四类 reduction totals 各 `65,536`；transport total `262,144`。
- [ ] DMA read/write bytes 各 `67,108,864`；全部 retry/rejected/stale 为 0。
- [ ] 固定 network/VN/worker/chunk/retry/memory/timeout signature 无漂移。
- [ ] 完整 resume 不启动新 SST，所有 child evidence 和 output hash 可重建。
- [ ] capacity source CSV 和中文 Markdown summary 确定性生成。
- [ ] 完整 Softmax focused suite、syntax、compile 和 diff hygiene PASS。
- [ ] Phase 4F 历史矩阵和 GEMM 路径未修改。

## 2026-07-16 执行状态更新

- [x] `512x4096` 真实 SST PASS，child wall `1462s`，golden `2,097,152/0`。
- [x] `1024x4096` 真实 SST PASS，child wall `2890s`，golden `4,194,304/0`。
- [ ] `2048x4096` 和 `4096x4096` 按用户决定延期；不得自动继续。
- [ ] 在恢复容量阶梯前，另行规划并验证不改变模拟语义的 wall-clock 优化。

本更新取代任务 5 的“立即继续完整阶梯”调度，但不降低最终 `4096x4096` 的验收合同。

## 恢复容量阶梯前的 wall-time 优化任务

- [ ] **W0：修复 custom guest 构建隔离。** 先写 focused contract test，证明 Softmax
  custom `VANADIS_EXE` 不重编/覆盖 GEMM guest 和 metadata；实现 opt-in 跳过默认 guest
  build，默认 GEMM dry-run 完全不变，然后运行原 `run_noc_dma_pipeline.sh` 回归。
- [ ] **W1：关闭 Vanadis pipeline trace。** 以 `64x4096` 单点 A/B，watchdog `600s`；
  要求 output SHA、simulated time 和全部计数完全一致，记录 wall-time speedup。
- [ ] **W2：关闭宿主逐事件文本。** 分别关闭 NoC debug、GM verbose、DMA trace 和 band
  trace，artifact gate 改用一次性配置证据与 stats；仍要求模拟结果完全一致。
- [ ] **W3：实现独立 benchmark quiet 模式。** 只屏蔽 guest 成功路径 debug `printf`，
  保留错误/PASS；把新 simulated-time 口径单独标识，不覆盖 Phase 4F 历史结果。
- [ ] **W4：评估 selective stats 和 SST `2/4` host threads。** 每次只改一项；紧耦合
  mesh 若无收益即保留单线程，不增加复杂度。
- [ ] **W5：用 `256x4096` 复验最佳低风险组合。** PASS 且收益稳定后，再决定是否另行
  设计事件驱动 SFU wait；未经批准不运行 1024/2048/4096。

每个 SST A/B 超过其 watchdog 必须取消并记录，不扩大 timeout 重跑。W0-W5 不得改变
网络带宽、VN、worker/band、chunk、retry、memory 或 Softmax 数学。
