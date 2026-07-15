# SFU Phase 4F 大规模 Explicit-NoC Softmax 实施计划

> **面向 agentic workers：** 必须逐任务使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 执行；所有步骤使用复选框跟踪，未经测试和审查不得跨任务推进。

**目标：** 在成熟 GEMM 实际使用的固定网络配置下，完成 8 个大规模 unified-job `explicit_noc` softmax 实验点，并生成可追溯的父 manifest、源数据和单张 16:9 英文结果图。

**架构：** 新增一个 shell parent runner，逐点调用现有 `run_sfu_unified_job_distributed_scaling.sh`。新增一个 Python 证据模块，统一负责 child artifact 复核、父 manifest 原子更新、派生指标和绘图；runner 与报告共用同一解析口径。所有实现局限在 softmax 测试目录，不修改 GEMM runner、architecture、SFU、GlobalMemory 或 SimpleNetwork。

**技术栈：** Bash、Python 3 标准库、matplotlib、unittest、现有 SST runner、现有 golden verifier、CSV/SST 日志与统计 artifact。

## 全局约束

- 设计依据：`docs/superpowers/specs/2026-07-15-sfu-phase4f-large-scale-explicit-noc-softmax-design.md`。
- transport 固定为 `explicit_noc`；固定 `num_vns=3`、`reduction_vn=0`、`dma_response_vn=0`；不运行 `modeled_noc`，不扫描 VN 或 bandwidth。
- 实际 GEMM 网络参数必须显式固定并逐点验证：

```text
GOLEM_NOC_LINK_BW=1200GB/s
GOLEM_NOC_XBAR_BW=1200GB/s
GOLEM_DIRCTRL_HIGHLINK_BW=1200GB/s
GOLEM_NOC_INPUT_BUF_SIZE=512KB
GOLEM_NOC_OUTPUT_BUF_SIZE=512KB
GOLEM_NOC_FLIT_SIZE=128B
GOLEM_GM_BUFFER_LENGTH=1024KB
GOLEM_NOC_INTER_ROUTER_NO_CUT=0
GOLEM_NOC_LOCAL_NO_CUT=0
```

- softmax 参数固定为 `chunk=256`、`staging_rows=4`、`job_rows=4`、`retry_ticks=1024`、`max_retries=8`。
- 默认矩阵严格为：`16:512:16:16 16:1024:16:16 16:2048:16:16 16:4096:16:16 16:4096:4:4 16:4096:8:8 64:4096:16:16 256:4096:16:16`。
- `dim<=1024` 使用 `134217728` bytes/node；`dim>=2048` 使用 `268435456` bytes/node。
- timeout 为：`16x512=900s`、`16x1024=1800s`、`16x2048=2400s`、`16x4096=3600s`、`64x4096=7200s`、`256x4096=14400s`。
- 每个 PASS 点必须通过 full-row golden、SFU worker/band、四类 reduction counter、explicit transport、GlobalMemory lifecycle、DMA lifecycle、output size/hash 和运行时网络配置 gate。
- queueing 可记录为结果，但 rejected、stale、DMA retry/exhaustion/write retry 必须为 0。
- 不得修改 `run_noc_dma_pipeline.sh`、`tests/architecture/*`、GEMM guest binary 或 production component。

---

### 任务 1：用测试锁定数据模型、矩阵和配置解析

**文件：**
- 新建：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_phase4f_large_scale.py`
- 新建：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/plot_sfu_phase4f_large_scale.py`

**接口：**

```python
@dataclasses.dataclass(frozen=True)
class PointSpec:
    stage: str
    rows: int
    dim: int
    worker_cores: int
    band_cores: int
    mem_node_size: int
    timeout_sec: int

@dataclasses.dataclass(frozen=True)
class PointRecord:
    spec: PointSpec
    run_id: str
    chunk_elems: int
    cooperative_groups: int
    transport: str
    reduction_vn: int
    num_vns: int
    dma_response_vn: int
    noc_link_bw: str
    noc_xbar_bw: str
    dirctrl_highlink_bw: str
    noc_input_buffer: str
    noc_output_buffer: str
    gm_buffer: str
    flit_size: str
    retry_ticks: int
    max_retries: int
    status: str
    exit_code: int
    artifact_validation: str
    golden_checked: int | None
    golden_mismatches: int | None
    transport_events: int | None
    transport_immediate: int | None
    transport_queued: int | None
    transport_rejected: int | None
    transport_stale: int | None
    inbox_high_water: int | None
    latency_avg_cycles: float | None
    latency_max_cycles: int | None
    total_send_packets: int | None
    total_send_bits: int | None
    total_xbar_stalls: int | None
    simulated_time_us: float | None
    wall_time_sec: float | None
    dma_timeout_retry: int | None
    dma_timeout_exhausted: int | None
    dma_write_timeout_retry: int | None
    output_sha256: str | None
    child_root: str
```

- [ ] **步骤 1：写 RED 测试。** 精确断言 9 个 canonical network 值、8 点顺序、唯一 anchor、stage `A/A/A/A/B/B/C/C`、memory 和 timeout 映射；拒绝矩阵外 shape、重复点和非法整数。
- [ ] **步骤 2：运行 RED。**

```bash
python3 -m unittest src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_phase4f_large_scale.py -v
```

预期因报告模块或接口缺失而失败。

- [ ] **步骤 3：实现 `CANONICAL_NETWORK`、`DEFAULT_POINTS`、`resolve_point()` 和 `parse_point_list()`，其值逐字复制全局约束。
- [ ] **步骤 4：重跑测试并确认 GREEN。**
- [ ] **步骤 5：提交。**

```bash
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/{test_sfu_phase4f_large_scale.py,plot_sfu_phase4f_large_scale.py}
git commit -m "test: define Phase 4F large-scale contracts"
```

---

### 任务 2：实现统一的 child artifact 解析与强制 gate

**文件：** 修改任务 1 的两个 Python 文件。

**接口：**

- `select_child_manifest_row(child_root: pathlib.Path, spec: PointSpec) -> dict[str, str]`
- `parse_child_point(child_root: pathlib.Path, spec: PointSpec, verifier: pathlib.Path) -> PointRecord`
- `upsert_parent_manifest(manifest: pathlib.Path, record: PointRecord) -> None`
- `load_parent_manifest(manifest: pathlib.Path) -> list[PointRecord]`

- [ ] **步骤 1：创建按 shape 推导计数的 synthetic fixture。** 生成 child manifest、唯一 SST log、stats/NoC/DMA/run summary、输入、正确尺寸 output 和 verifier stub。
- [ ] **步骤 2：写 manifest/golden RED 测试。** 只接受唯一 `PASS/PASS/exit_code=0` row；允许后续 `PASS/CACHED`；拒绝 duplicate canonical row、旧 schema、错误 shape/VN/transport、错误 checked、mismatch 和 hash/size 不一致。
- [ ] **步骤 3：写运行时网络 RED 测试。** SST log 必须精确包含：

```text
[NoC] input_buf_size=512KB, output_buf_size=512KB, link_bw=1200GB/s, xbar_bw=1200GB/s, flit_size=128B
[NoC] inter_router_no_cut=0, local_no_cut=0
[GOLEM] GlobalMemory link buffer_length=1024KB
GlobalMemory VN mapping: request_vn=0 response_vn=1 reduction_vn=0 (num_vns=3)
resolved golem_dma_response_vn=0 num_vns=3 explicit=1
```

`run_summary.csv` 的 link/xbar/flit/directory highlink/memory size 也必须精确匹配。

- [ ] **步骤 4：写 counter RED 测试。** 四类 reduction totals=`rows*workers`；transport=`4*rows*workers`；immediate+queued 等于 transport；rejected/stale=0；DMA operations=`rows*workers`、bytes=`rows*dim*4`、三类 retry=0。
- [ ] **步骤 5：写性能字段 RED 测试。** 聚合 latency average/max、NoC packets/bits/xbar stalls、wall time，并把完成日志中的 us/ms/s 转为微秒。
- [ ] **步骤 6：实现解析器。** 使用 `csv.DictReader(strict=True)`、`pathlib`、`re` 和 `subprocess.run()`；异常必须包含 child root、run ID 和 field。
- [ ] **步骤 7：实现父 manifest 原子 upsert。** 完整 identity 为 key，临时文件写完后 `os.replace()`；旧 schema 直接失败。
- [ ] **步骤 8：提供 runner 使用的 collect CLI。** 命令
  `python3 plot_sfu_phase4f_large_scale.py collect --child-root ROOT --stage A --rows 16 --dim 512 --workers 16 --bands 16 --parent-manifest MANIFEST`
  必须调用同一个 `parse_child_point()` 并原子 upsert；stdout 只返回规范化 run ID 和 output SHA-256。
- [ ] **步骤 9：重跑测试并提交。**

```bash
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/{test_sfu_phase4f_large_scale.py,plot_sfu_phase4f_large_scale.py}
git commit -m "feat: validate Phase 4F large-scale artifacts"
```

---

### 任务 3：实现固定网络配置的 parent runner

**文件：**
- 新建：`src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_phase4f_large_scale_explicit_noc.sh`
- 修改：`test_sfu_phase4f_large_scale.py`

**公开控制：**
- `GOLEM_PHASE4F_LARGE_SCALE_ROOT=<fresh absolute root>`
- `GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN=1`
- `GOLEM_PHASE4F_LARGE_SCALE_STOP_ON_FAIL=1`
- `GOLEM_PHASE4F_LARGE_SCALE_POINT_LIST="rows:dim:workers:bands [more-points]"`

- [ ] **步骤 1：写 runner RED 测试。** 真实 child dry-run 必须生成 8 个唯一点，且 stage、memory、timeout、child root identity 正确。
- [ ] **步骤 2：写 conflict RED 测试。** 注入错误 transport、VN、link/xbar/highlink、buffer、flit、no-cut、chunk、staging/job rows、retry 和自定义 pipeline args；必须在 child artifact 创建前退出 2。
- [ ] **步骤 3：写 lifecycle RED 测试。** 覆盖 parent flock、旧 schema、重复/非法点、损坏 marker、签名/hash 漂移、dry-run、resume 和 stop-on-fail。
- [ ] **步骤 4：实现 `require_unset_or_equal NAME EXPECTED`，检查通过后 export 全部 canonical network；拒绝继承的 child pipeline args。
- [ ] **步骤 5：逐点调用 child runner，核心环境必须为：

```bash
GOLEM_SWEEP_ROOT="$child_root" \
GOLEM_DRY_RUN_SWEEP="$DRY_RUN" \
GOLEM_STOP_ON_FAIL=1 \
GOLEM_SFU_DISTRIBUTED_POINT_LIST="$rows:$dim:$workers:$bands" \
GOLEM_SFU_DISTRIBUTED_REDUCTION_TRANSPORT=explicit_noc \
GOLEM_SFU_VN_SWEEP=1 GOLEM_SFU_REDUCTION_VN=0 GOLEM_DMA_RESPONSE_VN=0 \
GOLEM_SFU_DISTRIBUTED_CHUNK_ELEMS=256 \
GOLEM_SFU_DISTRIBUTED_STAGING_ROWS=4 GOLEM_SFU_DISTRIBUTED_JOB_ROWS=4 \
GOLEM_SFU_DISTRIBUTED_RETRY_TICKS=1024 GOLEM_SFU_DISTRIBUTED_MAX_RETRIES=8 \
GOLEM_TIMEOUT_512="$timeout_sec" GOLEM_TIMEOUT_1024="$timeout_sec" \
GOLEM_SFU_DISTRIBUTED_PIPELINE_ARGS="--noc-in-buf 512KB --noc-out-buf 512KB --noc-link-bw 1200GB/s --noc-xbar-bw 1200GB/s --noc-flit-size 128B --gm-buf 1024KB --mem-node-size $mem_node_size" \
bash "$SCRIPT_DIR/run_sfu_unified_job_distributed_scaling.sh"
```

directory highlink 和 no-cut 通过 parent 已 export 环境进入 architecture。
`GOLEM_SFU_VN_SWEEP=1` 只用于复用 child runner 已有的严格 runtime VN/DMA-VN
验证；parent 始终固定 `reduction_vn=0` 且不迭代 VN，因此这不是 VN 性能扫描。

- [ ] **步骤 6：实现 signature/marker。** 包含 schema、stage、shape、workers、transport/VN、全部网络值、memory、timeout、softmax 参数、child runner SHA 和 pipeline args SHA；缓存命中前重新运行 Python artifact gate。
- [ ] **步骤 7：实现状态规则。** TIMEOUT 正式记录；Stage C 任一点非 PASS 立即停止；不得自动降低 rows、network 或 retry。
- [ ] **步骤 8：运行语法和 focused tests。**

```bash
bash -n src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_phase4f_large_scale_explicit_noc.sh
python3 -m unittest src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/test_sfu_phase4f_large_scale.py -v
```

- [ ] **步骤 9：提交。**

```bash
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/{run_sfu_phase4f_large_scale_explicit_noc.sh,test_sfu_phase4f_large_scale.py}
git commit -m "feat: orchestrate Phase 4F large-scale runs"
```

---

### 任务 4：实现 source data、派生指标和 16:9 结果图

**文件：** 修改两个 Python 文件。

**接口：**

- `validate_complete_matrix(records: list[PointRecord]) -> None`
- `write_source_csv(records: list[PointRecord], path: pathlib.Path) -> None`
- `load_source_csv(path: pathlib.Path) -> list[PointRecord]`
- `derive_metrics(records: list[PointRecord]) -> dict[str, float]`
- `render_figure(records: list[PointRecord], output_prefix: pathlib.Path) -> None`
- `write_qa(records: list[PointRecord], output_path: pathlib.Path) -> None`

- [ ] **步骤 1：写 RED 测试。** 报告必须识别 8 个唯一 outcome；PASS 点检查 time/row、time/element、以 workers=4 为基准的 speedup/efficiency；TIMEOUT/FAIL 点保留在 source CSV 和 QA，但不进入趋势连线。缺失 identity、重复点、PASS 点网络漂移或 lifecycle error 必须失败。
- [ ] **步骤 2：写 CSV round-trip 和 export RED 测试。** CSV 逐字段可逆且确定性排序；SVG 保留 text，PDF TrueType，PNG 300 dpi、16:9；重复生成的 source CSV 和 SVG 字节一致。
- [ ] **步骤 3：实现 CLI。** 必须先重新解析 child evidence，再写 source CSV 和图；任一 gate 失败时不得 export。
- [ ] **步骤 4：实现英文四面板图。** `figsize=(13.333,7.5)`、`svg.fonttype=none`、`pdf.fonttype=42`；展示 dimension runtime/latency、worker speedup/efficiency、row total/normalized time、NoC pressure 与 correctness/lifecycle，并标注 fixed GEMM network profile。
- [ ] **步骤 5：实现 QA Markdown。** 列出 8 点 identity、固定网络、每点状态、golden/transport/DMA gate 和输出 hash；TIMEOUT/FAIL 必须注明停止原因和最后有效 shape。不得出现 modeled-NoC、bandwidth comparison 或 fusion roadmap。
- [ ] **步骤 6：使用 `/data4/jjgong/.venvs/golem-plot/bin/python` 运行测试并提交。**

```bash
git add src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/{plot_sfu_phase4f_large_scale.py,test_sfu_phase4f_large_scale.py}
git commit -m "feat: report Phase 4F large-scale results"
```

---

### 任务 5：执行 dry-run、focused 回归和 GEMM 隔离 gate

- [ ] **步骤 1：运行 8 点 dry-run。**

```bash
GOLEM_PHASE4F_LARGE_SCALE_ROOT=/tmp/sfu_phase4f_dryrun GOLEM_PHASE4F_LARGE_SCALE_DRY_RUN=1 \
bash src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_phase4f_large_scale_explicit_noc.sh
```

预期 8 个唯一 DRYRUN 点，network/memory/timeout/stage 正确，无 SST 进程。

- [ ] **步骤 2：运行完整 softmax focused suite。**

```bash
python3 -m unittest discover -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu -p 'test_*.py' -v
bash -n src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/run_sfu_phase4f_large_scale_explicit_noc.sh
python3 -m py_compile src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/{plot_sfu_phase4f_large_scale.py,test_sfu_phase4f_large_scale.py}
git diff --check
```

- [ ] **步骤 3：运行 GEMM 隔离 gate：`git diff --name-only 0f2e2b4..HEAD`。** 预期只出现 Phase 4F 文件和记录。若出现 shared/production 文件，先停止实验；production 源变化时重编 `libgolem.so`，并运行既有 `run_noc_dma_pipeline.sh` 默认 GEMM 回归，要求 exit 0、VERIFY-C PASS、无 SFU/reduction activity、DMA retry=0。

---

### 任务 6：分阶段执行真实 SST

**artifact root：** `src/sst/elements/golem/tests/artifacts/sweeps/sfu_phase4f_large_scale_explicit_noc_20260715`。

- [ ] **步骤 1：Stage A。** point list=`16:512:16:16 16:1024:16:16 16:2048:16:16 16:4096:16:16`。
- [ ] **步骤 2：Stage B。** 同一 root，point list=`16:4096:4:4 16:4096:8:8`；16-worker anchor 只验证复用。
- [ ] **步骤 3：Stage C。** 同一 root，point list=`64:4096:16:16 256:4096:16:16`；首个 TIMEOUT/FAIL 后停止，不调整参数。
- [ ] **步骤 4：每点检查。** manifest PASS、golden=`rows*dim/0`、四类 reduction=`rows*workers`、transport=`4*rows*workers`、DMA retry=0、runtime network 精确匹配。
- [ ] **步骤 5：重新运行完整默认矩阵验证 resume。** 不得启动新 SST，父 manifest 仍只有 8 个 identity。

---

### 任务 7：生成报告、视觉 QA 和更新记录

**文件：** 更新 `README.md/findings.md/progress.md`；生成 report 下的 CSV/SVG/PDF/PNG/QA。

- [ ] **步骤 1：运行报告命令。**

```bash
/data4/jjgong/.venvs/golem-plot/bin/python \
  src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu/plot_sfu_phase4f_large_scale.py \
  --root src/sst/elements/golem/tests/artifacts/sweeps/sfu_phase4f_large_scale_explicit_noc_20260715 \
  --output-dir src/sst/elements/golem/tests/artifacts/sweeps/sfu_phase4f_large_scale_explicit_noc_20260715/report
```

- [ ] **步骤 2：视觉检查 PNG/SVG。** 检查 16:9、标题/轴/图例不重叠、标签不越界、fixed network profile 可读、无 modeled-NoC 或 bandwidth sweep 文案。
- [ ] **步骤 3：从 source CSV 重建图。** SVG 和 QA 数值/hash 必须确定性一致。
- [ ] **步骤 4：更新中文记录。** 明确每点状态、artifact root、规模趋势和瓶颈；不得把 TIMEOUT 写成 PASS。
- [ ] **步骤 5：重跑任务 5 的所有验证后提交记录。**

## 完成判据

- parent runner、parser/report 和 focused tests 全部通过。
- 8 个默认 identity 均有明确 PASS/TIMEOUT/FAIL，不存在重复 anchor 或参数静默回退。
- 每个可用点的运行时网络配置与实际 GEMM 参数完全一致。
- 每个 PASS 点通过 golden、reduction、explicit transport、DMA、output hash 和 artifact gate。
- source CSV 可重建英文 16:9 SVG/PDF/PNG，QA 明确列出证据和限制。
- 主矩阵不包含 modeled-NoC、VN sweep 或 bandwidth sweep。
- GEMM/shared 路径未修改；若发生 shared 修改，则既有 GEMM 回归必须重新 PASS。
