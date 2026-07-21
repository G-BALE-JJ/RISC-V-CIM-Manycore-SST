# Softmax Row Engine Engineering Handoff

**Status:** causal `1024x4096` Row Engine path implemented and verified on
2026-07-21

**Branch:** `softmax-update`

**Final artifact:**
`/data4/jjgong/tmp/muticore_softmax_causal_dedupe_r1024_d4096`

## Current Result

The production path is:

`descriptor -> 16 NoC band dispatches -> per-row input DMA -> MAX -> EXP/SUM -> NORMALIZE -> per-row output DMA ACK -> 16 unique band completions -> accelerator ready -> guest wait return`

For FP32 `1024x4096` row-wise Softmax:

| Metric | Result |
| --- | ---: |
| Actual descriptor-to-accelerator completion | 66,958 cycles |
| Analytical compute reference | 66,061 cycles |
| Clean guest kernel window | 73,309 cycles |
| Whole SST interval at 2.3 GHz | 640,921 cycles / 278.661 us |
| Golden values checked / mismatches | 4,194,304 / 0 |
| Input DMA / output DMA ACK | 1,024 / 1,024 |
| MAX / EXP-SUM / NORMALIZE events | 1,024 / 1,024 / 1,024 |
| Unique band completions | 16 |
| Reduction requests / wait polls | 0 / 0 |
| Maximum NoC port utilization | 1.257% |

The non-overlapping critical path is `11 + 256 + 66,549 + 88 + 54 = 66,958`
cycles. The historical `66,062` value was an independently predicted ready
tick and is not end-to-end completion.

## Implementation Boundaries

Production changes are concentrated in:

- `src/sst/elements/golem/sfu/sfu.{h,cc}`: tensor controller, physical row
  contexts, stage events, resource scheduling, and completion aggregation.
- `src/sst/elements/golem/globalmemory/globalmemory.{h,cc}`: tensor control
  transport and DMA completion support.
- `src/sst/elements/golem/rocc/roccAnalog.h`: completion-aware waits and
  waitable output stores.
- `src/sst/elements/golem/tests/architecture/cpu_builder.py` and the archive
  architecture shim: parameter and endpoint wiring.
- `src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu`: guest ABI, runtime,
  build isolation, runner contracts, and focused source tests.
- This directory: dedicated runner, parser, layout tests, figures, and current
  result documentation.

Completion safety requires output DMA ACK plus 16 distinct, identity-checked
band completions. Job, row, worker, shape, and band mismatches are rejected;
unsafe transport/scratch mappings and overlapping tensor jobs cannot silently
share physical contexts.

## Reproduction And Verification

Run the focused source/model tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu \
  -p 'test_sfu_softmax_*.py'

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/muticore_softmax \
  -p 'test_*.py'
```

Build the production component and guest:

```bash
make -C build/sst-elements/src/sst/elements/golem -j2
make -C src/sst/elements/golem/tests/small/muticore_softmax all
```

Run a smoke or the target workload:

```bash
src/sst/elements/golem/tests/small/muticore_softmax/run_muticore_softmax.sh \
  --rows 16 --cols 4096 --timeout 600

src/sst/elements/golem/tests/small/muticore_softmax/run_muticore_softmax.sh \
  --rows 1024 --cols 4096 --timeout 3600
```

The target artifact is intentionally outside Git. Do not stage generated HBM
images, tensors, SST logs, build products, or `/data4/jjgong/tmp` data.

## Performance Conclusion

EXP/SUM contributes 65,536 of 98,304 aggregate active service cycles and is
the current compute throughput bottleneck. The default NoC is lightly loaded,
but remains causal: at `16x4096`, lowering NoC/DirCtrl bandwidth from 1200 to
64 GB/s increases completion from 2,076 to 4,294 cycles. Reducing EXP lanes
from four to two increases it to 3,100 cycles.

The next optimization should model a higher-throughput pipelined EXP datapath
and validate area/throughput assumptions before widening it. After compute is
reduced, output streaming and DMA are expected to become the next boundary.

## Known Limits

- This is causal functional simulation with event-modeled stage latency, not
  RTL or gate-level floating-point execution.
- The algorithm is stable max/exp-sum/normalize Softmax, not a strict online
  running-max/running-sum implementation.
- Rows wider than 4096 may require the planned 2/4-tile pair collective, which
  is not implemented on the accepted row-local path.
- The final closeout runs cover focused tests and component/guest builds. Run
  the canonical long GEMM regression before merging changes to shared Golem
  production components if it has not been run in the new session.
- Local TIFF exports exceed 200 MiB in aggregate. Stage PNG/SVG/CSV sources by
  default; add TIFF files only when a publication workflow explicitly needs
  them. PDFs are ignored by the repository `.gitignore`.

## GitHub Boundary

Use explicit path staging. Do not use `git add .`; group-meeting reports, PPT,
Draw.io, rendered presentation assets, local TIFF exports, and temporary run
artifacts are outside this engineering closeout unless explicitly requested.

Before committing, review `git status --short`, `git diff --cached --stat`, and
`git diff --cached --check`. Push the existing branch to the contributor fork
with `git push -u gbale softmax-update`; do not force-push.

This checkout currently has a stale, empty `.git/index.lock` dated 2026-07-16.
Confirm that no Git process is active, then remove only that lock before
staging. Do not remove it while another Git command is running.

## New-Session Prompt

```text
请继续处理 RISC-V-CIM-Manycore-SST 的新任务。仓库位于
/data4/jjgong/RISC-V-CIM-Manycore-SST，当前分支是 softmax-update，工作区不是干净的，
请保留所有现有修改，不要 reset、checkout 或覆盖用户改动。

开始前请阅读：
1. src/sst/elements/golem/tests/small/muticore_softmax/PROJECT_HANDOFF.md
2. src/sst/elements/golem/tests/small/muticore_softmax/README.md
3. src/sst/elements/golem/tests/small/muticore_softmax/findings.md
4. src/sst/elements/golem/tests/small/muticore_softmax/progress.md
5. docs/superpowers/specs/2026-07-16-sfu-softmax-row-engine-noc-architecture-design.md

当前有效基线是 1024x4096 实际 accelerator completion 66,958 cycles；66,061 只
是分析计算参考，历史 66,062 不是当前端到端完成时间。真实路径必须保持
input DMA -> MAX -> EXP/SUM -> NORMALIZE -> output DMA ACK -> unique band completion
的因果顺序。除非我明确要求，不要修改组会报告、PPT、Draw.io 或汇报素材。

我的新任务是：<在这里填写新任务>。

请先检查 git status 和相关实现，再完成修改、测试与结果总结。如果改动共享的 SFU、
GlobalMemory 或 RoCC 组件，请运行对应 focused tests、重编 libgolem，并评估 GEMM 回归。
```
