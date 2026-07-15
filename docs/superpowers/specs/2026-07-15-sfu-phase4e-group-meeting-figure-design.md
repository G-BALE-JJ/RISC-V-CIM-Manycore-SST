# SFU Phase 4E Group-Meeting Figure Design

## Purpose

Create one English 16:9 figure for a group meeting that preserves the Phase 4E
controlled-matrix evidence and communicates one conclusion:

> VN0, VN1, and VN2 are equivalent without backpressure; increasing worker
> count raises reduction latency and NoC pressure while end-to-end runtime
> remains nearly unchanged.

The figure reports completed experiments only. It does not include the future
softmax roadmap or GEMM+softmax fusion plans.

## Figure Contract

- **Archetype:** quantitative grid with a dominant latency panel.
- **Backend:** Python with matplotlib only.
- **Canvas:** 16:9 slide, 13.333 x 7.5 inches.
- **Language:** English.
- **Exports:** editable SVG, editable-text PDF, and 300 dpi PNG preview.
- **Source data:** a generated CSV parsed directly from the four Phase 4E
  artifact roots.
- **Statistics:** deterministic single SST runs; no error bars, confidence
  intervals, or inferential tests. The figure must label these values as single
  simulation outcomes.

## Source Artifacts

Parse these roots without modifying them:

- `sfu_phase4e_explicit_vn0_matrix_20260715`
- `sfu_phase4e_explicit_vn1_matrix_20260715`
- `sfu_phase4e_explicit_vn2_matrix_20260715`
- `sfu_phase4e_modeled_control_matrix_20260715`

For each point, read:

- `sweep_manifest.csv` for configuration and PASS status;
- the SST log for simulated time;
- `stats_selfcom.txt` for transport received, latency, inbox high-water, and
  stale-drop statistics;
- `noc_summary.csv` for packet, bit, and xbar-stall totals;
- `dma_summary.csv` for lifecycle and retry gates.

The parser must fail if any selected row is not `PASS/PASS`, golden evidence is
missing, DMA retry/exhaustion is nonzero, or an explicit point has an unexpected
transport-event total.

## Panel Design

### Panel a: End-to-End Runtime

- X-axis: worker cores, `4`, `8`, and `16`.
- Y-axis: simulated time in microseconds.
- Show modeled-NoC as a neutral gray line with diamond markers.
- Show explicit VN0/VN1/VN2 as compact, slightly offset markers at each worker
  count so their identical values remain visible.
- Use one restrained blue family and distinct marker shapes for the three VNs;
  do not imply that the VNs are different methods.
- Directly annotate: `VN0/VN1/VN2 overlap exactly`.
- Add a compact note: `max |explicit - modeled| < 0.061%`.

This panel establishes that the real event path has negligible effect on total
runtime in the current no-backpressure profile.

### Panel b: Reduction Transport Latency (Hero)

- X-axis: worker cores.
- Y-axis: latency in cycles.
- Plot aggregate average latency as a solid signal line.
- Plot maximum observed latency as a dashed accent line.
- Because all three VNs are identical, plot one explicit-NoC latency trace and
  state `identical for VN0/VN1/VN2` rather than drawing three overlapping lines.
- Label the average endpoints: `9,440` and `15,583 cycles`.
- Add a concise callout: `+65% average latency from 4 to 16 workers`.

This is the hero panel because it carries the primary scaling result.

### Panel c: NoC Pressure and Validation

- Use grouped bars for explicit-NoC and modeled-NoC xbar stalls at each worker
  count.
- Keep modeled bars neutral gray and explicit bars in the same signal blue used
  elsewhere.
- Place transport-event totals `256 / 512 / 1024` above or below the explicit
  groups without creating a second quantitative axis.
- Include a compact evidence block inside the panel:
  - `Inbox high-water = 4`
  - `Queued / rejected / stale = 0`
  - `Golden = 8192 checked, 0 mismatches (all points)`
  - `DMA retry / exhaustion = 0`

This panel demonstrates increasing network pressure while retaining complete
correctness and lifecycle validation.

## Visual System

- White background; no gradient or decorative effects.
- Dark neutral text and axes.
- Neutral gray for modeled-NoC.
- Low-saturation blue for explicit-NoC primary evidence.
- Muted cyan and navy marker variations distinguish VN0/VN1/VN2 only where
  required in panel a.
- Orange is reserved for the maximum-latency trace and the `+65%` callout.
- Minimum slide text size: 11 pt; panel labels: bold 15 pt; main title: 20-22 pt.
- No repeated legends. Use direct labels where practical and one compact shared
  legend only if panel a becomes ambiguous.
- Preserve stable panel geometry so annotations cannot resize the layout.

## Output Layout

Use an asymmetric GridSpec:

```text
+----------------------+----------------------------------+
| a  End-to-end time   | b  Reduction latency (hero)     |
|                      |                                  |
+----------------------+----------------------------------+
| c  NoC pressure and validation (full width)             |
+---------------------------------------------------------+
```

Panel b receives the largest plotting area. Panel c is shallow but wide enough
for grouped bars and the validation block.

## Deliverables

Store the reusable bundle under
`tests/artifacts/sweeps/sfu_phase4e_group_report_20260715/figures/`:

- `plot_sfu_phase4e_group_figure.py`
- `sfu_phase4e_group_figure_source_data.csv`
- `sfu_phase4e_group_figure.svg`
- `sfu_phase4e_group_figure.pdf`
- `sfu_phase4e_group_figure.png`
- `sfu_phase4e_group_figure_qa.md`

## QA Requirements

- Open the PNG preview and inspect the full 16:9 composition.
- Confirm all labels are readable at normal slide size and no text overlaps.
- Confirm VN markers in panel a remain distinguishable despite equal values.
- Confirm panel b is visually dominant and the `+65%` statement matches parsed
  source data.
- Confirm method spelling is consistent as `explicit-NoC` and
  `modeled-NoC`.
- Confirm the SVG keeps text editable and the PDF uses TrueType text.
- Confirm the source-data CSV reconstructs every plotted number.
- Do not report statistical significance because each configuration has one
  deterministic run.

## Scope After This Figure

The technical roadmap remains softmax-first: explore softmax performance and
large-scale configurations before considering GEMM+softmax fusion. That roadmap
is documented separately and is intentionally excluded from this result-only
figure.
