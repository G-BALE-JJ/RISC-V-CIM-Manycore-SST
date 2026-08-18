# PPTX QA Report

## Output

- PPTX: `final_presentation_cn.pptx`
- Slides: 16
- Format: 16:9
- Speaker notes: 16/16 slides
- Visuals: all architecture diagrams, timelines and charts are editable PowerPoint-native shapes
- External figure assets: none

## Evidence and scope

- Architecture and data-flow claims were taken from the Row Engine design and implementation documents.
- Final cycle, timeline, scaling and stage-overlap values were taken from fresh `16/64/256/1024 x 4096` result JSON files.
- GPU values are labelled as user-supplied nominal-clock estimates and are not presented as hardware counter measurements.
- The `16.9k` next-stage target is explicitly labelled as a model estimate.

## Self-review

### High severity

- Slide 12 first render: early timeline event labels overlapped. Fixed by combining ACK/completion and removing redundant near-zero labels from the overview axis. The exact events remain in speaker notes and interval bars.

### Medium severity

- Slides 2, 4, 5, 10, 11 and 15 were flagged by a character-count heuristic. Rendered inspection confirmed that text remains inside boxes and readable at 16:9 presentation scale; no clipping was observed.

### Low severity

- Mixed Chinese/English typography depends on Office font substitution. The deck uses `Noto Sans CJK SC` and `Aptos`; LibreOffice rendering was checked successfully.

## Verification

- Reopened successfully with `python-pptx`.
- PPTX ZIP integrity: no errors.
- Shape bounds: zero shapes outside slide canvas.
- Rendered with headless LibreOffice to PDF.
- All 16 rendered slides inspected as a contact sheet; architecture, data-flow and result slides were additionally inspected at full size.
- Revised slide 12 was rendered again and confirmed free of label overlap.

## Known limitations

- The deck does not claim silicon area, power or real-hardware throughput because the current work is an SST timing model.
- Whole-system cycles from the expanded clean instrumentation build are intentionally excluded from performance comparisons because post-kernel diagnostics change Vanadis retirement time.

