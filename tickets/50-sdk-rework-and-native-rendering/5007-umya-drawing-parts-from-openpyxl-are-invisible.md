# Ticket: 5007 Drawing Parts From openpyxl-Authored Workbooks Are Invisible

## Depends On
- none (found during 5004)

## Why
umya matches drawing and chart XML by literal prefixed names (`xdr:wsDr`, `c:chartSpace`) and expects relative relationship targets. openpyxl writes default-namespaced roots and absolute `Target="/xl/..."` paths. umya therefore reports no drawings for such workbooks, so the renderer cannot even raise `chart_omitted` or `image_omitted`, and every other surface that inspects drawings is blind to them. The 5004 fixture generator rewrites both forms to what Excel writes as a workaround.

## Scope
- Reproduce with an unmodified openpyxl chart and image workbook.
- Fix in the PSU3D0 umya fork (namespace-aware matching, absolute target normalization) and bump the workspace patch rev.
- Remove the rewrite step from `crates/agent-spreadsheet-render/tests/fixtures/generate.py` and confirm the chart and image warnings still fire.

## Definition of Done
- openpyxl-authored charts and images produce the structured warnings without fixture rewriting.

## Findings (repro 2026-09-02, umya fork rev 4b64d65)

Repro: an unmodified openpyxl 3.1 workbook with one `BarChart` added via `ws.add_chart(chart, "E2")`. `screenshot_sheet` returns `fidelity: full` and `warnings: []`, so the omission is silent, which is the worst outcome the warning taxonomy was built to prevent.

There are three independent gaps, not two. Each one alone is enough to lose the chart.

1. **Absolute relationship targets.** `RawFile::set_attributes` (`src/structs/raw/raw_file.rs:75`) joins `base_path` and `Target` with `join_paths` and never handles a leading `/`, so `/xl/drawings/drawing1.xml` resolves to a zip entry that does not exist and the drawing part is skipped.
2. **Default-namespaced roots.** `reader/xlsx/drawing.rs:23` matches `b"xdr:wsDr"` literally and `reader/xlsx/chart.rs:21` matches `b"c:chartSpace"`. The reader has 685 literal prefixed-name matches overall, so a fully namespace-aware reader is a large change; the bounded fix is to resolve the prefix for the root and anchor element names that gate detection, or to canonicalize the part text (strip the default `xmlns`, add the conventional prefix) before handing it to the existing parser.
3. **Charts in `oneCellAnchor`.** `structs/drawing/spreadsheet/one_cell_anchor.rs` handles `from`, `grpSp`, `sp`, and `pic` but not `graphicFrame`, while `two_cell_anchor.rs:216` does. openpyxl anchors charts with `oneCellAnchor` by default, so a chart is dropped even after gaps 1 and 2 are fixed. The fixture generator works around this too by forcing `TwoCellAnchor` in `chart_bar`; the repro above with the generator's `excelize` rewrite applied still produced no warning until the anchor was changed.

Scope amendment: the Definition of Done must include a chart in openpyxl's default `oneCellAnchor`, and removing the `excelize` step from `generate.py` must be paired with removing the forced `TwoCellAnchor`. Workbooks written by Excel, xlsxwriter, and LibreOffice use prefixed names, relative targets, and `twoCellAnchor`, so they are not affected; this is specifically about openpyxl output, which is the common case for agent-authored workbooks.
