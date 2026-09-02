# Ticket: 5003 Renderer Bake-off

## Depends On
- none

## Why
Before building `agent-spreadsheet-render` we want measured evidence on the display-list plus tiny-skia architecture, the text stack cost, and fidelity against a comparison oracle. Three independent projects converged on the same design (BetterOffice `xlsx-render`/`xlsx-raster`, readany-render, and office2pdf which paints through Typst over umya). None is a runtime dependency candidate for the WASM path, but each is a reference.

## Owner / Effort / Risk
- Owner: rendering
- Effort: M (about one week)
- Risk: Low

## Scope
- Corpus: pick about 40 workbooks. Sources: `crates/agent-spreadsheet/tests/fixtures`, `../formualizer/benchmarks/corpus/synthetic`, `../calamine/tests`, a few IronCalc calc tests, and generated styled fixtures produced with `asp` style, border, merge, column-size, and sheet-layout batch writes so that fonts, fills, borders, merges, wrapping, hidden rows/columns, and number formats are all exercised.
- Reference A: BetterOffice `betteroffice-xlsx` parse plus `betteroffice-xlsx-raster` `render_png` on the corpus, native only. Record parse time, render time, output size, failures, and what its parser rejects.
- Reference B: `readany-render` on the same corpus. Same measurements.
- Reference C: office2pdf at 2x raster as a high-fidelity oracle, plus LibreOffice PNGs from the `:full` image where available.
- Spike D: a throwaway `spike/render` crate that walks a umya `Worksheet` for a bounded range into a scene (tracks, merges, resolved style per cell, formatted text, borders) and rasterizes with tiny-skia. Reuse office2pdf's umya style and merge extraction logic as a reading reference. Measure wasm32 size delta with rustybuzz versus ab_glyph, and with subset versus full Carlito.
- Scoring: simple pixel diff and a structural checklist per workbook (values present, widths, heights, merges, fills, borders, alignment, number formats, hidden rows/cols).

## Deliverable
- `docs/architecture/renderer-bake-off.md` with the tables, the text-stack size numbers, and a recommendation for the 5004 scene model and text stack.
- The spike crate stays under `scratch/` or a `spike/` directory and is not published.

## Definition of Done
- 5004 can start with a pinned text stack, font strategy, and scene model, backed by numbers.
