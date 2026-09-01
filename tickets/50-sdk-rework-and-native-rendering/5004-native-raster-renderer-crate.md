# Ticket: 5004 Native Raster Renderer Crate + Canonical Wiring

## Depends On
- 5003

## Why
Screenshots currently require LibreOffice via a macro to PDF to PNG pipeline that only ships in the Docker `:full` image. Most inspection and verification workflows need a bounded, honest rendering of cached values and formatting, not Excel pixel parity.

## Owner / Effort / Risk
- Owner: rendering / core
- Effort: L (3 to 5 weeks)
- Risk: Med

## Scope
- New crate `crates/agent-spreadsheet-render`. No filesystem, process, tokio, MCP, or Typst dependencies. Two entry points: scene extraction from a `umya_spreadsheet::Worksheet` and a bounded range, and PNG rasterization returning bytes plus a report.
- MVP coverage: cached values and formatted text, row heights and column widths, hidden rows and columns, merged cells, fonts (embedded subset Carlito regular and bold), fills and colors including theme colors, alignment and wrap, borders, gridlines, and row/column headings.
- Report: `renderer` (`native-raster/1`), `fidelity` (`full`, `partial`), and `warnings` from a closed enum: `conditional_format_omitted`, `chart_omitted`, `image_omitted`, `font_substituted`, `rich_text_flattened`, `number_format_approximated`, `formulas_unevaluated`.
- Never recalculate. Surface calculation state from the resource revision.
- Core wiring: `ScreenshotBackend { NativeRaster, LibreOffice }`. `screenshot_rendering` capability is true whenever the `render` feature is compiled in. `screenshot_sheet` picks native by default; optional `backend` input. Output gains `renderer`, `fidelity`, `warnings`, `calculation` additively. Registry adapter metadata flips wasm and just_bash to supported for `screenshot_sheet`; this registry edit is reviewed in the main loop.
- CLI: human command `asp render <file> --sheet <name> [--range A1:F40] --output out.png`, and `asp op screenshot_sheet --bind file.xlsx --output out.png` writing the artifact bytes.
- Fixtures: golden PNG hashes and JSON scene goldens per fixture. Deterministic because no system fonts are used.
- LibreOffice remains behind `recalc-libreoffice` as an opt-in oracle.

## Non-Goals
- Charts, conditional formatting evaluation, rich text runs, CJK/RTL shaping quality, exact number-format parity. These produce warnings.

## Tests
- Scene golden tests per fixture.
- PNG hash goldens per fixture on native and wasm32 (same bytes).
- Canonical parity fixtures for the extended `screenshot_sheet` envelope on CLI, MCP, and WASM.

## Definition of Done
- Slim MCP image and the CLI render screenshots without LibreOffice.
- Warnings are structured and tested; nothing unsupported disappears silently.
