# agent-spreadsheet-render

A bounded native raster renderer for spreadsheet ranges. Turns a
`umya_spreadsheet::Worksheet` and a cell range into PNG bytes plus a structured
report of everything it did *not* reproduce.

It exists so that screenshots stop requiring LibreOffice. Most inspection and
verification workflows need a bounded, honest rendering of cached values and
formatting — not Excel pixel parity.

## Two entry points

```rust
use agent_spreadsheet_render::{RangeBounds, RasterOptions, RenderOptions};

let scene = agent_spreadsheet_render::extract_scene(
    sheet,        // &umya_spreadsheet::Worksheet
    &workbook,    // &umya_spreadsheet::Spreadsheet, for the theme colour scheme
    &RangeBounds::new(1, 13, 1, 40),   // A1:M40, 1-based inclusive
    &RenderOptions { styles_xml, ..Default::default() },
)?;
let output = agent_spreadsheet_render::rasterize(&scene, &RasterOptions::default())?;

output.png;             // Vec<u8>
output.width;           // u32
output.report.renderer; // "native-raster/1"
output.report.fidelity; // Full | Partial
output.report.warnings; // sorted, deduplicated, closed enum
```

`Scene` is a flat, ordered command list in absolute device pixels —
`FillRect`, `Line` and `Text`, nothing else — and it is `Serialize` +
`Deserialize`, which is what the scene goldens compare.

### `styles_xml`

Excel derives every column print metric from the workbook's Normal font, which
is the first `<font>` in `xl/styles.xml`. umya does not expose the stylesheet
and this crate opens no archives, so the caller passes the decompressed part
bytes. Passing `None` is supported: the renderer then assumes Excel's own
default, Calibri 11, giving a 6 pt column unit. A workbook whose Normal font is
something else will render at slightly different column widths in that case.

## Dependency budget

`umya-spreadsheet`, `tiny-skia`, `png`, `ab_glyph`, `quick-xml`, `serde`,
`thiserror`. No filesystem, process, tokio, MCP, `image` or Typst dependency.

```sh
cargo check -p agent-spreadsheet-render --target wasm32-unknown-unknown
```

builds. (Set `RUSTFLAGS='--cfg getrandom_backend="wasm_js"'` — umya pulls
`ahash`, which pulls `getrandom` 0.3, which requires an explicit wasm backend.)

## What it renders

Cached values and their formatted text, row heights and column widths, hidden
rows and columns, merged cells, fonts (bold real, italic synthesised by shear),
fills including patterns and theme colours with tint, alignment and wrap,
borders, gridlines, and row/column headings.

**It never recalculates.** A formula cell with no cached value renders empty and
raises `formulas_unevaluated`.

## What it warns about

Nothing unsupported disappears silently. Every omission or approximation raises
one of nine closed `Warning` variants, and any warning at all downgrades
`fidelity` from `full` to `partial`.

| warning | raised when |
|---|---|
| `conditional_format_omitted` | the sheet carries conditional formatting rules; they are never evaluated |
| `chart_omitted` | the sheet carries a chart |
| `image_omitted` | the sheet carries an image |
| `font_substituted` | a cell names a family other than Calibri/Carlito, or uses a codepoint outside the embedded subset |
| `rich_text_flattened` | a cell holds rich text; it renders in the cell-level font |
| `number_format_approximated` | a format code falls outside the bounded formatter and umya's own formatting is used |
| `formulas_unevaluated` | a formula cell has no cached value |
| `text_rotation_omitted` | a cell declares a non-zero text rotation |
| `pattern_fill_approximated` | a hatch pattern is composited to a flat colour at its measured ink coverage |

## Fonts

Subset Carlito Regular and Bold, 149,492 bytes for the pair, compiled in. No
system fonts are ever loaded, which is what makes the PNG goldens reproducible
across machines and across native and wasm32. A codepoint outside the subset
draws a visible `.notdef` box and raises `font_substituted` — it is never
dropped. See `assets/README.md` for the licence and the `pyftsubset` command.

## Number formats

The crate ships its own bounded formatter rather than using umya's
`get_formatted_value`, which mis-renders multi-section formats: under
`0.0%;[Red]-0.0%` umya gives `57.0%` for 0.5727 and `3.0%` for -0.032, losing
the rounding, the sign and the colour. This formatter gives `57.3%` and a red
`-3.2%`.

Supported: section splitting (`positive;negative;zero;text`), `[Red]` and the
other named colours plus `[Color n]`, the `0 # ?` digit placeholders, the
decimal point, the thousands comma, `%`, scientific notation, quoted literals,
backslash escapes, `_x` and `*x`, `@`, and `General`.

Out of scope, and declared as `number_format_approximated`: dates and times,
conditional sections, locale and currency codes, elapsed time, fractions, and
trailing-comma scaling.

## PNG encoding

Encoding is the dominant cost — 35.6 ms of a 51.4 ms median render in the 5003
bake-off. `RasterOptions::png_level` exposes it; the default is `Balanced`,
because for MCP image content the bytes are base64-inflated into a model
context. `Fast` is roughly 8x quicker and 5x larger.

## Tests

```sh
cargo test -p agent-spreadsheet-render --config 'build.rustc-wrapper=""'
```

* `tests/goldens.rs` — a scene JSON golden and a PNG sha256 golden per fixture.
  Regenerate with `UPDATE_GOLDENS=1`; inspect with `DUMP_PNG_DIR=/some/dir`.
* `tests/determinism.rs` — every fixture rendered twice in one process and once
  more in a freshly spawned process, with all three hashes required to agree.
* `tests/text_shape.rs` — the multi-contour glyph golden and the `.notdef` path.
* `tests/warnings.rs` — one test per warning variant.
* `tests/fixtures/generate.py` — regenerates the fixtures the bake-off corpus
  does not cover.

## Provenance

Design decisions here are pinned by the 5003 renderer bake-off
(`docs/architecture/renderer-bake-off.md`): the scene model, ab_glyph behind a
`Fonts` facade, the subset font strategy, the office2pdf metric ports, the
determinism requirement, and the PNG compression knob. The modules carry the
reasoning inline.
