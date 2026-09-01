# Rendering

`screenshot_sheet` renders a bounded sheet range to a PNG. Since 0.15 it does
that in process, with no LibreOffice, no subprocess and no temporary file. The
renderer lives in the `agent-spreadsheet-render` crate; this document is about
the product surface.

For the renderer's own internals — the scene model, the font strategy, the
number formatter, the goldens — see
[`crates/agent-spreadsheet-render/README.md`](../crates/agent-spreadsheet-render/README.md).
For why each decision is what it is, see
[`docs/architecture/renderer-bake-off.md`](architecture/renderer-bake-off.md).

## Backends

| backend | needs | default |
|---|---|---|
| `native` | nothing beyond the binary | yes, whenever the `render` feature is compiled in |
| `libreoffice` | `soffice`, a JRE, a writable profile, the `:full` Docker image | opt-in only |

`screenshot_sheet` takes an optional `backend` input:

```json
{"resource_id": "wb:...", "sheet_name": "Sheet1", "range": "A1:M40", "backend": "native"}
```

Omit it and you get `native` on any build with the `render` feature, which is
on by default. A build with `--no-default-features` behaves exactly as before:
`screenshot_rendering` is backed only when the LibreOffice probe succeeds, and
`backend` defaults to `libreoffice`.

LibreOffice therefore remains available as an oracle — it renders the workbook
with Excel's own layout engine's closest free equivalent — but nothing depends
on it any more. The slim MCP image and the CLI render screenshots without it.

## What the response says

The output grew additively. Existing consumers keep working; nothing was
renamed or removed.

```json
{
  "sheet_name": "Sheet1",
  "range": "A1:M40",
  "artifact": {"handle": "artifact:sha256:...", "hash": "sha256:...", "bytes": 32201, "media_type": "image/png"},
  "duration_ms": 12,
  "renderer": "native-raster/1",
  "fidelity": "partial",
  "warnings": ["chart_omitted", "formulas_unevaluated"],
  "calculation": {"state": "not_evaluated", "revision_id": "sha256:..."}
}
```

* `renderer` — `native-raster/1`, or `libreoffice`.
* `fidelity` — `full` when the renderer reproduced everything it found,
  `partial` when it did not.
* `warnings` — a closed, sorted, deduplicated set. See below.
* `calculation` — the workbook's calculation state at the rendered revision,
  sourced exactly the way `read_cells` sources its own calculation block.

Image bytes never travel inside the canonical envelope. The envelope carries a
content-addressed handle; bytes cross at adapter boundaries — MCP attaches
image content, the HTTP route serves the artifact, the CLI writes it to
`--output`.

## Rendering never recalculates

This is the single most important thing to know about the output. The renderer
draws the values that are cached in the file. A formula cell whose cached value
is missing renders **empty** and raises `formulas_unevaluated`; it does not
show the formula text, and it does not evaluate anything.

If you want a screenshot of current values, recalculate first:

```sh
asp workbook recalculate book.xlsx --in-place
asp render book.xlsx --sheet Sheet1 --output sheet.png
```

The `calculation` block in the response is how a caller learns which situation
it is in without having to guess.

## What is rendered

Cached values and their formatted text, row heights and column widths, hidden
rows and columns, merged cells, fonts (real bold, synthetic italic), fills
including hatch patterns and theme colours with tint, alignment and wrap,
borders, gridlines, and row and column headings.

Rows with no `customHeight` are auto-fitted to their tallest content, the way
Excel does, so an 18 pt title is not clipped into a 15 pt row.

## What warns

Nothing unsupported disappears silently.

| warning | meaning |
|---|---|
| `conditional_format_omitted` | the sheet has conditional formatting rules; they are not evaluated and their formatting is not applied |
| `chart_omitted` | the sheet has a chart; the data behind it still renders, the chart does not |
| `image_omitted` | the sheet has an embedded image |
| `font_substituted` | a cell names a family other than Calibri/Carlito, or uses a codepoint outside the embedded font subset. Out-of-subset codepoints draw a visible `.notdef` box, never a blank |
| `rich_text_flattened` | a cell holds per-run formatting; it renders in the cell-level font |
| `number_format_approximated` | the format code is outside the bounded formatter (dates, times, conditions, locale codes, fractions) and umya's formatting was used instead |
| `formulas_unevaluated` | at least one formula cell had no cached value and rendered empty |
| `text_rotation_omitted` | a cell declares a non-zero text rotation; the text renders horizontally |
| `pattern_fill_approximated` | a hatch pattern was composited to a flat colour at its measured ink coverage |

Charts and conditional formatting are explicit non-goals for this renderer.
When you need them, `backend: "libreoffice"` on a `:full` image is the escape
hatch — at the cost of a subprocess and about 600 MB of image.

## CLI

Human surface:

```sh
asp render book.xlsx --sheet Sheet1 --output sheet.png
asp render book.xlsx --sheet Sheet1 --range A1:F40 --png-level fast --output sheet.png
asp render book.xlsx --sheet Sheet1 --backend libreoffice --output sheet.png
```

`--png-level fast|balanced|best` applies to the native backend. PNG encoding
dominates render time — `fast` is roughly eight times quicker and five times
larger — so the level is an explicit knob, defaulting to `balanced` because
image bytes are base64-inflated into a model context.

Machine surface, printing the canonical envelope and writing the artifact
bytes:

```sh
asp op screenshot_sheet --bind book.xlsx \
  --json '{"sheet_name":"Sheet1","range":"A1:F40"}' \
  --output sheet.png
```

Both write byte-identical PNGs; a CLI test asserts it.

## Determinism

No system fonts are ever loaded and no geometry is inferred from font
resolution on the host, so the same workbook renders to the same pixels on any
machine and on `wasm32`. The renderer's goldens pin the decoded pixels rather
than the PNG bytes, because the PNG's deflate stream depends on which backend
Cargo's feature unification selects for `flate2`. One practical consequence:
content-addressed screenshot artifacts are stable within a deployment, but two
builds with different `flate2` backends can produce different artifact hashes
for identical images.

## Bounds

Unchanged from before: a screenshot range is at most 100 rows by 30 columns,
the artifact is at most 16 MiB, and the sheet name and range inputs are length
and pattern bounded. Oversized ranges come back as `INVALID_REQUEST` with a
tiling suggestion.
