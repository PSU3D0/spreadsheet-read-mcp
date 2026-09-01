# Renderer Bake-off (ticket 5003)

Status: complete, 2026-09-01. Spike code and fixtures live in `spike/render-bakeoff/`; it is its own cargo workspace and is not published.

## What was measured

Four renderers over a 45-workbook corpus, on one machine (Linux 6.17, x86-64), single-threaded, release builds.

- Reference A: `betteroffice-xlsx` 0.1.0 parse plus `betteroffice-xlsx-raster` 0.1.0 `render_png_for`, bounded to A1:M40.
- Reference B: `readany-render` 0.1.3 `render()` plus `rasterise_rect()`, clamped to a 1400x1000 px window (it has no cell-range API).
- Reference C: `office2pdf` 0.6.7 `convert()` to PDF, rastered afterwards with `pdftoppm -r 192` (2x). Also LibreOffice 25.x `--convert-to png` as the second oracle.
- Spike D: a throwaway umya `Worksheet` -> scene -> tiny-skia renderer written for this ticket, bounded to A1:M40 with row/column headings, 1440 lines across five files.

Every number below was produced by a command listed in the "Commands" section. Nothing is estimated.

## Corpus

45 workbooks. 9 from `crates/agent-spreadsheet/tests/fixtures/f1`, 14 from `../calamine/tests` (picked by styles.xml size, merge count and `<col>` count), 5 from `../formualizer/benchmarks/corpus/synthetic`, 5 from `../IronCalc/xlsx/tests/calc_tests`, and 12 generated styled fixtures.

Structural feature coverage across the corpus, read straight out of the xlsx by `score.py`:

| feature | workbooks | feature | workbooks |
|---|---|---|---|
| cached values | 38 | number formats | 9 |
| column widths | 25 | fills | 8 |
| alignment | 18 | merges | 6 |
| wrap | 12 | images | 3 |
| row heights | 11 | gridlines off | 2 |
| borders | 9 | hidden rows / hidden cols | 1 / 1 |

Conditional formatting and charts are at zero: no corpus workbook carries either, so neither the references nor spike D were exercised on them here. That is a gap 5004 should close with a purpose-built fixture.

### asp write-surface gap found while generating fixtures

`asp` 0.14.0 has no canonical write operation for row heights or for hiding rows and columns. `write batch column-size` carries only `{auto|width}`; `write batch sheet-layout` carries only freeze / zoom / gridlines / page-setup / margins / breaks / print-area; `write batch structure` carries only sheet and row/column insert-delete, merge/unmerge, copy/move/clone. Ten of the twelve generated fixtures went through `asp`; `gen_08_rowheights` and `gen_09_hidden` had to be produced with openpyxl. 5004 renders both features, so the renderer will ship fixtures the canonical surface cannot itself author.

## Reference A - BetterOffice (`betteroffice-xlsx` 0.1.0 + `betteroffice-xlsx-raster` 0.1.0)

| metric | median | mean | p95 | max |
|---|---|---|---|---|
| parse ms (`Workbook::open_for_read`) | 1.15 | 39.91 | 96.08 | 1093.84 |
| render ms (display list + raster + png, fused) | 30.65 | 40.22 | 77.25 | 181.03 |
| total ms | 35.45 | 80.13 | - | 1116.52 |
| png bytes | 37,408 | 55,046 | 112,973 | 231,612 |

42 of 45 succeeded. Three parser rejections:

| workbook | error |
|---|---|
| `calamine/date_iso.xlsx` | `Spreadsheet(Malformed("bad number \"2021-01-01\""))` - rejects an ISO-8601 date written into a numeric cell value |
| `calamine/issue_261.xlsx` | `Spreadsheet(Xml("syntax error: attribute value not closed"))` - stricter XML than Excel accepts; the Excel-repaired sibling `issue_261_fixed_by_excel.xlsx` parses fine |
| `synthetic/lookup_cross_sheet_dim_fact.xlsx` | `CollaborativeState("update array length exceeds its payload")` - the yrs CRDT layer fails on a 2.0 MB workbook even through the read-only entry point |

Fidelity was the best of the three non-oracle renderers by eye: correct merges, fills, borders, alignment, number formats including `[Red]` negatives, and correct spill behaviour. It draws no row/column headings.

Disqualifying constraints for our use, independent of quality:

- `yrs` 0.27.3 and 0.27.4 use `if let` guards, which are still unstable. `betteroffice-xlsx` therefore **does not build on the repository's pinned rustc 1.92**; the whole bake-off harness had to be built with `cargo +1.97.1`. `yrs` is a mandatory dependency even for read-only rendering.
- The raster crate embeds exactly one face, `Carlito-Regular.ttf` (628,032 bytes). Bold is faked by re-filling the glyph offset 0.35 px; italic is a 0.21 shear. `DrawCmd::Text::font_family` is parsed and then ignored.
- Its own README says the raster crate is "never part of the wasm build".

## Reference B - readany-render 0.1.3

| metric | median | mean | p95 | max |
|---|---|---|---|---|
| parse + layout ms (`render()`, fused) | 0.97 | 104.97 | 514.20 | 2749.08 |
| raster ms (`rasterise_rect`) | 2.50 | 12.52 | 65.42 | 162.34 |
| png encode ms | 1.58 | 5.62 | 38.57 | 44.19 |
| total ms | 10.76 | 123.12 | - | 2749.51 |
| png bytes | 8,799 | 29,766 | 88,709 | 208,849 |

42 of 45 succeeded. All three failures are the same bug:

| workbooks | error |
|---|---|
| `calamine/date_iso.xlsx`, `generated/gen_08_rowheights.xlsx`, `generated/gen_09_hidden.xlsx` | `MalformedDocument: required document part xl/xl/worksheets/sheet1.xml is missing` |

The doubled `xl/xl/` is a relative-part-path resolution bug in its OPC layer: when `xl/workbook.xml.rels` gives a target of `worksheets/sheet1.xml` in a form it does not expect, it re-prefixes `xl/`. Both openpyxl-written fixtures hit it, which means it fails on a very common writer.

It is the only reference that emits structured unrendered-content warnings, and the taxonomy is close to the one 5004 pins: across the corpus it reported `FormulaWithoutCachedValue` (4), `PivotTable` (1), `HiddenSheet` (1), `UnsupportedGlyphs` (1).

Two fidelity defects seen on `gen_11_dashboard`: cells with the number format `"$"#,##0` render as the literal `&quot` (it does not XML-unescape `formatCode`), and a 18 pt title in a default-height row is vertically clipped rather than overflowing.

Its font strategy is the opposite of ours: 21 faces (Carlito, Caladea, Liberation Sans/Serif/Mono, DejaVu Sans) totalling **7.8 MB** compiled in behind a non-optional-in-practice `fonts` feature. Disabling it makes `render()` return `NoFonts` for every non-image format. There is no cell-range API at all; the only bound is a geometric `Rect` in document pixels.

## Reference C - office2pdf 0.6.7 (oracle) and LibreOffice (oracle)

| oracle | median ms | mean ms | p95 ms | max ms | median output bytes |
|---|---|---|---|---|---|
| office2pdf `convert()` to PDF | 206.4 | 1044.5 | 3973.0 | 24393.7 | 14,965 (PDF) |
| office2pdf PDF -> PNG at 192 dpi (`pdftoppm`) | - | 160 (7217 ms / 45 files) | - | - | - |
| LibreOffice `--convert-to png` | 1944 | 2409 | - | 17196 | 22,411 |

Both oracles succeeded on 45 of 45. office2pdf emitted 53 `FallbackUsed` warnings across the corpus (charts downgraded to data tables).

office2pdf is the highest-fidelity output in the set - correct merged and centred titles, correct wrap into a merged range, correct `[Red]` negatives, real Times New Roman for a Times cell. It is also print-shaped: no gridlines, no row/column headings, and its fit-to-page dropped the last column of `gen_11_dashboard` entirely. It builds in 2 m 34 s and pulls typst 0.14, typst-kit, typst-pdf, comemo, docx-rs, zip, image; the linked binary is 76 MB. It is a fine oracle and a fine reading reference. It is not a dependency candidate, which is what decision 6 already says.

LibreOffice needs a writable `/tmp` for its named pipe (it fails with `ERROR: no valid pipe path found` under a sandbox that only allows `/tmp/claude`), and a writable `-env:UserInstallation` profile.

## Fidelity scoring

Neither oracle produces a registered image. LibreOffice and office2pdf both paginate to a print page with their own fit-to-width and no headings, while the candidates render a bounded A1:M40 grid window. A per-pixel diff between them measures the page geometry mismatch, not renderer quality, so **pixel scores against the oracles are not reported**. What survives, after cropping to the inked region exactly as `crop_png_in_place` in `crates/agent-spreadsheet/src/recalc/screenshot.rs` does, is the shape of the ink. It is reported as a weak signal only.

| candidate | vs LibreOffice: aspect / density (median) | vs office2pdf: aspect / density (median) |
|---|---|---|
| ref-a BetterOffice | 0.329 / 0.534 | 0.329 / 0.470 |
| ref-b readany-render | 0.539 / 0.263 | 0.526 / 0.259 |
| spike D | 0.392 / 0.589 | 0.380 / 0.508 |

Read this only as "no candidate is anywhere near page-registered with an oracle, and spike D's ink density is the closest of the three to both oracles". Do not read it as a fidelity ranking.

The real fidelity evidence is the structural checklist plus targeted visual inspection, both of which put office2pdf first, BetterOffice second, spike D third, readany-render fourth.

## Spike D

| metric | median | mean | p95 | max |
|---|---|---|---|---|
| parse ms (`umya reader::xlsx::read`) | 0.64 | 32.24 | 167.11 | 896.65 |
| scene ms (Worksheet -> Scene) | 0.04 | 3.08 | 27.92 | 49.97 |
| raster ms (Scene -> Pixmap, tiny-skia) | 7.13 | 13.11 | 31.74 | 66.14 |
| png encode ms | 35.60 | 39.87 | 56.46 | 87.10 |
| total ms | 51.38 | 88.30 | - | 931.52 |
| png bytes | 18,422 | 24,866 | 45,652 | 102,207 |

45 of 45 succeeded. It is the only renderer in the bake-off with no parse failures, because umya accepts everything the product already accepts, by construction.

Warnings raised across the corpus: `NumberFormatApproximated` 22, `FontSubstituted` 8, `ImageOmitted` 3, `PatternFillApproximated` 1, `TextRotationOmitted` 1.

### PNG encoding is the dominant cost, not rasterisation

Encode is 35.6 ms of a 51.4 ms median. Measured at three zlib levels on the identical pixmaps:

| png::Compression | median ms | mean ms | median bytes | corpus total bytes |
|---|---|---|---|---|
| `Balanced` (what `Pixmap::encode_png` uses) | 34.23 | 38.64 | 18,422 | 1,118,953 |
| `Fast` | 26.22 | 29.34 | 43,640 | 2,660,227 |
| `Fastest` | 4.11 | 4.21 | 99,701 | 8,699,798 |

`Fastest` is 8.3x faster and 5.4x larger in the aggregate. For MCP image content the bytes are base64-inflated into a model context, so the default should stay `Balanced` and the level should be an explicit knob rather than whatever `tiny_skia::Pixmap::encode_png` picks.

### Defects found in spike D that 5004 must not inherit

- **Non-deterministic geometry from umya.** A "dominant font" heuristic over `Worksheet::get_cell_collection()` produced a different normal font, and therefore a different column unit and a different image width (1122 px vs 1299 px on `gen_01_fonts`), between runs of the *same binary*. `get_cell_collection()` iteration order is not stable. Ticket 5004 requires PNG hash goldens; this is fatal to that requirement. Fixed in the spike by a deterministic tie-break, but the right fix is office2pdf's: read the first `<font>` out of `xl/styles.xml` rather than inferring it.
- **umya mis-formats multi-section number formats.** With `0.0%;[Red]-0.0%`, `Cell::get_formatted_value()` returned `57.0%` for 0.5727 (Excel, LibreOffice, BetterOffice and readany-render all give `57.3%`) and `3.0%` for -0.032 (correct is `-3.2%`, in red). Both the rounding and the sign are wrong. That is why 22 of 45 workbooks raise `NumberFormatApproximated`.
- ab_glyph's `Font::outline` returns a flat segment list; contours must be stitched by continuity before handing the path to tiny-skia or the winding fill collapses to hairline fragments. The first ab_glyph measurement run was invalid for exactly this reason and was redone.

## Text stack and font strategy

wasm32-unknown-unknown, release, `opt-level = "z"`, `lto = true`, `codegen-units = 1`, `panic = "abort"`, `strip = true`. The probe crate (`text-wasm`) links one text stack and one font set plus tiny-skia and png, and nothing else; `baseline` is the same shell with no renderer at all, to subtract the wasm prelude.

| variant | .wasm bytes | gzip bytes |
|---|---|---|
| baseline (no renderer) | 80 | 111 |
| rustybuzz + full Carlito R+B | 2,294,138 | 910,294 |
| rustybuzz + subset Carlito R+B | 1,133,129 | 436,215 |
| ab_glyph + full Carlito R+B | 1,859,952 | 749,465 |
| ab_glyph + subset Carlito R+B | 698,943 | 275,116 |

Derived deltas, which are exact because the two axes are orthogonal in these numbers:

| delta | bytes | gzip bytes |
|---|---|---|
| rustybuzz over ab_glyph, full fonts | +434,186 | +160,829 |
| rustybuzz over ab_glyph, subset fonts | +434,186 | +161,099 |
| full over subset fonts, rustybuzz | +1,161,009 | +474,079 |
| full over subset fonts, ab_glyph | +1,161,009 | +474,349 |

Font asset sizes, subset built with `pyftsubset` over Latin-1 plus Latin Extended-A punctuation, currency (including the euro and the rupee), arrows, maths signs and the fi/fl ligatures:

| face | full bytes | subset bytes | ratio |
|---|---|---|---|
| Carlito-Regular | 628,032 | 70,580 | 11.2% |
| Carlito-Bold | 682,468 | 78,912 | 11.6% |
| pair total | 1,310,500 | 149,492 | 11.4% |

### What the subset actually costs

Spike D rendered the whole corpus with full and with subset Carlito. Both runs are pixel-registered (identical geometry), so this diff is exact.

| comparison | files | pixel-identical | median differing px | mean | max |
|---|---|---|---|---|---|
| subset vs full Carlito | 45 | **43** | 0.0000% | 0.0011% | 0.0270% |
| ab_glyph vs rustybuzz (full Carlito both) | 45 | 0 | 0.6075% | 1.2920% | 4.2820% |

Only two workbooks change at all under the subset: `calamine/rph.xlsx` (Japanese ruby text) at 0.0257% of pixels, and `generated/gen_12_unicode.xlsx` at 0.0224%. Visual inspection of `gen_12_unicode` shows exactly one row lost: the Greek row `α β γ Ω` renders blank. Everything else in that fixture - `£ € ¥ ₹`, guillemets, em/en dashes, smart quotes, `± ° ½ ¼ ¾`, arrows, `ß æ ø å`, `© ™` - renders identically to the full font. The subset drops missing glyphs **silently**, with no `.notdef` box, which is the part that must not ship.

ab_glyph differs from rustybuzz on every workbook, but by a small margin and without visible quality loss on Latin text: side-by-side crops of `gen_01_fonts` (7 families x regular/bold/italic/bold-italic/underline/strike) are visually indistinguishable. The differences are sub-pixel positioning from kerning-only advance accumulation versus full shaping.

## Recommendation for 5004

**Scene model: a flat, ordered command list with absolute pixel coordinates, in a `scene` module with no rendering imports.** Three variants are enough for the MVP: `FillRect`, `Line` (axis-aligned, with a width, a colour and a dash rhythm), and `Text` (with its own box, a separate clip rect, halign/valign, wrap, and bold/italic/underline/strike flags). All three independent projects converged on this shape - BetterOffice's `DrawCmd` is `FillRect`/`Line`/`Path`/`Text`, readany-render's `Item` is `Glyphs`/`Path`/`Image`/`Group`. Do not add `Path` or `Group` in the MVP; nothing in the 5004 scope needs them, and `Path` is where BetterOffice's chart code lives. Do add a `Warning` set on the scene, populated during extraction, so the report is a by-product of extraction rather than a second pass. Paint order must be background, then fills, then gridlines, then borders, then text, then headings; spike D gets legible results only because headings paint last.

Colours should be a packed RGBA struct, not the `#rrggbb` strings BetterOffice uses - it re-parses every string in the raster loop.

**Text stack: ab_glyph.** It costs 434,186 fewer wasm bytes (160,829 gzipped) than rustybuzz, which is 38% of the rustybuzz+subset bundle, for no visible quality difference on Latin text. 5004's non-goals already exclude CJK and RTL shaping quality, and the warning taxonomy already has `font_substituted` to declare it. If a future ticket needs real shaping, the `Fonts` façade in the spike (`measure` / `ascent_px` / `draw`) is the seam to swap behind - keep it. Note the contour-stitching requirement above; write a golden test for a glyph with two contours (`o`, `8`) on day one.

**Font strategy: subset Carlito Regular and Bold, embedded, plus an explicit missing-glyph path.** 149,492 bytes instead of 1,310,500, and 43 of 45 corpus workbooks are pixel-identical to the full font. But the subset must never drop a glyph silently: emit `font_substituted` and draw a visible `.notdef` box whenever a codepoint is not in the subset. Do not follow readany-render's 7.8 MB of 21 faces, and do not follow BetterOffice's single regular face with faked bold - a real bold face costs 78,912 bytes subsetted and removes a whole class of fidelity complaints. Synthetic italic by shear is fine and is what both references do.

**Port from office2pdf, verbatim, into extraction:**

1. The column metric model - `column_unit_pt` = `round_half_up(max_digit_advance_em * normal_font_size_pt)`, `column_width_to_pt` = `round_half_up(width_chars * unit_pt)`, the default-width rule `baseColWidth(8) * unit + 5.0`, the hardcoded `reference_digit_advance_em` table (`CALIBRI_DIGIT_ADVANCE_EM = 0.506836`, arial 0.556152, verdana 0.635742, courier new 0.600098, times new roman 0.500000), and the `> 0` guards that read umya's zero as "absent". These encode 17 native-Excel probes we cannot re-derive, and the hardcoded table is exactly what makes the geometry identical under wasm.
2. `pattern_ink_coverage` and `blend_color`, and the decision to read `get_fill() -> get_pattern_fill() -> get_pattern_type()/get_foreground_color()/get_background_color()` rather than `Style::get_background_color()`, which returns `fgColor` regardless of pattern type.
3. `border_style_to_width` (hair/thin/dashed/dotted/dashDot/dashDotDot/double = 1.0 pt, medium family and slantDashDot = 2.0 pt, thick = 3.0 pt) and `border_style_to_line_style`, plus the "borders are bands anchored to the grid boundary, not centred strokes" model.
4. `merged_range_border` - Excel writes a merged range's border onto its constituent cells, so a rule under a two-row header lives on the bottom row. Composing each edge from the first member that declares it is not obvious and we would get it wrong.
5. `compute_spill_width` and `cell_wraps_past_one_line`, with the crude `estimate_text_width_pt` replaced by real ab_glyph measurement.
6. `XLSX_CELL_PADDING` = top 1.0, right 3.0, bottom 1.5, left 3.0 pt.
7. `extract_normal_font` - the first `<font>` out of `xl/styles.xml`. This is the fix for the non-determinism above, and umya does not expose the stylesheet, so it has to be a small quick-xml read of the archive.

Do **not** port `native_excel_pdf_row_height` (the `* 0.92` factor). It compensates for Excel's compacted *print* grid; we raster for screen and should use raw points at 96/72.

Do **not** port `resolve_style_color` - it is a one-line delegation to umya's `get_argb_with_theme`, and umya's implementation short-circuits on indexed colours and silently loses the tint. If 5004 wants correct theme colours it needs its own `INDEXED_COLORS` table and its own HLS-based `calc_tint`, and should fix the indexed+tint case rather than inherit it.

office2pdf handles hidden rows and columns nowhere at all - zero references to `hidden` in its parser. 5004 has them in the MVP list, so that part is ours to write; spike D's `get_column_dimension_by_number(..).get_hidden()` / `get_row_dimension(..).get_hidden()` reading a zero-width track works and is one line each.

**Encoding: expose the png compression level; default `Balanced`.** And do not use `Pixmap::encode_png`, which fixes the level.

**Two things not in 5004's scope that this bake-off says should be:**

- `number_format_approximated` will fire on roughly half of all real workbooks, not on an exotic tail, because umya mis-formats multi-section formats. Either the warning has to be documented as common, or 5004 needs its own section-splitting formatter for the `positive;negative;zero;text` shape and `[Red]`. That is a scope decision for the lead.
- Determinism needs a test, not an assumption. `get_cell_collection()` order is unstable; any extraction that aggregates over it will produce non-reproducible goldens.

## Commands

Corpus and fixtures.

```
spike/render-bakeoff/assemble_corpus.sh
ASP=<repo>/target/debug/asp spike/render-bakeoff/gen_fixtures.sh
UV_CACHE_DIR=$TMPDIR/uvcache uv run --with openpyxl python3 spike/render-bakeoff/gen_fixtures_openpyxl.py corpus/generated
```

Font subsets.

```
UNI='U+0020-007E,U+00A0-00FF,U+0152-0153,U+0160-0161,U+0178,U+017D-017E,U+0192,U+02C6,U+02DC,U+2013-2014,U+2018-201A,U+201C-201E,U+2020-2022,U+2026,U+2030,U+2039-203A,U+2044,U+20AC,U+20B9,U+20BD,U+2122,U+2190-2193,U+2202,U+2206,U+220F,U+2211-2212,U+2215,U+221A,U+221E,U+222B,U+2248,U+2260,U+2264-2265,U+25CA,U+FB01-FB02'
uv run --with fonttools pyftsubset Carlito-Regular.ttf --unicodes="$UNI" --layout-features='kern,liga,ccmp,locl' --output-file=Carlito-Regular-subset.ttf
uv run --with fonttools pyftsubset Carlito-Bold.ttf    --unicodes="$UNI" --layout-features='kern,liga,ccmp,locl' --output-file=Carlito-Bold-subset.ttf
```

References A, B and spike D.

```
cd spike/render-bakeoff
cargo fetch --config 'build.rustc-wrapper=""'                      # sandbox disabled; registry cache is read-only
cargo update -p yrs --precise 0.27.3 --config 'build.rustc-wrapper=""'
cargo +1.97.1 build --release -p harness --config 'build.rustc-wrapper=""'
./target/release/bakeoff corpus out/run1 ref-a ref-b spike-d > out/results.jsonl
```

Spike D text-stack and font-strategy variants.

```
cargo +1.97.1 build --release -p harness --config 'build.rustc-wrapper=""' --no-default-features --features spike-d,sd-rustybuzz,sd-font-subset
./target/release/bakeoff-subset corpus out/run-subset spike-d > out/results-subset.jsonl
cargo +1.97.1 build --release -p harness --config 'build.rustc-wrapper=""' --no-default-features --features spike-d,sd-abglyph,sd-font-full
./target/release/bakeoff-abglyph corpus out/run-abglyph spike-d > out/results-abglyph.jsonl
```

Reference C and the two oracles.

```
cd spike/render-bakeoff/refc && cargo +1.97.1 build --release --config 'build.rustc-wrapper=""'
./refc/target/release/refc corpus out/ref-c > out/results-refc.jsonl
for f in out/ref-c/*.pdf; do pdftoppm -png -r 192 -f 1 -l 1 -singlefile "$f" "out/ref-c-png/$(basename "$f" .pdf)"; done
spike/render-bakeoff/oracle_libreoffice.sh          # sandbox disabled; soffice needs /tmp for its named pipe
```

wasm sizes and png levels.

```
spike/render-bakeoff/measure_wasm.sh                # writes out/wasm-sizes.tsv
cargo +1.97.1 build --release -p harness --bin pngbench --config 'build.rustc-wrapper=""'
./target/release/pngbench corpus                    # writes out/pngbench.tsv
```

Scoring.

```
UV_CACHE_DIR=$TMPDIR/uvcache uv run --with pillow --with numpy python3 spike/render-bakeoff/score.py > out/scores.jsonl
```

## Not measured, and why

- **Pixel or SSIM parity against an oracle.** Neither LibreOffice nor office2pdf produces a registered image against a bounded grid window - both paginate to a print page with their own fit-to-width and no headings. Cropping does not fix the mismatch. Reported as ink shape only, and flagged as weak.
- **Conditional formatting and chart fidelity.** No corpus workbook carries either, so no renderer was exercised on them. Reference A and Reference C both implement conditional formatting; office2pdf's `cond_fmt.rs` (1013 lines) is the reference to read if 5004 ever scopes it in.
- **Reference A on the repository's pinned rustc 1.92.** It does not build there (`yrs` needs unstable `if let` guards). All numbers for Reference A are from rustc 1.97.1.
- **wasm *runtime* performance for any renderer.** Only wasm binary size was measured. No wasm execution timings were taken.
- **Spike D on wasm32 end-to-end.** The `text-wasm` probe links the scene, text and raster paths and builds for wasm32, but the umya extraction layer was not built for wasm32 in this spike, so "spike D runs in wasm" is not a claim this document makes.
- **Multi-sheet, frozen panes and 2x scale rendering.** Spike D renders sheet index 0 at scale 1.0 only.
- **Memory high-water marks.** Not instrumented for any backend.
