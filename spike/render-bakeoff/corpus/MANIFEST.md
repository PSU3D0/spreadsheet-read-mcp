# Bake-off corpus (45 workbooks)

Only `generated/` is committed. Everything else is copied from sibling checkouts at bake-off time and is reproducible; re-create with `assemble_corpus.sh`.

| group | n | source |
|---|---|---|
| `fixtures/` | 9 | `crates/agent-spreadsheet/tests/fixtures/f1/*.xlsx` |
| `calamine/` | 14 | `../calamine/tests/`, picked by styles.xml size, merge count and `<col>` count |
| `synthetic/` | 5 | `../formualizer/benchmarks/corpus/synthetic/` |
| `ironcalc/` | 5 | `../IronCalc/xlsx/tests/calc_tests/` |
| `generated/` | 12 | `gen_fixtures.sh` (asp CLI) and `gen_fixtures_openpyxl.py` |

## Generated fixture coverage against the 5004 MVP list

| fixture | exercises |
|---|---|
| `gen_01_fonts` | 7 font families, 6 sizes, bold/italic/underline/strikethrough, font colour |
| `gen_02_fills` | 12 solid fills, 5 non-solid pattern types with fg+bg |
| `gen_03_borders` | 12 border styles per edge, plus an inner horizontal/vertical grid |
| `gen_04_merges` | 5 merges incl. row-spanning and column-spanning, centred in merge |
| `gen_05_numfmt` | 18 number formats: currency, accounting, percent, scientific, fraction, 5 date/time codes, red-negative |
| `gen_06_align_wrap` | 6x3 horizontal x vertical alignment matrix, wrap, 6 text rotations, spill |
| `gen_07_colwidths` | 8 column widths from 2.0 to 60.0 character units |
| `gen_08_rowheights` | 8 row heights 8-90pt, plus one auto-height wrapped row |
| `gen_09_hidden` | 3 hidden rows, 3 hidden columns |
| `gen_10_gridlines` | gridlines off, frozen panes |
| `gen_11_dashboard` | all of the above combined in a realistic report, plus uncached formulas |
| `gen_12_unicode` | Latin-1, currency (£ € ¥ ₹), punctuation, arrows, Greek, ligatures |

## asp write-surface gap found while generating these

`asp` 0.14.0 has **no canonical write operation for row heights or for hiding rows/columns**. `write batch column-size` carries only `{auto|width}`, `write batch sheet-layout` carries only freeze/zoom/gridlines/print-setup/margins/breaks/print-area, and `write batch structure` carries only sheet/row/column insert-delete, merge/unmerge, copy/move/clone. `gen_08_rowheights` and `gen_09_hidden` are therefore produced with openpyxl. Ticket 5004 needs both features rendered, so the renderer will have fixtures it cannot itself author through the canonical surface.
