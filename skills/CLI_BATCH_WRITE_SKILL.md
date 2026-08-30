# CLI Batch Write Skill — Canonical Batch Workflow

Use this skill for stateless batch writes through `asp`.

## Short checklist

1. Choose the right command family
2. Discover the exact payload shape
3. Pick exactly one mutation mode
4. Dry-run first when the change is non-trivial
5. Recalculate if formulas were affected
6. Diff and inspect critical cells

## Discoverability first

When unsure of payload shape, ask the CLI directly:

```bash
asp schema write batch transform
asp example write batch transform
asp schema write batch structure
asp example write batch structure
asp schema session op transform.write_matrix
asp example session op transform.write_matrix
```

## Mutation modes

Every batch write command requires exactly **one** of:

| Mode | Flag | Behavior |
|------|------|----------|
| Dry run | `--dry-run` | Validate without mutation |
| In-place | `--in-place` | Atomically replace source file |
| Output | `--output <PATH>` | Write to a new file (`--force` to overwrite) |

## Canonical commands

```bash
asp write batch transform workbook.xlsx --ops @transform_ops.json --dry-run
asp write batch style workbook.xlsx --ops @style_ops.json --in-place
asp write batch formula-pattern workbook.xlsx --ops @formula_ops.json --in-place
asp write batch structure workbook.xlsx --ops @structure_ops.json --dry-run --impact-report --show-formula-delta
asp write batch column-size workbook.xlsx --ops @column_ops.json --in-place
asp write batch sheet-layout workbook.xlsx --ops @layout_ops.json --in-place
asp write batch rules workbook.xlsx --ops @rules_ops.json --in-place
asp write formulas replace workbook.xlsx Sheet1 --find '$64' --replace '$65' --dry-run
```

## Canonical machine mode and just-bash

In automation hosts, including the optional `agent-spreadsheet-sdk/just-bash` adapter, use the canonical machine protocol rather than a human command alias:

```bash
asp op list_sheets --bind /workbook.xlsx --json '{}'
asp op write --bind /workbook.xlsx --json '{"expected_revision":"<revision>","mode":"preview","ops":[...]}'
asp op write --bind /workbook.xlsx --output /result.xlsx --json '{"expected_revision":"<same-revision>","mode":"apply","ops":[...]}'
asp op recalculate --bind /result.xlsx --in-place --json '{"expected_revision":"<result-revision>"}'
asp op verify_workbook --bind /result.xlsx --baseline /workbook.xlsx --json '{}'
```

Use `asp operations`, `asp schema write`, and `asp example write` for registry-backed discovery. Paths in just-bash are VFS paths; do not assume host filesystem access. Preview is pure and takes no output flag, while apply and recalculate require exactly one of `--output` or `--in-place`.

## Exact payload conventions

### Most batch commands
Use a top-level `ops` array:

```json
{
  "ops": [
    { "kind": "...", ... }
  ]
}
```

### `column-size-batch`
Preferred canonical form includes top-level `sheet_name`:

```json
{
  "sheet_name": "Sheet1",
  "ops": [
    {
      "target": { "kind": "columns", "range": "A:C" },
      "size": { "kind": "width", "width_chars": 18.0 }
    }
  ]
}
```

## Exact examples

### `transform-batch`

```json
{
  "ops": [{
    "kind": "fill_range",
    "sheet_name": "Sheet1",
    "target": { "kind": "range", "range": "B2:B4" },
    "value": "0"
  }]
}
```

### `style-batch`

```json
{
  "ops": [{
    "sheet_name": "Sheet1",
    "target": { "kind": "range", "range": "B2:B2" },
    "patch": { "font": { "bold": true } }
  }]
}
```

### `structure-batch`

```json
{
  "ops": [{
    "kind": "rename_sheet",
    "old_name": "Summary",
    "new_name": "Dashboard"
  }]
}
```

## Post-write checklist

1. Run `asp workbook recalculate` if formulas changed
2. Run `asp verify proof <baseline> <current> --targets <Sheet!A1,...>` for explicit proof (target classifications + new/resolved/preexisting errors). Use `--errors-only` for a sheet-scoped QA pass or `--targets-only` for pure target proof.
3. Run `asp diff` to confirm intent. Add `--exclude-recalc-result` when you want a lower-noise review focused on direct edits.
4. Use `asp write append ...` when you need to append tabular rows before totals/subtotals without hand-calculating insertion points. It accepts either `--rows @rows.json` or `--from-csv rows.csv --header`, and it can target either `--region-id` or `--table-name` with `--footer-policy auto|before-footer|append-at-end`.
5. Use `asp write clone-template-row ...` when you need another modeled row like a nearby template row. Start with `--dry-run` and inspect `formula_targets`, `likely_patch_targets`, merge warnings, and confidence before applying.
6. Use `asp write clone-row-band ...` when the template spans multiple contiguous rows and you want repeated blocks with the same preview-first safety contract.
7. Use `asp read cells` on critical cells/ranges
5. Use `asp workbook recalculate --changed-cells` for a change summary

## Session integration

Use sessions for multi-step edits or anything that needs undo/redo, branching, or staged apply:

```bash
asp session start --base workbook.xlsx --workspace <dir>
asp example session op transform.write_matrix
asp session op --session <id> --ops @edits.json --workspace <dir>
asp session apply --session <id> <staged_id> --workspace <dir>
asp session materialize --session <id> --output result.xlsx --workspace <dir>
```

### Session kind mapping

| Batch command | Session `kind` |
|---|---|
| `transform-batch` | `transform.clear_range`, `transform.fill_range`, `transform.replace_in_range` |
| write_matrix | `transform.write_matrix` |
| `structure-batch` | `structure.insert_rows`, `structure.clone_row`, etc. |
| `style-batch` | `style.apply` |
| `apply-formula-pattern` | `formula.apply_pattern` |
| `replace-in-formulas` | `formula.replace_in_formulas` |
| `column-size-batch` | `column.size` |
| `sheet-layout-batch` | `layout.apply` |
| `rules-batch` | `rules.apply` |
| named range CRUD | `name.define`, `name.update`, `name.delete` |

## Hard rules

- Never mix mutation modes.
- Never guess a payload shape when `asp schema` / `asp example` can tell you.
- Always dry-run structure changes first.
- Recalculate after formula-affecting writes.
