# agent-spreadsheet

[![Crates.io](https://img.shields.io/crates/v/agent-spreadsheet.svg)](https://crates.io/crates/agent-spreadsheet)
[![License](https://img.shields.io/crates/l/agent-spreadsheet.svg)](https://github.com/PSU3D0/agent-spreadsheet/blob/main/LICENSE)

**`agent-spreadsheet` is the shared engine crate behind the agent-spreadsheet tool interaction service for agent-based spreadsheet work.**

It powers:
- the `asp` / `agent-spreadsheet` stateless CLI
- the `agent-spreadsheet-mcp` stateful MCP server
- the in-repo `agent-spreadsheet-wasm` wrapper
- shared verification, diff, session, and mutation logic

If the root project is the product surface, this crate is the semantic core.

---

## What this crate contains

### Shared workbook engine
- workbook loading and repository abstractions
- sheet metrics and region detection
- table reads, range reads, paging, and analysis tools
- diff and recalc pipelines
- formula-aware write helpers
- style / layout / rules / structure mutation logic

### Shared contracts
- request/response models used across CLI, MCP, and WASM-facing work
- diagnostic types and formula parse policy types
- diff result models
- verification and warning models
- session/runtime abstractions

### CLI binaries
This crate also builds the stateless CLI binaries:
- `asp`
- `agent-spreadsheet`

Those binaries are feature-gated on recalc support.

---

## What this crate is good for

Use `agent-spreadsheet` directly when you want:
- Rust-native access to the shared spreadsheet engine
- direct workbook/session manipulation without going through MCP transport
- common semantic behavior across your own higher-level adapters
- reuse of the same diff, verification, and mutation core used by the CLI/MCP surfaces

Use companion packages instead when you want:
- **CLI** ergonomics -> `agent-spreadsheet` / `asp`
- **MCP** transport -> `agent-spreadsheet-mcp`
- **JS integration** -> `agent-spreadsheet-sdk`

---

## Highlights

### Read and inspect
- workbook metadata and sheet summaries
- region detection and structured table reads
- targeted cell inspection
- layout-aware range rendering
- named range discovery

### Analyze
- value and formula search
- formula tracing and volatility scans
- sheet/table profiling
- structural reference impact preview

### Mutate safely
- shorthand cell edit normalization
- transform/style/structure/layout/rules batch helpers
- formula-only replace flows
- named range define/update/delete
- recalc-aware write paths

### Verify and review
- grouped diff summaries
- post-edit proof surfaces
- explicit warnings and diagnostics

### Stateful workflows
- session abstractions for staged operations, apply, replay, and materialization

---

## Example

```toml
[dependencies]
agent-spreadsheet = "0.10"
```

```rust
use spreadsheet_kit::write::normalize_shorthand_edit;

let edit = normalize_shorthand_edit("A1=42").unwrap();
assert_eq!(edit.address, "A1");
assert_eq!(edit.value, "42");
assert!(!edit.is_formula);

let formula = normalize_shorthand_edit("B2==SUM(A1:A2)").unwrap();
assert_eq!(formula.address, "B2");
assert_eq!(formula.value, "SUM(A1:A2)");
assert!(formula.is_formula);
```

---

## Important types and capabilities

Examples of notable shared types/functions exposed by this crate include:
- `CellEdit`
- `CoreWarning`
- `FormulaParsePolicy`
- `FormulaParseDiagnostics`
- `validate_formula()`
- diff/result models
- session/runtime traits and helpers
- edit normalization and workbook apply helpers

The exact surface evolves with the shared engine, but the theme is constant: **one semantic spreadsheet core, many agent-facing surfaces**.

---

## Feature flags

| Feature | Purpose |
| --- | --- |
| `recalc-formualizer` | Default native Rust recalc backend |
| `recalc-libreoffice` | LibreOffice-backed recalc support |
| `recalc` | Shared recalc-related functionality used by the binaries |

Default builds include Formualizer-backed recalc.

---

## What this crate intentionally does not contain

This crate is **not** the MCP transport/server layer.

For that, see [`agent-spreadsheet-mcp`](../agent-spreadsheet-mcp/).

It is also not the npm distribution wrapper or JS SDK surface.

---

## Related packages

| Package | Role |
| --- | --- |
| [`agent-spreadsheet-mcp`](../agent-spreadsheet-mcp/) | Stateful MCP server adapter |
| `agent-spreadsheet` / `asp` | Stateless CLI built from this crate |
| `agent-spreadsheet-sdk` | JS SDK for MCP/WASM-style integrations |
| `agent-spreadsheet-wasm` | In-repo WASM-facing wrapper |

---

## More documentation

- Root README: <https://github.com/PSU3D0/agent-spreadsheet#readme>
- Packaging/versioning: <https://github.com/PSU3D0/agent-spreadsheet/blob/main/docs/PACKAGING.md>
- Heuristics: <https://github.com/PSU3D0/agent-spreadsheet/blob/main/docs/HEURISTICS.md>
- Recalc architecture: <https://github.com/PSU3D0/agent-spreadsheet/blob/main/docs/RECALC.md>

---

## License

Apache-2.0 — see [LICENSE](../../LICENSE).
