# Spreadsheet Read MCP

`spreadsheet-read-mcp` is a Model Context Protocol (MCP) server that lets LLM agents explore spreadsheet workbooks safely and deterministically. It focuses on high-signal, read-only insights (structure, formulas, styles, statistics) without mutating the source files. The server is optimized for XLSX first and can discover `.xls`/`.xlsb` files, while keeping the backend pluggable for future formats.

## What It Does
- Enumerates workbooks inside a workspace, exposing stable short IDs that are easy for an LLM to reference.
- Streams sheet pages, highlights formulas, and surfaces cached values so models can inspect data slices without loading entire files.
- Maps formula clusters, traces precedents/dependents with pagination-friendly summaries, and tags volatile functions.
- Reports workbook metadata, sheet classifications, style usage, and named ranges to give agents a comprehensive mental model.
- Provides manifest stubs and bookkeeping helpers so downstream harnesses can integrate the results quickly.

## What It Is Not
- No spreadsheet writing, mutation, or recalculation; everything is read-only.
- No XLS macro execution, VBA inspection, or automation beyond surface metadata.
- No on-the-fly format conversion; ODS and other backends will require future CAPS-enabled adapters.
- Not a generic file browser — it focuses strictly on spreadsheet-aware inspection.

## Quick Start
```bash
# Run directly from the repository
cargo run --release -- \
  --workspace-root /path/to/workbooks \
  --cache-capacity 10
```

To install the binary locally:
```bash
cargo install --path spreadsheet-read-mcp
spreadsheet-read-mcp --workspace-root /path/to/workbooks
```

The server speaks MCP over stdio; pair it with an MCP-aware client to issue tool calls once it is running.

## Configuration Options
You can configure the server through CLI flags, environment variables, or a YAML/JSON config file.

### CLI / Environment
| Flag | Env Var | Description |
| --- | --- | --- |
| `--workspace-root <DIR>` | `SPREADSHEET_MCP_WORKSPACE` | Root directory to scan for workbooks (defaults to current directory). |
| `--cache-capacity <N>` | `SPREADSHEET_MCP_CACHE_CAPACITY` | Maximum in-memory workbook cache size (minimum 1, default 5). |
| `--extensions ext1,ext2` | `SPREADSHEET_MCP_EXTENSIONS` | Allowed file extensions (defaults to `xlsx,xls,xlsb`). |
| `--workbook <FILE>` | `SPREADSHEET_MCP_WORKBOOK` | Lock the server to a single workbook without scanning the workspace. |
| `--enabled-tools tool1,tool2` | `SPREADSHEET_MCP_ENABLED_TOOLS` | Restrict execution to the named tools; others return an MCP `invalid_request`. |
| `--config <FILE>` | – | Load settings from a YAML or JSON file. CLI/env values override file entries. |

### Config File Example (`config.yaml`)
```yaml
workspace_root: /data/spreadsheets
cache_capacity: 8
extensions: ["xlsx", "xlsb"]
```

Start the server with `spreadsheet-read-mcp --config config.yaml`.

## Tool Surface
| Tool | Why It Matters |
| --- | --- |
| `list_workbooks` | Lists discoverable workbooks with slug + short ID so agents can choose targets without remembering long hashes. |
| `describe_workbook` | Returns workbook-level metadata (size, sheet count, CAPS) to gauge complexity before drilling in. |
| `list_sheets` | Presents sheet summaries, visibility, metrics, and tags to help prioritize inspection order. |
| `sheet_overview` | Offers classification, headline stats, and highlights (tables, named items) for a single sheet. |
| `sheet_page` | Pages through tabular data with optional formula/style payloads, enabling high-signal slices for LLM review. |
| `sheet_formula_map` | Groups identical formulas and aggregates ranges so the agent can spot patterns without sifting cell-by-cell. |
| `formula_trace` | Walks precedents/dependents recursively (with safe pagination) to explain how values propagate across sheets. |
| `named_ranges` | Surfaces named items, their scope, and target ranges to anchor reasoning in business terminology. |
| `sheet_statistics` | Captures distribution metrics, data density, and heuristics that describe how "busy" a sheet is. |
| `find_formula` | Searches formulas by text/regex, ideal for locating specific functions or references quickly. |
| `scan_volatiles` | Flags volatile functions and high-churn ranges so models can reason about recalculation risk. |
| `sheet_styles` | Summarizes style reuse and annotations, revealing which cells carry semantic emphasis or commentary. |
| `get_manifest_stub` | Emits a structured stub that downstream pipelines can drop into corpus manifests. |
| `close_workbook` | Explicitly evicts a workbook from the cache to free memory between exploratory sessions. |

## Workspace Semantics
- Workbooks are discovered relative to the configured workspace root. Subdirectories are preserved; use `list_workbooks` filters (`slug_prefix`, `folder`, `path_glob`) to focus the scan.
- Single-workbook mode (`--workbook`) skips directory traversal and indexes only the specified file, while still providing the usual short ID aliases.
- XLSX files are fully parsed through `umya-spreadsheet`. XLS/XLSB are enumerated and validated before load; unsupported structures are reported as MCP errors instead of crashing the server.
- A bounded LRU cache keeps recently accessed workbooks warm while respecting memory limits.

## Development
- Format / lint using standard Rust tooling (`cargo fmt`, `cargo clippy`).
- Run the full test suite with `cargo test` from the project root; integration tests synthesize workbooks on the fly via `umya-spreadsheet` fixtures.

When opening pull requests, GitHub Actions will automatically run the cross-platform test + build workflow defined in `.github/workflows/ci.yml` and publish release binaries as artifacts.

## Related Documentation
Design notes and deeper architectural context live under `docs/`:
- `mcp-server-design.md` — server architecture and module responsibilities.
- `mcp-server-plan.md` — roadmap, tool contracts, and CAPS approach.
- `mcp-rust-umya-analysis.md` — backend decision record and XLSX-first rationale.
- `formualizer-parse-integration.md` — formula parser integration details.
