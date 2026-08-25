# agent-spreadsheet-mcp

[![Crates.io](https://img.shields.io/crates/v/agent-spreadsheet-mcp.svg)](https://crates.io/crates/agent-spreadsheet-mcp)
[![License](https://img.shields.io/crates/l/agent-spreadsheet-mcp.svg)](https://github.com/PSU3D0/agent-spreadsheet/blob/main/LICENSE)

**`agent-spreadsheet-mcp` is the stateful MCP server for agent-spreadsheet — the tool interaction service for agent-based spreadsheet usage.**

Use it when your agent needs more than one-shot file commands:
- workbook caching across calls
- fork-based editing
- staged changes
- recalculation and screenshots
- long-lived, multi-turn spreadsheet workflows

---

## Why this server exists

The CLI is great for stateless jobs.

The MCP server exists for cases where an agent needs a **persistent spreadsheet workspace** with server-managed state:
- open a workbook once
- inspect it repeatedly without reloading every time
- fork, edit, recalculate, checkpoint, diff, and save
- stay inside a normal MCP tool-calling loop

This is the stateful half of the agent-spreadsheet product story.

---

## Install

```bash
cargo install agent-spreadsheet-mcp
```

Formualizer-backed recalc is included by default.

Or use Docker:

```bash
# Read-only / slim
docker pull ghcr.io/psu3d0/agent-spreadsheet-mcp:latest

# Write + recalc + screenshots
docker pull ghcr.io/psu3d0/agent-spreadsheet-mcp:latest-full
```

---

## Quickstart

### Stdio transport

```bash
agent-spreadsheet-mcp --workspace-root /path/to/workbooks --transport stdio
```

### Claude Code / Claude Desktop

Add to `~/.claude.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "spreadsheet": {
      "command": "agent-spreadsheet-mcp",
      "args": ["--workspace-root", "/path/to/workbooks", "--transport", "stdio"]
    }
  }
}
```

### Docker

```json
{
  "mcpServers": {
    "spreadsheet": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/path/to/workbooks:/data",
        "ghcr.io/psu3d0/agent-spreadsheet-mcp:latest-full",
        "--transport", "stdio"
      ]
    }
  }
}
```

`:latest` is the read-only slim image (write/fork/recalc tools disabled); `:latest-full` includes the write tools and recalculation.

### HTTP transport

```bash
agent-spreadsheet-mcp --workspace-root /path/to/workbooks
# -> http://127.0.0.1:8079  (POST /mcp)
```

---

## What this crate provides

### Read and discovery tools
- workbook listing and metadata
- sheet listing and summaries
- sheet overview and region detection
- sheet/page/range/table reads
- named ranges
- style inspection and workbook style summary

### Search and analysis tools
- value search
- formula search
- formula tracing
- table and sheet profiling
- volatile function scans
- SheetPort manifest stub generation

### Stateful write workflow
- fork lifecycle
- checkpoints and restore
- edit / transform / style / structure operations
- column size, layout, and rules operations
- recalc
- changeset review
- staged change management
- save/export of forks

### Optional extras
- VBA inspection
- screenshots via LibreOffice-backed deployments

---

## Tool families

### Always-available read tools
- `list_workbooks`
- `describe_workbook`
- `list_sheets`
- `workbook_summary`
- `sheet_overview`
- `sheet_page`
- `read_table`
- `range_values`
- `named_ranges`
- `sheet_styles`
- `workbook_style_summary`
- `find_value`
- `find_formula`
- `sheet_formula_map`
- `formula_trace`
- `scan_volatiles`
- `table_profile`
- `sheet_statistics`
- `get_manifest_stub`

### Opt-in write/recalc tools
Enable with `--recalc-enabled`.

Includes:
- fork lifecycle
- checkpoints
- `edit_batch`
- `transform_batch`
- `style_batch`
- `apply_formula_pattern`
- `structure_batch`
- `column_size_batch`
- `sheet_layout_batch`
- `rules_batch`
- `recalculate`
- `get_changeset`
- `save_fork`
- staged-change management
- `screenshot_sheet`

### Opt-in VBA tools
Enable with `--vba-enabled`.

Includes:
- `vba_project_summary`
- `vba_module_source`

---

## Deployment modes

| Mode | State | Recalc | Notes |
| --- | --- | --- | --- |
| stdio MCP | stateful | optional | best for local MCP clients |
| HTTP MCP | stateful | optional | best for remote/client-server setups |
| Docker `latest` | stateful | no | slim read-only image |
| Docker `full` | stateful | yes | LibreOffice-backed write/recalc/screenshot image |

---

## Key configuration

All flags also support `SPREADSHEET_MCP_` environment variables.

| Flag | Purpose |
| --- | --- |
| `--workspace-root <DIR>` | Workbook root directory |
| `--transport stdio|http` | Transport mode |
| `--cache-capacity <N>` | LRU workbook cache size |
| `--recalc-enabled` | Enable write/recalc tools |
| `--vba-enabled` | Enable VBA tools |
| `--output-profile token-dense|verbose` | Output verbosity profile |
| `--http-bind <ADDR>` | HTTP bind address |
| `--enabled-tools <csv>` | Tool allowlist |
| `--tool-timeout-ms <MS>` | Per-tool timeout |
| `--max-response-bytes <N>` | Response size guard |

See the root README for the full configuration matrix.

---

## Recalc backends

| Backend | Default | Best for |
| --- | --- | --- |
| Formualizer | yes | fast native recalc with no external dependency |
| LibreOffice | opt-in / Docker `full` | maximum compatibility and screenshots |

---

## When to choose MCP vs CLI

Choose **`agent-spreadsheet-mcp`** when you want:
- one workbook loaded across many tool calls
- long-lived agent sessions
- fork/checkpoint/save workflows
- server-managed state
- richer stateful automation loops

Choose **`asp` / `agent-spreadsheet`** when you want:
- one-shot commands
- shell pipelines
- CI jobs
- stateless file operations

---

## Related packages

| Package | Role |
| --- | --- |
| [`agent-spreadsheet`](../agent-spreadsheet/) | Shared semantic core |
| `agent-spreadsheet` / `asp` | Stateless CLI |
| `agent-spreadsheet-sdk` | JS SDK |

---

## Full documentation

See the [root README](https://github.com/PSU3D0/agent-spreadsheet#readme) for:
- product overview
- CLI workflows
- verification and session guidance
- Docker deployment examples
- architecture and recalc notes

---

## License

Apache-2.0 — see [LICENSE](../../LICENSE).
