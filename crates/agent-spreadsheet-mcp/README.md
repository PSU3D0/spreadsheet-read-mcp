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

## Canonical tool surface

The default router is generated from the canonical operation registry. Each available descriptor becomes one MCP tool with the descriptor's closed input schema, canonical envelope, and worst-case risk annotations.

The baseline write-capable surface has 27 operations:

- discovery/read: `list_workbooks`, `describe_workbook`, `list_sheets`, `sheet_overview`, `read_cells`, `inspect_cells`, `read_table`, `read_layout`, `export_grid`, `named_ranges`
- analysis: `analyze_styles`, `search_values`, `search_formulas`, `formula_trace`, `formula_map`, `profile_table`, `sheet_statistics`
- write/lifecycle: `write`, `create_fork`, `list_forks`, `recalculate`, `verify_workbook`, `export_fork`, `discard_fork`, `get_changes`, `checkpoint`, `staged_change`

Capability-backed deployments can add `screenshot_sheet`, `sheetport_manifest`, `execute_sheetport`, and `inspect_vba`, for up to 31 tools. Read-only deployments omit write/lifecycle descriptors, screenshot requires its rendering backend, and VBA requires `--vba-enabled`. `close_workbook` is not canonical because cache eviction is runtime administration.

### 0.13 compatibility

Set `SPREADSHEET_MCP_SLIM_SURFACE=false` (or `--slim-surface=false`) during the compatibility window to add the legacy 0.13 tool names. Legacy routes win when a name is shared, preserving 0.13 schemas and envelopes without registering names twice. The default slim surface contains canonical tools only.

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
| `--slim-surface <BOOL>` | Canonical-only by default; `false` adds legacy 0.13 compatibility tools |
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
