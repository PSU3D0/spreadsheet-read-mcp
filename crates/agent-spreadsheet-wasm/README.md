# agent-spreadsheet-wasm

**`agent-spreadsheet-wasm` is the in-repo WASM-facing wrapper around the agent-spreadsheet semantic core.**

It provides a byte/session-oriented bridge for embedded spreadsheet workflows that want the same underlying behavior as the CLI and MCP surfaces.

---

## Current status

This crate exists and is tested in-repo, but it is currently:
- **experimental / internal-facing**
- **not published to crates.io**
- part of the broader WASM/backend strategy for agent-spreadsheet

So the correct expectation is:
- the WASM surface is real in the codebase
- its public packaging and distribution story is still evolving

---

## What it is for

Use this layer when you need:
- workbook bytes rather than filesystem-only flows
- session-oriented in-process spreadsheet execution
- a bridge for browser/embedded integrations
- a backend target for the JS SDK's WASM/session path

---

## Relationship to the rest of the stack

| Package | Role |
| --- | --- |
| `agent-spreadsheet` | Shared spreadsheet engine and semantics |
| `agent-spreadsheet-wasm` | WASM-facing wrapper over the shared core |
| `agent-spreadsheet-mcp` | Stateful MCP server |
| `agent-spreadsheet` / `asp` | Stateless CLI |
| `agent-spreadsheet-sdk` | JS SDK capable of targeting MCP or WASM/session backends |

---

## In-repo development

This crate is part of the workspace and participates in the shared validation flow:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

For the broader product story and current workflow surfaces, see the root README:
<https://github.com/PSU3D0/agent-spreadsheet#readme>
