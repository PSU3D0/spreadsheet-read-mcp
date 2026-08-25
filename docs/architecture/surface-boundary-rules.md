# Surface Boundary Rules (Non-Negotiable)

Status: active guardrails (Tranche 35 foundation)
Owner: Tranche 35 (tickets/35-js-surface-migration)

This document defines the **hard boundaries** across CLI, MCP, WASM, and SDK surfaces.
If a change conflicts with these rules, the change must be redesigned (not waived ad hoc).

## Rule set

### 1) CLI is stateless and path-driven

- CLI commands **must** operate on explicit file paths or explicit output paths.
- CLI commands **must not** require long-lived server sessions, fork IDs, staged-change IDs, or MCP transport envelopes.
- CLI mutation mode flags (`--dry-run`, `--in-place`, `--output`, `--force`) are **CLI adapter concerns** and are not parity requirements for MCP/WASM.

Primary implementation boundary:
- `crates/agent-spreadsheet/src/cli/**`
- `crates/agent-spreadsheet/src/runtime/stateless.rs`

### 2) MCP owns session/fork/staging orchestration

- Fork lifecycle (`create_fork`, `save_fork`, `discard_fork`, checkpoints, staged changes) is **MCP-first orchestration**.
- These workflows **must not** leak into CLI as required behavior.
- MCP transport contracts (tool envelopes, tool-level guardrails/timeouts) are **adapter-mcp concerns**.

Primary implementation boundary:
- `crates/agent-spreadsheet-mcp/src/server.rs`
- `crates/agent-spreadsheet/src/tools/fork.rs`

### 3) WASM is byte/session oriented (not workspace/path oriented)

- WASM surface **must** expose in-memory/session APIs suitable for browser/runtime embedding.
- WASM surface **must not** require workspace roots, repository scanning, or host path mapping semantics.
- Any host-specific path/workspace policy remains outside WASM adapter boundaries.

Primary architecture projection boundary:
- `core.read.*`, `core.write.*`, `core.analysis.*` semantics are reusable.
- Path/workspace semantics remain adapter-owned (CLI/MCP hosts).

### 4) SDK is backend-abstraction, not a fourth semantics fork

- SDK **must** normalize on shared capability semantics across `McpBackend` and `WasmBackend` where the matrix says `ALL`.
- SDK **must not** expose backend-specific orchestration primitives as default cross-backend APIs.
- Backend-specific operations (e.g., MCP fork lifecycle) must be explicitly namespaced/opt-in.

### 5) Shared semantics live in core

- Capabilities marked `ALL` in the matrix are expected to converge on shared core behavior.
- Adapters may shape UX/transport, but semantic divergence requires explicit matrix/boundary doc updates.
- New capabilities must declare classification (`ALL`, `CLI_ONLY`, `MCP_ONLY`, `SHARED_PARTIAL`) before implementation.

## Enforcement hooks

- Capability inventory source of truth:
  - `docs/architecture/surface-capability-matrix.md`
- Drift check (CLI/MCP coverage against matrix):
  - `scripts/check_surface_matrix_drift.py`
- Local enforcement command:
  - `python3 scripts/check_surface_matrix_drift.py`
  - `cargo test -p agent-spreadsheet surface_matrix_drift_check`

## Change control

Any boundary exception proposal must include:
1. Which rule is being changed and why.
2. Matrix updates (`surface-capability-matrix.md`) for affected rows.
3. Drift-check/test updates so the new rule is enforced automatically.
