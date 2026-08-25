# agent-spreadsheet-sdk

**`agent-spreadsheet-sdk` is the JavaScript integration layer for agent-spreadsheet — the tool interaction service for agent-based spreadsheet work.**

It lets one application target multiple spreadsheet backends behind a consistent API:
- **MCP backend** for remote/stateful workflows
- **WASM/session backend** for embedded or in-process workflows

The goal is not “another transport wrapper.”
The goal is **one app-facing spreadsheet interaction model** for agent systems.

---

## Install

```bash
npm i agent-spreadsheet-sdk
```

Node: `>=18`

---

## What this SDK normalizes

- shared method names across backends
- tolerance for common input aliases (`camelCase` / `snake_case`)
- top-level output normalization
- capability checks for backend-specific flows
- typed errors for unsupported features and backend failures

---

## Backends

### `McpBackend`
Use when your app talks to a running `agent-spreadsheet-mcp` server or MCP client transport.

Best for:
- multi-turn agent workflows
- remote/stateful spreadsheet execution
- fork lifecycle and staged changes

### `WasmBackend`
Use when your app embeds a WASM/session runtime.

Best for:
- local/in-process execution
- browser or embedded app scenarios
- session-oriented workflows without MCP transport

---

## Quick start

```js
const { McpBackend, WasmBackend } = require("agent-spreadsheet-sdk")

const mcp = new McpBackend({ transport: myMcpTransport })
const wasm = new WasmBackend({ bindings: myWasmBindings })

async function sharedReadFlow(backend, ctx) {
  const workbook = await backend.describeWorkbook(ctx)
  const sheets = await backend.listSheets(ctx)
  const overview = await backend.sheetOverview({ ...ctx, sheetName: sheets[0] })
  const page = await backend.sheetPage({
    ...ctx,
    sheetName: sheets[0],
    startRow: 1,
    pageSize: 50,
    format: "compact"
  })

  return { workbook, sheets, overview, page }
}
```

Typical context identity:
- MCP: `{ workbookId: "..." }`
- WASM/session: `{ sessionId: "..." }`
- higher-level helper code may use `{ contextId: "..." }`

---

## Shared methods

### Read / inspection
- `describeWorkbook(input)`
- `listSheets(input)`
- `sheetOverview(input)`
- `namedRanges(input)`
- `rangeValues(input)`
- `readTable(input)`
- `sheetPage(input)`
- `findValue(input)`
- `gridExport(input)`

### Write
- `transformBatch(input)`

### Verification helpers
- `verifyWorkbook(input)`
- `verifyTargets(input)`
- `verifyErrors(input)`

### Backend-specific lifecycle
- MCP-oriented: `createFork`, `listForks`, `saveFork`, `discardFork`, staged-change methods
- WASM/session-oriented: `createSession`, `exportWorkbook`, `disposeSession`

Always branch on capabilities before using backend-specific methods.

---

## Capability model

```js
const caps = backend.getCapabilities()

if (caps.supportsForkLifecycle) {
  // MCP path
}

if (caps.supportsVerification) {
  // verifyWorkbook / verifyTargets / verifyErrors
}

if (caps.supportsSessionLifecycle) {
  // WASM/session path
}
```

Unsupported backend-specific calls throw `CapabilityError` with code:
- `UNSUPPORTED_CAPABILITY`

This is a feature, not a footgun: **capabilities are the contract for mixed-backend safety**.

---

## Error model

SDK exports:
- `SpreadsheetSdkError`
- `CapabilityError`
- `BackendOperationError`

Common normalized codes:
- `INVALID_ARGUMENT`
- `INVALID_RESPONSE`
- `UNSUPPORTED_CAPABILITY`
- backend-provided machine codes when available

---

## MCP transport contract

`McpBackend` accepts either:
- a generic `invoke(operation, params)` transport
- or transport objects with per-operation methods

Example:

```js
const backend = new McpBackend({
  transport: {
    async invoke(operation, params) {
      return client.callTool(operation, params)
    }
  }
})
```

---

## WASM bindings contract

`WasmBackend` expects bindings shaped like:
- `createSession`
- `describeWorkbook`
- `namedRanges`
- `sheetOverview`
- `listSheets`
- `rangeValues`
- `findValue`
- `readTable`
- `sheetPage`
- `gridExport`
- `transformBatch`
- `exportWorkbook`
- `disposeSession`

These map to the in-repo `agent-spreadsheet-wasm` work.

---

## When to use this SDK

Use `agent-spreadsheet-sdk` when you are building:
- an app that may swap between MCP and embedded execution
- a higher-level agent workflow system that should not care about transport details
- a UI or service that needs one normalized spreadsheet interaction model

Use the CLI directly when you only need shell/file workflows.
Use MCP directly when your system already lives entirely inside an MCP runtime.

---

## Release lanes

This package is published from `sdk-vX.Y.Z` tags.

Dist-tag policy:
- stable -> `latest`
- `-rc.N` -> `rc`
- `-beta.N` -> `beta`
- `-alpha.N` -> `alpha`

---

## Related packages

- `agent-spreadsheet` — npm CLI wrapper
- `agent-spreadsheet` — Rust semantic core
- `agent-spreadsheet-mcp` — MCP server
- `agent-spreadsheet-wasm` — in-repo WASM-facing crate

Repo: <https://github.com/PSU3D0/agent-spreadsheet>
