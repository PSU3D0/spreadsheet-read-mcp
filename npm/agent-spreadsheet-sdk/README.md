# agent-spreadsheet-sdk

Backend adapters for the agent-spreadsheet canonical operation protocol.

```bash
npm install agent-spreadsheet-sdk
```

Node.js 18 or newer is required.

## Canonical dispatch

Both backends expose the same primary API and return the canonical envelope unchanged:

```js
const { McpBackend, WasmBackend } = require("agent-spreadsheet-sdk")

const result = await backend.execute("read_cells", {
  resource_id: "wb:workbook-1",
  sheet_name: "Revenue",
  selection: { kind: "range", ranges: ["A1:F50"] }
})
```

Convenience methods are installed from the checked-in canonical registry. For example, `read_cells` provides `readCells(input)`. Use `execute` when operation names are dynamic.

The Rust operation registry remains the taxonomy and schema source of truth. `src/generated/canonical-registry.json` is generated, not independently authored:

```bash
ASP_BINARY=../../target/debug/asp npm run generate:registry
npm test
```

Generation consumes `asp operations` and `asp schema <operation>`. The Node drift test compares every checked-in descriptor and schema with those commands when `ASP_BINARY`, `target/debug/asp`, or `target/release/asp` is available.

## MCP

```js
const backend = new McpBackend({
  transport: {
    invoke(operation, input) {
      return mcpClient.callTool(operation, input)
    }
  },
  operations: ["describe_workbook", "read_cells", "write"]
})
```

`operations` should contain the server's advertised canonical operations. When omitted, MCP defaults to the generated native registry for compatibility. A transport may instead implement methods named after canonical operations.

## WASM

WASM capabilities are explicit: no canonical operation is advertised unless `executeOperation` exists and `operations` (or `bindings.supportedOperations`) declares it.

```js
const backend = new WasmBackend({
  bindings: wasmBindings,
  operations: ["describe_workbook", "read_cells"]
})

const { resource_id } = await backend.bindWorkbook({ workbookBytes })
const result = await backend.execute("describe_workbook", { resource_id })
```

The WASM JSON binding contract is:

```text
executeOperation(sessionId, operationName, paramsJson) -> resultJson
```

The adapter strips the `session:` prefix for the binding, serializes only for transport, and parses the returned JSON. It does not reshape canonical responses. Resource binding, byte export, and session disposal remain adapter-specific.

## Capabilities and errors

`backend.getCapabilities().operations` is the authoritative supported operation list. Compatibility booleans such as `supportsVerification` are derived from that list rather than independently configured. Resource capabilities (`resourceBinding`, `resourceExport`) are derived from real adapter bindings.

Unsupported calls throw `CapabilityError` with code `UNSUPPORTED_CAPABILITY`. Canonical success and error envelopes, including revision and proof metadata, pass through without response normalization. Transport rejections also remain unchanged.

## Deprecated compatibility methods

The 0.13 method names remain as compatibility projections where they compile directly to a canonical operation. They translate legacy inputs, call `execute`, and return the canonical envelope's `data` field. Examples include `rangeValues` -> `read_cells`, `findValue` -> `search_values`, and batch/name helpers -> `write`.

New code should use canonical inputs with `execute` or the canonical convenience methods. Backend-specific lifecycle methods (`createSession`, `exportWorkbook`, and `disposeSession`) remain available for WASM resource handling.
