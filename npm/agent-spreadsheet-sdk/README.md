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

Convenience methods are installed for all 31 operations in the checked-in canonical registry. For example, `read_cells` provides `readCells(input)`. Each method is still gated by the backend's live operation set. Use `execute` when operation names are dynamic.

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
    listTools() {
      return mcpClient.listTools()
    },
    invoke(operation, input) {
      return mcpClient.callTool(operation, input)
    }
  }
})
await backend.initialize()
```

MCP support is negotiated from the live server. The transport may expose `listOperations()`, `listTools()`, `"tools/list"()`, or `request({ method: "tools/list" })`. `execute` waits for initial negotiation automatically; `initialize()` makes it explicit and `refresh()` updates support after the server changes. Before negotiation, synchronous `getCapabilities()` truthfully returns `initialized: false` and an empty operation list.

For transports without discovery, pass an explicit `supportedOperations` array. There is no fallback that treats all manifest operations as available. A transport may dispatch through `invoke(operation, input)` or methods named after canonical operations.

## WASM

WASM capabilities come directly from the generated binding's live `operations()` descriptors. Callers do not provide or default an operation list.

```js
const backend = new WasmBackend({ bindings: wasmBindings })

const { resource_id } = await backend.bindWorkbook({ workbookBytes })
const result = await backend.execute("describe_workbook", { resource_id })
```

The WASM JSON binding contract is:

```text
executeOperation(sessionId, operationName, paramsJson) -> resultJson
```

The binding returns a typed `session:` resource ID. The adapter passes that exact ID unchanged through canonical execution, export, and disposal, serializes only the canonical input for transport, and parses the returned JSON. It does not reshape canonical responses. Resource binding, byte export, and session disposal remain adapter-specific.

Run `npm run test:wasm` to build the actual wasm-bindgen Node package with `wasm-pack` and exercise SDK read, write, recalculate, verify, export, and disposal end to end.

## Capabilities and errors

`backend.getCapabilities().operations` is the authoritative supported operation list. Compatibility booleans such as `supportsVerification` are derived from that list rather than independently configured. Resource capabilities (`resourceBinding`, `resourceExport`) are derived from real adapter bindings.

Unsupported calls throw `CapabilityError` with code `UNSUPPORTED_CAPABILITY`. Canonical success and error envelopes, including revision and proof metadata, pass through without response normalization. Transport rejections also remain unchanged.

## Deprecated compatibility methods

The 0.13 method names remain as compatibility projections where they compile directly to a canonical operation. They translate legacy inputs, call `execute`, and return the canonical envelope's `data` field. Examples include `rangeValues` -> `read_cells`, `findValue` -> `search_values`, and batch/name helpers -> `write`.

New code should use canonical inputs with `execute` or the canonical convenience methods. Backend-specific lifecycle methods (`createSession`, `exportWorkbook`, and `disposeSession`) remain available for WASM resource handling.
