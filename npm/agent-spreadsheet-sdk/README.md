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

Generation consumes the host-independent `asp registry --all` projection. The Node drift test compares every checked-in descriptor with that registry and separately verifies the adapter-projected `asp schema <operation>` output when `ASP_BINARY`, `target/debug/asp`, or `target/release/asp` is available.

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

Generic `tools/list` discovery accepts a descriptor only when `_meta["agent-spreadsheet/canonical"]` contains `schema_version: "1"` and an `operation` exactly matching the tool name. This prevents legacy compatibility routes that share canonical names, including the legacy screenshot route, from being misidentified as canonical. An explicit `listOperations()` result or `supportedOperations` array remains a trusted canonical API contract.

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

From a repository checkout, run `node scripts/run-generated-wasm-integration.js` to build the actual wasm-bindgen Node package with `wasm-pack` and exercise SDK read, write, recalculate, verify, export, and disposal end to end. This repository-only harness is intentionally not part of the published package.

## just-bash

The optional `agent-spreadsheet-sdk/just-bash` subpath registers one trusted host command over the generated WASM binding. Install `just-bash` explicitly; it is an optional peer so the core SDK does not pull the sandbox and its runtimes into other applications. just-bash 3.4.2 requires Node.js 20.18.1 or newer. Use an ESM host (`.mjs` or `"type":"module"`) when enabling its `js-exec` bridge: the upstream 3.4.2 CommonJS export fails that bridge with `Invalid URL`. CommonJS remains supported for ordinary custom-command execution when `js-exec` is not used.

```js
import { createRequire } from "node:module"
import { Bash } from "just-bash"

const require = createRequire(import.meta.url)
const { createAspCommand } = require("agent-spreadsheet-sdk/just-bash")

const bash = new Bash({
  files: { "/workbook.xlsx": workbookBytes },
  customCommands: [createAspCommand({ bindings: wasmBindings })]
})
const result = await bash.exec(
  "asp op read_cells --bind /workbook.xlsx",
  { stdin: JSON.stringify({ sheet_name: "Sheet1", selection: { kind: "range", ranges: ["A1:C10"] } }) }
)
```

The command accepts only `asp op <operation> [--bind VFS_PATH] [--baseline VFS_PATH] [--json JSON] [--output VFS_PATH|--in-place]`. It reads and writes workbook bytes only through `ctx.fs`. The reusable `agent-spreadsheet-sdk/stateless-byte-adapter` module owns ephemeral resource binding, canonical execution, export, status-to-exit mapping, and disposal; it is backend-oriented and has no dependency on just-bash or a filesystem. Preview never exports. Apply and recalculate require exactly one output target and use an adjacent VFS temporary file plus `mv`.

Writes to the same resolved target are serialized per `asp` command instance from the no-clobber check through `mv`. Concurrent `--output` calls therefore have exactly one winner, while `--in-place` replacements cannot interleave, and failed writes remove their temporary file.

The defaults match WASM's 64 MiB workbook and 1 MiB parameter ceilings. Override them with `maxWorkbookBytes` and `maxParamsBytes`; stat and payload limits are checked before creating a WASM session. `asp operations` is the intersection of the registry's explicit `adapters.just_bash` support plan and the backend's live operation capabilities. `asp schema <operation>` and `asp example <operation>` remain generic projections of the checked-in canonical registry. With `javascript: true`, `js-exec` can call `asp` through its `child_process.execSync` or `spawnSync` bridge without another tool projection.

## Capabilities and errors

`backend.getCapabilities().operations` is the authoritative supported operation list. Compatibility booleans such as `supportsVerification` are derived from that list rather than independently configured. Resource capabilities (`resourceBinding`, `resourceExport`) are derived from real adapter bindings.

Unsupported calls throw `CapabilityError` with code `UNSUPPORTED_CAPABILITY`. Canonical success and error envelopes, including revision and proof metadata, pass through without response normalization. Transport rejections also remain unchanged.

## Deprecated compatibility methods

The 0.13 method names remain as compatibility projections where they compile directly to a canonical operation. They translate legacy inputs, call `execute`, and return the canonical envelope's `data` field. Examples include `rangeValues` -> `read_cells`, `findValue` -> `search_values`, and batch/name helpers -> `write`.

Nine camel-case names collide with generated canonical convenience methods: `describeWorkbook`, `namedRanges`, `sheetOverview`, `listSheets`, `readTable`, `createFork`, `listForks`, `verifyWorkbook`, and `discardFork`. Legacy-shaped input preserves the 0.13 data-only result. An explicit canonical `resource_id` selects envelope-preserving canonical dispatch. `listForks({})` is ambiguous and therefore remains data-only; use `execute("list_forks", {})` for its canonical envelope.

New code should use canonical inputs with `execute`. Backend-specific lifecycle methods (`createSession`, `exportWorkbook`, and `disposeSession`) remain available for WASM resource handling.
