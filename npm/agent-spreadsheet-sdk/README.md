# agent-spreadsheet-sdk

A TypeScript SDK for driving spreadsheets through the agent-spreadsheet canonical
operation protocol. One object model, two runtimes:

- **local** — the operations run in-process on WebAssembly. You own workbook bytes.
- **server** — the operations run in an `agent-spreadsheet-mcp` process over its
  canonical `/v1` HTTP route, sharing that process's workspace, forks, and checkpoints.

```bash
npm install agent-spreadsheet-sdk agent-spreadsheet-wasm   # local runtime
npm install agent-spreadsheet-sdk                          # server runtime only
```

Node.js 18 or newer. Ships CommonJS, ESM, and declarations; no runtime dependencies.

Every JavaScript block in this file is executed by `test/readme.test.js`.

## Local runtime

`local.open(bytes)` returns a `LocalWorkbook` that owns a WASM session and tracks its
own `resource_id` and `revision_id`. Compare-and-swap revisions are filled in for you.

```js
const fs = require("node:fs")
const { createWasmRuntime } = require("agent-spreadsheet-wasm")
const { createLocalSpreadsheet } = require("agent-spreadsheet-sdk")

const local = createLocalSpreadsheet({ runtime: createWasmRuntime({}) })
const workbook = await local.open(fs.readFileSync("book.xlsx"))

const sheets = await workbook.listSheets()
console.log(workbook.resourceId, sheets.operation)

// expected_revision defaults to the revision this workbook last saw.
const written = await workbook.write({
  mode: "apply",
  ops: [{
    kind: "set_cells",
    sheet_name: "Sheet1",
    cells: { A1: { kind: "value", value: "e" } }
  }]
})
console.log(written.data.status, workbook.revisionId)

await workbook.recalculate({ backend: "formualizer" })
fs.writeFileSync("book.out.xlsx", await workbook.exportBytes())
await workbook.dispose()
```

`dispose()` releases the session. `LocalWorkbook` also implements `Symbol.asyncDispose`,
so `await using workbook = await local.open(bytes)` releases it at the end of the scope.

Verification binds two sessions in one call:

```js
const fs = require("node:fs")
const { createWasmRuntime } = require("agent-spreadsheet-wasm")
const { createLocalSpreadsheet } = require("agent-spreadsheet-sdk")

const local = createLocalSpreadsheet({ runtime: createWasmRuntime({}) })
const bytes = fs.readFileSync("book.xlsx")
const current = await local.open(bytes)
const baseline = await local.open(bytes)

const proof = await current.verifyAgainst(baseline, {
  targets: ["Sheet1!A1"],
  targets_only: true
})
console.log(proof.operation)

await current.dispose()
await baseline.dispose()
```

## Server runtime

`connectSpreadsheetServer` speaks the canonical `/v1` route of a running
`agent-spreadsheet-mcp` process. Reads go straight to a workbook; writes go to a fork.

```js
const { connectSpreadsheetServer } = require("agent-spreadsheet-sdk")

const client = connectSpreadsheetServer({ baseUrl: "http://127.0.0.1:8079" })

const listed = await client.listWorkbooks({ limit: 10 })
const workbook = client.workbook(listed.data.workbooks[0].resource_id)

const described = await workbook.describe()
console.log(described.resource_id, described.revision_id)

const fork = await workbook.createFork()
await fork.write({
  mode: "apply",
  ops: [{
    kind: "set_cells",
    sheet_name: "Sheet1",
    cells: { A1: { kind: "value", value: "e" } }
  }]
})
await fork.checkpoint({ action: "create", label: "after-write" })

const proof = await fork.verifyAgainst(workbook, { targets_only: true })
console.log(proof.operation)

await fork.discard()
```

`RemoteWorkbook` is a non-owning read handle: it never disposes anything. `RemoteFork`
adds `write`, `recalculate`, `verifyAgainst`, `getChanges`, `checkpoint`, `stagedChange`,
`exportFork`, and `discard`, and its `Symbol.asyncDispose` discards the fork.

The `/v1` route has no authentication; its loopback bind is the security boundary. Pass
`fetch` and `headers` if you front it with an authenticating proxy:

```js
const { connectSpreadsheetServer } = require("agent-spreadsheet-sdk")

const client = connectSpreadsheetServer({
  baseUrl: "http://127.0.0.1:8079",
  headers: { "x-proxy-token": "local-only" },
  fetch: (url, init) => fetch(url, init)
})
console.log((await client.capabilities()).includes("read_cells"))
```

## The shared read surface

Every workbook-shaped object — `LocalWorkbook`, `RemoteWorkbook`, `RemoteFork` — exposes
the same generated read surface, one method per single-resource read operation
(`describeWorkbook`, `listSheets`, `sheetOverview`, `readCells`, `inspectCells`,
`readTable`, `readLayout`, `exportGrid`, `namedRanges`, `analyzeStyles`, `searchValues`,
`searchFormulas`, `formulaTrace`, `formulaMap`, `profileTable`, `sheetStatistics`,
`screenshotSheet`, `sheetportManifest`, `executeSheetport`, `inspectVba`). The methods are
generated as real declarations from the registry, so editors complete them and their
inputs and outputs are typed. Responses are canonical envelopes, unmodified.

```js
const { connectSpreadsheetServer, READ_SURFACE_OPERATIONS } = require("agent-spreadsheet-sdk")

const client = connectSpreadsheetServer({ baseUrl: "http://127.0.0.1:8079" })
const workbook = client.workbook("wb:wb-1")

const cells = await workbook.readCells({
  sheet_name: "Sheet1",
  selection: { kind: "range", ranges: ["A1:B2"] },
  format: "dense"
})
console.log(cells.operation, READ_SURFACE_OPERATIONS.includes("read_cells"))
```

`resource_id` is injected by the object, never by you. When you do want the raw protocol,
`client.canonical.execute` is the typed escape hatch and takes the full canonical input:

```js
const { connectSpreadsheetServer } = require("agent-spreadsheet-sdk")

const client = connectSpreadsheetServer({ baseUrl: "http://127.0.0.1:8079" })
const response = await client.canonical.execute("read_cells", {
  resource_id: "wb:wb-1",
  sheet_name: "Sheet1",
  selection: { kind: "range", ranges: ["A1:B2"] }
})
console.log(response.schema_version, response.operation)
```

In TypeScript, `execute<K extends OperationName>(operation: K, input: InputOf<K>)` returns
`Promise<OutputOf<K>>`. A wrong operation name or a wrong input shape is a compile error.

## Types

`OperationName`, `InputOf<K>`, `OutputOf<K>`, and `CanonicalErrorEnvelope` are generated
from `src/generated/canonical-registry.json`, which is itself generated from the Rust
registry. The generated TypeScript is checked in and drift-tested.

```bash
ASP_BINARY=../../target/debug/asp npm run generate:registry
npm run generate:types
npm test
```

## Rendering

`renderSheet` returns PNG bytes plus what the renderer reported. Image bytes cross the
adapter boundary rather than a canonical operation: the server runtime fetches
`GET /v1/artifacts/{handle}`, and the local runtime uses the WASM byte binding.

```js
const { connectSpreadsheetServer } = require("agent-spreadsheet-sdk")

const client = connectSpreadsheetServer({ baseUrl: "http://127.0.0.1:8079" })
const rendered = await client.workbook("wb:wb-1").renderSheet({
  sheet_name: "Sheet1",
  range: "A1:H40"
})
console.log(rendered.png.byteLength, rendered.fidelity, rendered.warnings.length)
```

`fidelity` is `"unknown"` and `warnings` is empty against a server that predates fidelity
reporting. A local runtime whose bindings have no artifact binding throws
`CapabilityError` rather than pretending.

## Errors

One hierarchy, rooted at `SpreadsheetError`:

| Class | Thrown when |
| --- | --- |
| `CanonicalOperationError` | An adapter returned a canonical error envelope. Carries `code`, `operation`, `path`, `details`, and the raw `envelope`. |
| `CapabilityError` | The live runtime does not advertise the operation. Thrown before any transport. |
| `TransportError` | A non-canonical failure: an unreachable host, a proxy error page, a body that is not an envelope. Carries `status` and `body`. |

WASM rejections are decoded into `CanonicalOperationError`, never rethrown as raw strings.
HTTP failures use the route's pinned status table (`CANONICAL_ERROR_STATUS`).

```js
const { CanonicalOperationError, CapabilityError, connectSpreadsheetServer } =
  require("agent-spreadsheet-sdk")

const client = connectSpreadsheetServer({ baseUrl: "http://127.0.0.1:8079" })
const fork = await client.workbook("wb:wb-1").createFork()

try {
  await fork.write({ expected_revision: "stale", mode: "apply", ops: [] })
} catch (error) {
  if (error instanceof CanonicalOperationError) {
    console.log(error.code, error.canonicalStatus, error.path)
  } else if (error instanceof CapabilityError) {
    console.log("unsupported:", error.operation)
  } else {
    throw error
  }
}
```

## Capabilities

`client.capabilities()` is the authoritative live operation list: `GET /v1/operations` for
the server runtime, the binding's `operations()` for the local one. Operations outside it
throw `CapabilityError` before any bytes move.

```js
const { connectSpreadsheetServer } = require("agent-spreadsheet-sdk")

const client = connectSpreadsheetServer({ baseUrl: "http://127.0.0.1:8079" })
const capabilities = await client.capabilities()
console.log(capabilities.includes("create_fork"), capabilities.includes("inspect_vba"))
```

## just-bash

`agent-spreadsheet-sdk/just-bash` registers one `asp` custom command on the local runtime.
Install `just-bash` explicitly; it is an optional peer dependency. It accepts only
`asp op <operation> [--bind VFS_PATH] [--baseline VFS_PATH] [--json JSON]
[--output VFS_PATH|--in-place]`, plus `asp operations`, `asp schema <op>`, and
`asp example <op>`. It reads and writes workbook bytes only through `ctx.fs`, binds an
ephemeral session per call, and exports atomically through a temporary file plus `mv`.

```js
const { Bash } = require("just-bash")
const { createWasmRuntime } = require("agent-spreadsheet-wasm")
const { createAspCommand } = require("agent-spreadsheet-sdk/just-bash")

const bash = new Bash({
  files: { "/workbook.xlsx": Uint8Array.from([1, 2, 3]) },
  customCommands: [createAspCommand({ bindings: createWasmRuntime({}) })]
})
const result = await bash.exec("asp op list_sheets --bind /workbook.xlsx", {
  stdin: JSON.stringify({})
})
console.log(JSON.parse(result.stdout).operation)
```

just-bash 3.4.2 needs Node.js 20.18.1 or newer, and its `js-exec` bridge needs an ESM host.
Defaults match the WASM ceilings (64 MiB per workbook, 1 MiB per parameter document);
override them with `maxWorkbookBytes` and `maxParamsBytes`.

## Deprecated: `agent-spreadsheet-sdk/compat`

The 0.14 surface — `McpBackend`, `WasmBackend`, the legacy camel-case method layer, and
`stateless-byte-adapter` — moved to `agent-spreadsheet-sdk/compat` for one release. Every
export there is `@deprecated` and will be removed in the release after 0.15.

Migration: replace `new WasmBackend({ bindings })` with
`createLocalSpreadsheet({ runtime })` plus `local.open(bytes)`. Replace
`new McpBackend({ transport })` with `connectSpreadsheetServer({ baseUrl })`, or with a
real MCP client if you want the MCP transport — the SDK no longer pretends to be one.
Legacy methods that flattened envelopes to `data` have no replacement: canonical envelopes
are returned whole, and `client.canonical.execute` is the typed escape hatch.

## Development

```bash
npm install
npm run build            # dist/cjs, dist/esm, dist/types
npm run typecheck        # includes the type tests under test-types/
npm test                 # build + typecheck + node --test

# integration harnesses
node scripts/run-generated-wasm-integration.js          # builds/reuses the WASM package
SPREADSHEET_MCP_BINARY=../../target/debug/agent-spreadsheet-mcp npm run test:server
```

The Rust operation registry is the source of truth for the taxonomy, schemas, and adapter
support. This package never hand-normalizes canonical semantics; see
`docs/architecture/surface-boundary-rules.md` rule 5.
