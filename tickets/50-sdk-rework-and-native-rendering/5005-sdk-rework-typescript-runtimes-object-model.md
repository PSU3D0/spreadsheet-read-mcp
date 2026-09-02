# Ticket: 5005 SDK 0.15 Rework: TypeScript, Runtimes, Object Model

## Depends On
- 5001 (route shape)
- 5002 (binding shape and npm runtime package)

## Why
The 0.14 SDK is a canonical protocol adapter, not an application SDK: no declarations, untyped `execute`, resource IDs in every call, two backends with different lifecycles behind one API, methods installed by `defineProperty`, a README whose first example uses an undefined variable, and an MCP transport that does not match a standard MCP client.

## Owner / Effort / Risk
- Owner: SDK
- Effort: L
- Risk: Med

## Scope

### Build and types
- TypeScript source, tsc build to `dist/` CJS and ESM, `exports` map with `types`. Generated `OperationName`, `InputOf<K>`, `OutputOf<K>` from the checked-in registry JSON via json-schema-to-typescript as a dev dependency, with a drift test against `asp registry --all`. Generated static methods file for editor discoverability.
- `execute<K extends OperationName>(operation: K, input: InputOf<K>): Promise<OutputOf<K>>` as the typed escape hatch under `client.canonical`.

### Object model
- `createLocalSpreadsheet({ runtime })` with `LocalRuntime` over the `agent-spreadsheet-wasm` package. `LocalWorkbook` owns a session: read/analysis methods, `write`, `recalculate`, `verifyAgainst(other)`, `exportBytes()`, `dispose()`, `Symbol.asyncDispose`.
- `connectSpreadsheetServer({ baseUrl, fetch? })` with `ServerRuntime` over `/v1/op/{operation}`. `listWorkbooks()`, `RemoteWorkbook` (non-owning read handle), `RemoteFork` (write, recalculate, verify, checkpoint, stagedChange, getChanges, export, discard).
- Shared `WorkbookView` read surface generated from the registry's single-read operations. Unsupported operations throw `CapabilityError` before transport.
- Resource IDs and revisions are held by the objects. Compare-and-swap uses the object's tracked revision unless the caller overrides.
- `renderSheet()` returns `{ png: Uint8Array, fidelity, warnings, calculation }` on both runtimes: local via the bytes binding (5006), remote via the artifact bytes route.

### Errors and compat
- One `SpreadsheetError` hierarchy wrapping canonical error envelopes with `code`, `operation`, `path`, `details`. WASM rejections are decoded into it, never thrown as raw objects.
- `McpBackend`, `WasmBackend`, the legacy method layer, and `stateless-byte-adapter` move to `agent-spreadsheet-sdk/compat` for one release with `@deprecated` declarations. The just-bash command is rebuilt on `LocalRuntime`.
- README rewritten; every code block is executed by a test.

## Non-Goals
- An MCP client adapter. MCP users use an MCP client.
- Any change to canonical semantics.

## Tests
- Type tests (`tsd` or `expect-type`) proving wrong operation names and wrong input shapes fail to compile.
- Local runtime integration against the built npm wasm package.
- Server runtime integration against a spawned `agent-spreadsheet-mcp` HTTP process.
- Registry drift and generated-methods drift.

## Definition of Done
- The acceptance gate in the tranche README holds for the SDK bullets.
