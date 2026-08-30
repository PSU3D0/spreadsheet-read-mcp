# agent-spreadsheet-wasm

`agent-spreadsheet-wasm` is the byte-session adapter for the shared canonical operation dispatcher. It does not expose workspace paths, forks, or host process assumptions.

## Canonical API

```js
const sessionId = createSession(workbookBytes); // session:<opaque-id>
const resultJson = await executeOperation(
  sessionId,
  "read_cells",
  JSON.stringify({
    sheet_name: "Sheet1",
    selection: { kind: "range", ranges: ["A1:C20"] }
  })
);
const outputBytes = exportWorkbook(sessionId);
disposeSession(sessionId);
```

`paramsJson` uses the canonical operation schema. The adapter injects `resource_id` when omitted; if supplied, it must exactly match `sessionId`. Success is a serialized canonical response envelope. Rejections from `executeOperation` are canonical error envelopes.

Call `operations()` for serialized canonical discovery filtered to capabilities actually backed by this runtime. The byte-session runtime supports the advertised canonical reads plus `write`, `recalculate`, and `verify_workbook`. Writes use revision CAS; preview is pure, apply atomically replaces the session bytes by default, and non-atomic apply returns canonical partial results. `mode:"stage"` is rejected with `INVALID_REQUEST` because WASM sessions do not expose durable staged bundles. Recalculation uses Formualizer entirely in memory, and verification binds the current session to another `session:` id supplied as `baseline_resource_id`.

Workspace discovery, fork/checkpoint/history lifecycle, LibreOffice rendering, SheetPort, and VBA inspection remain hidden. `exportWorkbook(sessionId)` always returns the latest applied or recalculated bytes.

The older named bindings remain for compatibility. New integrations should use `executeOperation`; `exportWorkbook` and `disposeSession` own the adapter lifecycle.

## Limits

Limits are checked before workbook bytes are copied into WASM or parsed and before parameter JSON is parsed:

- 64 MiB per workbook
- 1 MiB per canonical parameter document
- 16 concurrent sessions
- 256 MiB aggregate source workbook bytes

## Development

```bash
cargo test -p agent-spreadsheet-wasm
cargo check -p agent-spreadsheet-wasm --target wasm32-unknown-unknown
cargo test -p agent-spreadsheet-wasm --target wasm32-unknown-unknown --test wasm_node
cargo test -p agent-spreadsheet-wasm --target wasm32-unknown-unknown --test wasm_write_kinds
cargo test -p agent-spreadsheet-wasm --target wasm32-unknown-unknown --test wasm_browser
cargo clippy -p agent-spreadsheet-wasm --all-targets -- -D warnings
```

The browser test requires a configured WebDriver/browser environment. The Node test uses `wasm-bindgen-test-runner`.
