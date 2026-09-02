# agent-spreadsheet-wasm

WebAssembly byte/session runtime for [agent-spreadsheet](https://github.com/PSU3D0/agent-spreadsheet).

It creates a workbook resource from bytes, dispatches canonical operations
against it, and exports bytes back. It never touches a workspace root, a host
path, or an MCP fork identity.

## Install

```sh
npm install agent-spreadsheet-wasm
```

## Use

```js
import { createWasmRuntime } from "agent-spreadsheet-wasm"

const runtime = await createWasmRuntime()

const resourceId = runtime.createSession(new Uint8Array(workbookBytes))
const described = JSON.parse(
  await runtime.executeOperation(resourceId, "describe_workbook", "{}")
)
const exported = runtime.exportWorkbook(resourceId)
runtime.disposeSession(resourceId)
```

`createWasmRuntime()` reads the `.wasm` next to the package on Node 18+ and
fetches it in browsers. Override the source when your bundler moves the asset:

```js
await createWasmRuntime({ wasmUrl: new URL("./agent_spreadsheet_wasm_bg.wasm", import.meta.url) })
await createWasmRuntime({ wasmBytes })
```

The module is instantiated once per process; later calls return the same
bindings, so sessions stay visible across callers.

## With the SDK

The resolved object is exactly what `agent-spreadsheet-sdk` expects:

```js
import { WasmBackend } from "agent-spreadsheet-sdk"
import { createWasmRuntime } from "agent-spreadsheet-wasm"

const backend = new WasmBackend({ bindings: await createWasmRuntime() })
```

## Building from source

```sh
npm run build   # regenerates pkg/ via wasm-pack + wasm-opt
npm test
```

`pkg/` is generated; do not edit it by hand.
