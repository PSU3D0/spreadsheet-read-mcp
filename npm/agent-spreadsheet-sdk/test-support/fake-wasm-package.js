// Stands in for the `agent-spreadsheet-wasm` package while running README examples.

const { createFakeBindings } = require("./fake-bindings.js")

function createWasmRuntime(_options = {}) {
  return createFakeBindings().bindings
}

module.exports = { createWasmRuntime }
