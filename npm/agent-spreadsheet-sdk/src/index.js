const { MCP_CAPABILITIES, WASM_CAPABILITIES, freezeCapabilities } = require("./capabilities")
const {
  SpreadsheetSdkError,
  CapabilityError,
  BackendOperationError,
  normalizeBackendError
} = require("./errors")
const { McpBackend } = require("./mcp-backend")
const { WasmBackend } = require("./wasm-backend")
const { registry: CANONICAL_REGISTRY, OPERATION_NAMES } = require("./backend")
const {
  executeStatelessByteOperation,
  supportsStatelessBytePlan,
  validateStatelessByteRequest
} = require("./stateless-byte-adapter")

module.exports = {
  McpBackend,
  WasmBackend,
  CANONICAL_REGISTRY,
  OPERATION_NAMES,
  MCP_CAPABILITIES,
  WASM_CAPABILITIES,
  freezeCapabilities,
  executeStatelessByteOperation,
  supportsStatelessBytePlan,
  validateStatelessByteRequest,
  SpreadsheetSdkError,
  CapabilityError,
  BackendOperationError,
  normalizeBackendError
}
