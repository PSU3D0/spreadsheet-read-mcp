/**
 * Deprecated 0.14 surface, kept for one release.
 *
 * Migration: the 0.14 backends were protocol adapters that made the caller carry
 * resource ids, revisions, and lifecycle. Replace `new WasmBackend({ bindings })` with
 * `createLocalSpreadsheet({ runtime })` and `local.open(bytes)`, which returns a
 * `LocalWorkbook` that owns its session, tracks `resource_id` and `revision_id`, and
 * exposes the generated read surface plus `write`, `recalculate`, `verifyAgainst`,
 * `renderSheet`, `exportBytes`, and `dispose`. Replace `new McpBackend({ transport })`
 * with `connectSpreadsheetServer({ baseUrl })` against the server's canonical `/v1`
 * route, or with a real MCP client if you want the MCP transport; the SDK no longer
 * pretends to be one. Legacy camel-case methods that flattened envelopes to `data` have
 * no replacement: canonical envelopes are returned whole, and
 * `client.canonical.execute(operation, input)` is the typed escape hatch. Every export
 * below is removed in the release after 0.15.
 *
 * @deprecated Use the 0.15 object model exported from `agent-spreadsheet-sdk`.
 */

export {
  BackendOperationError,
  CapabilityError,
  SpreadsheetSdkError,
  normalizeBackendError
} from "./errors.js"
export type { SpreadsheetSdkErrorOptions } from "./errors.js"

export {
  MCP_CAPABILITIES,
  WASM_CAPABILITIES,
  createCapabilities,
  freezeCapabilities
} from "./capabilities.js"
export type { LegacyCapabilities } from "./capabilities.js"

export {
  CANONICAL_METHODS,
  OPERATION_NAMES,
  OPERATION_SET,
  declaredOperations,
  discoveredCanonicalTools,
  discoveredOperations,
  installCanonicalMethods,
  legacyResourceId,
  projectData,
  registry,
  registry as CANONICAL_REGISTRY,
  requireOperation
} from "./backend.js"

export { McpBackend, installLegacyMethods } from "./mcp-backend.js"
export { WasmBackend } from "./wasm-backend.js"

export {
  executeStatelessByteOperation,
  supportsStatelessBytePlan,
  validateStatelessByteRequest
} from "./stateless-byte-adapter.js"
