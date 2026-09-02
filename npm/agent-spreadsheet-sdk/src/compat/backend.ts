/* Legacy 0.14 backend plumbing: registry lookup, discovery normalization, and the
 * `defineProperty` method layer. Superseded by the generated read surface. */

import { canonicalRegistry } from "../generated/registry-data.js"
import { CapabilityError, SpreadsheetSdkError } from "./errors.js"

/** @deprecated Use `canonicalRegistry` from `agent-spreadsheet-sdk`. */
export const registry = canonicalRegistry as unknown as Record<string, any>

/** @deprecated Use `OPERATION_NAMES` from `agent-spreadsheet-sdk`. */
export const OPERATION_NAMES: readonly string[] = Object.freeze(
  canonicalRegistry.operations.map(({ name }) => name)
)

const OPERATION_SET_INTERNAL = new Set<string>(OPERATION_NAMES)

/** @deprecated Use `isOperationName` from `agent-spreadsheet-sdk`. */
export const OPERATION_SET: ReadonlySet<string> = OPERATION_SET_INTERNAL

const CANONICAL_TOOL_META_KEY = "agent-spreadsheet/canonical"
const CANONICAL_SCHEMA_VERSION = canonicalRegistry.schema_version

function canonicalMethodName(operation: string): string {
  return operation.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase())
}

/** @deprecated The 0.14 camel-case method map. */
export const CANONICAL_METHODS: Readonly<Record<string, string>> = Object.freeze(
  Object.fromEntries(OPERATION_NAMES.map((operation) => [canonicalMethodName(operation), operation]))
)

/** @deprecated Legacy declared-operation normalization. */
export function declaredOperations(value: any, fallback: string[] = []): string[] {
  const operations = Array.isArray(value)
    ? value
    : value?.operations || value?.supportedOperations || fallback
  if (!Array.isArray(operations)) {
    throw new SpreadsheetSdkError("capabilities.operations must be an array", {
      code: "INVALID_ARGUMENT"
    })
  }
  return [...new Set(operations.filter((operation: string) => OPERATION_SET_INTERNAL.has(operation)))]
}

function discoveryEntries(value: any): any[] {
  let entries = value
  if (typeof entries === "string") {
    try {
      entries = JSON.parse(entries)
    } catch (cause) {
      throw new SpreadsheetSdkError("operation discovery returned invalid JSON", {
        code: "INVALID_RESPONSE",
        cause
      })
    }
  }
  entries = entries?.result?.operations || entries?.result?.tools ||
    entries?.operations || entries?.tools || entries
  if (!Array.isArray(entries)) {
    throw new SpreadsheetSdkError(
      "operation discovery must return an array of names or descriptors",
      { code: "INVALID_RESPONSE" }
    )
  }
  return entries
}

// Explicit operation registries and WASM operations() are trusted canonical API
// contracts. Generic MCP tools/list descriptors are not: compatibility routers
// intentionally reuse several canonical names with legacy schemas and outputs.
/** @deprecated Legacy discovery normalization. */
export function discoveredOperations(value: any): string[] {
  const entries = discoveryEntries(value)
  const names = entries.map((entry) => (typeof entry === "string" ? entry : entry?.name))
  if (names.some((name) => typeof name !== "string")) {
    throw new SpreadsheetSdkError("operation discovery contains a descriptor without a name", {
      code: "INVALID_RESPONSE"
    })
  }
  return [...new Set(names.filter((name: string) => OPERATION_SET_INTERNAL.has(name)))]
}

/** @deprecated Legacy `tools/list` canonical marker filter. */
export function discoveredCanonicalTools(value: any): string[] {
  return [...new Set(discoveryEntries(value)
    .filter((entry) => {
      if (!entry || typeof entry !== "object" || typeof entry.name !== "string") return false
      const marker = entry._meta?.[CANONICAL_TOOL_META_KEY] ?? entry.meta?.[CANONICAL_TOOL_META_KEY]
      return marker?.schema_version === CANONICAL_SCHEMA_VERSION &&
        marker?.operation === entry.name && OPERATION_SET_INTERNAL.has(entry.name)
    })
    .map(({ name }) => name as string))]
}

/** @deprecated Legacy capability gate. */
export function requireOperation(backend: any, operation: string, method = "execute"): void {
  if (!backend._operationSet.has(operation)) {
    throw new CapabilityError({
      backend: backend.kind,
      capability: operation,
      method,
      operation
    })
  }
}

/** @deprecated The `defineProperty` method layer the 0.15 generated surface replaces. */
export function installCanonicalMethods(Backend: any): void {
  for (const [method, operation] of Object.entries(CANONICAL_METHODS)) {
    if (Object.prototype.hasOwnProperty.call(Backend.prototype, method)) continue
    Object.defineProperty(Backend.prototype, method, {
      configurable: true,
      value(this: any, input: any = {}) {
        return this.execute(operation, input)
      }
    })
  }
}

/** @deprecated Legacy envelope flattening. */
export function projectData(response: any): any {
  return response && typeof response === "object" && "data" in response ? response.data : response
}

function prefixedResourceId(value: any, prefix: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new SpreadsheetSdkError("missing resource identity", { code: "INVALID_ARGUMENT" })
  }
  return /^(wb|fork|session):/.test(value) ? value : `${prefix}:${value}`
}

/** @deprecated Legacy resource-id inference. */
export function legacyResourceId(kind: string, input: any, mutation = false): string {
  if (input.resource_id) return input.resource_id
  if (kind === "wasm") {
    const session = input.sessionId || input.session_id || input.contextId || input.context_id
    if (session) return prefixedResourceId(session, "session")
    const legacy = mutation
      ? input.forkId || input.fork_id || input.workbookId || input.workbook_id
      : input.workbookId || input.workbook_id
    return prefixedResourceId(legacy, mutation ? "fork" : "session")
  }
  const value = mutation
    ? input.forkId || input.fork_id || input.workbookId || input.workbook_id || input.contextId
    : input.workbookId || input.workbook_id || input.contextId
  return prefixedResourceId(value, mutation ? "fork" : "wb")
}
