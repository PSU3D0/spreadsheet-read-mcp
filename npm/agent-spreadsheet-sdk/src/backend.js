const registry = require("./generated/canonical-registry.json")
const { CapabilityError, SpreadsheetSdkError } = require("./errors")

const OPERATION_NAMES = Object.freeze(registry.operations.map(({ name }) => name))
const OPERATION_SET = new Set(OPERATION_NAMES)

function canonicalMethodName(operation) {
  return operation.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase())
}

const CANONICAL_METHODS = Object.freeze(Object.fromEntries(
  OPERATION_NAMES.map((operation) => [canonicalMethodName(operation), operation])
))

function declaredOperations(value, fallback = []) {
  const operations = Array.isArray(value)
    ? value
    : value?.operations || value?.supportedOperations || fallback
  if (!Array.isArray(operations)) {
    throw new SpreadsheetSdkError("capabilities.operations must be an array", {
      code: "INVALID_ARGUMENT"
    })
  }
  return [...new Set(operations.filter((operation) => OPERATION_SET.has(operation)))]
}

function requireOperation(backend, operation, method = "execute") {
  if (!backend._operationSet.has(operation)) {
    throw new CapabilityError({
      backend: backend.kind,
      capability: operation,
      method,
      operation
    })
  }
}

function installCanonicalMethods(Backend) {
  for (const [method, operation] of Object.entries(CANONICAL_METHODS)) {
    if (Object.prototype.hasOwnProperty.call(Backend.prototype, method)) continue
    Object.defineProperty(Backend.prototype, method, {
      configurable: true,
      value(input = {}) {
        return this.execute(operation, input)
      }
    })
  }
}

function projectData(response) {
  return response && typeof response === "object" && "data" in response
    ? response.data
    : response
}

function prefixedResourceId(value, prefix) {
  if (typeof value !== "string" || value.length === 0) {
    throw new SpreadsheetSdkError("missing resource identity", { code: "INVALID_ARGUMENT" })
  }
  return /^(wb|fork|session):/.test(value) ? value : `${prefix}:${value}`
}

function legacyResourceId(kind, input, mutation = false) {
  if (input.resource_id) return input.resource_id
  if (kind === "wasm") {
    return prefixedResourceId(input.sessionId || input.session_id || input.contextId, "session")
  }
  const value = mutation
    ? input.forkId || input.fork_id || input.workbookId || input.workbook_id || input.contextId
    : input.workbookId || input.workbook_id || input.contextId
  return prefixedResourceId(value, mutation ? "fork" : "wb")
}

module.exports = {
  registry,
  OPERATION_NAMES,
  OPERATION_SET,
  CANONICAL_METHODS,
  declaredOperations,
  requireOperation,
  installCanonicalMethods,
  projectData,
  legacyResourceId
}
