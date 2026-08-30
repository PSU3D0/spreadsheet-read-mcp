const { createCapabilities } = require("./capabilities")
const {
  declaredOperations,
  requireOperation,
  installCanonicalMethods
} = require("./backend")
const { installLegacyMethods } = require("./mcp-backend")
const { CapabilityError, SpreadsheetSdkError } = require("./errors")

function parseBindingResult(value, operation) {
  if (typeof value !== "string") return value
  try {
    return JSON.parse(value)
  } catch (cause) {
    throw new SpreadsheetSdkError("WASM canonical dispatcher returned invalid JSON", {
      code: "INVALID_RESPONSE",
      backend: "wasm",
      operation,
      cause
    })
  }
}

class WasmBackend {
  constructor(params) {
    if (!params || !params.bindings || typeof params.bindings !== "object") {
      throw new SpreadsheetSdkError("WasmBackend requires bindings object", {
        code: "INVALID_ARGUMENT",
        backend: "wasm"
      })
    }
    this.kind = "wasm"
    this._bindings = params.bindings
    const canExecute = typeof params.bindings.executeOperation === "function"
    const advertised = params.operations || params.capabilities || params.bindings
    const operations = canExecute ? declaredOperations(advertised, []) : []
    this._operationSet = new Set(operations)
    this._capabilities = createCapabilities(this.kind, operations, {
      transport: "wasm-json",
      resourceBinding: typeof params.bindings.createSession === "function",
      resourceExport: typeof params.bindings.exportWorkbook === "function",
      sessionLifecycle: typeof params.bindings.createSession === "function" &&
        typeof params.bindings.disposeSession === "function"
    })
  }

  getCapabilities() {
    return this._capabilities
  }

  async execute(operation, input = {}) {
    requireOperation(this, operation)
    if (!input || typeof input !== "object" || Array.isArray(input)) {
      throw new SpreadsheetSdkError("canonical operation input must be an object", {
        code: "INVALID_ARGUMENT",
        backend: this.kind,
        operation
      })
    }
    const resourceId = input.resource_id
    const sessionId = typeof resourceId === "string" && resourceId.startsWith("session:")
      ? resourceId.slice("session:".length)
      : resourceId
    try {
      const result = await this._bindings.executeOperation(
        sessionId,
        operation,
        JSON.stringify(input)
      )
      return parseBindingResult(result, operation)
    } catch (error) {
      if (typeof error !== "string") throw error
      let decoded
      try {
        decoded = JSON.parse(error)
      } catch (_) {
        throw error
      }
      throw decoded
    }
  }

  async bindWorkbook(input = {}) {
    if (typeof this._bindings.createSession !== "function") {
      throw new CapabilityError({
        backend: this.kind,
        capability: "resource_binding",
        method: "bindWorkbook"
      })
    }
    const bytes = input.workbookBytes || input.bytes || input
    if (!bytes) {
      throw new SpreadsheetSdkError("missing required field 'workbookBytes'", {
        code: "INVALID_ARGUMENT",
        backend: this.kind,
        operation: "bindWorkbook"
      })
    }
    const sessionId = await this._bindings.createSession(bytes)
    return { resource_id: String(sessionId).startsWith("session:") ? sessionId : `session:${sessionId}` }
  }

  async createSession(input = {}) {
    const bound = await this.bindWorkbook(input)
    return bound.resource_id.slice("session:".length)
  }

  async exportWorkbook(input = {}) {
    if (typeof this._bindings.exportWorkbook !== "function") {
      throw new CapabilityError({
        backend: this.kind,
        capability: "resource_export",
        method: "exportWorkbook"
      })
    }
    const resourceId = input.resource_id || input.sessionId || input.session_id || input.contextId
    if (typeof resourceId !== "string" || resourceId.length === 0) {
      throw new SpreadsheetSdkError("missing resource identity", {
        code: "INVALID_ARGUMENT",
        backend: this.kind,
        operation: "exportWorkbook"
      })
    }
    return this._bindings.exportWorkbook(resourceId.replace(/^session:/, ""))
  }

  async disposeSession(input = {}) {
    if (typeof this._bindings.disposeSession !== "function") {
      throw new CapabilityError({
        backend: this.kind,
        capability: "resource_binding",
        method: "disposeSession"
      })
    }
    const resourceId = input.resource_id || input.sessionId || input.session_id || input.contextId
    if (typeof resourceId !== "string" || resourceId.length === 0) {
      throw new SpreadsheetSdkError("missing resource identity", {
        code: "INVALID_ARGUMENT",
        backend: this.kind,
        operation: "disposeSession"
      })
    }
    return this._bindings.disposeSession(resourceId.replace(/^session:/, ""))
  }
}

installCanonicalMethods(WasmBackend)
installLegacyMethods(WasmBackend)

module.exports = { WasmBackend }
