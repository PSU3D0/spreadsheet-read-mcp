const { createCapabilities } = require("./capabilities")
const {
  discoveredOperations,
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
    if (canExecute && typeof params.bindings.operations !== "function") {
      throw new SpreadsheetSdkError("WASM bindings with executeOperation must provide operations()", {
        code: "INVALID_ARGUMENT",
        backend: "wasm"
      })
    }
    const advertised = canExecute ? params.bindings.operations() : []
    if (advertised && typeof advertised.then === "function") {
      throw new SpreadsheetSdkError("WASM bindings operations() must be synchronous", {
        code: "INVALID_RESPONSE",
        backend: "wasm"
      })
    }
    const operations = canExecute ? discoveredOperations(advertised) : []
    this._operationSet = new Set(operations)
    this._capabilities = createCapabilities(this.kind, operations, {
      transport: "wasm-json",
      initialized: canExecute,
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
    try {
      const result = await this._bindings.executeOperation(
        resourceId,
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
    const resourceId = await this._bindings.createSession(bytes)
    return { resource_id: resourceId }
  }

  async createSession(input = {}) {
    const bound = await this.bindWorkbook(input)
    return bound.resource_id
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
    return this._bindings.exportWorkbook(resourceId)
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
    return this._bindings.disposeSession(resourceId)
  }
}

installCanonicalMethods(WasmBackend)
installLegacyMethods(WasmBackend)

module.exports = { WasmBackend }
