class SpreadsheetSdkError extends Error {
  /**
   * @param {string} message
   * @param {{
   *   code?: string,
   *   backend?: string,
   *   operation?: string,
   *   capability?: string,
   *   details?: Record<string, unknown>,
   *   cause?: unknown
   * }} [options]
   */
  constructor(message, options = {}) {
    super(message)
    this.name = "SpreadsheetSdkError"
    this.code = options.code || "SDK_ERROR"
    this.backend = options.backend
    this.operation = options.operation
    this.capability = options.capability
    this.details = options.details || {}

    if (options.cause !== undefined) {
      this.cause = options.cause
    }
  }
}

class CapabilityError extends SpreadsheetSdkError {
  /**
   * @param {{ backend: string, capability: string, method?: string }} params
   */
  constructor(params) {
    super(
      `${params.backend} backend does not support capability '${params.capability}'`,
      {
        code: "UNSUPPORTED_CAPABILITY",
        backend: params.backend,
        operation: params.method,
        capability: params.capability,
        details: { method: params.method }
      }
    )
    this.name = "CapabilityError"
  }
}

class BackendOperationError extends SpreadsheetSdkError {
  /**
   * @param {string} message
   * @param {{ backend: string, operation: string, cause?: unknown, code?: string }} params
   */
  constructor(message, params) {
    super(message, {
      code: params.code || "BACKEND_OPERATION_FAILED",
      backend: params.backend,
      operation: params.operation,
      cause: params.cause
    })
    this.name = "BackendOperationError"
  }
}

/**
 * @param {unknown} error
 * @param {{ backend: string, operation: string }} params
 */
function normalizeBackendError(error, params) {
  if (error instanceof SpreadsheetSdkError) {
    return error
  }

  if (error && typeof error === "object") {
    const code = typeof error.code === "string" ? error.code : "BACKEND_OPERATION_FAILED"
    const message = typeof error.message === "string" ? error.message : "backend operation failed"
    return new BackendOperationError(message, {
      code,
      backend: params.backend,
      operation: params.operation,
      cause: error
    })
  }

  if (error instanceof Error) {
    return new BackendOperationError(error.message, {
      backend: params.backend,
      operation: params.operation,
      cause: error
    })
  }

  return new BackendOperationError("backend operation failed", {
    backend: params.backend,
    operation: params.operation,
    cause: error
  })
}

module.exports = {
  SpreadsheetSdkError,
  CapabilityError,
  BackendOperationError,
  normalizeBackendError
}
