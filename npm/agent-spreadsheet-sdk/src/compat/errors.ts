/* Legacy 0.14 error hierarchy. Superseded by the `SpreadsheetError` hierarchy in the
 * package root. Kept byte-compatible for one release. */

export interface SpreadsheetSdkErrorOptions {
  code?: string
  backend?: string
  operation?: string
  capability?: string
  details?: Record<string, unknown>
  cause?: unknown
}

/** @deprecated Use `SpreadsheetError` from `agent-spreadsheet-sdk`. */
export class SpreadsheetSdkError extends Error {
  code: string
  backend: string | undefined
  operation: string | undefined
  capability: string | undefined
  details: Record<string, unknown>

  constructor(message: string, options: SpreadsheetSdkErrorOptions = {}) {
    super(message)
    this.name = "SpreadsheetSdkError"
    this.code = options.code || "SDK_ERROR"
    this.backend = options.backend
    this.operation = options.operation
    this.capability = options.capability
    this.details = options.details || {}
    if (options.cause !== undefined) (this as { cause?: unknown }).cause = options.cause
  }
}

/** @deprecated Use `CapabilityError` from `agent-spreadsheet-sdk`. */
export class CapabilityError extends SpreadsheetSdkError {
  constructor(params: { backend: string; capability: string; method?: string; operation?: string }) {
    super(`${params.backend} backend does not support capability '${params.capability}'`, {
      code: "UNSUPPORTED_CAPABILITY",
      backend: params.backend,
      operation: params.operation || params.method,
      capability: params.capability,
      details: { method: params.method, operation: params.operation }
    })
    this.name = "CapabilityError"
  }
}

/** @deprecated Use `CanonicalOperationError` from `agent-spreadsheet-sdk`. */
export class BackendOperationError extends SpreadsheetSdkError {
  constructor(
    message: string,
    params: { backend: string; operation: string; cause?: unknown; code?: string }
  ) {
    super(message, {
      code: params.code || "BACKEND_OPERATION_FAILED",
      backend: params.backend,
      operation: params.operation,
      cause: params.cause
    })
    this.name = "BackendOperationError"
  }
}

/** @deprecated Use `decodeRejection` from `agent-spreadsheet-sdk`. */
export function normalizeBackendError(
  error: unknown,
  params: { backend: string; operation: string }
): SpreadsheetSdkError {
  if (error instanceof SpreadsheetSdkError) return error

  if (error && typeof error === "object") {
    const candidate = error as { code?: unknown; message?: unknown }
    const code = typeof candidate.code === "string" ? candidate.code : "BACKEND_OPERATION_FAILED"
    const message = typeof candidate.message === "string" ? candidate.message : "backend operation failed"
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
