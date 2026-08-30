const { decodeBytesToUtf8, defineCommand } = require("just-bash")
const { WasmBackend } = require("./wasm-backend")
const {
  executeStatelessByteOperation,
  supportsStatelessBytePlan,
  validateStatelessByteRequest
} = require("./stateless-byte-adapter")
const {
  assertLimit,
  canonicalError,
  descriptorFor,
  discover,
  errorEnvelope,
  invalid,
  jsonResult,
  parseOperationArgs,
  utf8Bytes
} = require("./just-bash-parser")
const { createVfsWriter, readWorkbook, resolveVfsPath } = require("./just-bash-vfs")

const DEFAULT_MAX_WORKBOOK_BYTES = 64 * 1024 * 1024
const DEFAULT_MAX_PARAMS_BYTES = 1024 * 1024

function createAspCommand({
  bindings,
  maxWorkbookBytes = DEFAULT_MAX_WORKBOOK_BYTES,
  maxParamsBytes = DEFAULT_MAX_PARAMS_BYTES
} = {}) {
  assertLimit(maxWorkbookBytes, "maxWorkbookBytes")
  assertLimit(maxParamsBytes, "maxParamsBytes")
  const backend = new WasmBackend({ bindings })
  const capabilities = backend.getCapabilities()
  const availableOperations = new Set(capabilities.operations.filter((operation) => {
    const descriptor = descriptorFor(operation)
    return descriptor && supportsStatelessBytePlan(descriptor.adapters.just_bash, capabilities)
  }))
  const { atomicWrite } = createVfsWriter()

  return defineCommand("asp", async (args, ctx) => {
    let operation
    try {
      const discovery = discover(args, availableOperations)
      if (discovery !== null) return jsonResult(discovery)

      const options = parseOperationArgs(args)
      operation = options.operation
      const descriptor = descriptorFor(operation)
      if (!descriptor) throw canonicalError(
        "UNKNOWN_OPERATION", `unknown operation '${operation}'`, operation, "$.operation"
      )
      const plan = descriptor.adapters.just_bash
      if (plan.support_status !== "supported" || !availableOperations.has(operation)) {
        throw canonicalError(
          "CAPABILITY_UNAVAILABLE",
          `canonical operation '${operation}' is unavailable in the just-bash adapter`,
          operation,
          "adapter"
        )
      }

      let payloadText
      try {
        payloadText = options.json === undefined
          ? decodeBytesToUtf8(ctx.stdin, maxParamsBytes)
          : options.json
      } catch (_) {
        throw canonicalError(
          "INVALID_REQUEST", `params JSON exceeds the ${maxParamsBytes}-byte adapter limit`,
          operation, "$.params"
        )
      }
      if (utf8Bytes(payloadText) > maxParamsBytes) throw canonicalError(
        "INVALID_REQUEST", `params JSON exceeds the ${maxParamsBytes}-byte adapter limit`,
        operation, "$.params"
      )
      let params
      try { params = JSON.parse(payloadText) } catch (error) {
        throw canonicalError("INVALID_REQUEST", error.message, operation, "$")
      }
      validateStatelessByteRequest(plan, params)

      const needsBind = plan.binding_kind !== "none"
      const twoResource = plan.binding_kind === "two_resource"
      if (needsBind !== Boolean(options.bind)) invalid(
        needsBind ? `canonical operation '${operation}' requires --bind <VFS_PATH>`
          : `canonical operation '${operation}' does not accept --bind`,
        operation, "--bind"
      )
      if (twoResource !== Boolean(options.baseline)) invalid(
        twoResource ? `canonical operation '${operation}' requires --baseline <VFS_PATH>`
          : "--baseline is only accepted by two-resource operations",
        operation, "--baseline"
      )

      const exports = plan.persistence === "export_required"
      const preview = params.mode === "preview"
      if (!exports && (options.output || options.inPlace)) invalid(
        "--output and --in-place require a bound workbook-write operation",
        operation, "adapter_flags"
      )
      if (exports && preview && (options.output || options.inPlace)) invalid(
        "preview does not accept file export flags", operation, "adapter_flags"
      )
      if (exports && !preview && Boolean(options.output) === Boolean(options.inPlace)) invalid(
        "a mutating operation requires exactly one of --output <VFS_PATH> or --in-place",
        operation, "adapter_flags"
      )
      if (options.output && resolveVfsPath(ctx, options.output) === resolveVfsPath(ctx, options.bind)) {
        invalid("--output must differ from --bind; use --in-place", operation, "--output")
      }

      const sources = []
      if (options.bind) sources.push(readWorkbook(ctx, options.bind, maxWorkbookBytes, "--bind"))
      if (options.baseline) {
        sources.push(readWorkbook(ctx, options.baseline, maxWorkbookBytes, "--baseline"))
      }
      const result = await executeStatelessByteOperation({
        backend,
        operation,
        params,
        plan,
        workbooks: await Promise.all(sources)
      })
      if (result.workbookBytes) {
        await atomicWrite(
          ctx,
          options.inPlace ? options.bind : options.output,
          result.workbookBytes,
          options.inPlace
        )
      }
      return jsonResult(result.response, result.exitCode)
    } catch (error) {
      return jsonResult(errorEnvelope(error, operation), 1, true)
    }
  })
}

module.exports = {
  createAspCommand,
  DEFAULT_MAX_WORKBOOK_BYTES,
  DEFAULT_MAX_PARAMS_BYTES
}
