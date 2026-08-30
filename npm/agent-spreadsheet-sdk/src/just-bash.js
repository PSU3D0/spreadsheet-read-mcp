const { decodeBytesToUtf8, defineCommand } = require("just-bash")
const { WasmBackend } = require("./wasm-backend")
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
const { atomicWrite, readWorkbook, resolveVfsPath } = require("./just-bash-vfs")

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

  return defineCommand("asp", async (args, ctx) => {
    let operation
    const sessions = []
    try {
      const discovery = discover(args)
      if (discovery !== null) return jsonResult(discovery)

      const options = parseOperationArgs(args)
      operation = options.operation
      const descriptor = descriptorFor(operation)
      if (!descriptor) throw canonicalError(
        "UNKNOWN_OPERATION", `unknown operation '${operation}'`, operation, "$.operation"
      )
      const adapter = descriptor.adapters.wasm
      if (adapter.support_status !== "supported") throw canonicalError(
        "CAPABILITY_UNAVAILABLE",
        `canonical operation '${operation}' is unavailable in the WASM adapter`,
        operation,
        "adapter"
      )

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
      if (!params || typeof params !== "object" || Array.isArray(params)) {
        throw canonicalError("INVALID_REQUEST", "canonical request payload must be a JSON object", operation, "$")
      }

      const needsBind = adapter.binding_kind !== "none"
      const twoResource = adapter.binding_kind === "two_resource"
      if (needsBind !== Boolean(options.bind)) invalid(
        needsBind ? `canonical operation '${operation}' requires --bind <VFS_PATH>`
          : `canonical operation '${operation}' does not accept --bind`,
        operation,
        "--bind"
      )
      if (twoResource !== Boolean(options.baseline)) invalid(
        twoResource ? `canonical operation '${operation}' requires --baseline <VFS_PATH>`
          : "--baseline is only accepted by two-resource operations",
        operation,
        "--baseline"
      )

      const mutable = descriptor.adapters.cli.persistence === "export_required"
      const preview = params.mode === "preview"
      if (!mutable && (options.output || options.inPlace)) invalid(
        "--output and --in-place require a bound workbook-write operation", operation, "adapter_flags"
      )
      if (mutable && params.mode === "stage") throw canonicalError(
        "CAPABILITY_UNAVAILABLE",
        "stage requires durable orchestration and is unavailable in the just-bash adapter",
        operation,
        "$.mode"
      )
      if (mutable && preview && (options.output || options.inPlace)) invalid(
        "preview does not accept file export flags", operation, "adapter_flags"
      )
      if (mutable && !preview && Boolean(options.output) === Boolean(options.inPlace)) invalid(
        "a mutating operation requires exactly one of --output <VFS_PATH> or --in-place",
        operation,
        "adapter_flags"
      )
      if (options.output && resolveVfsPath(ctx, options.output) === resolveVfsPath(ctx, options.bind)) {
        invalid("--output must differ from --bind; use --in-place", operation, "--output")
      }

      const reads = []
      if (options.bind) reads.push(readWorkbook(ctx, options.bind, maxWorkbookBytes, operation, "--bind"))
      if (options.baseline) reads.push(readWorkbook(
        ctx, options.baseline, maxWorkbookBytes, operation, "--baseline"
      ))
      const workbooks = await Promise.all(reads)
      for (const workbook of workbooks) {
        const resourceId = await backend.createSession({ workbookBytes: workbook.bytes })
        sessions.push(resourceId)
      }
      if (sessions[0]) {
        if (params.resource_id && params.resource_id !== sessions[0]) invalid(
          "payload resource_id does not match the ephemeral --bind resource", operation, "$.resource_id"
        )
        params.resource_id = sessions[0]
      }
      if (sessions[1]) {
        if (params.baseline_resource_id && params.baseline_resource_id !== sessions[1]) invalid(
          "payload baseline_resource_id does not match the ephemeral --baseline resource",
          operation,
          "$.baseline_resource_id"
        )
        params.baseline_resource_id = sessions[1]
      }

      const response = await backend.execute(operation, params)
      const status = response?.data?.status
      const exportResult = mutable && !preview && !["failed", "rolled_back"].includes(status)
      if (exportResult) {
        const bytes = await backend.exportWorkbook({ resource_id: sessions[0] })
        await atomicWrite(ctx, options.inPlace ? options.bind : options.output, bytes, options.inPlace)
      }
      return jsonResult(response, status === "partial" ? 2 : ["failed", "rolled_back"].includes(status) ? 1 : 0)
    } catch (error) {
      return jsonResult(errorEnvelope(error, operation), 1, true)
    } finally {
      for (const resourceId of sessions.reverse()) {
        try { await backend.disposeSession({ resource_id: resourceId }) } catch (_) {}
      }
    }
  })
}

module.exports = {
  createAspCommand,
  DEFAULT_MAX_WORKBOOK_BYTES,
  DEFAULT_MAX_PARAMS_BYTES
}
