import { decodeBytesToUtf8, defineCommand } from "just-bash"

import { createLocalSpreadsheet, type WasmBindings } from "../local.js"
import {
  adapterEnvelopeFor,
  availableEphemeralOperations,
  executeEphemeralOperation,
  validateEphemeralRequest
} from "./ephemeral.js"
import {
  assertLimit,
  canonicalError,
  descriptorFor,
  discover,
  errorEnvelope,
  invalid,
  jsonResult,
  parseOperationArgs,
  utf8Bytes
} from "./parser.js"
import { createVfsWriter, readWorkbook, resolveVfsPath } from "./vfs.js"

const DEFAULT_MAX_WORKBOOK_BYTES = 64 * 1024 * 1024
const DEFAULT_MAX_PARAMS_BYTES = 1024 * 1024

/** Options for {@link createAspCommand}. */
export interface AspCommandOptions {
  bindings: WasmBindings
  maxWorkbookBytes?: number
  maxParamsBytes?: number
}

/**
 * Register the `asp` just-bash custom command on the local runtime.
 *
 * Surface: `asp op <operation> [--bind PATH] [--baseline PATH] [--json JSON]
 * [--output PATH|--in-place]`, plus `asp operations`, `asp schema <op>`, and
 * `asp example <op>`. It binds bytes through `ctx.fs` and carries no operation taxonomy.
 */
export function createAspCommand(options: AspCommandOptions): unknown {
  const { bindings } = options ?? ({} as AspCommandOptions)
  const maxWorkbookBytes = options?.maxWorkbookBytes ?? DEFAULT_MAX_WORKBOOK_BYTES
  const maxParamsBytes = options?.maxParamsBytes ?? DEFAULT_MAX_PARAMS_BYTES
  assertLimit(maxWorkbookBytes, "maxWorkbookBytes")
  assertLimit(maxParamsBytes, "maxParamsBytes")

  const local = createLocalSpreadsheet({ runtime: bindings })
  const { atomicWrite } = createVfsWriter()
  let availability: Promise<Set<string>> | undefined

  function available(): Promise<Set<string>> {
    if (!availability) {
      availability = local.capabilities().then((operations) => availableEphemeralOperations("just_bash", {
        operations: new Set(operations),
        canExport: typeof bindings?.exportWorkbook === "function",
        canDispose: typeof bindings?.disposeSession === "function"
      }))
    }
    return availability
  }

  return defineCommand("asp", async (args: string[], ctx: any) => {
    let operation: string | undefined
    try {
      const availableOperations = await available()
      const discovery = discover(args, availableOperations)
      if (discovery !== null) return jsonResult(discovery)

      const opts = parseOperationArgs(args)
      operation = opts.operation
      const descriptor = descriptorFor(operation!)
      if (!descriptor) throw canonicalError(
        "UNKNOWN_OPERATION", `unknown operation '${operation}'`, operation, "$.operation"
      )
      const plan = descriptor.adapters.just_bash
      if (plan.support_status !== "supported" || !availableOperations.has(operation!)) {
        throw canonicalError(
          "CAPABILITY_UNAVAILABLE",
          `canonical operation '${operation}' is unavailable in the just-bash adapter`,
          operation,
          "adapter"
        )
      }

      let payloadText: string
      try {
        payloadText = opts.json === undefined
          ? decodeBytesToUtf8(ctx.stdin, maxParamsBytes)
          : opts.json
      } catch {
        throw canonicalError(
          "INVALID_REQUEST", `params JSON exceeds the ${maxParamsBytes}-byte adapter limit`,
          operation, "$.params"
        )
      }
      if (utf8Bytes(payloadText) > maxParamsBytes) throw canonicalError(
        "INVALID_REQUEST", `params JSON exceeds the ${maxParamsBytes}-byte adapter limit`,
        operation, "$.params"
      )
      let params: any
      try { params = JSON.parse(payloadText) } catch (error: any) {
        throw canonicalError("INVALID_REQUEST", error.message, operation, "$")
      }
      validateEphemeralRequest(plan, params)

      const needsBind = plan.binding_kind !== "none"
      const twoResource = plan.binding_kind === "two_resource"
      if (needsBind !== Boolean(opts.bind)) invalid(
        needsBind ? `canonical operation '${operation}' requires --bind <VFS_PATH>`
          : `canonical operation '${operation}' does not accept --bind`,
        operation, "--bind"
      )
      if (twoResource !== Boolean(opts.baseline)) invalid(
        twoResource ? `canonical operation '${operation}' requires --baseline <VFS_PATH>`
          : "--baseline is only accepted by two-resource operations",
        operation, "--baseline"
      )

      const exports = plan.persistence === "export_required"
      const preview = params.mode === "preview"
      if (!exports && (opts.output || opts.inPlace)) invalid(
        "--output and --in-place require a bound workbook-write operation",
        operation, "adapter_flags"
      )
      if (exports && preview && (opts.output || opts.inPlace)) invalid(
        "preview does not accept file export flags", operation, "adapter_flags"
      )
      if (exports && !preview && Boolean(opts.output) === Boolean(opts.inPlace)) invalid(
        "a mutating operation requires exactly one of --output <VFS_PATH> or --in-place",
        operation, "adapter_flags"
      )
      if (opts.output && resolveVfsPath(ctx, opts.output) === resolveVfsPath(ctx, opts.bind)) {
        invalid("--output must differ from --bind; use --in-place", operation, "--output")
      }

      const sources = []
      if (opts.bind) sources.push(readWorkbook(ctx, opts.bind, maxWorkbookBytes, "--bind"))
      if (opts.baseline) {
        sources.push(readWorkbook(ctx, opts.baseline, maxWorkbookBytes, "--baseline"))
      }
      const result = await executeEphemeralOperation({
        local,
        operation: operation!,
        params,
        plan,
        workbooks: await Promise.all(sources)
      })
      if (result.workbookBytes) {
        await atomicWrite(
          ctx,
          opts.inPlace ? opts.bind : opts.output,
          result.workbookBytes,
          Boolean(opts.inPlace)
        )
      }
      return jsonResult(result.response, result.exitCode)
    } catch (error: any) {
      return jsonResult(errorEnvelope(adapterEnvelopeFor(error) ?? error, operation), 1, true)
    }
  })
}

export { DEFAULT_MAX_PARAMS_BYTES, DEFAULT_MAX_WORKBOOK_BYTES }
