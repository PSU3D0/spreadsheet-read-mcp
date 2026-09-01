/**
 * Ephemeral byte binding for stateless adapters on the local runtime.
 *
 * The CLI-shaped adapters bind workbook bytes to a session for exactly one canonical
 * call, export when the operation mutates, and dispose in reverse order.
 */

import { CanonicalOperationError, CapabilityError, SpreadsheetError } from "../errors.js"
import type { LocalSpreadsheet, LocalWorkbook } from "../local.js"
import type { OperationName } from "../generated/operations.js"
import { descriptorFor } from "../registry.js"

const RESOURCE_COUNTS: Readonly<Record<string, number>> = Object.freeze({
  none: 0,
  single_read: 1,
  single_mutable: 1,
  two_resource: 2
})

function adapterError(code: string, message: string, path: string): Error {
  return Object.assign(new Error(message), { aspCode: code, aspPath: path })
}

/** Adapter plan fields the ephemeral flow understands. */
export interface AdapterPlan {
  binding_kind: string
  persistence: string
  support_status: string
}

/** Runtime lifecycle facts the plan gate needs. */
export interface EphemeralRuntimeFacts {
  operations: ReadonlySet<string>
  canExport: boolean
  canDispose: boolean
}

/** True when `plan` can run against a runtime with these lifecycle facts. */
export function supportsEphemeralPlan(plan: AdapterPlan | undefined, facts: EphemeralRuntimeFacts): boolean {
  if (!plan || plan.support_status !== "supported") return false
  if (!Object.hasOwn(RESOURCE_COUNTS, plan.binding_kind)) return false
  if (!["none", "export_required"].includes(plan.persistence)) return false
  if (RESOURCE_COUNTS[plan.binding_kind] > 0 && !facts.canDispose) return false
  return plan.persistence !== "export_required" || facts.canExport
}

/** Operation names the ephemeral adapter can run for `adapter` on this runtime. */
export function availableEphemeralOperations(
  adapter: string,
  facts: EphemeralRuntimeFacts
): Set<string> {
  const available = new Set<string>()
  for (const operation of facts.operations) {
    const plan = descriptorFor(operation)?.adapters[adapter] as AdapterPlan | undefined
    if (supportsEphemeralPlan(plan, facts)) available.add(operation)
  }
  return available
}

/** Reject requests the ephemeral flow cannot honour before any allocation. */
export function validateEphemeralRequest(plan: AdapterPlan, params: unknown): void {
  if (!plan || plan.support_status !== "supported" ||
      !Object.hasOwn(RESOURCE_COUNTS, plan.binding_kind) ||
      !["none", "export_required"].includes(plan.persistence)) {
    throw adapterError("CAPABILITY_UNAVAILABLE", "operation is unavailable in this stateless adapter", "adapter")
  }
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    throw adapterError("INVALID_REQUEST", "canonical request payload must be a JSON object", "$")
  }
  if (plan.persistence === "export_required" && (params as { mode?: unknown }).mode === "stage") {
    throw adapterError(
      "CAPABILITY_UNAVAILABLE",
      "stage requires durable orchestration and is unavailable in a stateless adapter",
      "$.mode"
    )
  }
}

function bindResource(params: Record<string, unknown>, key: string, resourceId: string): void {
  if (params[key] && params[key] !== resourceId) {
    throw adapterError(
      "INVALID_REQUEST",
      `payload ${key} does not match the ephemeral bound resource`,
      `$.${key}`
    )
  }
  params[key] = resourceId
}

/** The result of one ephemeral canonical call. */
export interface EphemeralResult {
  response: unknown
  workbookBytes?: Uint8Array
  exitCode: number
}

/** Bind bytes, dispatch one canonical operation, export when required, dispose. */
export async function executeEphemeralOperation(params: {
  local: LocalSpreadsheet
  operation: string
  params: Record<string, unknown>
  plan: AdapterPlan
  workbooks?: Uint8Array[]
}): Promise<EphemeralResult> {
  validateEphemeralRequest(params.plan, params.params)
  const expected = RESOURCE_COUNTS[params.plan.binding_kind]
  const workbooks = params.workbooks ?? []
  if (workbooks.length !== expected) {
    throw adapterError(
      "INVALID_REQUEST",
      `binding kind '${params.plan.binding_kind}' requires ${expected} workbook byte source(s)`,
      "adapter_binding"
    )
  }

  const request: Record<string, unknown> = { ...params.params }
  const opened: LocalWorkbook[] = []
  try {
    for (const bytes of workbooks) opened.push(await params.local.open(bytes))
    if (opened[0]) bindResource(request, "resource_id", opened[0].resourceId)
    if (opened[1]) bindResource(request, "baseline_resource_id", opened[1].resourceId)

    const response = await params.local.canonical.execute(
      params.operation as OperationName,
      request as never
    )
    const status = (response as { data?: { status?: string } })?.data?.status
    const failed = status === "failed" || status === "rolled_back"
    let workbookBytes: Uint8Array | undefined
    if (params.plan.persistence === "export_required" && request["mode"] !== "preview" && !failed) {
      workbookBytes = await opened[0]!.exportBytes()
    }
    return { response, workbookBytes, exitCode: status === "partial" ? 2 : failed ? 1 : 0 }
  } finally {
    for (const workbook of opened.reverse()) {
      try { await workbook.dispose() } catch { /* best effort */ }
    }
  }
}

/** Project any SDK error onto the adapter's canonical envelope shape. */
export function adapterEnvelopeFor(error: unknown): unknown {
  if (error instanceof CanonicalOperationError) return error.envelope
  if (error instanceof CapabilityError) {
    return {
      schema_version: "1",
      error: {
        code: "CAPABILITY_UNAVAILABLE",
        message: error.message,
        ...(error.operation ? { operation: error.operation } : {}),
        path: "adapter"
      }
    }
  }
  if (error instanceof SpreadsheetError) {
    return {
      schema_version: "1",
      error: { code: "OPERATION_FAILED", message: error.message }
    }
  }
  return undefined
}
