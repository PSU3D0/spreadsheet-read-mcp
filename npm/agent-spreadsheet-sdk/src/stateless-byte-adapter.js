const RESOURCE_COUNTS = Object.freeze({
  none: 0,
  single_read: 1,
  single_mutable: 1,
  two_resource: 2
})

function adapterError(code, message, path) {
  return Object.assign(new Error(message), { aspCode: code, aspPath: path })
}

function assertPlan(plan) {
  if (!plan || plan.support_status !== "supported") {
    throw adapterError("CAPABILITY_UNAVAILABLE", "operation is unavailable in this stateless adapter", "adapter")
  }
  if (!Object.hasOwn(RESOURCE_COUNTS, plan.binding_kind)) {
    throw adapterError(
      "CAPABILITY_UNAVAILABLE",
      `binding kind '${plan.binding_kind}' requires durable orchestration`,
      "adapter"
    )
  }
  if (!["none", "export_required"].includes(plan.persistence)) {
    throw adapterError(
      "CAPABILITY_UNAVAILABLE",
      `persistence '${plan.persistence}' requires durable orchestration`,
      "adapter"
    )
  }
}

function supportsStatelessBytePlan(plan, capabilities) {
  if (!plan || plan.support_status !== "supported" ||
      !Object.hasOwn(RESOURCE_COUNTS, plan.binding_kind) ||
      !["none", "export_required"].includes(plan.persistence)) return false
  if (RESOURCE_COUNTS[plan.binding_kind] > 0 &&
      (!capabilities?.resourceBinding || !capabilities?.supportsSessionLifecycle)) return false
  return plan.persistence !== "export_required" || Boolean(capabilities?.resourceExport)
}

function bindResource(params, key, resourceId) {
  if (params[key] && params[key] !== resourceId) {
    throw adapterError(
      "INVALID_REQUEST",
      `payload ${key} does not match the ephemeral bound resource`,
      `$.${key}`
    )
  }
  params[key] = resourceId
}

function validateStatelessByteRequest(plan, params) {
  assertPlan(plan)
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    throw adapterError("INVALID_REQUEST", "canonical request payload must be a JSON object", "$")
  }
  if (plan.persistence === "export_required" && params.mode === "stage") {
    throw adapterError(
      "CAPABILITY_UNAVAILABLE",
      "stage requires durable orchestration and is unavailable in a stateless adapter",
      "$.mode"
    )
  }
}

async function executeStatelessByteOperation({ backend, operation, params, plan, workbooks = [] }) {
  validateStatelessByteRequest(plan, params)
  const expected = RESOURCE_COUNTS[plan.binding_kind]
  if (!Array.isArray(workbooks) || workbooks.length !== expected) {
    throw adapterError(
      "INVALID_REQUEST",
      `binding kind '${plan.binding_kind}' requires ${expected} workbook byte source(s)`,
      "adapter_binding"
    )
  }

  const request = { ...params }
  const sessions = []
  try {
    for (const workbookBytes of workbooks) {
      sessions.push(await backend.createSession({ workbookBytes }))
    }
    if (sessions[0]) bindResource(request, "resource_id", sessions[0])
    if (sessions[1]) bindResource(request, "baseline_resource_id", sessions[1])

    const response = await backend.execute(operation, request)
    const status = response?.data?.status
    const failed = ["failed", "rolled_back"].includes(status)
    let workbookBytes
    if (plan.persistence === "export_required" && params.mode !== "preview" && !failed) {
      workbookBytes = await backend.exportWorkbook({ resource_id: sessions[0] })
    }
    return {
      response,
      workbookBytes,
      exitCode: status === "partial" ? 2 : failed ? 1 : 0
    }
  } finally {
    for (const resourceId of sessions.reverse()) {
      try { await backend.disposeSession({ resource_id: resourceId }) } catch (_) {}
    }
  }
}

module.exports = {
  executeStatelessByteOperation,
  supportsStatelessBytePlan,
  validateStatelessByteRequest
}
