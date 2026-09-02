const assert = require("node:assert/strict")
const test = require("node:test")

const {
  executeStatelessByteOperation,
  validateStatelessByteRequest
} = require("agent-spreadsheet-sdk/compat")

function backend(status = "applied") {
  const state = { created: [], disposed: [], request: null }
  return {
    state,
    adapter: {
      async createSession({ workbookBytes }) {
        const id = `resource:${state.created.length + 1}`
        state.created.push({ id, workbookBytes })
        return id
      },
      async execute(operation, params) {
        state.request = { operation, params }
        return { data: { status } }
      },
      async exportWorkbook({ resource_id }) {
        return Uint8Array.from([...state.created.find(({ id }) => id === resource_id).workbookBytes, 9])
      },
      async disposeSession({ resource_id }) {
        state.disposed.push(resource_id)
      }
    }
  }
}

test("stateless byte lifecycle binds, exports, maps status, and disposes without a VFS", async () => {
  const { adapter, state } = backend("partial")
  const result = await executeStatelessByteOperation({
    backend: adapter,
    operation: "write",
    params: { mode: "apply" },
    plan: { binding_kind: "single_mutable", persistence: "export_required", support_status: "supported" },
    workbooks: [Uint8Array.from([1, 2])]
  })
  assert.equal(state.request.params.resource_id, "resource:1")
  assert.deepEqual(Array.from(result.workbookBytes), [1, 2, 9])
  assert.equal(result.exitCode, 2)
  assert.deepEqual(state.disposed, ["resource:1"])
})

test("stateless byte lifecycle preserves two resources and disposes in reverse", async () => {
  const { adapter, state } = backend()
  const result = await executeStatelessByteOperation({
    backend: adapter,
    operation: "verify_workbook",
    params: {},
    plan: { binding_kind: "two_resource", persistence: "none", support_status: "supported" },
    workbooks: [Uint8Array.of(1), Uint8Array.of(2)]
  })
  assert.equal(state.request.params.resource_id, "resource:1")
  assert.equal(state.request.params.baseline_resource_id, "resource:2")
  assert.equal(result.workbookBytes, undefined)
  assert.deepEqual(state.disposed, ["resource:2", "resource:1"])
})

test("stateless byte plans reject durable staging before allocation", () => {
  assert.throws(
    () => validateStatelessByteRequest(
      { binding_kind: "single_mutable", persistence: "export_required", support_status: "supported" },
      { mode: "stage" }
    ),
    (error) => error.aspCode === "CAPABILITY_UNAVAILABLE" && error.aspPath === "$.mode"
  )
})
