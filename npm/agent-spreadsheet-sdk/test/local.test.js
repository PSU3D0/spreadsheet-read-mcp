const test = require("node:test")
const assert = require("node:assert/strict")

const {
  CanonicalOperationError,
  CapabilityError,
  TransportError,
  createLocalSpreadsheet
} = require("agent-spreadsheet-sdk")
const { createFakeBindings } = require("../test-support/fake-bindings.js")

test("createLocalSpreadsheet validates its runtime", async () => {
  assert.throws(() => createLocalSpreadsheet({}), TypeError)
  assert.throws(() => createLocalSpreadsheet(), TypeError)
  const broken = createLocalSpreadsheet({ runtime: { operations() { return [] } } })
  await assert.rejects(() => broken.capabilities(), TypeError)
})

test("a promised runtime is resolved once and reused", async () => {
  const { bindings, state } = createFakeBindings()
  const local = createLocalSpreadsheet({ runtime: Promise.resolve(bindings) })
  assert.deepEqual(await local.capabilities(), [
    "describe_workbook", "list_sheets", "read_cells", "screenshot_sheet",
    "write", "recalculate", "verify_workbook"
  ])
  const workbook = await local.open(Uint8Array.from([1]))
  assert.equal(workbook.resourceId, "session:s1")
  assert.equal(state.created.length, 1)
})

test("open rejects anything that is not workbook bytes", async () => {
  const { bindings } = createFakeBindings()
  const local = createLocalSpreadsheet({ runtime: bindings })
  await assert.rejects(() => local.open("/tmp/book.xlsx"), TypeError)
})

test("the read surface injects the resource id and tracks the revision", async () => {
  const { bindings, state } = createFakeBindings()
  const local = createLocalSpreadsheet({ runtime: bindings })
  const workbook = await local.open(Uint8Array.from([1]))
  assert.equal(workbook.revisionId, undefined)

  const response = await workbook.readCells({
    sheet_name: "Sheet1",
    selection: { kind: "range", ranges: ["A1:B2"] }
  })
  assert.equal(response.operation, "read_cells")
  assert.equal(state.executed[0].params.resource_id, "session:s1")
  assert.equal(state.executed[0].sessionId, "session:s1")
  assert.equal(workbook.revisionId, "rev-1")
})

test("write and recalculate default expected_revision to the tracked revision", async () => {
  const { bindings, state } = createFakeBindings()
  const local = createLocalSpreadsheet({ runtime: bindings })
  const workbook = await local.open(Uint8Array.from([1]))

  await workbook.write({
    mode: "apply",
    ops: [{ kind: "set_cells", sheet_name: "Sheet1", cells: { A1: { kind: "value", value: "e" } } }]
  })
  // An untracked revision is fetched with describe_workbook first.
  assert.deepEqual(state.executed.map(({ operation }) => operation), ["describe_workbook", "write"])
  assert.equal(state.executed[1].params.expected_revision, "rev-1")
  assert.equal(workbook.revisionId, "rev-2")

  await workbook.recalculate()
  assert.equal(state.executed[2].params.expected_revision, "rev-2")

  await workbook.recalculate({ expected_revision: "rev-3" })
  assert.equal(state.executed[3].params.expected_revision, "rev-3")
})

test("verifyAgainst binds the baseline resource without touching its revision", async () => {
  const { bindings, state } = createFakeBindings()
  const local = createLocalSpreadsheet({ runtime: bindings })
  const current = await local.open(Uint8Array.from([1]))
  const baseline = await local.open(Uint8Array.from([2]))

  await current.verifyAgainst(baseline, { targets: ["Sheet1!A1"], targets_only: true })
  const call = state.executed.at(-1)
  assert.equal(call.params.resource_id, current.resourceId)
  assert.equal(call.params.baseline_resource_id, baseline.resourceId)
  assert.equal(baseline.revisionId, undefined)
})

test("unsupported operations throw CapabilityError before any transport", async () => {
  const { bindings, state } = createFakeBindings({ operations: ["list_sheets"] })
  const local = createLocalSpreadsheet({ runtime: bindings })
  const workbook = await local.open(Uint8Array.from([1]))

  await assert.rejects(() => workbook.readCells({ sheet_name: "S", selection: {} }), (error) => {
    assert.ok(error instanceof CapabilityError)
    assert.equal(error.operation, "read_cells")
    assert.deepEqual(error.available, ["list_sheets"])
    return true
  })
  await assert.rejects(() => local.listWorkbooks(), CapabilityError)
  await assert.rejects(
    () => local.canonical.execute("not_an_operation", {}),
    (error) => error instanceof CapabilityError
  )
  assert.deepEqual(state.executed, [])
})

test("WASM string rejections decode into CanonicalOperationError", async () => {
  const { bindings } = createFakeBindings()
  const local = createLocalSpreadsheet({ runtime: bindings })
  const workbook = await local.open(Uint8Array.from([1]))

  await assert.rejects(
    () => workbook.write({ expected_revision: "rev-9", mode: "apply", ops: [] }),
    (error) => {
      assert.ok(error instanceof CanonicalOperationError)
      assert.equal(error.code, "REVISION_CONFLICT")
      assert.equal(error.operation, "write")
      assert.equal(error.path, "$.expected_revision")
      assert.equal(error.canonicalStatus, 409)
      assert.equal(error.envelope.error.code, "REVISION_CONFLICT")
      assert.equal(error.details.runtime, "local")
      return true
    }
  )
})

test("non-canonical rejections and payloads become TransportError", async () => {
  const plain = createLocalSpreadsheet({
    runtime: createFakeBindings({ respond: () => { throw "boom, not json" } }).bindings
  })
  const plainWorkbook = await plain.open(Uint8Array.from([1]))
  await assert.rejects(() => plainWorkbook.listSheets(), (error) => {
    assert.ok(error instanceof TransportError)
    assert.equal(error.message, "boom, not json")
    return true
  })

  const invalidJson = createLocalSpreadsheet({
    runtime: createFakeBindings({ respond: () => "{" }).bindings
  })
  const invalidWorkbook = await invalidJson.open(Uint8Array.from([1]))
  await assert.rejects(() => invalidWorkbook.listSheets(), (error) => {
    assert.ok(error instanceof TransportError)
    assert.match(error.message, /invalid JSON/)
    return true
  })

  const notEnvelope = createLocalSpreadsheet({
    runtime: createFakeBindings({ respond: () => ({ sheets: [] }) }).bindings
  })
  const notEnvelopeWorkbook = await notEnvelope.open(Uint8Array.from([1]))
  await assert.rejects(() => notEnvelopeWorkbook.listSheets(), (error) => {
    assert.ok(error instanceof TransportError)
    assert.match(error.message, /non-canonical response/)
    return true
  })

  const thrower = createLocalSpreadsheet({
    runtime: createFakeBindings({ respond: () => new Error("host exploded") }).bindings
  })
  const throwerWorkbook = await thrower.open(Uint8Array.from([1]))
  await assert.rejects(() => throwerWorkbook.listSheets(), (error) => {
    assert.ok(error instanceof TransportError)
    assert.equal(error.message, "host exploded")
    return true
  })
})

test("canonical.execute is the typed escape hatch and takes the full input", async () => {
  const { bindings, state } = createFakeBindings()
  const local = createLocalSpreadsheet({ runtime: bindings })
  const workbook = await local.open(Uint8Array.from([1]))
  const response = await local.canonical.execute("list_sheets", {
    resource_id: workbook.resourceId
  })
  assert.equal(response.operation, "list_sheets")
  assert.equal(state.executed[0].params.resource_id, workbook.resourceId)
  assert.ok((await local.canonical.operations()).includes("list_sheets"))
})

test("renderSheet fetches bytes through the binding when it exists", async () => {
  const { bindings, state } = createFakeBindings()
  const local = createLocalSpreadsheet({ runtime: bindings })
  const workbook = await local.open(Uint8Array.from([1]))

  const rendered = await workbook.renderSheet({ sheet_name: "Sheet1", range: "A1:C3" })
  assert.deepEqual(Array.from(rendered.png), [137, 80, 78, 71])
  assert.equal(rendered.fidelity, "approximate")
  assert.deepEqual(rendered.warnings, [{ code: "font_substituted", message: "Calibri" }])
  assert.deepEqual(rendered.calculation, { state: "cached" })
  assert.equal(rendered.sheet_name, "Sheet1")
  assert.equal(rendered.range, "A1:C3")
  assert.equal(rendered.renderer, "native")
  assert.deepEqual(state.artifacts, [{
    sessionId: workbook.resourceId,
    handle: `artifact:sha256:${"a".repeat(64)}`
  }])
  // The slot is released as soon as the bytes have crossed, so a render loop
  // cannot evict its own earlier artifacts.
  assert.deepEqual(state.disposedArtifacts, [{
    sessionId: workbook.resourceId,
    handle: `artifact:sha256:${"a".repeat(64)}`
  }])
  assert.equal(rendered.width, 640)
  assert.equal(rendered.height, 480)
  assert.equal(rendered.png_level, "balanced")
})

test("renderSheet passes png_level through to the canonical request", async () => {
  const { bindings, state } = createFakeBindings()
  const local = createLocalSpreadsheet({ runtime: bindings })
  const workbook = await local.open(Uint8Array.from([1]))

  const rendered = await workbook.renderSheet({ sheet_name: "Sheet1", png_level: "fast" })
  assert.equal(state.executed[0].params.png_level, "fast")
  assert.equal(rendered.png_level, "fast")
})

test("renderSheet throws CapabilityError when the binding has no readArtifact", async () => {
  const { bindings } = createFakeBindings({ readArtifact: false })
  const local = createLocalSpreadsheet({ runtime: bindings })
  const workbook = await local.open(Uint8Array.from([1]))

  await assert.rejects(() => workbook.renderSheet({ sheet_name: "Sheet1" }), (error) => {
    assert.ok(error instanceof CapabilityError)
    assert.equal(error.capability, "readArtifact")
    return true
  })
})

test("export, dispose, and async disposal are capability-gated and idempotent", async () => {
  const { bindings, state } = createFakeBindings()
  const local = createLocalSpreadsheet({ runtime: bindings })
  const workbook = await local.open(Uint8Array.from([1]))

  assert.deepEqual(Array.from(await workbook.exportBytes()), [80, 75, 3, 4])
  await workbook.dispose()
  await workbook.dispose()
  assert.deepEqual(state.disposed, [workbook.resourceId])
  assert.equal(workbook.disposed, true)
  await assert.rejects(() => workbook.exportBytes(), CapabilityError)
  await assert.rejects(() => workbook.listSheets(), CapabilityError)

  const scoped = await local.open(Uint8Array.from([2]))
  await scoped[Symbol.asyncDispose]()
  assert.deepEqual(state.disposed, [workbook.resourceId, scoped.resourceId])

  const readOnly = createLocalSpreadsheet({
    runtime: createFakeBindings({ exportWorkbook: false }).bindings
  })
  const readOnlyWorkbook = await readOnly.open(Uint8Array.from([1]))
  await assert.rejects(() => readOnlyWorkbook.exportBytes(), (error) => {
    assert.ok(error instanceof CapabilityError)
    assert.equal(error.capability, "exportWorkbook")
    return true
  })
})
