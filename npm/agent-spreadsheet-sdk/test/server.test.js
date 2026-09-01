const test = require("node:test")
const assert = require("node:assert/strict")

const {
  CANONICAL_ERROR_STATUS,
  CanonicalOperationError,
  CapabilityError,
  RemoteFork,
  TransportError,
  connectSpreadsheetServer,
  statusForCanonicalCode
} = require("agent-spreadsheet-sdk")
const {
  canonicalFailure,
  createFakeFetch,
  response,
  STATUS_BY_CODE
} = require("../test-support/fake-fetch.js")

function connect(options = {}) {
  const fake = createFakeFetch(options)
  return { fake, client: connectSpreadsheetServer({ baseUrl: "http://127.0.0.1:8079", fetch: fake.fetch }) }
}

test("the status table matches the pinned canonical route document", () => {
  assert.deepEqual({ ...CANONICAL_ERROR_STATUS }, STATUS_BY_CODE)
  assert.equal(statusForCanonicalCode("REVISION_CONFLICT"), 409)
  assert.equal(statusForCanonicalCode("NOT_A_CODE"), undefined)
})

test("baseUrl is normalized to the /v1 root", () => {
  const { fetch } = createFakeFetch()
  assert.equal(connectSpreadsheetServer({ baseUrl: "http://h:1/", fetch }).baseUrl, "http://h:1/v1")
  assert.equal(connectSpreadsheetServer({ baseUrl: "http://h:1/v1", fetch }).baseUrl, "http://h:1/v1")
  assert.equal(connectSpreadsheetServer({ baseUrl: "http://h:1/v1/", fetch }).baseUrl, "http://h:1/v1")
  assert.throws(() => connectSpreadsheetServer({ baseUrl: "", fetch }), TypeError)
  assert.throws(
    () => connectSpreadsheetServer({ baseUrl: "http://h:1", fetch: "nope" }),
    TypeError
  )
})

test("capabilities come from GET /v1/operations and are cached until refresh", async () => {
  const { fake, client } = connect()
  const capabilities = await client.capabilities()
  assert.ok(capabilities.includes("create_fork"))
  await client.capabilities()
  assert.equal(fake.state.operations, 1)
  client.refresh()
  await client.capabilities()
  assert.equal(fake.state.operations, 2)
})

test("client-level operations post to /v1/op and return the envelope", async () => {
  const { fake, client } = connect()
  const listed = await client.listWorkbooks({ limit: 5 })
  assert.equal(listed.operation, "list_workbooks")
  assert.deepEqual(listed.data.workbooks, [{ resource_id: "wb:wb-1" }])
  assert.deepEqual(fake.state.calls[0], {
    operation: "list_workbooks",
    input: { limit: 5 },
    method: "POST"
  })
})

test("a remote workbook is a non-owning read handle that tracks its revision", async () => {
  const { fake, client } = connect()
  const workbook = client.workbook("wb:wb-1")
  assert.throws(() => client.workbook(""), TypeError)

  const described = await workbook.describe()
  assert.equal(described.operation, "describe_workbook")
  assert.equal(workbook.revisionId, "rev-1")
  assert.equal(fake.state.calls[0].input.resource_id, "wb:wb-1")

  const read = await workbook.readCells({
    sheet_name: "Sheet1",
    selection: { kind: "range", ranges: ["A1:B2"] }
  })
  assert.equal(read.operation, "read_cells")
})

test("createFork defaults to the tracked revision and returns a fork handle", async () => {
  const { fake, client } = connect()
  const workbook = client.workbook("wb:wb-1")

  const fork = await workbook.createFork()
  assert.ok(fork instanceof RemoteFork)
  assert.equal(fork.resourceId, "fork:f1")
  assert.equal(fork.revisionId, "rev-1")
  // The fork's identity never overwrites the source workbook's.
  assert.equal(workbook.resourceId, "wb:wb-1")
  assert.deepEqual(fake.state.calls.map(({ operation }) => operation), [
    "describe_workbook",
    "create_fork"
  ])
  assert.equal(fake.state.calls[1].input.expected_revision, "rev-1")

  const explicit = await workbook.createFork({ expected_revision: "rev-1" })
  assert.equal(explicit.resourceId, "fork:f2")
})

test("fork mutations default expected_revision and track the new revision", async () => {
  const { fake, client } = connect()
  const fork = await client.workbook("wb:wb-1").createFork({ expected_revision: "rev-1" })

  const written = await fork.write({
    mode: "apply",
    ops: [{ kind: "set_cells", sheet_name: "Sheet1", cells: { A1: { kind: "value", value: "e" } } }]
  })
  assert.equal(written.operation, "write")
  assert.equal(fake.state.calls.at(-1).input.expected_revision, "rev-1")
  assert.equal(fork.revisionId, "rev-2")

  await fork.recalculate({ backend: "formualizer" })
  assert.equal(fake.state.calls.at(-1).input.expected_revision, "rev-2")
  assert.equal(fork.revisionId, "rev-3")

  await fork.getChanges({ view: { kind: "operations" } })
  assert.deepEqual(fake.state.calls.at(-1).input, { view: { kind: "operations" }, resource_id: "fork:f1" })

  await fork.verifyAgainst(client.workbook("wb:wb-1"), { targets_only: true })
  assert.equal(fake.state.calls.at(-1).input.baseline_resource_id, "wb:wb-1")
})

test("checkpoint and stagedChange inject the revision only when the action needs one", async () => {
  const { fake, client } = connect()
  const fork = await client.workbook("wb:wb-1").createFork({ expected_revision: "rev-1" })

  await fork.checkpoint({ action: "list" })
  assert.equal(fake.state.calls.at(-1).input.expected_revision, undefined)

  await fork.checkpoint({ action: "create", label: "before" })
  assert.equal(fake.state.calls.at(-1).input.expected_revision, "rev-1")

  await fork.stagedChange({ action: "list" })
  assert.equal(fake.state.calls.at(-1).input.expected_revision, undefined)

  await fork.stagedChange({ action: "apply", change_id: "c1" })
  assert.equal(fake.state.calls.at(-1).input.expected_revision, "rev-1")
})

test("exportFork, discard, and async disposal use the tracked revision once", async () => {
  const { fake, client } = connect()
  const fork = await client.workbook("wb:wb-1").createFork({ expected_revision: "rev-1" })

  await fork.exportFork({ destination: { kind: "in_place" } })
  assert.equal(fake.state.calls.at(-1).input.destination.kind, "in_place")

  const discarded = await fork.discard()
  assert.equal(discarded.operation, "discard_fork")
  assert.equal(fork.discarded, true)
  const callsAfterDiscard = fake.state.calls.length
  assert.equal(await fork.discard(), undefined)
  assert.equal(fake.state.calls.length, callsAfterDiscard)

  const scoped = await client.workbook("wb:wb-1").createFork({ expected_revision: "rev-3" })
  await scoped[Symbol.asyncDispose]()
  assert.equal(scoped.discarded, true)
  assert.equal(fake.state.calls.at(-1).operation, "discard_fork")
})

test("canonical error envelopes map to CanonicalOperationError with their status", async () => {
  const { client } = connect()
  const fork = await client.workbook("wb:wb-1").createFork({ expected_revision: "rev-1" })

  await assert.rejects(
    () => fork.write({ expected_revision: "rev-99", mode: "apply", ops: [] }),
    (error) => {
      assert.ok(error instanceof CanonicalOperationError)
      assert.equal(error.code, "REVISION_CONFLICT")
      assert.equal(error.details.status, 409)
      assert.equal(error.details.runtime, "server")
      assert.equal(error.path, "$.expected_revision")
      return true
    }
  )
})

test("adapter policy failures keep the canonical envelope", async () => {
  const { client } = connect({
    handler: ({ operation }) => operation === "read_cells"
      ? canonicalFailure("CAPABILITY_UNAVAILABLE", "excluded by SPREADSHEET_MCP_ENABLED_TOOLS", operation)
      : undefined
  })
  await assert.rejects(
    () => client.workbook("wb:wb-1").readCells({ sheet_name: "S", selection: {} }),
    (error) => {
      assert.ok(error instanceof CanonicalOperationError)
      assert.equal(error.code, "CAPABILITY_UNAVAILABLE")
      assert.equal(error.details.status, 501)
      return true
    }
  )
})

test("non-canonical HTTP failures become TransportError", async () => {
  const proxy = connect({
    handler: () => response(502, "<html>bad gateway</html>")
  })
  await assert.rejects(() => proxy.client.workbook("wb:wb-1").describe(), (error) => {
    assert.ok(error instanceof TransportError)
    assert.equal(error.status, 502)
    assert.match(error.body, /bad gateway/)
    assert.equal(error.operation, "describe_workbook")
    return true
  })

  const garbage = connect({ handler: () => response(200, "not json at all") })
  await assert.rejects(() => garbage.client.workbook("wb:wb-1").describe(), (error) => {
    assert.ok(error instanceof TransportError)
    assert.match(error.message, /invalid JSON/)
    return true
  })

  const notEnvelope = connect({ handler: () => response(200, { sheets: [] }) })
  await assert.rejects(() => notEnvelope.client.workbook("wb:wb-1").describe(), (error) => {
    assert.ok(error instanceof TransportError)
    assert.match(error.message, /non-canonical response/)
    return true
  })

  const offline = connectSpreadsheetServer({
    baseUrl: "http://127.0.0.1:1",
    fetch: async () => { throw new Error("ECONNREFUSED") }
  })
  await assert.rejects(() => offline.capabilities(), (error) => {
    assert.ok(error instanceof TransportError)
    assert.match(error.message, /GET http:\/\/127\.0\.0\.1:1\/v1\/operations failed/)
    return true
  })
  // A failed discovery is not cached.
  await assert.rejects(() => offline.capabilities(), TransportError)
})

test("operations missing from GET /v1/operations throw CapabilityError before transport", async () => {
  const { fake, client } = connect({ operations: ["list_workbooks", "describe_workbook"] })
  await assert.rejects(
    () => client.workbook("wb:wb-1").createFork({ expected_revision: "rev-1" }),
    (error) => {
      assert.ok(error instanceof CapabilityError)
      assert.equal(error.operation, "create_fork")
      assert.match(error.message, /server runtime/)
      return true
    }
  )
  assert.deepEqual(fake.state.calls, [])
})

test("renderSheet fetches artifact bytes from GET /v1/artifacts", async () => {
  const { fake, client } = connect()
  const rendered = await client.workbook("wb:wb-1").renderSheet({ sheet_name: "Sheet1" })
  assert.deepEqual(Array.from(rendered.png), [137, 80, 78, 71])
  assert.equal(rendered.handle, `artifact:sha256:${"b".repeat(64)}`)
  assert.equal(rendered.sheet_name, "Sheet1")
  assert.equal(rendered.range, "A1:D20")
  // The 0.14 registry has no fidelity, warnings, or calculation fields yet.
  assert.equal(rendered.fidelity, "unknown")
  assert.deepEqual(rendered.warnings, [])
  assert.equal(rendered.calculation, null)
  assert.equal(rendered.renderer, undefined)
  assert.deepEqual(fake.state.artifacts, [`artifact:sha256:${"b".repeat(64)}`])
})

test("a missing artifact keeps the canonical envelope", async () => {
  const { client } = connect({ artifact: null })
  await assert.rejects(() => client.workbook("wb:wb-1").renderSheet({ sheet_name: "S" }), (error) => {
    assert.ok(error instanceof CanonicalOperationError)
    assert.equal(error.code, "RESOURCE_NOT_FOUND")
    assert.equal(error.details.status, 404)
    return true
  })
})

test("canonical.execute takes the full input including resource_id", async () => {
  const { fake, client } = connect()
  const response = await client.canonical.execute("read_cells", {
    resource_id: "wb:wb-1",
    sheet_name: "Sheet1",
    selection: { kind: "range", ranges: ["A1:A2"] }
  })
  assert.equal(response.operation, "read_cells")
  assert.equal(fake.state.calls[0].input.resource_id, "wb:wb-1")
})
