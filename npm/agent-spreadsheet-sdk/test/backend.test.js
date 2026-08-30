const test = require("node:test")
const assert = require("node:assert/strict")

const {
  McpBackend,
  WasmBackend,
  CapabilityError,
  CANONICAL_REGISTRY,
  OPERATION_NAMES
} = require("../src")

function envelope(operation, resourceId, data = {}) {
  return {
    schema_version: "1",
    operation,
    resource_id: resourceId,
    revision_id: "rev-1",
    data
  }
}

test("MCP execute dispatches canonical input and preserves the envelope", async () => {
  const input = {
    resource_id: "wb:wb-1",
    sheet_name: "Sheet1",
    selection: { kind: "range", ranges: ["A1:B2"] }
  }
  const expected = envelope("read_cells", "wb:wb-1", { blocks: [], next_cursor: null })
  const backend = new McpBackend({
    operations: ["read_cells"],
    transport: {
      async invoke(operation, received) {
        assert.equal(operation, "read_cells")
        assert.strictEqual(received, input)
        return expected
      }
    }
  })

  assert.strictEqual(await backend.execute("read_cells", input), expected)
  assert.strictEqual(await backend.readCells(input), expected)
})

test("MCP supports per-operation canonical transports", async () => {
  const expected = envelope("list_workbooks", undefined, { items: [] })
  const backend = new McpBackend({
    operations: ["list_workbooks"],
    transport: {
      list_workbooks(input) {
        assert.deepEqual(input, {})
        return expected
      }
    }
  })

  assert.strictEqual(await backend.execute("list_workbooks", {}), expected)
})

test("canonical error envelopes pass through without SDK normalization", async () => {
  const errorEnvelope = {
    schema_version: "1",
    error: {
      code: "REVISION_CONFLICT",
      message: "revision changed",
      operation: "write",
      path: "$.expected_revision"
    }
  }
  const returned = new McpBackend({
    operations: ["write"],
    transport: { invoke: async () => errorEnvelope }
  })
  assert.strictEqual(await returned.execute("write", {}), errorEnvelope)

  const rejected = new McpBackend({
    operations: ["write"],
    transport: { invoke: async () => { throw errorEnvelope } }
  })
  await assert.rejects(rejected.execute("write", {}), (error) => error === errorEnvelope)
})

test("WASM execute uses the canonical JSON dispatch binding", async () => {
  const expected = envelope("read_cells", "session:session-1", { blocks: [] })
  const backend = new WasmBackend({
    operations: ["read_cells"],
    bindings: {
      async executeOperation(sessionId, operation, paramsJson) {
        assert.equal(sessionId, "session-1")
        assert.equal(operation, "read_cells")
        assert.deepEqual(JSON.parse(paramsJson), {
          resource_id: "session:session-1",
          sheet_name: "Sheet1",
          selection: { kind: "range", ranges: ["A1"] }
        })
        return JSON.stringify(expected)
      }
    }
  })

  assert.deepEqual(await backend.execute("read_cells", {
    resource_id: "session:session-1",
    sheet_name: "Sheet1",
    selection: { kind: "range", ranges: ["A1"] }
  }), expected)
})

test("WASM decodes canonical error envelopes from its JSON transport", async () => {
  const errorEnvelope = {
    schema_version: "1",
    error: { code: "RESOURCE_NOT_FOUND", message: "missing", operation: "read_cells" }
  }
  const backend = new WasmBackend({
    operations: ["read_cells"],
    bindings: {
      async executeOperation() {
        throw JSON.stringify(errorEnvelope)
      }
    }
  })

  await assert.rejects(
    backend.execute("read_cells", { resource_id: "session:missing" }),
    (error) => error.schema_version === "1" && error.error.code === "RESOURCE_NOT_FOUND"
  )
})

test("WASM capabilities require a real dispatcher and explicit operations", async () => {
  const missingDispatcher = new WasmBackend({
    operations: ["verify_workbook"],
    bindings: {}
  })
  assert.deepEqual(missingDispatcher.getCapabilities().operations, [])
  assert.equal(missingDispatcher.getCapabilities().supportsVerification, false)

  await assert.rejects(
    missingDispatcher.verifyWorkbook({ sessionId: "session-1" }),
    (error) => {
      assert.ok(error instanceof CapabilityError)
      assert.equal(error.capability, "verify_workbook")
      return true
    }
  )

  const explicit = new WasmBackend({
    operations: ["read_cells"],
    bindings: { executeOperation() {} }
  })
  assert.deepEqual(explicit.getCapabilities().operations, ["read_cells"])
  assert.equal(explicit.getCapabilities().supportsRangeValues, true)
  assert.equal(explicit.getCapabilities().supportsVerification, false)
})

test("unsupported canonical operations throw CapabilityError before transport", async () => {
  let called = false
  const backend = new McpBackend({
    operations: ["list_sheets"],
    transport: { invoke() { called = true } }
  })

  await assert.rejects(backend.execute("write", {}), (error) => {
    assert.ok(error instanceof CapabilityError)
    assert.equal(error.code, "UNSUPPORTED_CAPABILITY")
    assert.equal(error.operation, "write")
    return true
  })
  assert.equal(called, false)
})

test("legacy read methods are deprecated data projections over execute", async () => {
  const seen = []
  const backend = new McpBackend({
    operations: ["list_sheets", "read_cells", "search_values"],
    transport: {
      async invoke(operation, input) {
        seen.push({ operation, input })
        return envelope(operation, input.resource_id, { marker: operation })
      }
    }
  })

  assert.deepEqual(await backend.listSheets({ workbookId: "wb-1" }), { marker: "list_sheets" })
  assert.deepEqual(await backend.rangeValues({
    workbookId: "wb-1",
    sheetName: "Sheet1",
    ranges: "A1:B2"
  }), { marker: "read_cells" })
  assert.deepEqual(await backend.findValue({ workbookId: "wb-1", query: "alpha" }), {
    marker: "search_values"
  })

  assert.deepEqual(seen[0], {
    operation: "list_sheets",
    input: { resource_id: "wb:wb-1" }
  })
  assert.deepEqual(seen[1].input.selection, { kind: "range", ranges: ["A1:B2"] })
  assert.equal(seen[2].input.resource_id, "wb:wb-1")
  assert.equal(seen[2].input.query, "alpha")
})

test("legacy write methods compile to canonical write operations", async () => {
  let received
  const backend = new McpBackend({
    operations: ["write"],
    transport: {
      async invoke(operation, input) {
        received = { operation, input }
        return envelope("write", input.resource_id, { status: "previewed" })
      }
    }
  })

  assert.deepEqual(await backend.replaceInFormulas({
    forkId: "fork-1",
    expectedRevision: "rev-1",
    sheetName: "Sheet1",
    find: "C2:C10",
    replace: "D2:D20",
    options: { dryRun: true }
  }), { status: "previewed" })
  assert.equal(received.operation, "write")
  assert.equal(received.input.resource_id, "fork:fork-1")
  assert.equal(received.input.expected_revision, "rev-1")
  assert.equal(received.input.mode, "preview")
  assert.equal(received.input.ops[0].kind, "replace_in_formulas")
})

test("WASM resource lifecycle capabilities reflect actual bindings", async () => {
  const calls = []
  const backend = new WasmBackend({
    bindings: {
      createSession(bytes) {
        calls.push(["create", bytes])
        return "session-1"
      },
      exportWorkbook(id) {
        calls.push(["export", id])
        return Uint8Array.from([1, 2])
      },
      disposeSession(id) {
        calls.push(["dispose", id])
      }
    }
  })

  assert.equal(backend.getCapabilities().resourceBinding, true)
  assert.equal(backend.getCapabilities().resourceExport, true)
  assert.deepEqual(await backend.bindWorkbook({ bytes: Uint8Array.from([1]) }), {
    resource_id: "session:session-1"
  })
  assert.deepEqual(await backend.exportWorkbook({ resource_id: "session:session-1" }), Uint8Array.from([1, 2]))
  await backend.disposeSession({ resource_id: "session:session-1" })
  assert.deepEqual(calls.map((call) => call[0]), ["create", "export", "dispose"])
})

test("checked-in registry drives canonical convenience methods", () => {
  assert.equal(CANONICAL_REGISTRY.schema_version, "1")
  assert.deepEqual(OPERATION_NAMES, CANONICAL_REGISTRY.operations.map(({ name }) => name))
  for (const operation of OPERATION_NAMES) {
    const method = operation.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase())
    assert.equal(typeof McpBackend.prototype[method], "function")
    assert.equal(typeof WasmBackend.prototype[method], "function")
  }
})
