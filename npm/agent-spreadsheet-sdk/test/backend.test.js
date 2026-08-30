const test = require("node:test")
const assert = require("node:assert/strict")

const {
  McpBackend,
  WasmBackend,
  CapabilityError,
  CANONICAL_REGISTRY,
  OPERATION_NAMES
} = require("../src")

function canonicalTool(operation, markerOperation = operation) {
  return {
    name: operation,
    _meta: {
      "agent-spreadsheet/canonical": {
        schema_version: "1",
        operation: markerOperation
      }
    }
  }
}

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
    supportedOperations: ["read_cells"],
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
    supportedOperations: ["list_workbooks"],
    transport: {
      list_workbooks(input) {
        assert.deepEqual(input, {})
        return expected
      }
    }
  })

  assert.strictEqual(await backend.execute("list_workbooks", {}), expected)
})

test("MCP negotiates tools before execute and refreshes live capabilities", async () => {
  let advertised = ["read_cells"]
  let invokes = 0
  const backend = new McpBackend({
    transport: {
      async listTools() {
        return { tools: advertised.map((name) => canonicalTool(name)) }
      },
      async invoke(operation, input) {
        invokes += 1
        return envelope(operation, input.resource_id)
      }
    }
  })

  assert.equal(backend.getCapabilities().initialized, false)
  assert.deepEqual(backend.getCapabilities().operations, [])
  assert.equal(backend.getCapabilities().supportsRangeValues, false)

  await backend.readCells({ resource_id: "wb:one" })
  assert.equal(invokes, 1)
  assert.equal(backend.getCapabilities().initialized, true)
  assert.deepEqual(backend.getCapabilities().operations, ["read_cells"])
  assert.equal(backend.getCapabilities().supportsRangeValues, true)

  advertised = ["write"]
  await backend.refresh()
  assert.deepEqual(backend.getCapabilities().operations, ["write"])
  assert.equal(backend.getCapabilities().supportsRangeValues, false)
  assert.equal(backend.getCapabilities().supportsTransformBatch, true)
  await assert.rejects(backend.readCells({ resource_id: "wb:one" }), CapabilityError)
  assert.equal(invokes, 1)
})

test("MCP tools/list trusts only matching canonical markers", async () => {
  const backend = new McpBackend({
    transport: {
      async request(request) {
        assert.deepEqual(request, { method: "tools/list", params: {} })
        return {
          result: {
            tools: [
              canonicalTool("read_cells"),
              { name: "list_sheets" },
              { name: "screenshot_sheet" },
              canonicalTool("describe_workbook", "list_sheets"),
              canonicalTool("not_an_operation")
            ]
          }
        }
      },
      async invoke() {}
    }
  })

  await backend.initialize()
  assert.deepEqual(backend.getCapabilities().operations, ["read_cells"])
})

test("MCP explicit listOperations remains a trusted canonical contract", async () => {
  const backend = new McpBackend({
    transport: {
      listOperations: () => JSON.stringify(["list_sheets", { name: "read_cells" }]),
      async invoke() {}
    }
  })

  await backend.initialize()
  assert.deepEqual(backend.getCapabilities().operations, ["list_sheets", "read_cells"])
})

test("MCP without discovery or explicit support does not fall back to the manifest", async () => {
  const backend = new McpBackend({ transport: { invoke() { throw new Error("must not invoke") } } })
  assert.deepEqual(backend.getCapabilities().operations, [])
  await assert.rejects(backend.execute("read_cells", {}), (error) => {
    assert.equal(error.code, "INVALID_ARGUMENT")
    return true
  })
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
    supportedOperations: ["write"],
    transport: { invoke: async () => errorEnvelope }
  })
  assert.strictEqual(await returned.execute("write", {}), errorEnvelope)

  const rejected = new McpBackend({
    supportedOperations: ["write"],
    transport: { invoke: async () => { throw errorEnvelope } }
  })
  await assert.rejects(rejected.execute("write", {}), (error) => error === errorEnvelope)
})

test("WASM execute uses the canonical JSON dispatch binding", async () => {
  const expected = envelope("read_cells", "session:session-1", { blocks: [] })
  const backend = new WasmBackend({
    bindings: {
      operations: () => JSON.stringify([{ name: "read_cells" }]),
      async executeOperation(sessionId, operation, paramsJson) {
        assert.equal(sessionId, "session:session-1")
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
    bindings: {
      operations: () => [{ name: "read_cells" }],
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

test("WASM capabilities come from live binding discovery", async () => {
  const missingDispatcher = new WasmBackend({ bindings: {} })
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
    bindings: {
      operations: () => [{ name: "read_cells" }],
      executeOperation() {}
    }
  })
  assert.deepEqual(explicit.getCapabilities().operations, ["read_cells"])
  assert.equal(explicit.getCapabilities().supportsRangeValues, true)
  assert.equal(explicit.getCapabilities().supportsVerification, false)
})

test("unsupported canonical operations throw CapabilityError before transport", async () => {
  let called = false
  const backend = new McpBackend({
    supportedOperations: ["list_sheets"],
    transport: { invoke() { called = true } }
  })

  await assert.rejects(backend.execute("write", {}), (error) => {
    assert.ok(error instanceof CapabilityError)
    assert.equal(error.code, "UNSUPPORTED_CAPABILITY")
    assert.equal(error.operation, "write")
    return true
  })
  assert.equal(called, false)

  const wasmBackend = new WasmBackend({
    bindings: {
      operations: () => [{ name: "list_sheets" }],
      executeOperation() { called = true }
    }
  })
  await assert.rejects(wasmBackend.execute("write", {}), CapabilityError)
  assert.equal(called, false)
})

test("legacy read methods are deprecated data projections over execute", async () => {
  const seen = []
  const backend = new McpBackend({
    supportedOperations: ["read_cells", "search_values"],
    transport: {
      async invoke(operation, input) {
        seen.push({ operation, input })
        return envelope(operation, input.resource_id, { marker: operation })
      }
    }
  })

  assert.deepEqual(await backend.rangeValues({
    workbookId: "wb-1",
    sheetName: "Sheet1",
    ranges: "A1:B2"
  }), { marker: "read_cells" })
  assert.deepEqual(await backend.findValue({ workbookId: "wb-1", query: "alpha" }), {
    marker: "search_values"
  })

  assert.deepEqual(seen[0].input.selection, { kind: "range", ranges: ["A1:B2"] })
  assert.equal(seen[1].input.resource_id, "wb:wb-1")
  assert.equal(seen[1].input.query, "alpha")
})

const COLLIDING_COMPAT_METHODS = [
  ["describeWorkbook", "describe_workbook", { workbookId: "book" }, "wb:book"],
  ["namedRanges", "named_ranges", { workbookId: "book" }, "wb:book"],
  ["sheetOverview", "sheet_overview", { workbookId: "book", sheetName: "Sheet1" }, "wb:book"],
  ["listSheets", "list_sheets", { workbookId: "book" }, "wb:book"],
  ["readTable", "read_table", { workbookId: "book", sheetName: "Sheet1" }, "wb:book"],
  ["createFork", "create_fork", { workbookOrForkId: "book", expectedRevision: "rev-1" }, "wb:book"],
  ["listForks", "list_forks", {}, undefined],
  ["verifyWorkbook", "verify_workbook", {
    currentWorkbookOrForkId: "current",
    baselineWorkbookOrForkId: "baseline"
  }, "fork:current"],
  ["discardFork", "discard_fork", { forkId: "fork-1", expectedRevision: "rev-1" }, "fork:fork-1"]
]

for (const kind of ["MCP", "WASM"]) {
  test(`${kind} preserves all nine colliding 0.13 methods for legacy-shaped input`, async () => {
    const operations = COLLIDING_COMPAT_METHODS.map(([, operation]) => operation)
    const calls = []
    const invoke = async (operation, input) => {
      calls.push({ operation, input })
      return envelope(operation, input.resource_id, { marker: `${kind}:${operation}` })
    }
    const backend = kind === "MCP"
      ? new McpBackend({ supportedOperations: operations, transport: { invoke } })
      : new WasmBackend({
          bindings: {
            operations: () => operations,
            executeOperation: async (_resourceId, operation, params) => JSON.stringify(
              await invoke(operation, JSON.parse(params))
            )
          }
        })

    for (const [method, operation, input, expectedResourceId] of COLLIDING_COMPAT_METHODS) {
      const legacyInput = kind === "WASM" && input.workbookId
        ? { ...input, workbookId: undefined, sessionId: input.workbookId }
        : input
      assert.deepEqual(await backend[method](legacyInput), { marker: `${kind}:${operation}` }, method)
      const call = calls.at(-1)
      assert.equal(call.operation, operation, method)
      assert.equal(call.input.resource_id, kind === "WASM" && expectedResourceId?.startsWith("wb:")
        ? expectedResourceId.replace(/^wb:/, "session:")
        : expectedResourceId, method)
    }
  })

  test(`${kind} colliding methods preserve canonical envelopes only for explicit canonical input`, async () => {
    const operations = COLLIDING_COMPAT_METHODS.map(([, operation]) => operation)
    const invoke = async (operation, input) => envelope(operation, input.resource_id, { canonical: true })
    const backend = kind === "MCP"
      ? new McpBackend({ supportedOperations: operations, transport: { invoke } })
      : new WasmBackend({
          bindings: {
            operations: () => operations,
            executeOperation: async (_resourceId, operation, params) => JSON.stringify(
              await invoke(operation, JSON.parse(params))
            )
          }
        })

    for (const [method, operation] of COLLIDING_COMPAT_METHODS) {
      if (method === "listForks") {
        assert.deepEqual(await backend.listForks({}), { canonical: true })
        continue
      }
      const input = operation === "verify_workbook"
        ? { resource_id: "fork:current", baseline_resource_id: "wb:baseline" }
        : { resource_id: operation === "discard_fork" ? "fork:one" : "wb:one" }
      const result = await backend[method](input)
      assert.equal(result.schema_version, "1", method)
      assert.equal(result.operation, operation, method)
    }
  })
}

test("legacy write methods compile to canonical write operations", async () => {
  let received
  const backend = new McpBackend({
    supportedOperations: ["write"],
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
        return "session:session-1"
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
  assert.equal(await backend.createSession({ bytes: Uint8Array.from([2]) }), "session:session-1")
  assert.deepEqual(await backend.exportWorkbook({ resource_id: "session:session-1" }), Uint8Array.from([1, 2]))
  await backend.disposeSession({ resource_id: "session:session-1" })
  assert.deepEqual(calls, [
    ["create", Uint8Array.from([1])],
    ["create", Uint8Array.from([2])],
    ["export", "session:session-1"],
    ["dispose", "session:session-1"]
  ])
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
