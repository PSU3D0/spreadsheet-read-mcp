// Local runtime integration against the real built WASM package.
//
// Set AGENT_SPREADSHEET_WASM_PACKAGE, or build it once with
// `node scripts/run-generated-wasm-integration.js`, which reuses
// ../../target/sdk-wasm-node when it exists.

const test = require("node:test")
const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")

const {
  CapabilityError,
  CanonicalOperationError,
  createLocalSpreadsheet,
  operationsForAdapter
} = require("agent-spreadsheet-sdk")

const repositoryRoot = path.resolve(__dirname, "..", "..", "..")
const defaultPackage = path.join(repositoryRoot, "target", "sdk-wasm-node")
const generatedPackage = process.env.AGENT_SPREADSHEET_WASM_PACKAGE ||
  (fs.existsSync(path.join(defaultPackage, "package.json")) ? defaultPackage : undefined)
const fixture = path.join(
  repositoryRoot, "crates", "agent-spreadsheet", "tests", "fixtures", "f1", "baseline.xlsx"
)

test("local runtime drives the real WASM package end to end", {
  skip: generatedPackage ? false : "set AGENT_SPREADSHEET_WASM_PACKAGE to run the local integration"
}, async (t) => {
  const local = createLocalSpreadsheet({ runtime: require(generatedPackage) })
  const capabilities = await local.capabilities()
  assert.deepEqual([...capabilities].sort(), [...operationsForAdapter("wasm")].sort())

  const bytes = fs.readFileSync(fixture)
  const workbook = await local.open(bytes)
  const baseline = await local.open(bytes)
  t.after(async () => {
    await baseline.dispose()
    await workbook.dispose()
  })

  assert.match(workbook.resourceId, /^session:/)

  await t.test("read surface tracks resource and revision", async () => {
    const described = await workbook.describeWorkbook()
    assert.equal(described.operation, "describe_workbook")
    assert.equal(described.resource_id, workbook.resourceId)
    assert.equal(workbook.revisionId, described.revision_id)

    const read = await workbook.readCells({
      sheet_name: "Sheet1",
      selection: { kind: "range", ranges: ["A1:B2"] },
      format: "dense"
    })
    assert.equal(read.operation, "read_cells")
  })

  await t.test("write defaults expected_revision to the tracked revision", async () => {
    const before = workbook.revisionId
    const written = await workbook.write({
      mode: "apply",
      ops: [{
        kind: "set_cells",
        sheet_name: "Sheet1",
        cells: { A1: { kind: "value", value: "e" } }
      }]
    })
    assert.equal(written.data.status, "applied")
    assert.equal(written.resource_id, workbook.resourceId)
    assert.notEqual(workbook.revisionId, before)
  })

  await t.test("a stale expected_revision surfaces a canonical error", async () => {
    await assert.rejects(
      workbook.write({
        expected_revision: "0".repeat(64),
        mode: "apply",
        ops: [{
          kind: "set_cells",
          sheet_name: "Sheet1",
          cells: { A1: { kind: "value", value: "stale" } }
        }]
      }),
      (error) => {
        assert.ok(error instanceof CanonicalOperationError, `${error}`)
        assert.equal(error.code, "REVISION_CONFLICT")
        assert.equal(error.canonicalStatus, 409)
        assert.equal(error.envelope.schema_version, "1")
        return true
      }
    )
  })

  await t.test("recalculate and verifyAgainst run on the same session", async () => {
    const recalculated = await workbook.recalculate({ backend: "formualizer" })
    assert.equal(recalculated.operation, "recalculate")
    assert.equal(recalculated.data.evaluation_coverage.source, "formualizer")

    const verified = await workbook.verifyAgainst(baseline, {
      targets: ["Sheet1!A1"],
      targets_only: true
    })
    assert.equal(verified.data.proof_status, "differences_found")
  })

  await t.test("exportBytes returns the mutated workbook", async () => {
    const exported = await workbook.exportBytes()
    assert.ok(exported instanceof Uint8Array)
    assert.ok(exported.byteLength > 0)
  })

  await t.test("unsupported operations fail before transport", async () => {
    await assert.rejects(
      () => local.listWorkbooks(),
      (error) => error instanceof CapabilityError && error.operation === "list_workbooks"
    )
  })

  await t.test("renderSheet returns PNG bytes and releases the slot", async () => {
    const rendered = await workbook.renderSheet({ sheet_name: "Sheet1", range: "A1:C6" })
    assert.deepEqual(Array.from(rendered.png.subarray(0, 4)), [0x89, 0x50, 0x4e, 0x47])
    assert.equal(rendered.renderer, "native-raster/1")
    assert.equal(rendered.range, "A1:C6")
    assert.equal(rendered.png_level, "balanced")
    assert.ok(rendered.width > 0 && rendered.height > 0)
    assert.match(rendered.handle, /^artifact:sha256:[0-9a-f]{64}$/)
    assert.ok(["full", "partial"].includes(rendered.fidelity), rendered.fidelity)

    // The object model disposes the slot once the bytes have crossed, so the
    // handle is gone from the session.
    const runtime = require(generatedPackage)
    assert.throws(() => runtime.readArtifact(workbook.resourceId, rendered.handle))

    const fast = await workbook.renderSheet({
      sheet_name: "Sheet1", range: "A1:C6", png_level: "fast"
    })
    assert.equal(fast.png_level, "fast")
    assert.equal(fast.width, rendered.width)
    assert.ok(fast.png.byteLength > rendered.png.byteLength)
  })

  await t.test("a disposed workbook refuses further work", async () => {
    const scratch = await local.open(bytes)
    await scratch.dispose()
    assert.equal(scratch.disposed, true)
    await scratch.dispose()
    await assert.rejects(() => scratch.listSheets(), (error) => error instanceof CapabilityError)
  })
})

test("worker mode drives the real WASM package off the main thread", {
  skip: generatedPackage ? false : "set AGENT_SPREADSHEET_WASM_PACKAGE to run the local integration"
}, async (t) => {
  // The runtime is named, not passed: a live bindings object cannot cross the
  // worker boundary, so worker mode takes the module the worker should import.
  const local = createLocalSpreadsheet({
    runtime: { module: generatedPackage },
    worker: true
  })
  assert.ok((await local.capabilities()).includes("screenshot_sheet"))
  const workbook = await local.open(fs.readFileSync(fixture))
  // Registration order matters: the session must be released before the worker
  // that owns it goes away.
  t.after(() => workbook.dispose())
  t.after(() => local.close())
  assert.equal(local.worker, true)

  const sheets = await workbook.listSheets()
  assert.equal(sheets.operation, "list_sheets")

  const rendered = await workbook.renderSheet({ sheet_name: "Sheet1", range: "A1:C6" })
  assert.deepEqual(Array.from(rendered.png.subarray(0, 4)), [0x89, 0x50, 0x4e, 0x47])
  assert.equal(rendered.renderer, "native-raster/1")

  await assert.rejects(
    () => workbook.write({
      expected_revision: "0".repeat(64),
      mode: "apply",
      ops: [{
        kind: "set_cells",
        sheet_name: "Sheet1",
        cells: { A1: { kind: "value", value: "stale" } }
      }]
    }),
    (error) => {
      assert.ok(error instanceof CanonicalOperationError, `${error}`)
      assert.equal(error.code, "REVISION_CONFLICT")
      return true
    }
  )
})
