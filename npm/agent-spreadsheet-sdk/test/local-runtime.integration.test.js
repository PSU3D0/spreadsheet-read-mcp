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
    await assert.rejects(
      () => workbook.renderSheet({ sheet_name: "Sheet1" }),
      (error) => error instanceof CapabilityError
    )
  })

  await t.test("a disposed workbook refuses further work", async () => {
    const scratch = await local.open(bytes)
    await scratch.dispose()
    assert.equal(scratch.disposed, true)
    await scratch.dispose()
    await assert.rejects(() => scratch.listSheets(), (error) => error instanceof CapabilityError)
  })
})
