const test = require("node:test")
const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")

const { WasmBackend, CANONICAL_REGISTRY } = require("../src")

const generatedPackage = process.env.AGENT_SPREADSHEET_WASM_PACKAGE

test("SDK drives the generated wasm-bindgen Node package", {
  skip: generatedPackage ? false : "run npm run test:wasm to build the generated Node package"
}, async () => {
  const bindings = require(generatedPackage)
  const backend = new WasmBackend({ bindings })
  const expectedOperations = CANONICAL_REGISTRY.operations
    .filter((descriptor) => descriptor.adapters.wasm.support_status === "supported")
    .map((descriptor) => descriptor.name)
  assert.deepEqual(backend.getCapabilities().operations, expectedOperations)

  const fixture = path.resolve(
    __dirname,
    "..", "..", "..",
    "crates", "agent-spreadsheet", "tests", "fixtures", "f1", "baseline.xlsx"
  )
  const workbookBytes = fs.readFileSync(fixture)
  const resourceId = await backend.createSession({ workbookBytes })
  const baselineResourceId = await backend.createSession({ workbookBytes })
  assert.match(resourceId, /^session:/)

  try {
    const read = await backend.readCells({
      resource_id: resourceId,
      sheet_name: "Sheet1",
      selection: { kind: "range", ranges: ["A1:B2"] },
      format: "dense"
    })
    assert.equal(read.operation, "read_cells")
    assert.equal(read.resource_id, resourceId)

    const written = await backend.write({
      resource_id: resourceId,
      expected_revision: read.revision_id,
      mode: "apply",
      ops: [{
        kind: "set_cells",
        sheet_name: "Sheet1",
        cells: { A1: { kind: "value", value: "sdk-wasm" } }
      }]
    })
    assert.equal(written.data.status, "applied")
    assert.equal(written.resource_id, resourceId)

    const recalculated = await backend.recalculate({
      resource_id: resourceId,
      expected_revision: written.revision_id,
      backend: "formualizer"
    })
    assert.equal(recalculated.operation, "recalculate")
    assert.equal(recalculated.resource_id, resourceId)
    assert.equal(recalculated.data.evaluation_coverage.source, "formualizer")

    const verified = await backend.verifyWorkbook({
      resource_id: resourceId,
      baseline_resource_id: baselineResourceId,
      targets: ["Sheet1!A1"],
      targets_only: true
    })
    assert.equal(verified.operation, "verify_workbook")
    assert.equal(verified.resource_id, resourceId)
    assert.equal(verified.data.proof_status, "differences_found")

    const exported = await backend.exportWorkbook({ resource_id: resourceId })
    assert.ok(exported instanceof Uint8Array)
    assert.ok(exported.byteLength > 0)
  } finally {
    assert.equal(await backend.disposeSession({ resource_id: baselineResourceId }), true)
    assert.equal(await backend.disposeSession({ resource_id: resourceId }), true)
  }
})
