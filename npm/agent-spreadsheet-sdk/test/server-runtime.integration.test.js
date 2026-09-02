// Server runtime integration against a spawned agent-spreadsheet-mcp process.
//
//   SPREADSHEET_MCP_BINARY=../../target/debug/agent-spreadsheet-mcp node --test \
//     test/server-runtime.integration.test.js
//
// The server's fork registry writes outside the repository, so run this without a
// filesystem sandbox. The test skips cleanly when SPREADSHEET_MCP_BINARY is absent.

const test = require("node:test")
const assert = require("node:assert/strict")
const { spawn } = require("node:child_process")
const fs = require("node:fs")
const net = require("node:net")
const os = require("node:os")
const path = require("node:path")

const {
  CanonicalOperationError,
  CapabilityError,
  connectSpreadsheetServer
} = require("agent-spreadsheet-sdk")

const binary = process.env.SPREADSHEET_MCP_BINARY
const fixture = path.resolve(
  __dirname, "..", "..", "..",
  "crates", "agent-spreadsheet", "tests", "fixtures", "f1", "baseline.xlsx"
)

function freePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer()
    probe.on("error", reject)
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address()
      probe.close(() => resolve(port))
    })
  })
}

async function waitForRoute(baseUrl, deadlineMs = 30_000) {
  const started = Date.now()
  for (;;) {
    try {
      const response = await fetch(`${baseUrl}/v1/operations`)
      if (response.ok) return
    } catch {
      // not listening yet
    }
    if (Date.now() - started > deadlineMs) throw new Error(`${baseUrl} never came up`)
    await new Promise((resolve) => setTimeout(resolve, 150))
  }
}

test("server runtime drives a live canonical /v1 route", {
  skip: binary && fs.existsSync(binary)
    ? false
    : "set SPREADSHEET_MCP_BINARY to the built agent-spreadsheet-mcp binary"
}, async (t) => {
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "asp-sdk-server-"))
  fs.copyFileSync(fixture, path.join(workspace, "baseline.xlsx"))
  const port = await freePort()
  const child = spawn(binary, [], {
    env: {
      ...process.env,
      SPREADSHEET_MCP_WORKSPACE: workspace,
      SPREADSHEET_MCP_TRANSPORT: "http",
      SPREADSHEET_MCP_HTTP_BIND: `127.0.0.1:${port}`,
      SPREADSHEET_MCP_RECALC_ENABLED: "true"
    },
    stdio: ["ignore", "pipe", "pipe"]
  })
  t.after(() => {
    child.kill("SIGTERM")
    fs.rmSync(workspace, { recursive: true, force: true })
  })

  const baseUrl = `http://127.0.0.1:${port}`
  await waitForRoute(baseUrl)
  const client = connectSpreadsheetServer({ baseUrl })

  const capabilities = await client.capabilities()
  assert.ok(capabilities.includes("create_fork"))
  assert.ok(capabilities.includes("read_cells"))

  const listed = await client.listWorkbooks({})
  assert.equal(listed.operation, "list_workbooks")
  const [descriptor] = listed.data.workbooks
  assert.match(descriptor.resource_id, /^wb:/)

  const workbook = client.workbook(descriptor.resource_id)

  await t.test("the read surface returns whole envelopes", async () => {
    const described = await workbook.describe()
    assert.equal(described.operation, "describe_workbook")
    assert.equal(described.resource_id, workbook.resourceId)
    assert.equal(typeof described.revision_id, "string")
    assert.equal(workbook.revisionId, described.revision_id)

    const sheets = await workbook.listSheets()
    assert.ok(Array.isArray(sheets.data.sheets))
  })

  await t.test("a fork writes, reads back, and reports its changes", async () => {
    const fork = await workbook.createFork()
    assert.match(fork.resourceId, /^fork:/)

    const written = await fork.write({
      mode: "apply",
      ops: [{
        kind: "set_cells",
        sheet_name: "Sheet1",
        cells: { A1: { kind: "value", value: "e" } }
      }]
    })
    assert.equal(written.data.status, "applied")
    assert.equal(fork.revisionId, written.revision_id)

    const read = await fork.readCells({
      sheet_name: "Sheet1",
      selection: { kind: "range", ranges: ["A1:A1"] },
      format: "dense"
    })
    assert.equal(read.operation, "read_cells")
    assert.match(JSON.stringify(read.data), /"e"/)

    const changes = await fork.getChanges({ view: { kind: "operations" } })
    assert.equal(changes.operation, "get_changes")

    const created = await fork.checkpoint({ action: "create", label: "after-write" })
    assert.equal(created.operation, "checkpoint")
    const listedCheckpoints = await fork.checkpoint({ action: "list" })
    assert.ok(JSON.stringify(listedCheckpoints.data).includes("after-write"))

    const verified = await fork.verifyAgainst(workbook, {
      targets: ["Sheet1!A1"],
      targets_only: true
    })
    assert.equal(verified.operation, "verify_workbook")

    const discarded = await fork.discard()
    assert.equal(discarded.operation, "discard_fork")
    assert.equal(fork.discarded, true)
  })

  await t.test("a stale revision maps 409 onto CanonicalOperationError", async () => {
    const fork = await workbook.createFork()
    await assert.rejects(
      fork.write({
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
        assert.equal(error.details.status, 409)
        assert.equal(error.details.status, error.canonicalStatus)
        return true
      }
    )
    await fork[Symbol.asyncDispose]()
  })

  await t.test("an unknown resource maps 404 onto CanonicalOperationError", async () => {
    await assert.rejects(
      client.workbook("wb:wb-does-not-exist").describe(),
      (error) => {
        assert.ok(error instanceof CanonicalOperationError, `${error}`)
        assert.equal(error.code, "RESOURCE_NOT_FOUND")
        assert.equal(error.details.status, 404)
        return true
      }
    )
  })

  await t.test("operations the live process does not serve fail before transport", async () => {
    assert.equal(capabilities.includes("inspect_vba"), false)
    await assert.rejects(
      () => client.canonical.execute("inspect_vba", { resource_id: workbook.resourceId }),
      (error) => error instanceof CapabilityError && error.operation === "inspect_vba"
    )
  })
})
