const assert = require("node:assert/strict")
const { spawnSync } = require("node:child_process")
const fs = require("node:fs")
const os = require("node:os")
const path = require("node:path")
const test = require("node:test")

const { Bash } = require("just-bash")
const { createAspCommand } = require("agent-spreadsheet-sdk/just-bash")

const generatedPackage = process.env.AGENT_SPREADSHEET_WASM_PACKAGE
const asp = process.env.ASP_BINARY || path.resolve(__dirname, "..", "..", "..", "target", "debug", "asp")
const fixture = path.resolve(__dirname, "..", "..", "..", "crates", "agent-spreadsheet", "tests", "fixtures", "f1", "baseline.xlsx")
const partialFixture = path.resolve(path.dirname(fixture), "partial.xlsx")

function native(operation, options) {
  const args = ["op", operation, "--bind", options.bind, "--json", JSON.stringify(options.params)]
  if (options.baseline) args.push("--baseline", options.baseline)
  if (options.output) args.push("--output", options.output)
  if (options.inPlace) args.push("--in-place")
  const result = spawnSync(asp, args, { encoding: "utf8" })
  const stream = result.status === 0 || result.status === 2 ? result.stdout : result.stderr
  return { ...result, json: JSON.parse(stream) }
}

function scrub(value) {
  if (Array.isArray(value)) return value.map(scrub)
  if (!value || typeof value !== "object") return value
  return Object.fromEntries(Object.entries(value).map(([key, child]) => {
    if (key.includes("revision")) return [key, "<revision>"]
    if (key.endsWith("resource_id")) return [key, "<resource>"]
    if (key === "workbook_id") return [key, "<workbook>"]
    if (key === "max_payload_bytes") return [key, "<adapter-budget>"]
    if (key === "duration_ms") return [key, "<duration>"]
    return [key, scrub(child)]
  }))
}

async function execute(bash, operation, flags, params) {
  const result = await bash.exec(`asp op ${operation} ${flags}`, { stdin: JSON.stringify(params) })
  const stream = result.exitCode === 0 || result.exitCode === 2 ? result.stdout : result.stderr
  return { ...result, json: JSON.parse(stream) }
}

test("generated WASM just-bash adapter matches native canonical JSON and workbook-byte goldens", {
  skip: generatedPackage && fs.existsSync(asp) ? false : "run the generated WASM integration harness"
}, async () => {
  const bindings = require(generatedPackage)
  const baselineBytes = fs.readFileSync(fixture)
  const partialBytes = fs.readFileSync(partialFixture)
  const bash = new Bash({
    files: {
      "/vfs-only.xlsx": baselineBytes,
      "/partial.xlsx": partialBytes
    },
    customCommands: [createAspCommand({ bindings })]
  })
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "asp-just-bash-"))
  try {
    const nativeRead = native("read_cells", {
      bind: fixture,
      params: { sheet_name: "Sheet1", selection: { kind: "range", ranges: ["A1:B2"] } }
    })
    const adapterRead = await execute(
      bash,
      "read_cells",
      "--bind /vfs-only.xlsx",
      { sheet_name: "Sheet1", selection: { kind: "range", ranges: ["A1:B2"] } }
    )
    assert.equal(adapterRead.exitCode, 0)
    assert.match(adapterRead.json.resource_id, /^session:/)
    assert.match(nativeRead.json.resource_id, /^wb:/)
    assert.deepEqual(scrub(adapterRead.json), scrub(nativeRead.json))

    const op = {
      kind: "set_cells",
      sheet_name: "Sheet1",
      cells: { A1: { kind: "value", value: 77 } }
    }
    const adapterPreview = await execute(bash, "write", "--bind /vfs-only.xlsx", {
      expected_revision: adapterRead.json.revision_id,
      mode: "preview",
      ops: [op]
    })
    const nativePreview = native("write", {
      bind: fixture,
      params: {
        expected_revision: nativeRead.json.revision_id,
        mode: "preview",
        ops: [op]
      }
    })
    assert.deepEqual(scrub(adapterPreview.json), scrub(nativePreview.json))
    assert.deepEqual(Buffer.from(await bash.fs.readFileBuffer("/vfs-only.xlsx")), baselineBytes)

    const nativeAppliedPath = path.join(temp, "native-applied.xlsx")
    const adapterApply = await execute(bash, "write", "--bind /vfs-only.xlsx --output /applied.xlsx", {
      expected_revision: adapterRead.json.revision_id,
      mode: "apply",
      ops: [op]
    })
    const nativeApply = native("write", {
      bind: fixture,
      output: nativeAppliedPath,
      params: {
        expected_revision: nativeRead.json.revision_id,
        mode: "apply",
        ops: [op]
      }
    })
    assert.deepEqual(scrub(adapterApply.json), scrub(nativeApply.json))
    assert.deepEqual(
      Buffer.from(await bash.fs.readFileBuffer("/applied.xlsx")),
      fs.readFileSync(nativeAppliedPath)
    )

    const raceFlags = "--bind /vfs-only.xlsx --output /race.xlsx"
    const raceParams = {
      expected_revision: adapterRead.json.revision_id,
      mode: "apply",
      ops: [op]
    }
    const raced = await Promise.all([
      execute(bash, "write", raceFlags, raceParams),
      execute(bash, "write", raceFlags, raceParams)
    ])
    assert.deepEqual(raced.map(({ exitCode }) => exitCode).sort(), [0, 1])
    assert.equal(raced.find(({ exitCode }) => exitCode === 1).json.error.path, "--output")
    assert.deepEqual(
      Buffer.from(await bash.fs.readFileBuffer("/race.xlsx")),
      fs.readFileSync(nativeAppliedPath)
    )
    assert.equal((await bash.fs.getAllPaths()).some((entry) => entry.includes(".asp-tmp-")), false)

    const nativePartialRead = native("list_sheets", { bind: partialFixture, params: {} })
    const adapterPartialRead = await execute(bash, "list_sheets", "--bind /partial.xlsx", {})
    const nativeRecalcPath = path.join(temp, "native-recalc.xlsx")
    const nativeRecalc = native("recalculate", {
      bind: partialFixture,
      output: nativeRecalcPath,
      params: { expected_revision: nativePartialRead.json.revision_id }
    })
    const adapterRecalc = await execute(bash, "recalculate", "--bind /partial.xlsx --in-place", {
      expected_revision: adapterPartialRead.json.revision_id
    })
    assert.deepEqual(scrub(adapterRecalc.json), scrub(nativeRecalc.json))
    const adapterRecalcBytes = Buffer.from(await bash.fs.readFileBuffer("/partial.xlsx"))
    assert.notDeepEqual(adapterRecalcBytes, partialBytes)
    const adapterRecalcPath = path.join(temp, "adapter-recalc.xlsx")
    fs.writeFileSync(adapterRecalcPath, adapterRecalcBytes)
    const recalcReadParams = {
      sheet_name: "Sheet1",
      selection: { kind: "range", ranges: ["A1:C1"] }
    }
    assert.deepEqual(
      scrub(native("read_cells", { bind: adapterRecalcPath, params: recalcReadParams }).json),
      scrub(native("read_cells", { bind: nativeRecalcPath, params: recalcReadParams }).json)
    )

    const nativeVerify = native("verify_workbook", {
      bind: nativeAppliedPath,
      baseline: fixture,
      params: { targets: ["Sheet1!A1"], targets_only: true }
    })
    const adapterVerify = await execute(
      bash,
      "verify_workbook",
      "--bind /applied.xlsx --baseline /vfs-only.xlsx",
      { targets: ["Sheet1!A1"], targets_only: true }
    )
    for (const field of [
      "proof_status",
      "baseline_state",
      "current_state",
      "target_deltas",
      "summary",
      "new_errors",
      "resolved_errors",
      "preexisting_errors"
    ]) {
      assert.deepEqual(scrub(adapterVerify.json.data[field]), scrub(nativeVerify.json.data[field]))
    }
    assert.match(adapterVerify.json.data.current_resource_id, /^session:/)
    assert.match(adapterVerify.json.data.baseline_resource_id, /^session:/)

    // Rendering: the adapter writes PNG bytes into the VFS and the envelope
    // stays byte-identical to the native one, which writes the same artifact to
    // a real path.
    const nativeShotPath = path.join(temp, "native.png")
    const nativeShot = native("screenshot_sheet", {
      bind: fixture,
      output: nativeShotPath,
      params: { sheet_name: "Sheet1", range: "A1:C6" }
    })
    const adapterShot = await execute(
      bash,
      "screenshot_sheet",
      "--bind /vfs-only.xlsx --output /shot.png",
      { sheet_name: "Sheet1", range: "A1:C6" }
    )
    assert.equal(adapterShot.exitCode, 0)
    assert.deepEqual(scrub(adapterShot.json), scrub(nativeShot.json))
    assert.deepEqual(
      Buffer.from(await bash.fs.readFileBuffer("/shot.png")),
      fs.readFileSync(nativeShotPath)
    )
    assert.equal((await bash.fs.getAllPaths()).some((entry) => entry.includes(".asp-tmp-")), false)

    const nativeError = native("read_cells", {
      bind: fixture,
      params: { sheet_name: "Missing", selection: { kind: "range", ranges: ["A1"] } }
    })
    const adapterError = await execute(
      bash,
      "read_cells",
      "--bind /vfs-only.xlsx",
      { sheet_name: "Missing", selection: { kind: "range", ranges: ["A1"] } }
    )
    assert.deepEqual(adapterError.json, nativeError.json)

    const limited = new Bash({
      files: { "/only-in-vfs.xlsx": baselineBytes },
      customCommands: [createAspCommand({ bindings, maxWorkbookBytes: 1, maxParamsBytes: 1 })]
    })
    const oversizedParams = await limited.exec("asp op read_cells --bind /only-in-vfs.xlsx --json '{}'")
    assert.equal(JSON.parse(oversizedParams.stderr).error.code, "INVALID_REQUEST")
    const oversizedWorkbook = new Bash({
      files: { "/only-in-vfs.xlsx": baselineBytes },
      customCommands: [createAspCommand({ bindings, maxWorkbookBytes: 1 })]
    })
    const oversizedFile = await oversizedWorkbook.exec(
      "asp op read_cells --bind /only-in-vfs.xlsx --json '{}'"
    )
    assert.equal(JSON.parse(oversizedFile.stderr).error.path, "--bind")
  } finally {
    fs.rmSync(temp, { recursive: true, force: true })
  }
})
