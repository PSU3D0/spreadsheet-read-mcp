const test = require("node:test")
const assert = require("node:assert/strict")
const { execFileSync } = require("node:child_process")
const fs = require("node:fs")
const path = require("node:path")

const manifest = require("../src/generated/canonical-registry.json")

function aspBinary() {
  const candidates = [
    process.env.ASP_BINARY,
    path.resolve(__dirname, "..", "..", "..", "target", "debug", "asp"),
    path.resolve(__dirname, "..", "..", "..", "target", "release", "asp")
  ].filter(Boolean)
  return candidates.find((candidate) => fs.existsSync(candidate))
}

function run(asp, args) {
  return JSON.parse(execFileSync(asp, args, { encoding: "utf8" }))
}

test("checked-in host-independent registry has all 31 asp operations", (t) => {
  const asp = aspBinary()
  if (!asp) {
    t.skip("set ASP_BINARY (or build target/debug/asp) to run registry drift verification")
    return
  }

  const actual = run(asp, ["registry", "--all"])
  const { generated_by: _generatedBy, ...checkedIn } = manifest
  assert.deepEqual(checkedIn, actual, "run npm run generate:registry after registry changes")
  assert.equal(actual.operations.length, 31)

  for (const descriptor of actual.operations) {
    const schema = run(asp, ["schema", descriptor.name])
    assert.deepEqual(descriptor.input_schema, schema.input_schema, `${descriptor.name} input schema drifted`)
    assert.deepEqual(descriptor.output_schema, schema.output_schema, `${descriptor.name} output schema drifted`)
    assert.deepEqual(actual.error_schema, schema.error_schema, `${descriptor.name} error schema drifted`)
  }
})

const EXPECTED_WASM_OPERATIONS = [
  "describe_workbook",
  "list_sheets",
  "sheet_overview",
  "read_cells",
  "inspect_cells",
  "read_table",
  "read_layout",
  "export_grid",
  "named_ranges",
  "analyze_styles",
  "search_values",
  "search_formulas",
  "formula_trace",
  "formula_map",
  "profile_table",
  "sheet_statistics",
  "write",
  "recalculate",
  "verify_workbook"
]

function supported(adapter) {
  return manifest.operations
    .filter((descriptor) => descriptor.adapters[adapter].support_status === "supported")
    .map((descriptor) => descriptor.name)
}

test("checked-in MCP adapter subset remains intentional", () => {
  assert.deepEqual(supported("mcp"), manifest.operations.map(({ name }) => name))
  assert.equal(supported("mcp").length, 31)
})

test("checked-in WASM adapter subset remains intentional", () => {
  assert.deepEqual(supported("wasm"), EXPECTED_WASM_OPERATIONS)
})
