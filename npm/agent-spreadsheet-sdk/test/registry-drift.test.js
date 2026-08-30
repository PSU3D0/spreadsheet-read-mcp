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

test("checked-in canonical registry has not drifted from asp registry --all", (t) => {
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
