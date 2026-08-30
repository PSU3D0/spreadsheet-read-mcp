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

test("checked-in canonical registry has not drifted from asp operations/schema", (t) => {
  const asp = aspBinary()
  if (!asp) {
    t.skip("set ASP_BINARY (or build target/debug/asp) to run registry drift verification")
    return
  }

  const discovery = run(asp, ["operations"])
  const generatedDiscovery = manifest.operations.map(({ input_schema, output_schema, ...descriptor }) => descriptor)
  assert.deepEqual(generatedDiscovery, discovery, "run npm run generate:registry after registry changes")

  for (const descriptor of discovery) {
    const generated = manifest.operations.find(({ name }) => name === descriptor.name)
    const actual = run(asp, ["schema", descriptor.name])
    assert.deepEqual(generated.input_schema, actual.input_schema, `${descriptor.name} input schema drifted`)
    assert.deepEqual(generated.output_schema, actual.output_schema, `${descriptor.name} output schema drifted`)
    assert.deepEqual(manifest.error_schema, actual.error_schema, `${descriptor.name} error schema drifted`)
  }
})
