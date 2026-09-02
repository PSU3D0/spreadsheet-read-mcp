// The generated TypeScript artifacts are checked in. This regenerates them from
// src/generated/canonical-registry.json and compares, so a registry change that is
// not followed by `npm run generate:types` fails here rather than in a consumer.

const test = require("node:test")
const assert = require("node:assert/strict")
const { spawnSync } = require("node:child_process")
const fs = require("node:fs")
const path = require("node:path")

const packageRoot = path.resolve(__dirname, "..")
const registry = require("../src/generated/canonical-registry.json")

test("checked-in generated types match the registry", () => {
  const result = spawnSync(
    process.execPath,
    [path.join(packageRoot, "scripts", "generate-types.js"), "--check"],
    { cwd: packageRoot, encoding: "utf8" }
  )
  assert.equal(result.status, 0, `${result.stdout}${result.stderr}`)
})

test("every registry operation has generated input and output types", () => {
  const operations = fs.readFileSync(
    path.join(packageRoot, "src", "generated", "operations.ts"),
    "utf8"
  )
  for (const { name } of registry.operations) {
    assert.match(operations, new RegExp(`^  ${name}: In`, "m"), `${name} input entry`)
    assert.match(operations, new RegExp(`^  ${name}: Out`, "m"), `${name} output entry`)
  }
})

test("the generated read surface covers every single-read operation", () => {
  const surface = fs.readFileSync(
    path.join(packageRoot, "src", "generated", "read-surface.ts"),
    "utf8"
  )
  const reads = registry.operations
    .filter(({ adapters }) => adapters.mcp.binding_kind === "single_read")
    .map(({ name }) => name)
  assert.ok(reads.length >= 20)
  for (const name of reads) {
    const method = name.replace(/_([a-z0-9])/g, (_, letter) => letter.toUpperCase())
    assert.match(surface, new RegExp(`  ${method}\\(input`), `${name} method`)
  }
})
