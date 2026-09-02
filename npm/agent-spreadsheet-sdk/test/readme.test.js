// Executes every JavaScript block in README.md. An example that references an
// undefined variable, calls a method that does not exist, or passes an input the
// runtime rejects fails here.

const test = require("node:test")
const assert = require("node:assert/strict")
const fs = require("node:fs")
const os = require("node:os")
const path = require("node:path")
const Module = require("node:module")

const { createFakeFetch } = require("../test-support/fake-fetch.js")

const packageRoot = path.resolve(__dirname, "..")
const readme = fs.readFileSync(path.join(packageRoot, "README.md"), "utf8")

function javascriptBlocks(markdown) {
  const blocks = []
  const pattern = /```js\n([\s\S]*?)```/g
  let match
  while ((match = pattern.exec(markdown)) !== null) {
    const line = markdown.slice(0, match.index).split("\n").length
    blocks.push({ line, source: match[1] })
  }
  return blocks
}

const ALIASES = {
  "agent-spreadsheet-sdk": path.join(packageRoot, "dist", "cjs", "index.js"),
  "agent-spreadsheet-sdk/just-bash": path.join(packageRoot, "dist", "cjs", "just-bash", "index.js"),
  "agent-spreadsheet-wasm": path.join(packageRoot, "test-support", "fake-wasm-package.js"),
  "just-bash": require.resolve("just-bash")
}

test("every README JavaScript block runs", async (t) => {
  const blocks = javascriptBlocks(readme)
  assert.ok(blocks.length >= 10, `expected the README examples to survive, found ${blocks.length}`)

  const workdir = fs.mkdtempSync(path.join(os.tmpdir(), "asp-readme-"))
  fs.writeFileSync(path.join(workdir, "book.xlsx"), Uint8Array.from([80, 75, 3, 4]))
  // Worker mode imports the runtime by name inside the worker, where this
  // process's module aliases do not apply, so the stand-in is installed into the
  // example working directory as a real package.
  const shim = path.join(workdir, "node_modules", "agent-spreadsheet-wasm")
  fs.mkdirSync(shim, { recursive: true })
  fs.writeFileSync(
    path.join(shim, "package.json"),
    JSON.stringify({ name: "agent-spreadsheet-wasm", version: "0.0.0", main: "index.js" })
  )
  fs.writeFileSync(
    path.join(shim, "index.js"),
    `module.exports = require(${JSON.stringify(ALIASES["agent-spreadsheet-wasm"])})\n`
  )

  const resolve = Module._resolveFilename
  const cwd = process.cwd()
  const realFetch = globalThis.fetch
  const realLog = console.log
  globalThis.fetch = createFakeFetch().fetch
  console.log = () => {}
  Module._resolveFilename = function (request, ...rest) {
    if (Object.hasOwn(ALIASES, request)) return ALIASES[request]
    return resolve.call(this, request, ...rest)
  }
  process.chdir(workdir)

  t.after(() => {
    Module._resolveFilename = resolve
    globalThis.fetch = realFetch
    console.log = realLog
    process.chdir(cwd)
    fs.rmSync(workdir, { recursive: true, force: true })
  })

  for (const [index, block] of blocks.entries()) {
    const file = path.join(workdir, `readme-${index}.js`)
    fs.writeFileSync(file, `module.exports = (async () => {\n${block.source}\n})()\n`)
    await t.test(`README.md line ${block.line}`, async () => {
      await require(file)
    })
  }
})
