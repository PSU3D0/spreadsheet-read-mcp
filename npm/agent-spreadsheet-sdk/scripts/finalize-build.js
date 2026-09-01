#!/usr/bin/env node
// Pins the module format of each build output so Node resolves dist/cjs as CommonJS
// and dist/esm as ESM regardless of the package's own "type".

const fs = require("node:fs")
const path = require("node:path")

const dist = path.resolve(__dirname, "..", "dist")
for (const [directory, type] of [["cjs", "commonjs"], ["esm", "module"]]) {
  const target = path.join(dist, directory)
  if (!fs.existsSync(target)) throw new Error(`missing build output: dist/${directory}`)
  fs.writeFileSync(path.join(target, "package.json"), `${JSON.stringify({ type }, null, 2)}\n`)
}
console.log("pinned dist/cjs (commonjs) and dist/esm (module)")
