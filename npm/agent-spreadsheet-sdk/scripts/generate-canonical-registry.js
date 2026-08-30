#!/usr/bin/env node

const { execFileSync } = require("node:child_process")
const fs = require("node:fs")
const path = require("node:path")

const packageRoot = path.resolve(__dirname, "..")
const outputPath = path.join(packageRoot, "src", "generated", "canonical-registry.json")
const asp = process.env.ASP_BINARY || process.argv[2] || path.resolve(packageRoot, "..", "..", "target", "debug", "asp")

function run(args) {
  return JSON.parse(execFileSync(asp, args, { encoding: "utf8" }))
}

const discovery = run(["operations"])
const operations = discovery.map((descriptor) => {
  const schema = run(["schema", descriptor.name])
  return {
    ...descriptor,
    input_schema: schema.input_schema,
    output_schema: schema.output_schema
  }
})

const manifest = {
  generated_by: "asp operations + asp schema <operation>",
  schema_version: operations[0]?.schema_version || "1",
  operations,
  error_schema: operations.length > 0
    ? run(["schema", operations[0].name]).error_schema
    : null
}

fs.mkdirSync(path.dirname(outputPath), { recursive: true })
fs.writeFileSync(outputPath, `${JSON.stringify(manifest, null, 2)}\n`)
console.log(`wrote ${operations.length} operations to ${outputPath}`)
