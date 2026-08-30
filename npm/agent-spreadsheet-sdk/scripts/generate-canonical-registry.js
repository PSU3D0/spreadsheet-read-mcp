#!/usr/bin/env node

const { execFileSync } = require("node:child_process")
const fs = require("node:fs")
const path = require("node:path")

const packageRoot = path.resolve(__dirname, "..")
const outputPath = path.join(packageRoot, "src", "generated", "canonical-registry.json")
const asp = process.env.ASP_BINARY || process.argv[2] || path.resolve(packageRoot, "..", "..", "target", "debug", "asp")

const manifest = JSON.parse(execFileSync(asp, ["registry", "--all"], { encoding: "utf8" }))
manifest.generated_by = "asp registry --all"

fs.mkdirSync(path.dirname(outputPath), { recursive: true })
fs.writeFileSync(outputPath, `${JSON.stringify(manifest, null, 2)}\n`)
console.log(`wrote ${manifest.operations.length} operations to ${outputPath}`)
