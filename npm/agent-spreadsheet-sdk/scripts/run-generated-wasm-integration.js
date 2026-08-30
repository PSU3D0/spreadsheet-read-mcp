#!/usr/bin/env node

const { spawnSync } = require("node:child_process")
const fs = require("node:fs")
const path = require("node:path")

const packageRoot = path.resolve(__dirname, "..")
const repositoryRoot = path.resolve(packageRoot, "..", "..")
const outputDirectory = path.join(repositoryRoot, "target", "sdk-wasm-node")

fs.rmSync(outputDirectory, { recursive: true, force: true })
const build = spawnSync("wasm-pack", [
  "build",
  "crates/agent-spreadsheet-wasm",
  "--target", "nodejs",
  "--dev",
  "--out-dir", outputDirectory
], {
  cwd: repositoryRoot,
  encoding: "utf8",
  stdio: "inherit"
})
if (build.error) throw build.error
if (build.status !== 0) process.exit(build.status || 1)

const test = spawnSync(process.execPath, [
  "--test",
  path.join(packageRoot, "test", "wasm-integration.test.js"),
  path.join(packageRoot, "test", "just-bash-integration.test.js")
], {
  cwd: packageRoot,
  env: { ...process.env, AGENT_SPREADSHEET_WASM_PACKAGE: outputDirectory },
  encoding: "utf8",
  stdio: "inherit"
})
if (test.error) throw test.error
process.exit(test.status || 0)
