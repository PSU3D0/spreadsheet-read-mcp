#!/usr/bin/env node
// Repository-only harness: build the wasm-bindgen Node package with wasm-pack and run
// the integration tests that need a real WASM runtime. Pass --rebuild to force a fresh
// build; otherwise an existing target/sdk-wasm-node is reused.

const { spawnSync } = require("node:child_process")
const fs = require("node:fs")
const path = require("node:path")

const packageRoot = path.resolve(__dirname, "..")
const repositoryRoot = path.resolve(packageRoot, "..", "..")
const outputDirectory = process.env.AGENT_SPREADSHEET_WASM_PACKAGE ||
  path.join(repositoryRoot, "target", "sdk-wasm-node")
const rebuild = process.argv.includes("--rebuild")

if (rebuild || !fs.existsSync(path.join(outputDirectory, "package.json"))) {
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
} else {
  console.log(`reusing ${outputDirectory} (pass --rebuild to rebuild)`)
}

const test = spawnSync(process.execPath, [
  "--test",
  path.join(packageRoot, "test", "local-runtime.integration.test.js"),
  path.join(packageRoot, "test", "just-bash-integration.test.js")
], {
  cwd: packageRoot,
  env: { ...process.env, AGENT_SPREADSHEET_WASM_PACKAGE: outputDirectory },
  encoding: "utf8",
  stdio: "inherit"
})
if (test.error) throw test.error
process.exit(test.status || 0)
