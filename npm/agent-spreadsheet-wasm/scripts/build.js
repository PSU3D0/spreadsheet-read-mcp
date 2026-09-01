#!/usr/bin/env node

// Regenerate npm/agent-spreadsheet-wasm/pkg from the Rust crate.
//
// The published package ships the `--target web` wasm-bindgen output because it
// is the only target that works unchanged in browsers, bundlers and Node 18+
// once the loader supplies the bytes itself.

import { spawnSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const repositoryRoot = path.resolve(packageRoot, "..", "..")
const outDir = path.join(packageRoot, "pkg")

const result = spawnSync(process.execPath, [
  path.join(repositoryRoot, "scripts", "build-wasm-package.js"),
  "--target", "web",
  "--profile", process.env.AGENT_SPREADSHEET_WASM_PROFILE || "wasm-release",
  "--out-dir", outDir
], { cwd: repositoryRoot, stdio: "inherit" })
if (result.error) throw result.error
if (result.status !== 0) process.exit(result.status || 1)

// wasm-pack writes an npm package manifest of its own; this directory is only
// the asset payload for the hand-written package at the parent level.
for (const stray of ["package.json", "README.md", ".gitignore"]) {
  fs.rmSync(path.join(outDir, stray), { force: true })
}

console.log(`wrote ${path.relative(repositoryRoot, outDir)}`)
