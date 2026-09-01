#!/usr/bin/env node

const { spawnSync } = require("node:child_process")
const path = require("node:path")

const REPO_ROOT = path.resolve(__dirname, "..")
const SDK_TEST_PATH = path.join(REPO_ROOT, "npm/agent-spreadsheet-sdk/test/compat/backend.test.js")

const CHECKS = [
  {
    id: "basic-read-parity",
    surfaces: ["core", "wasm", "mcp", "sdk"],
    status: "automated",
    note: "Deprecated 0.14 backend shared-method parity (agent-spreadsheet-sdk/compat)",
    runner: "sdk-backend-tests"
  },
  {
    id: "sheet-page-parity",
    surfaces: ["core", "wasm", "mcp", "sdk"],
    status: "automated",
    note: "sheetPage behavior wired through shared SDK backend surface",
    runner: "sdk-backend-tests"
  },
  {
    id: "capability-divergence-guards",
    surfaces: ["wasm", "mcp", "sdk"],
    status: "automated",
    note: "Typed capability misuse errors for unsupported backend-specific flows",
    runner: "sdk-backend-tests"
  },
  {
    id: "formula-diagnostics-parity",
    surfaces: ["core", "wasm", "mcp"],
    status: "planned",
    note: "Shared fixtures for off|warn|fail formula parse policy"
  },
  {
    id: "grid-roundtrip-parity",
    surfaces: ["core", "wasm", "mcp"],
    status: "planned",
    note: "Export/import roundtrip parity for values/formulas/styles/merges/column widths"
  },
  {
    id: "csv-edge-case-parity",
    surfaces: ["core", "wasm", "mcp"],
    status: "planned",
    note: "CSV import fixtures for quotes, CRLF, embedded newlines, blanks"
  }
]

const RUNNERS = {
  "sdk-backend-tests": {
    command: "node",
    args: ["--test", SDK_TEST_PATH],
    cwd: REPO_ROOT
  }
}

function printChecks() {
  for (const check of CHECKS) {
    const surfaces = check.surfaces.join(",")
    const runner = check.runner || "-"
    console.log(`${check.id}\t${check.status}\t${surfaces}\t${runner}\t${check.note}`)
  }
}

function runAutomated() {
  const automated = CHECKS.filter((check) => check.status === "automated" && check.runner)
  const runnerIds = [...new Set(automated.map((check) => check.runner))]
  let failed = false

  for (const runnerId of runnerIds) {
    const runner = RUNNERS[runnerId]
    if (!runner) {
      failed = true
      console.error(`[parity-harness] unknown runner: ${runnerId}`)
      continue
    }

    const covered = automated.filter((check) => check.runner === runnerId).map((check) => check.id)
    console.log(`\n[parity-harness] ${runnerId} covers: ${covered.join(", ")}`)
    console.log(`[parity-harness] ${runner.command} ${runner.args.join(" ")}`)

    const result = spawnSync(runner.command, runner.args, {
      stdio: "inherit",
      cwd: runner.cwd || process.cwd()
    })
    if (result.status !== 0) {
      failed = true
      console.error(`[parity-harness] failed runner: ${runnerId}`)
    }
  }

  if (failed) {
    process.exit(1)
  }

  console.log("\n[parity-harness] automated checks passed")
}

const args = new Set(process.argv.slice(2))
if (args.has("--list")) {
  printChecks()
  process.exit(0)
}

runAutomated()
