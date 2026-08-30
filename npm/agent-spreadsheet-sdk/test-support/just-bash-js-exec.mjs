import { createRequire } from "node:module"
import { Bash } from "just-bash"

const require = createRequire(import.meta.url)
const { createAspCommand } = require("agent-spreadsheet-sdk/just-bash")

let sequence = 0
const bindings = {
  operations: () => ["list_sheets"],
  createSession: () => `session:s${++sequence}`,
  executeOperation: (resourceId, operation) => JSON.stringify({
    schema_version: "1",
    operation,
    resource_id: resourceId,
    revision_id: "rev-1",
    data: { sheets: [{ name: "Sheet1" }] }
  }),
  disposeSession: () => true
}
const source = [
  "const { execSync } = require('child_process');",
  "const out = execSync(\"asp op list_sheets --bind /book.xlsx --json '{}'\");",
  "console.log(out.toString());"
].join("\n")
const bash = new Bash({
  files: {
    "/book.xlsx": Uint8Array.from([1, 2, 3]),
    "/invoke.js": source
  },
  javascript: true,
  customCommands: [createAspCommand({ bindings })]
})
const result = await bash.exec("js-exec /invoke.js")
if (result.stdout) process.stdout.write(result.stdout)
if (result.stderr) process.stderr.write(result.stderr)
process.exit(result.exitCode)
