#!/usr/bin/env node

const fs = require("node:fs")
const http = require("node:http")
const path = require("node:path")
const { chromium } = require("playwright-core")

const packageRoot = path.resolve(__dirname, "..")
const repositoryRoot = path.resolve(packageRoot, "..", "..")
const wasmDirectory = path.join(repositoryRoot, "target", "sdk-wasm-web")
const fixturePath = path.join(
  repositoryRoot,
  "crates", "agent-spreadsheet", "tests", "fixtures", "f1", "baseline.xlsx"
)
const chromeBinary = process.env.CHROME_BIN
const timeoutMs = 60_000

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function contentType(filePath) {
  if (filePath.endsWith(".js")) return "text/javascript; charset=utf-8"
  if (filePath.endsWith(".wasm")) return "application/wasm"
  if (filePath.endsWith(".xlsx")) {
    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
  }
  return "application/octet-stream"
}

function sendFile(response, filePath) {
  const body = fs.readFileSync(filePath)
  response.writeHead(200, {
    "Content-Type": contentType(filePath),
    "Content-Length": body.byteLength,
    "Cache-Control": "no-store"
  })
  response.end(body)
}

function createServer() {
  return http.createServer((request, response) => {
    try {
      const pathname = new URL(request.url, "http://localhost").pathname
      if (pathname === "/") {
        const body = "<!doctype html><meta charset=utf-8><title>agent-spreadsheet WASM browser test</title>"
        response.writeHead(200, {
          "Content-Type": "text/html; charset=utf-8",
          "Content-Length": Buffer.byteLength(body),
          "Cache-Control": "no-store"
        })
        response.end(body)
        return
      }
      if (pathname === "/favicon.ico") {
        response.writeHead(204).end()
        return
      }
      if (pathname === "/fixture.xlsx") {
        sendFile(response, fixturePath)
        return
      }
      if (pathname.startsWith("/wasm/")) {
        const name = pathname.slice("/wasm/".length)
        assert(/^[A-Za-z0-9_.-]+$/.test(name), "invalid WASM package path")
        const filePath = path.join(wasmDirectory, name)
        assert(fs.statSync(filePath).isFile(), `missing WASM package file: ${name}`)
        sendFile(response, filePath)
        return
      }
      response.writeHead(404).end("not found")
    } catch (error) {
      response.writeHead(500).end(error.message)
    }
  })
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject)
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject)
      resolve(server.address())
    })
  })
}

function close(server) {
  if (!server.listening) return Promise.resolve()
  server.closeAllConnections?.()
  return new Promise((resolve, reject) => {
    server.close((error) => error ? reject(error) : resolve())
  })
}

async function main() {
  assert(chromeBinary, "CHROME_BIN must point to the Chrome executable")
  assert(fs.existsSync(chromeBinary), `CHROME_BIN does not exist: ${chromeBinary}`)
  assert(fs.existsSync(path.join(wasmDirectory, "agent_spreadsheet_wasm.js")),
    `missing generated web package at ${wasmDirectory}; run wasm-pack build first`)
  assert(fs.existsSync(fixturePath), `missing workbook fixture: ${fixturePath}`)

  const server = createServer()
  let browser
  let timer
  try {
    const address = await listen(server)
    const origin = `http://127.0.0.1:${address.port}`
    browser = await chromium.launch({
      executablePath: chromeBinary,
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox"]
    })
    const page = await browser.newPage()
    page.setDefaultTimeout(30_000)

    const runtimeErrors = []
    let rejectRuntimeError
    const runtimeError = new Promise((_, reject) => { rejectRuntimeError = reject })
    const failRuntime = (message) => {
      const error = new Error(message)
      runtimeErrors.push(error)
      rejectRuntimeError(error)
    }
    page.on("console", (message) => {
      if (message.type() === "error") failRuntime(`browser console error: ${message.text()}`)
    })
    page.on("pageerror", (error) => failRuntime(`uncaught page error: ${error.message}`))
    page.on("crash", () => failRuntime("browser page crashed"))

    await page.goto(origin, { waitUntil: "load" })
    const timedOut = new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`browser test timed out after ${timeoutMs}ms`)), timeoutMs)
    })
    const result = await Promise.race([
      page.evaluate(async () => {
        const check = (condition, message) => {
          if (!condition) throw new Error(message)
        }
        const bindings = await import("/wasm/agent_spreadsheet_wasm.js")
        await bindings.default()

        const fixtureResponse = await fetch("/fixture.xlsx")
        check(fixtureResponse.ok, `fixture fetch failed: ${fixtureResponse.status}`)
        const fixture = new Uint8Array(await fixtureResponse.arrayBuffer())
        const operations = JSON.parse(bindings.operations())
        const operationNames = operations.map((descriptor) => descriptor.name)
        for (const required of ["list_sheets", "read_cells", "write"]) {
          check(operationNames.includes(required), `WASM discovery omitted ${required}`)
        }

        const execute = async (resourceId, operation, params) => {
          const response = JSON.parse(await bindings.executeOperation(
            resourceId,
            operation,
            JSON.stringify(params)
          ))
          check(response.schema_version === "1", `${operation} returned the wrong schema version`)
          check(response.operation === operation, `${operation} returned the wrong operation name`)
          check(response.resource_id === resourceId, `${operation} returned the wrong resource ID`)
          return response
        }

        const sessions = []
        let recalculation = "not-advertised"
        try {
          const resourceId = bindings.createSession(fixture)
          sessions.push(resourceId)
          check(/^session:/.test(resourceId), `createSession returned an untyped ID: ${resourceId}`)

          const listed = await execute(resourceId, "list_sheets", {})
          check(Array.isArray(listed.data?.sheets), "list_sheets omitted data.sheets")
          const read = await execute(resourceId, "read_cells", {
            sheet_name: "Sheet1",
            selection: { kind: "range", ranges: ["A1:B2"] },
            format: "dense"
          })
          const written = await execute(resourceId, "write", {
            expected_revision: read.revision_id,
            mode: "apply",
            ops: [{
              kind: "set_cells",
              sheet_name: "Sheet1",
              cells: { A1: { kind: "value", value: "sdk-browser" } }
            }]
          })
          check(written.data?.status === "applied", "canonical write was not applied")

          const exported = bindings.exportWorkbook(resourceId)
          check(exported instanceof Uint8Array && exported.byteLength > 0,
            "exportWorkbook did not return workbook bytes")
          const reboundId = bindings.createSession(exported)
          sessions.push(reboundId)
          check(/^session:/.test(reboundId), `rebind returned an untyped ID: ${reboundId}`)
          const rebound = await execute(reboundId, "read_cells", {
            sheet_name: "Sheet1",
            selection: { kind: "range", ranges: ["A1"] },
            format: "dense"
          })
          check(JSON.stringify(rebound.data).includes("sdk-browser"),
            "exported and rebound workbook did not preserve the canonical write")

          if (operationNames.includes("recalculate")) {
            const recalculated = await execute(reboundId, "recalculate", {
              expected_revision: rebound.revision_id,
              backend: "formualizer"
            })
            check(recalculated.data?.evaluation_coverage?.source === "formualizer",
              "Formualizer recalculation omitted Formualizer coverage")
            recalculation = recalculated.data.evaluation_coverage.source
          }
        } finally {
          for (const resourceId of sessions.reverse()) {
            check(bindings.disposeSession(resourceId) === true, `failed to dispose ${resourceId}`)
          }
        }

        return { operationCount: operationNames.length, recalculation }
      }),
      runtimeError,
      timedOut
    ])
    clearTimeout(timer)
    if (runtimeErrors.length > 0) throw runtimeErrors[0]
    console.log(`Browser WASM integration passed (${result.operationCount} operations, recalculation: ${result.recalculation})`)
  } finally {
    clearTimeout(timer)
    if (browser) await browser.close()
    await close(server)
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
