// WASM pixel goldens.
//
// The renderer compiles to wasm32 and the native goldens hash decoded pixels
// rather than PNG bytes, precisely so the same numbers can be checked from both
// sides. This harness renders every `agent-spreadsheet-render` fixture through
// the built WASM package and compares the pixel signature against
// crates/agent-spreadsheet-render/tests/goldens/pixels-sha256.txt.

const test = require("node:test")
const assert = require("node:assert/strict")
const crypto = require("node:crypto")
const fs = require("node:fs")
const path = require("node:path")
const zlib = require("node:zlib")

const { createLocalSpreadsheet } = require("agent-spreadsheet-sdk")

const repositoryRoot = path.resolve(__dirname, "..", "..", "..")
const defaultPackage = path.join(repositoryRoot, "target", "sdk-wasm-node")
const generatedPackage = process.env.AGENT_SPREADSHEET_WASM_PACKAGE ||
  (fs.existsSync(defaultPackage) ? defaultPackage : undefined)
const renderCrate = path.join(repositoryRoot, "crates", "agent-spreadsheet-render")
const goldenFile = path.join(renderCrate, "tests", "goldens", "pixels-sha256.txt")
// The golden window, mirroring `GOLDEN_RANGE` in tests/common/mod.rs.
const GOLDEN_RANGE = "A1:F12"

/** Parse `name <sha256> <width>x<height>` lines. */
function readGoldens() {
  return fs.readFileSync(goldenFile, "utf8")
    .split("\n")
    .filter((line) => line.trim() && !line.startsWith("#"))
    .map((line) => {
      const [name, hash, size] = line.trim().split(/\s+/)
      return { name, signature: `${hash} ${size}` }
    })
}

/**
 * Decode an 8-bit RGBA non-interlaced PNG to its raw pixel buffer.
 *
 * Node ships zlib but no image decoder, and the goldens are defined on the
 * decoded buffer, so the container is unwrapped here rather than hashed.
 */
function decodePixels(png) {
  assert.deepEqual(Array.from(png.subarray(0, 8)), [137, 80, 78, 71, 13, 10, 26, 10], "PNG signature")
  const chunks = []
  let header
  let offset = 8
  while (offset < png.length) {
    const length = png.readUInt32BE(offset)
    const type = png.toString("ascii", offset + 4, offset + 8)
    const body = png.subarray(offset + 8, offset + 8 + length)
    if (type === "IHDR") {
      header = {
        width: body.readUInt32BE(0),
        height: body.readUInt32BE(4),
        depth: body[8],
        colorType: body[9],
        interlace: body[12]
      }
    } else if (type === "IDAT") chunks.push(Buffer.from(body))
    else if (type === "IEND") break
    offset += 12 + length
  }
  assert.equal(header.depth, 8, "goldens are 8-bit")
  assert.equal(header.colorType, 6, "goldens are RGBA")
  assert.equal(header.interlace, 0, "goldens are not interlaced")

  const bytesPerPixel = 4
  const stride = header.width * bytesPerPixel
  const raw = zlib.inflateSync(Buffer.concat(chunks))
  const out = Buffer.alloc(stride * header.height)
  let previous = Buffer.alloc(stride)
  for (let row = 0; row < header.height; row += 1) {
    const filter = raw[row * (stride + 1)]
    const line = raw.subarray(row * (stride + 1) + 1, (row + 1) * (stride + 1))
    const current = Buffer.alloc(stride)
    for (let index = 0; index < stride; index += 1) {
      const left = index >= bytesPerPixel ? current[index - bytesPerPixel] : 0
      const up = previous[index]
      const upLeft = index >= bytesPerPixel ? previous[index - bytesPerPixel] : 0
      let value = line[index]
      if (filter === 1) value += left
      else if (filter === 2) value += up
      else if (filter === 3) value += (left + up) >> 1
      else if (filter === 4) {
        const estimate = left + up - upLeft
        const dLeft = Math.abs(estimate - left)
        const dUp = Math.abs(estimate - up)
        const dUpLeft = Math.abs(estimate - upLeft)
        value += dLeft <= dUp && dLeft <= dUpLeft ? left : dUp <= dUpLeft ? up : upLeft
      } else if (filter !== 0) assert.fail(`unsupported PNG filter ${filter}`)
      current[index] = value & 0xff
    }
    current.copy(out, row * stride)
    previous = current
  }
  return { pixels: out, width: header.width, height: header.height }
}

test("WASM renders the 5004 fixtures to the native pixel goldens", {
  skip: generatedPackage ? false : "set AGENT_SPREADSHEET_WASM_PACKAGE to run the WASM goldens"
}, async (t) => {
  const local = createLocalSpreadsheet({ runtime: require(generatedPackage) })
  const goldens = readGoldens()
  assert.ok(goldens.length >= 17, `expected the full fixture set, got ${goldens.length}`)

  const mismatched = []
  for (const golden of goldens) {
    const fixture = path.join(renderCrate, "tests", "fixtures", `${golden.name}.xlsx`)
    const workbook = await local.open(fs.readFileSync(fixture))
    try {
      const sheets = await workbook.listSheets()
      const rendered = await workbook.renderSheet({
        sheet_name: sheets.data.sheets[0].name,
        range: GOLDEN_RANGE
      })
      assert.equal(rendered.renderer, "native-raster/1")
      const decoded = decodePixels(Buffer.from(rendered.png))
      const signature = `${crypto.createHash("sha256").update(decoded.pixels).digest("hex")} ` +
        `${decoded.width}x${decoded.height}`
      if (signature !== golden.signature) {
        mismatched.push(`${golden.name}\n  wasm:   ${signature}\n  native: ${golden.signature}`)
      }
    } finally {
      await workbook.dispose()
    }
  }
  assert.deepEqual(mismatched, [], `WASM pixels differ from the native goldens:\n${mismatched.join("\n")}`)
  t.diagnostic(`${goldens.length} fixtures matched the native pixel goldens`)
})
