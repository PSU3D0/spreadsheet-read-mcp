const { OPERATION_NAMES } = require("./backend")

const LEGACY_OPERATION_CAPABILITIES = Object.freeze({
  supportsDescribeWorkbook: ["describe_workbook"],
  supportsNamedRanges: ["named_ranges"],
  supportsNamedRangeMutations: ["write"],
  supportsSheetOverview: ["sheet_overview"],
  supportsListSheets: ["list_sheets"],
  supportsRangeValues: ["read_cells"],
  supportsFindValue: ["search_values"],
  supportsReadTable: ["read_table"],
  supportsSheetPage: ["read_cells"],
  supportsGridExport: ["export_grid"],
  supportsTransformBatch: ["write"],
  supportsStructureBatch: ["write"],
  supportsReplaceInFormulas: ["write"],
  supportsVerification: ["verify_workbook"],
  supportsForkLifecycle: ["create_fork", "list_forks", "export_fork", "discard_fork"],
  supportsStaging: ["staged_change"]
})

function createCapabilities(kind, operations, adapter = {}) {
  const operationSet = new Set(operations)
  const capabilities = {
    schemaVersion: "1",
    backend: kind,
    operations: Object.freeze([...operations]),
    resourceBinding: Boolean(adapter.resourceBinding),
    resourceExport: Boolean(adapter.resourceExport),
    transport: adapter.transport || kind
  }

  for (const [name, required] of Object.entries(LEGACY_OPERATION_CAPABILITIES)) {
    capabilities[name] = required.every((operation) => operationSet.has(operation))
  }
  capabilities.supportsSessionLifecycle = Boolean(adapter.sessionLifecycle)
  capabilities.supportsExportWorkbook = Boolean(adapter.resourceExport)
  return Object.freeze(capabilities)
}

const MCP_CAPABILITIES = createCapabilities("mcp", OPERATION_NAMES, { transport: "mcp" })
const WASM_CAPABILITIES = createCapabilities("wasm", [], { transport: "wasm" })

function freezeCapabilities(capabilities) {
  return Object.freeze({
    ...capabilities,
    operations: Object.freeze([...(capabilities.operations || [])])
  })
}

module.exports = {
  MCP_CAPABILITIES,
  WASM_CAPABILITIES,
  createCapabilities,
  freezeCapabilities
}
