/* Legacy 0.14 capability booleans. Superseded by `client.capabilities()`. */

const LEGACY_OPERATION_CAPABILITIES: Readonly<Record<string, string[]>> = Object.freeze({
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

/** @deprecated The 0.14 capability record. */
export interface LegacyCapabilities {
  schemaVersion: string
  backend: string
  initialized: boolean
  operations: readonly string[]
  resourceBinding: boolean
  resourceExport: boolean
  transport: string
  [key: string]: unknown
}

/** @deprecated Use `client.capabilities()`. */
export function createCapabilities(
  kind: string,
  operations: readonly string[],
  adapter: {
    transport?: string
    initialized?: boolean
    resourceBinding?: boolean
    resourceExport?: boolean
    sessionLifecycle?: boolean
  } = {}
): LegacyCapabilities {
  const operationSet = new Set(operations)
  const capabilities: LegacyCapabilities = {
    schemaVersion: "1",
    backend: kind,
    initialized: adapter.initialized ?? true,
    operations: Object.freeze([...operations]),
    resourceBinding: Boolean(adapter.resourceBinding),
    resourceExport: Boolean(adapter.resourceExport),
    transport: adapter.transport || kind
  }

  for (const [name, required] of Object.entries(LEGACY_OPERATION_CAPABILITIES)) {
    capabilities[name] = required.every((operation) => operationSet.has(operation))
  }
  capabilities["supportsSessionLifecycle"] = Boolean(adapter.sessionLifecycle)
  capabilities["supportsExportWorkbook"] = Boolean(adapter.resourceExport)
  return Object.freeze(capabilities)
}

/** @deprecated Legacy MCP capability seed. */
export const MCP_CAPABILITIES: LegacyCapabilities =
  createCapabilities("mcp", [], { transport: "mcp", initialized: false })

/** @deprecated Legacy WASM capability seed. */
export const WASM_CAPABILITIES: LegacyCapabilities =
  createCapabilities("wasm", [], { transport: "wasm-json", initialized: false })

/** @deprecated Legacy capability freezer. */
export function freezeCapabilities(capabilities: LegacyCapabilities): LegacyCapabilities {
  return Object.freeze({
    ...capabilities,
    operations: Object.freeze([...(capabilities.operations || [])])
  })
}
