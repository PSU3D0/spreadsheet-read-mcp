const { createCapabilities } = require("./capabilities")
const {
  OPERATION_NAMES,
  declaredOperations,
  requireOperation,
  installCanonicalMethods,
  projectData,
  legacyResourceId
} = require("./backend")
const { CapabilityError, SpreadsheetSdkError } = require("./errors")

function snakeCaseInput(value) {
  if (Array.isArray(value)) return value.map(snakeCaseInput)
  if (!value || typeof value !== "object") return value
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [
    key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`),
    snakeCaseInput(item)
  ]))
}

function withResource(backend, input, mutation = false) {
  const result = snakeCaseInput(input)
  delete result.workbook_id
  delete result.fork_id
  delete result.context_id
  result.resource_id = legacyResourceId(backend.kind, input, mutation)
  return result
}

const LEGACY_METHODS = {
  describeWorkbook: ["describe_workbook", withResource],
  namedRanges: ["named_ranges", withResource],
  sheetOverview: ["sheet_overview", withResource],
  listSheets: ["list_sheets", withResource],
  readTable: ["read_table", withResource],
  findValue: ["search_values", withResource],
  gridExport: ["export_grid", withResource],
  createFork: ["create_fork", withResource],
  listForks: ["list_forks", (_, input) => snakeCaseInput(input)],
  saveFork: ["export_fork", (backend, input) => withResource(backend, input, true)],
  discardFork: ["discard_fork", (backend, input) => withResource(backend, input, true)],
  listStagedChanges: ["staged_change", (backend, input) => ({ ...withResource(backend, input, true), action: "list" })],
  applyStagedChange: ["staged_change", (backend, input) => ({ ...withResource(backend, input, true), action: "apply" })],
  discardStagedChange: ["staged_change", (backend, input) => ({ ...withResource(backend, input, true), action: "discard" })],
  rangeValues: ["read_cells", (backend, input) => ({
    resource_id: legacyResourceId(backend.kind, input),
    sheet_name: input.sheetName || input.sheet_name,
    selection: { kind: "range", ranges: Array.isArray(input.ranges) ? input.ranges : [input.ranges] },
    include_formulas: input.includeFormulas ?? input.include_formulas,
    format: input.format
  })],
  sheetPage: ["read_cells", (backend, input) => ({
    resource_id: legacyResourceId(backend.kind, input),
    sheet_name: input.sheetName || input.sheet_name,
    selection: {
      kind: "rows",
      start_row: input.startRow ?? input.start_row ?? 1,
      row_count: input.pageSize ?? input.page_size ?? 50,
      columns: input.columns,
      columns_by_header: input.columnsByHeader ?? input.columns_by_header,
      include_header: input.includeHeader ?? input.include_header
    },
    include_formulas: input.includeFormulas ?? input.include_formulas,
    include_styles: input.includeStyles ?? input.include_styles,
    format: input.format
  })],
  transformBatch: ["write", (backend, input) => ({
    resource_id: legacyResourceId(backend.kind, input, true),
    expected_revision: input.expectedRevision ?? input.expected_revision,
    mode: input.options?.dryRun ? "preview" : (input.mode || "apply"),
    atomic: input.atomic,
    ops: snakeCaseInput(input.ops || [])
  })],
  structureBatch: ["write", (backend, input) => ({
    ...withResource(backend, input, true),
    ops: snakeCaseInput(input.ops || [])
  })],
  replaceInFormulas: ["write", (backend, input) => ({
    resource_id: legacyResourceId(backend.kind, input, true),
    expected_revision: input.expectedRevision ?? input.expected_revision,
    mode: input.options?.dryRun ? "preview" : (input.mode || "apply"),
    ops: [{
      kind: "replace_in_formulas",
      sheet_name: input.sheetName || input.sheet_name,
      range: input.range,
      find: input.find,
      replace: input.replace || "",
      regex: input.regex ?? false,
      case_sensitive: input.caseSensitive ?? input.case_sensitive ?? true
    }]
  })],
  defineName: ["write", (backend, input) => nameWrite(backend, input, "define_name")],
  updateName: ["write", (backend, input) => nameWrite(backend, input, "update_name")],
  deleteName: ["write", (backend, input) => nameWrite(backend, input, "delete_name")],
  verifyWorkbook: ["verify_workbook", verifyInput],
  verifyTargets: ["verify_workbook", (backend, input) => ({ ...verifyInput(backend, input), targets_only: true })],
  verifyErrors: ["verify_workbook", (backend, input) => ({ ...verifyInput(backend, input), errors_only: true })]
}

function nameWrite(backend, input, kind) {
  return {
    resource_id: legacyResourceId(backend.kind, input, true),
    expected_revision: input.expectedRevision ?? input.expected_revision,
    mode: input.options?.dryRun ? "preview" : (input.mode || "apply"),
    ops: [{
      kind,
      name: input.name,
      refers_to: input.refersTo ?? input.refers_to,
      scope: input.scope,
      scope_sheet_name: input.scopeSheetName ?? input.scope_sheet_name
    }]
  }
}

function verifyInput(_backend, input) {
  const current = input.currentResourceId || input.current_resource_id || input.currentWorkbookOrForkId || input.current_workbook_or_fork_id || input.currentId || input.current_id
  const baseline = input.baselineResourceId || input.baseline_resource_id || input.baselineWorkbookOrForkId || input.baseline_workbook_or_fork_id || input.baselineId || input.baseline_id
  return {
    resource_id: /^(wb|fork):/.test(current || "") ? current : `fork:${current}`,
    baseline_resource_id: /^(wb|fork):/.test(baseline || "") ? baseline : `fork:${baseline}`,
    targets: input.targets || [],
    sheet_name: input.sheetName || input.sheet_name,
    include_named_range_deltas: input.includeNamedRangeDeltas ?? input.include_named_range_deltas,
    errors_only: input.errorsOnly ?? input.errors_only ?? false,
    targets_only: input.targetsOnly ?? input.targets_only ?? false
  }
}

class McpBackend {
  constructor(params) {
    if (!params || !params.transport || typeof params.transport !== "object") {
      throw new SpreadsheetSdkError("McpBackend requires a transport object", {
        code: "INVALID_ARGUMENT",
        backend: "mcp"
      })
    }
    this.kind = "mcp"
    this._transport = params.transport
    const operations = declaredOperations(params.capabilities || params, OPERATION_NAMES)
    this._operationSet = new Set(operations)
    this._capabilities = createCapabilities(this.kind, operations, { transport: "mcp" })
  }

  getCapabilities() {
    return this._capabilities
  }

  async execute(operation, input = {}) {
    requireOperation(this, operation)
    if (!input || typeof input !== "object" || Array.isArray(input)) {
      throw new SpreadsheetSdkError("canonical operation input must be an object", {
        code: "INVALID_ARGUMENT",
        backend: this.kind,
        operation
      })
    }
    if (typeof this._transport[operation] === "function") {
      return this._transport[operation](input)
    }
    if (typeof this._transport.invoke === "function") {
      return this._transport.invoke(operation, input)
    }
    throw new CapabilityError({
      backend: this.kind,
      capability: "canonical_transport",
      method: "execute",
      operation
    })
  }

  createSession() {
    throw new CapabilityError({ backend: this.kind, capability: "resource_binding", method: "createSession" })
  }

  exportWorkbook() {
    throw new CapabilityError({ backend: this.kind, capability: "resource_export", method: "exportWorkbook" })
  }

  disposeSession() {
    throw new CapabilityError({ backend: this.kind, capability: "resource_binding", method: "disposeSession" })
  }
}

function installLegacyMethods(Backend) {
  for (const [method, [operation, mapInput]] of Object.entries(LEGACY_METHODS)) {
    Object.defineProperty(Backend.prototype, method, {
      configurable: true,
      value(input = {}) {
        return this.execute(operation, mapInput(this, input)).then(projectData)
      }
    })
  }
}

installCanonicalMethods(McpBackend)
installLegacyMethods(McpBackend)

module.exports = { McpBackend, installLegacyMethods }
