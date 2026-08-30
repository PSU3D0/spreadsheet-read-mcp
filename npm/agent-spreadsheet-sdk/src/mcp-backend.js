const { createCapabilities } = require("./capabilities")
const {
  declaredOperations,
  discoveredOperations,
  discoveredCanonicalTools,
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
  createFork: ["create_fork", (backend, input) => {
    const source = input.resource_id || input.workbookOrForkId || input.workbook_or_fork_id ||
      input.workbookId || input.workbook_id || input.contextId || input.context_id
    const resourceId = legacyResourceId(backend.kind, {
      ...input,
      workbookId: source,
      sessionId: source
    })
    return {
      resource_id: resourceId,
      expected_revision: input.expectedRevision ?? input.expected_revision
    }
  }],
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

function verifyInput(backend, input) {
  const current = input.currentResourceId || input.current_resource_id || input.currentWorkbookOrForkId || input.current_workbook_or_fork_id || input.currentId || input.current_id
  const baseline = input.baselineResourceId || input.baseline_resource_id || input.baselineWorkbookOrForkId || input.baseline_workbook_or_fork_id || input.baselineId || input.baseline_id
  const prefix = backend.kind === "wasm" ? "session" : "fork"
  return {
    resource_id: /^(wb|fork|session):/.test(current || "") ? current : `${prefix}:${current}`,
    baseline_resource_id: /^(wb|fork|session):/.test(baseline || "") ? baseline : `${prefix}:${baseline}`,
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
    this._supportedOperations = params.supportedOperations === undefined
      ? null
      : declaredOperations(params.supportedOperations)
    this._operationSet = new Set()
    this._capabilities = createCapabilities(this.kind, [], {
      transport: "mcp",
      initialized: false
    })
    this._initialization = null
    if (this._supportedOperations) this._setOperations(this._supportedOperations)
  }

  getCapabilities() {
    return this._capabilities
  }

  _setOperations(operations) {
    this._operationSet = new Set(operations)
    this._capabilities = createCapabilities(this.kind, operations, {
      transport: "mcp",
      initialized: true
    })
    return this._capabilities
  }

  async _discoverOperations() {
    if (typeof this._transport.listOperations === "function") {
      return discoveredOperations(await this._transport.listOperations())
    }
    if (typeof this._transport.listTools === "function") {
      return discoveredCanonicalTools(await this._transport.listTools())
    }
    if (typeof this._transport["tools/list"] === "function") {
      return discoveredCanonicalTools(await this._transport["tools/list"]({}))
    }
    if (typeof this._transport.request === "function") {
      return discoveredCanonicalTools(await this._transport.request({ method: "tools/list", params: {} }))
    }
    throw new SpreadsheetSdkError(
      "McpBackend requires supportedOperations or a transport tools/list discovery method",
      { code: "INVALID_ARGUMENT", backend: this.kind }
    )
  }

  async initialize() {
    if (this._capabilities.initialized) return this._capabilities
    if (!this._initialization) {
      this._initialization = this.refresh().finally(() => {
        this._initialization = null
      })
    }
    return this._initialization
  }

  async refresh() {
    const operations = this._supportedOperations || await this._discoverOperations()
    return this._setOperations(operations)
  }

  async execute(operation, input = {}) {
    await this.initialize()
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

function isCanonicalInput(operation, input) {
  return operation !== "list_forks" &&
    input && typeof input === "object" && !Array.isArray(input) &&
    Object.prototype.hasOwnProperty.call(input, "resource_id")
}

function installLegacyMethods(Backend) {
  for (const [method, [operation, mapInput]] of Object.entries(LEGACY_METHODS)) {
    const collidesWithCanonicalMethod = Object.prototype.hasOwnProperty.call(Backend.prototype, method)
    Object.defineProperty(Backend.prototype, method, {
      configurable: true,
      value(input = {}) {
        if (collidesWithCanonicalMethod && isCanonicalInput(operation, input)) {
          return this.execute(operation, input)
        }
        return this.execute(operation, mapInput(this, input)).then(projectData)
      }
    })
  }
}

installCanonicalMethods(McpBackend)
installLegacyMethods(McpBackend)

module.exports = { McpBackend, installLegacyMethods }
