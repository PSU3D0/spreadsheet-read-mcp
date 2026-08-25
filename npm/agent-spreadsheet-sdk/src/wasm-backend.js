const { freezeCapabilities, WASM_CAPABILITIES } = require("./capabilities")
const {
  requireCapability,
  requiredString,
  normalizeSheetPageResult,
  normalizeGridExportResult,
  normalizeTransformBatchResult,
  normalizeDescribeWorkbookResult,
  normalizeNamedRangesResult,
  normalizeSheetOverviewResult,
  normalizeFindValueResult,
  normalizeReadTableResult
} = require("./backend")
const { SpreadsheetSdkError, normalizeBackendError } = require("./errors")

function normalizeSheetNames(items) {
  return items.map((item) => {
    if (typeof item === "string") {
      return item
    }
    if (item && typeof item === "object" && typeof item.name === "string") {
      return item.name
    }
    throw new SpreadsheetSdkError("invalid sheet summary in listSheets response", {
      code: "INVALID_RESPONSE",
      backend: "wasm",
      operation: "listSheets"
    })
  })
}

function normalizeListSheetsResult(result) {
  if (Array.isArray(result)) {
    return normalizeSheetNames(result)
  }
  if (result && typeof result === "object") {
    if (Array.isArray(result.sheets)) {
      return normalizeSheetNames(result.sheets)
    }
    if (Array.isArray(result.sheet_names)) {
      return normalizeSheetNames(result.sheet_names)
    }
  }
  throw new SpreadsheetSdkError("invalid listSheets response", {
    code: "INVALID_RESPONSE",
    backend: "wasm",
    operation: "listSheets"
  })
}

function normalizeRangeValuesResult(result, fallbackSheetName) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid rangeValues response", {
      code: "INVALID_RESPONSE",
      backend: "wasm",
      operation: "rangeValues"
    })
  }

  const sheetName = typeof result.sheetName === "string"
    ? result.sheetName
    : typeof result.sheet_name === "string"
      ? result.sheet_name
      : fallbackSheetName

  const values = Array.isArray(result.values) ? result.values : []
  return { sheetName, values }
}

class WasmBackend {
  /**
   * @param {{
   *   bindings: Record<string, unknown>,
   *   capabilities?: import("./capabilities").BackendCapabilities
   * }} params
   */
  constructor(params) {
    if (!params || !params.bindings || typeof params.bindings !== "object") {
      throw new SpreadsheetSdkError("WasmBackend requires bindings object", {
        code: "INVALID_ARGUMENT",
        backend: "wasm"
      })
    }

    this.kind = "wasm"
    this._bindings = params.bindings
    this._capabilities = freezeCapabilities(params.capabilities || WASM_CAPABILITIES)
  }

  getCapabilities() {
    return this._capabilities
  }

  async createSession(input = {}) {
    requireCapability(this, "supportsSessionLifecycle", "createSession")
    const workbookBytes = input.workbookBytes || input.bytes
    if (!workbookBytes) {
      throw new SpreadsheetSdkError("missing required field 'workbookBytes'", {
        code: "INVALID_ARGUMENT",
        backend: this.kind,
        operation: "createSession"
      })
    }
    return this._call("createSession", workbookBytes)
  }

  async describeWorkbook(input = {}) {
    requireCapability(this, "supportsDescribeWorkbook", "describeWorkbook")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const result = await this._call("describeWorkbook", sessionId)
    return normalizeDescribeWorkbookResult(result, sessionId)
  }

  async namedRanges(input = {}) {
    requireCapability(this, "supportsNamedRanges", "namedRanges")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const result = await this._call("namedRanges", sessionId)
    return normalizeNamedRangesResult(result, sessionId)
  }

  async verifyWorkbook(input = {}) {
    requireCapability(this, "supportsVerification", "verifyWorkbook")
    return input
  }

  async verifyTargets(input = {}) {
    requireCapability(this, "supportsVerification", "verifyTargets")
    return input
  }

  async verifyErrors(input = {}) {
    requireCapability(this, "supportsVerification", "verifyErrors")
    return input
  }

  async defineName(input = {}) {
    requireCapability(this, "supportsNamedRangeMutations", "defineName")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const name = requiredString(input.name, "name")
    const refersTo = requiredString(input.refersTo || input.refers_to, "refersTo")
    return this._call("defineName", sessionId, {
      name,
      refersTo: refersTo,
      scope: input.scope,
      scopeSheetName: input.scopeSheetName ?? input.scope_sheet_name
    })
  }

  async updateName(input = {}) {
    requireCapability(this, "supportsNamedRangeMutations", "updateName")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const name = requiredString(input.name, "name")
    return this._call("updateName", sessionId, {
      name,
      refersTo: input.refersTo ?? input.refers_to,
      scope: input.scope,
      scopeSheetName: input.scopeSheetName ?? input.scope_sheet_name
    })
  }

  async deleteName(input = {}) {
    requireCapability(this, "supportsNamedRangeMutations", "deleteName")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const name = requiredString(input.name, "name")
    return this._call("deleteName", sessionId, {
      name,
      scope: input.scope,
      scopeSheetName: input.scopeSheetName ?? input.scope_sheet_name
    })
  }

  async sheetOverview(input = {}) {
    requireCapability(this, "supportsSheetOverview", "sheetOverview")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const sheetName = requiredString(input.sheetName || input.sheet_name, "sheetName")
    const result = await this._call("sheetOverview", sessionId, {
      ...input,
      sheetName,
      maxRegions: input.maxRegions ?? input.max_regions,
      maxHeaders: input.maxHeaders ?? input.max_headers,
      includeHeaders: input.includeHeaders ?? input.include_headers
    })
    return normalizeSheetOverviewResult(result, sheetName, sessionId)
  }

  async listSheets(input = {}) {
    requireCapability(this, "supportsListSheets", "listSheets")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const result = await this._call("listSheets", sessionId)
    return normalizeListSheetsResult(result)
  }

  async rangeValues(input = {}) {
    requireCapability(this, "supportsRangeValues", "rangeValues")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const sheetName = requiredString(input.sheetName || input.sheet_name, "sheetName")
    const ranges = input.ranges

    const result = await this._call("rangeValues", sessionId, {
      sheetName,
      ranges
    })

    return normalizeRangeValuesResult(result, sheetName)
  }

  async findValue(input = {}) {
    requireCapability(this, "supportsFindValue", "findValue")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const query = requiredString(input.query, "query")

    const result = await this._call("findValue", sessionId, {
      ...input,
      query,
      sheetName: input.sheetName ?? input.sheet_name,
      caseSensitive: input.caseSensitive ?? input.case_sensitive,
      limit: input.limit,
      offset: input.offset
    })

    return normalizeFindValueResult(result, sessionId)
  }

  async readTable(input = {}) {
    requireCapability(this, "supportsReadTable", "readTable")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")

    const result = await this._call("readTable", sessionId, {
      ...input,
      sheetName: input.sheetName ?? input.sheet_name,
      includeHeaders: input.includeHeaders ?? input.include_headers,
      includeTypes: input.includeTypes ?? input.include_types
    })

    return normalizeReadTableResult(result, sessionId, input.sheetName || input.sheet_name)
  }

  async sheetPage(input = {}) {
    requireCapability(this, "supportsSheetPage", "sheetPage")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const sheetName = requiredString(input.sheetName || input.sheet_name, "sheetName")

    const result = await this._call("sheetPage", sessionId, {
      ...input,
      sheetName,
      startRow: input.startRow ?? input.start_row,
      pageSize: input.pageSize ?? input.page_size,
      columnsByHeader: input.columnsByHeader ?? input.columns_by_header,
      includeFormulas: input.includeFormulas ?? input.include_formulas,
      includeStyles: input.includeStyles ?? input.include_styles,
      includeHeader: input.includeHeader ?? input.include_header
    })

    return normalizeSheetPageResult(result, sheetName)
  }

  async gridExport(input = {}) {
    requireCapability(this, "supportsGridExport", "gridExport")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    const sheetName = requiredString(input.sheetName || input.sheet_name, "sheetName")

    const result = await this._call("gridExport", sessionId, {
      ...input,
      sheetName,
      range: input.range
    })

    return normalizeGridExportResult(result)
  }

  async transformBatch(input = {}) {
    requireCapability(this, "supportsTransformBatch", "transformBatch")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    
    // Convert SDK generic `mode: "preview"` to Wasm specific `dryRun` if missing options
    const options = input.options ?? {}
    if (input.mode === "preview" && options.dryRun === undefined) {
      options.dryRun = true
    }

    const result = await this._call("transformBatch", sessionId, input.ops || [], options)
    
    return normalizeTransformBatchResult(result)
  }

  async replaceInFormulas() {
    requireCapability(this, "supportsReplaceInFormulas", "replaceInFormulas")
    throw new SpreadsheetSdkError("replaceInFormulas is not implemented for WASM backend (fork-only)", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "replaceInFormulas"
    })
  }

  async exportWorkbook(input = {}) {
    requireCapability(this, "supportsExportWorkbook", "exportWorkbook")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    return this._call("exportWorkbook", sessionId)
  }

  async disposeSession(input = {}) {
    requireCapability(this, "supportsSessionLifecycle", "disposeSession")
    const sessionId = requiredString(input.sessionId || input.session_id || input.contextId, "sessionId")
    return this._call("disposeSession", sessionId)
  }

  async createFork() {
    requireCapability(this, "supportsForkLifecycle", "createFork")
    throw new SpreadsheetSdkError("createFork is not implemented for WASM backend", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "createFork"
    })
  }

  async listForks() {
    requireCapability(this, "supportsForkLifecycle", "listForks")
    throw new SpreadsheetSdkError("listForks is not implemented for WASM backend", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "listForks"
    })
  }

  async saveFork() {
    requireCapability(this, "supportsForkLifecycle", "saveFork")
    throw new SpreadsheetSdkError("saveFork is not implemented for WASM backend", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "saveFork"
    })
  }

  async discardFork() {
    requireCapability(this, "supportsForkLifecycle", "discardFork")
    throw new SpreadsheetSdkError("discardFork is not implemented for WASM backend", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "discardFork"
    })
  }

  async listStagedChanges() {
    requireCapability(this, "supportsStaging", "listStagedChanges")
    throw new SpreadsheetSdkError("listStagedChanges is not implemented for WASM backend", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "listStagedChanges"
    })
  }

  async applyStagedChange() {
    requireCapability(this, "supportsStaging", "applyStagedChange")
    throw new SpreadsheetSdkError("applyStagedChange is not implemented for WASM backend", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "applyStagedChange"
    })
  }

  async discardStagedChange() {
    requireCapability(this, "supportsStaging", "discardStagedChange")
    throw new SpreadsheetSdkError("discardStagedChange is not implemented for WASM backend", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "discardStagedChange"
    })
  }

  async _call(binding, ...args) {
    const fn = this._bindings[binding]
    if (typeof fn !== "function") {
      throw new SpreadsheetSdkError(`wasm bindings missing '${binding}'`, {
        code: "INVALID_ARGUMENT",
        backend: this.kind,
        operation: binding
      })
    }

    try {
      return await fn(...args)
    } catch (error) {
      throw normalizeBackendError(error, {
        backend: this.kind,
        operation: binding
      })
    }
  }
}

module.exports = {
  WasmBackend
}
