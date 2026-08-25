const { freezeCapabilities, MCP_CAPABILITIES } = require("./capabilities")
const {
  requireCapability,
  requiredString,
  normalizeSheetPageResult,
  normalizeGridExportResult,
  normalizeTransformBatchResult,
  normalizeStructureBatchResult,
  normalizeReplaceInFormulasResult,
  normalizeVerifyWorkbookResult,
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
    throw new SpreadsheetSdkError("invalid sheet summary in list_sheets response", {
      code: "INVALID_RESPONSE",
      backend: "mcp",
      operation: "list_sheets"
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
  throw new SpreadsheetSdkError("invalid list_sheets response", {
    code: "INVALID_RESPONSE",
    backend: "mcp",
    operation: "list_sheets"
  })
}

function normalizeRangeValuesResult(result, fallbackSheetName) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid range_values response", {
      code: "INVALID_RESPONSE",
      backend: "mcp",
      operation: "range_values"
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

class McpBackend {
  /**
   * @param {{
   *   transport: { invoke?: (operation: string, params?: Record<string, unknown>) => unknown, [k: string]: unknown },
   *   capabilities?: import("./capabilities").BackendCapabilities
   * }} params
   */
  constructor(params) {
    if (!params || !params.transport || typeof params.transport !== "object") {
      throw new SpreadsheetSdkError("McpBackend requires a transport object", {
        code: "INVALID_ARGUMENT",
        backend: "mcp"
      })
    }

    this.kind = "mcp"
    this._transport = params.transport
    this._capabilities = freezeCapabilities(params.capabilities || MCP_CAPABILITIES)
  }

  getCapabilities() {
    return this._capabilities
  }

  async describeWorkbook(input = {}) {
    requireCapability(this, "supportsDescribeWorkbook", "describeWorkbook")
    const workbookId = requiredString(
      input.workbookId || input.workbook_id || input.contextId,
      "workbookId"
    )
    const result = await this._call("describe_workbook", {
      ...input,
      workbook_id: workbookId
    })
    return normalizeDescribeWorkbookResult(result, workbookId)
  }

  async namedRanges(input = {}) {
    requireCapability(this, "supportsNamedRanges", "namedRanges")
    const workbookId = requiredString(
      input.workbookId || input.workbook_id || input.contextId,
      "workbookId"
    )
    const result = await this._call("named_ranges", {
      ...input,
      workbook_id: workbookId
    })
    return normalizeNamedRangesResult(result, workbookId)
  }

  async verifyWorkbook(input = {}) {
    requireCapability(this, "supportsVerification", "verifyWorkbook")
    const baselineWorkbookOrForkId = requiredString(
      input.baselineWorkbookOrForkId || input.baseline_workbook_or_fork_id || input.baselineId || input.baseline_id,
      "baselineWorkbookOrForkId"
    )
    const currentWorkbookOrForkId = requiredString(
      input.currentWorkbookOrForkId || input.current_workbook_or_fork_id || input.currentId || input.current_id,
      "currentWorkbookOrForkId"
    )
    const result = await this._call("verify_workbook", {
      baseline_workbook_or_fork_id: baselineWorkbookOrForkId,
      current_workbook_or_fork_id: currentWorkbookOrForkId,
      targets: Array.isArray(input.targets) ? input.targets : [],
      sheet_name: input.sheetName || input.sheet_name,
      include_named_range_deltas: Boolean(input.includeNamedRangeDeltas || input.include_named_range_deltas),
      errors_only: Boolean(input.errorsOnly || input.errors_only),
      targets_only: Boolean(input.targetsOnly || input.targets_only)
    })
    return normalizeVerifyWorkbookResult(result, baselineWorkbookOrForkId, currentWorkbookOrForkId)
  }

  async verifyTargets(input = {}) {
    return this.verifyWorkbook({
      ...input,
      targetsOnly: true,
      targets_only: true
    })
  }

  async verifyErrors(input = {}) {
    return this.verifyWorkbook({
      ...input,
      errorsOnly: true,
      errors_only: true
    })
  }

  async defineName(input = {}) {
    requireCapability(this, "supportsNamedRangeMutations", "defineName")
    const forkId = requiredString(
      input.forkId || input.fork_id || input.workbookId || input.workbook_id || input.contextId,
      "forkId"
    )
    const name = requiredString(input.name, "name")
    const refersTo = requiredString(input.refersTo || input.refers_to, "refersTo")
    return this._call("define_name", {
      fork_id: forkId,
      name,
      refers_to: refersTo,
      scope: input.scope,
      scope_sheet_name: input.scope_sheet_name ?? input.scopeSheetName
    })
  }

  async updateName(input = {}) {
    requireCapability(this, "supportsNamedRangeMutations", "updateName")
    const forkId = requiredString(
      input.forkId || input.fork_id || input.workbookId || input.workbook_id || input.contextId,
      "forkId"
    )
    const name = requiredString(input.name, "name")
    return this._call("update_name", {
      fork_id: forkId,
      name,
      refers_to: input.refersTo ?? input.refers_to,
      scope: input.scope,
      scope_sheet_name: input.scope_sheet_name ?? input.scopeSheetName
    })
  }

  async deleteName(input = {}) {
    requireCapability(this, "supportsNamedRangeMutations", "deleteName")
    const forkId = requiredString(
      input.forkId || input.fork_id || input.workbookId || input.workbook_id || input.contextId,
      "forkId"
    )
    const name = requiredString(input.name, "name")
    return this._call("delete_name", {
      fork_id: forkId,
      name,
      scope: input.scope,
      scope_sheet_name: input.scope_sheet_name ?? input.scopeSheetName
    })
  }

  async sheetOverview(input = {}) {
    requireCapability(this, "supportsSheetOverview", "sheetOverview")
    const workbookId = requiredString(
      input.workbookId || input.workbook_id || input.contextId,
      "workbookId"
    )
    const sheetName = requiredString(input.sheetName || input.sheet_name, "sheetName")
    const result = await this._call("sheet_overview", {
      ...input,
      workbook_id: workbookId,
      sheet_name: sheetName,
      max_regions: input.max_regions ?? input.maxRegions,
      max_headers: input.max_headers ?? input.maxHeaders,
      include_headers: input.include_headers ?? input.includeHeaders
    })

    return normalizeSheetOverviewResult(result, sheetName, workbookId)
  }

  async listSheets(input = {}) {
    requireCapability(this, "supportsListSheets", "listSheets")
    const workbookId = requiredString(
      input.workbookId || input.workbook_id || input.contextId,
      "workbookId"
    )
    const result = await this._call("list_sheets", {
      ...input,
      workbook_id: workbookId
    })
    return normalizeListSheetsResult(result)
  }

  async rangeValues(input = {}) {
    requireCapability(this, "supportsRangeValues", "rangeValues")
    const workbookId = requiredString(
      input.workbookId || input.workbook_id || input.contextId,
      "workbookId"
    )
    const sheetName = requiredString(input.sheetName || input.sheet_name, "sheetName")
    const ranges = input.ranges

    const result = await this._call("range_values", {
      ...input,
      workbook_id: workbookId,
      sheet_name: sheetName,
      ranges
    })
    return normalizeRangeValuesResult(result, sheetName)
  }

  async findValue(input = {}) {
    requireCapability(this, "supportsFindValue", "findValue")
    const workbookId = requiredString(
      input.workbookId || input.workbook_id || input.contextId,
      "workbookId"
    )
    const query = requiredString(input.query, "query")

    const result = await this._call("find_value", {
      ...input,
      workbook_id: workbookId,
      query,
      sheet_name: input.sheet_name ?? input.sheetName,
      case_sensitive: input.case_sensitive ?? input.caseSensitive,
      limit: input.limit,
      offset: input.offset
    })

    return normalizeFindValueResult(result, workbookId)
  }

  async readTable(input = {}) {
    requireCapability(this, "supportsReadTable", "readTable")
    const workbookId = requiredString(
      input.workbookId || input.workbook_id || input.contextId,
      "workbookId"
    )

    const result = await this._call("read_table", {
      ...input,
      workbook_id: workbookId,
      sheet_name: input.sheet_name ?? input.sheetName,
      include_headers: input.include_headers ?? input.includeHeaders,
      include_types: input.include_types ?? input.includeTypes
    })

    return normalizeReadTableResult(result, workbookId, input.sheetName || input.sheet_name)
  }

  async sheetPage(input = {}) {
    requireCapability(this, "supportsSheetPage", "sheetPage")
    const workbookId = requiredString(
      input.workbookId || input.workbook_id || input.contextId,
      "workbookId"
    )
    const sheetName = requiredString(input.sheetName || input.sheet_name, "sheetName")

    const result = await this._call("sheet_page", {
      ...input,
      workbook_id: workbookId,
      sheet_name: sheetName,
      start_row: input.start_row ?? input.startRow,
      page_size: input.page_size ?? input.pageSize,
      columns: input.columns,
      format: input.format,
      columns_by_header: input.columns_by_header ?? input.columnsByHeader,
      include_formulas: input.include_formulas ?? input.includeFormulas,
      include_styles: input.include_styles ?? input.includeStyles,
      include_header: input.include_header ?? input.includeHeader
    })

    return normalizeSheetPageResult(result, sheetName)
  }

  async gridExport(input = {}) {
    requireCapability(this, "supportsGridExport", "gridExport")
    const workbookId = requiredString(
      input.workbookId || input.workbook_id || input.contextId,
      "workbookId"
    )
    const sheetName = requiredString(input.sheetName || input.sheet_name, "sheetName")

    const result = await this._call("grid_export", {
      ...input,
      workbook_id: workbookId,
      sheet_name: sheetName,
      range: input.range
    })

    return normalizeGridExportResult(result)
  }

  async transformBatch(input = {}) {
    requireCapability(this, "supportsTransformBatch", "transformBatch")
    const forkId = requiredString(
      input.forkId || input.fork_id || input.workbookId || input.workbook_id || input.contextId,
      "forkId"
    )
    const result = await this._call("transform_batch", {
      ...input,
      fork_id: forkId,
      ops: input.ops,
      mode: input.options?.dryRun ? "preview" : (input.mode ?? "apply")
    })

    return normalizeTransformBatchResult(result)
  }

  async structureBatch(input = {}) {
    requireCapability(this, "supportsStructureBatch", "structureBatch")
    const forkId = requiredString(
      input.forkId || input.fork_id || input.workbookId || input.workbook_id || input.contextId,
      "forkId"
    )
    const result = await this._call("structure_batch", {
      ...input,
      fork_id: forkId,
      ops: input.ops,
      mode: input.mode ?? "apply",
      impact_report: input.impactReport ?? input.impact_report,
      show_formula_delta: input.showFormulaDelta ?? input.show_formula_delta
    })

    return normalizeStructureBatchResult(result)
  }

  async replaceInFormulas(input = {}) {
    requireCapability(this, "supportsReplaceInFormulas", "replaceInFormulas")
    const forkId = requiredString(
      input.forkId || input.fork_id || input.workbookId || input.workbook_id || input.contextId,
      "forkId"
    )
    const sheetName = requiredString(input.sheetName || input.sheet_name, "sheetName")
    const find = requiredString(input.find, "find")

    const result = await this._call("replace_in_formulas", {
      fork_id: forkId,
      sheet_name: sheetName,
      find,
      replace: input.replace ?? "",
      range: input.range,
      regex: input.regex ?? false,
      case_sensitive: input.caseSensitive ?? input.case_sensitive ?? true,
      mode: input.options?.dryRun ? "preview" : (input.mode ?? "apply"),
      label: input.label,
      formula_parse_policy: input.formulaParsePolicy ?? input.formula_parse_policy
    })

    return normalizeReplaceInFormulasResult(result)
  }

  async createFork(input = {}) {
    requireCapability(this, "supportsForkLifecycle", "createFork")
    const workbookOrForkId = requiredString(
      input.workbookOrForkId || input.workbook_or_fork_id || input.workbookId || input.workbook_id,
      "workbookOrForkId"
    )

    return this._call("create_fork", {
      ...input,
      workbook_or_fork_id: workbookOrForkId
    })
  }

  async listForks(input = {}) {
    requireCapability(this, "supportsForkLifecycle", "listForks")
    return this._call("list_forks", input)
  }

  async saveFork(input = {}) {
    requireCapability(this, "supportsForkLifecycle", "saveFork")
    return this._call("save_fork", input)
  }

  async discardFork(input = {}) {
    requireCapability(this, "supportsForkLifecycle", "discardFork")
    return this._call("discard_fork", input)
  }

  async listStagedChanges(input = {}) {
    requireCapability(this, "supportsStaging", "listStagedChanges")
    return this._call("list_staged_changes", input)
  }

  async applyStagedChange(input = {}) {
    requireCapability(this, "supportsStaging", "applyStagedChange")
    return this._call("apply_staged_change", input)
  }

  async discardStagedChange(input = {}) {
    requireCapability(this, "supportsStaging", "discardStagedChange")
    return this._call("discard_staged_change", input)
  }

  async createSession() {
    requireCapability(this, "supportsSessionLifecycle", "createSession")
    throw new SpreadsheetSdkError("createSession is not implemented for MCP backend", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "createSession"
    })
  }

  async exportWorkbook() {
    requireCapability(this, "supportsExportWorkbook", "exportWorkbook")
    throw new SpreadsheetSdkError("exportWorkbook is not implemented for MCP backend", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "exportWorkbook"
    })
  }

  async disposeSession() {
    requireCapability(this, "supportsSessionLifecycle", "disposeSession")
    throw new SpreadsheetSdkError("disposeSession is not implemented for MCP backend", {
      code: "UNSUPPORTED",
      backend: this.kind,
      operation: "disposeSession"
    })
  }

  async _call(operation, params) {
    try {
      if (typeof this._transport[operation] === "function") {
        return await this._transport[operation](params)
      }
      if (typeof this._transport.invoke === "function") {
        return await this._transport.invoke(operation, params)
      }
      throw new SpreadsheetSdkError(
        `mcp transport does not implement '${operation}' or invoke()`,
        {
          code: "INVALID_ARGUMENT",
          backend: this.kind,
          operation
        }
      )
    } catch (error) {
      throw normalizeBackendError(error, {
        backend: this.kind,
        operation
      })
    }
  }
}

module.exports = {
  McpBackend
}
